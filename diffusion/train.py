"""
=============================================================================
Step 4: Self-Supervised 학습 루프 (train.py)
=============================================================================

핵심 설계 철학:
    1. Self-supervised: 레이블 없이 "노이즈를 맞추는" 방식으로 학습.
       target 시퀀스에 노이즈를 추가하고, 모델이 그 노이즈를 예측하도록 훈련.
    2. EMA (Exponential Moving Average): 추론 시 EMA 가중치를 사용하면
       학습 불안정에 의한 성능 변동이 smoothing되어 일관된 품질.
    3. AMP (Automatic Mixed Precision): fp16 학습으로 메모리 50% 절감, 속도 ~1.5배.
    4. Gradient Clipping: 디퓨전 모델은 gradient 폭발 경향 -> max_norm=1.0.

학습 흐름 (매 배치):
    1. DataLoader에서 (context, target, coin_id) 로드
    2. target에서 랜덤 timestep t ~ Uniform[0, T) 샘플링
    3. 랜덤 노이즈 epsilon ~ N(0, I) 생성
    4. scheduler.add_noise(target, epsilon, t) -> noisy_target (= x_t)
    5. model(noisy_target, t, context, coin_id) -> predicted_noise
    6. loss = MSE(predicted_noise, epsilon)
    7. optimizer.step()

GPU 최적화 (RTX 4080, 16GB):
    - AMP(fp16) + batch_size=256 -> ~4GB VRAM
    - Gradient accumulation 불필요 (충분한 배치 크기)
    - persistent_workers=True -> DataLoader 오버헤드 최소화
"""

from __future__ import annotations

import os
import gc
import copy
import math
import time
import logging
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast

from diffusion.model import ConditionalDiffusionModel, ModelConfig
from diffusion.scheduler import DDIMScheduler, SchedulerConfig

logger = logging.getLogger("diffusion.train")


# ---------------------------------------------------------------------------
#  1. Configuration Dataclass
# ---------------------------------------------------------------------------


@dataclass
class TrainConfig:
    """
    학습 파이프라인의 모든 하이퍼파라미터.

    Key decisions:
        - lr=1e-4: 디퓨전 모델의 사실상 표준 (ADM, DDPM 등).
        - weight_decay=0.01: AdamW에서 정규화. 과적합 방지.
        - ema_decay=0.9999: 매우 느린 EMA. 학습 후반부의 안정적 가중치.
        - grad_clip=1.0: 디퓨전에서 gradient가 t에 따라 크게 변동 -> 클리핑 필수.
        - patience=15: 15 에포크 동안 val loss 개선 없으면 조기 종료.
    """
    # --- 최적화 ---
    epochs: int = 100
    lr: float = 1e-4
    weight_decay: float = 0.01
    grad_clip: float = 1.0          # max gradient norm

    # --- EMA ---
    ema_decay: float = 0.9999       # EMA 감쇠율 (0.9999 = 매우 느린 업데이트)
    ema_start_epoch: int = 5        # EMA 시작 에포크 (초기 불안정 구간 skip)

    # --- LR Schedule ---
    warmup_epochs: int = 5          # LR 워밍업 (0 -> lr까지 선형 증가)
    min_lr: float = 1e-6            # 코사인 스케줄 최소 LR

    # --- Early Stopping ---
    patience: int = 15              # val loss 개선 없으면 중단

    # --- 체크포인트 ---
    save_dir: str = "diffusion/checkpoints"
    save_every: int = 10            # N 에포크마다 체크포인트 저장

    # --- 로깅 ---
    log_every: int = 100            # N 스텝마다 학습 로그 출력

    # --- AMP (Mixed Precision) ---
    use_amp: bool = True            # fp16 학습 활성화

    # --- 디바이스 ---
    device: str = "auto"            # "auto", "cuda", "cpu"


# ---------------------------------------------------------------------------
#  2. EMA (Exponential Moving Average)
# ---------------------------------------------------------------------------


class EMAModel:
    """
    모델 파라미터의 지수이동평균(EMA)을 관리.

    Why EMA?
        - 학습 중 파라미터가 매 스텝 oscillation (진동)함.
        - EMA는 이 진동을 smooth하여 더 안정적인 가중치를 유지.
        - 추론 시 EMA 가중치를 사용하면:
          1. Validation loss가 ~5-10% 낮음 (일반적)
          2. 생성 품질이 더 높고 일관적
          3. 학습 후반부의 과적합 영향을 줄임
        - 디퓨전 모델(DDPM, ADM, Imagen 등)에서 사실상 필수 기법.

    수식:
        theta_ema = decay * theta_ema + (1 - decay) * theta_model

        decay=0.9999이면:
        - 최근 10,000 스텝의 가중 평균에 가까움.
        - 충분히 느려서 학습 진동을 효과적으로 smoothing.
        - 너무 느리면(0.99999) 초기 가중치에 편향.
    """

    def __init__(self, model: nn.Module, decay: float = 0.9999):
        self.decay = decay
        # 모델의 깊은 복사본을 EMA 파라미터로 유지
        self.ema_model = copy.deepcopy(model)
        self.ema_model.eval()
        self.ema_model.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        """
        EMA 파라미터 업데이트.

        theta_ema = decay * theta_ema + (1 - decay) * theta_model
        """
        for ema_param, model_param in zip(
            self.ema_model.parameters(), model.parameters()
        ):
            ema_param.data.mul_(self.decay).add_(
                model_param.data, alpha=1.0 - self.decay
            )

    def state_dict(self) -> dict:
        return self.ema_model.state_dict()

    def load_state_dict(self, state_dict: dict) -> None:
        self.ema_model.load_state_dict(state_dict)


# ---------------------------------------------------------------------------
#  3. Learning Rate Schedule
# ---------------------------------------------------------------------------


def get_lr(
    epoch: int,
    total_epochs: int,
    warmup_epochs: int,
    base_lr: float,
    min_lr: float,
) -> float:
    """
    Warmup + Cosine Annealing LR Schedule.

    Why this schedule?
        Phase 1 - Warmup (0 -> warmup_epochs):
            - 학습 초기에 큰 LR을 쓰면 파라미터가 발산할 수 있음.
            - 0에서 base_lr까지 선형으로 증가시켜 안전하게 시작.
            - 특히 디퓨전 모델은 초기 loss가 매우 불안정.

        Phase 2 - Cosine Decay (warmup_epochs -> total_epochs):
            - LR을 서서히 줄여 fine-grained 최적화.
            - Step decay (계단식)보다 smooth하여 학습 곡선이 안정적.
            - min_lr까지만 줄여 학습이 완전히 멈추지 않도록 함.

    수식 (Phase 2):
        lr = min_lr + 0.5 * (base_lr - min_lr) * (1 + cos(pi * progress))
        where progress = (epoch - warmup) / (total - warmup)
    """
    if epoch < warmup_epochs:
        # Warmup: 선형 증가
        return base_lr * (epoch + 1) / warmup_epochs
    else:
        # Cosine Annealing
        progress = (epoch - warmup_epochs) / max(total_epochs - warmup_epochs, 1)
        return min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * progress))


# ---------------------------------------------------------------------------
#  4. DiffusionTrainer
# ---------------------------------------------------------------------------


class DiffusionTrainer:
    """
    디퓨전 모델의 학습/검증/체크포인트 관리를 수행하는 메인 학습 클래스.

    역할:
        1. 매 에포크마다 train_epoch() -> validate() 반복
        2. EMA 파라미터 업데이트
        3. LR 스케줄 관리 (Warmup + Cosine)
        4. Early stopping (patience 기반)
        5. 체크포인트 저장/로드
        6. AMP (Mixed Precision) 관리

    Design Decisions:
        - Trainer를 별도 클래스로 분리:
          모델/스케줄러와 학습 로직을 분리하면 코드 재사용성 증가.
          예: 다른 모델로 교체해도 Trainer 코드 재사용 가능.
        - loss = MSE(predicted_noise, actual_noise):
          Self-supervised learning의 핵심. 레이블이 필요 없음.
          모델이 "추가된 노이즈가 무엇인지" 맞추도록 학습.
    """

    def __init__(
        self,
        model: ConditionalDiffusionModel,
        scheduler: DDIMScheduler,
        config: Optional[TrainConfig] = None,
    ):
        self.config = config or TrainConfig()
        self.scheduler = scheduler

        # ── 디바이스 설정 ──
        if self.config.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(self.config.device)

        self.model = model.to(self.device)
        logger.info(f"Device: {self.device}")
        if self.device.type == "cuda":
            gpu_name = torch.cuda.get_device_name(0)
            gpu_mem = torch.cuda.get_device_properties(0).total_mem / 1024**3
            logger.info(f"GPU: {gpu_name} ({gpu_mem:.1f} GB)")

        # ── Optimizer: AdamW ──
        # Why AdamW (not Adam)?
        #   - Adam에서 weight decay가 L2 정규화와 비동등한 문제 해결.
        #   - Loshchilov & Hutter (2017): "Decoupled Weight Decay Regularization"
        #   - 디퓨전 모델에서 사실상 표준 optimizer.
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.lr,
            weight_decay=self.config.weight_decay,
            betas=(0.9, 0.999),     # Adam 표준 값
            eps=1e-8,
        )

        # ── Loss Function ──
        # Why MSE (not L1 or Huber)?
        #   - DDPM 원 논문에서 MSE = simplified ELBO로 증명.
        #   - L1은 노이즈가 작은 스텝에서 gradient가 불연속.
        #   - Huber는 좋은 대안이지만, MSE가 표준이고 충분히 동작.
        self.criterion = nn.MSELoss()

        # ── AMP (Mixed Precision) ──
        # Why AMP?
        #   - forward/backward를 fp16으로 수행 -> VRAM 50% 절감.
        #   - RTX 4080의 Tensor Core 활용 -> 학습 속도 ~1.5배.
        #   - GradScaler로 fp16의 underflow 방지.
        self.use_amp = self.config.use_amp and self.device.type == "cuda"
        self.scaler = GradScaler(enabled=self.use_amp)

        # ── EMA ──
        self.ema: Optional[EMAModel] = None

        # ── Early Stopping 상태 ──
        self.best_val_loss = float("inf")
        self.patience_counter = 0

        # ── 학습 히스토리 ──
        self.history: Dict[str, list] = {
            "train_loss": [],
            "val_loss": [],
            "lr": [],
        }

    # -------------------------------------------------------------------
    #  학습 1 에포크
    # -------------------------------------------------------------------

    def train_epoch(self, train_loader: DataLoader, epoch: int) -> float:
        """
        1 에포크 학습.

        매 배치에서:
            1. target에 랜덤 노이즈 추가 -> noisy_target
            2. 모델이 노이즈 예측
            3. MSE loss로 역전파

        Args:
            train_loader: 학습 DataLoader
            epoch: 현재 에포크 번호

        Returns:
            평균 학습 loss
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for step, batch in enumerate(train_loader):
            # ── 데이터 로드 ──
            context = batch["context"].to(self.device)       # (B, 60, 5)
            target = batch["target"].to(self.device)         # (B, 15, 5)
            coin_id = batch["coin_id"].to(self.device)       # (B,)
            B = target.size(0)

            # ── 1) 랜덤 타임스텝 샘플링 ──
            # Why uniform sampling?
            #   - 모든 노이즈 레벨을 균일하게 학습.
            #   - t=0 (거의 원본)부터 t=T-1 (순수 노이즈)까지 고르게.
            #   - importance sampling도 가능하지만 uniform이 기본이고 충분.
            timesteps = torch.randint(
                0, self.scheduler.config.num_train_steps, (B,),
                device=self.device,
            )

            # ── 2) 랜덤 노이즈 생성 ──
            noise = torch.randn_like(target)

            # ── 3) Forward diffusion: x_0 -> x_t ──
            noisy_target = self.scheduler.add_noise(target, noise, timesteps)

            # ── 4) 모델 forward + loss 계산 ──
            # AMP context: forward/loss를 fp16으로 수행
            with autocast(enabled=self.use_amp):
                noise_pred = self.model(noisy_target, timesteps, context, coin_id)
                loss = self.criterion(noise_pred, noise)

            # ── 5) 역전파 ──
            self.optimizer.zero_grad(set_to_none=True)
            # Why set_to_none=True? grad를 None으로 설정하면
            # 0 텐서를 할당하는 것보다 메모리 효율적.
            self.scaler.scale(loss).backward()

            # ── 6) Gradient Clipping ──
            # Why? 디퓨전 모델은 t에 따라 gradient 크기가 크게 변동.
            # t~T-1 (많은 노이즈)에서 gradient가 폭발할 수 있음.
            # max_norm=1.0으로 클리핑하면 안정적.
            if self.config.grad_clip > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.grad_clip
                )

            self.scaler.step(self.optimizer)
            self.scaler.update()

            # ── 7) EMA 업데이트 ──
            if self.ema is not None and epoch >= self.config.ema_start_epoch:
                self.ema.update(self.model)

            # ── 로깅 ──
            total_loss += loss.item()
            num_batches += 1

            if (step + 1) % self.config.log_every == 0:
                avg = total_loss / num_batches
                logger.info(
                    f"  Epoch {epoch+1} | Step {step+1:>5}/{len(train_loader)} | "
                    f"Loss: {loss.item():.6f} | Avg: {avg:.6f}"
                )

        return total_loss / max(num_batches, 1)

    # -------------------------------------------------------------------
    #  검증
    # -------------------------------------------------------------------

    @torch.no_grad()
    def validate(self, val_loader: DataLoader) -> float:
        """
        검증 루프. 학습과 동일하지만 gradient 계산/업데이트 없음.

        Why EMA model for validation?
            - EMA 가중치가 활성화되어 있으면 EMA 모델로 검증.
            - EMA는 학습 진동을 smooth하여 더 안정적인 val loss를 보여줌.
            - 최종 추론도 EMA 가중치로 수행하므로, 검증과 일관.

        Returns:
            평균 검증 loss
        """
        eval_model = (
            self.ema.ema_model if self.ema is not None else self.model
        )
        eval_model.eval()

        total_loss = 0.0
        num_batches = 0

        for batch in val_loader:
            context = batch["context"].to(self.device)
            target = batch["target"].to(self.device)
            coin_id = batch["coin_id"].to(self.device)
            B = target.size(0)

            timesteps = torch.randint(
                0, self.scheduler.config.num_train_steps, (B,),
                device=self.device,
            )
            noise = torch.randn_like(target)
            noisy_target = self.scheduler.add_noise(target, noise, timesteps)

            with autocast(enabled=self.use_amp):
                noise_pred = eval_model(noisy_target, timesteps, context, coin_id)
                loss = self.criterion(noise_pred, noise)

            total_loss += loss.item()
            num_batches += 1

        return total_loss / max(num_batches, 1)

    # -------------------------------------------------------------------
    #  체크포인트 저장/로드
    # -------------------------------------------------------------------

    def save_checkpoint(self, epoch: int, val_loss: float, is_best: bool = False) -> str:
        """
        학습 상태 전체를 체크포인트로 저장.

        저장 내용:
            - 모델 가중치
            - EMA 가중치
            - Optimizer 상태
            - Scaler 상태
            - 학습 히스토리
            - 에포크/val_loss
        """
        os.makedirs(self.config.save_dir, exist_ok=True)

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scaler_state_dict": self.scaler.state_dict(),
            "val_loss": val_loss,
            "best_val_loss": self.best_val_loss,
            "history": self.history,
            "model_config": self.model.config.__dict__,
        }

        if self.ema is not None:
            checkpoint["ema_state_dict"] = self.ema.state_dict()

        # 정기 체크포인트
        path = os.path.join(self.config.save_dir, f"checkpoint_epoch{epoch+1}.pt")
        torch.save(checkpoint, path)

        # 최고 성능 체크포인트
        if is_best:
            best_path = os.path.join(self.config.save_dir, "best_model.pt")
            torch.save(checkpoint, best_path)
            logger.info(f"  [BEST] Saved to {best_path}")

        return path

    def load_checkpoint(self, path: str) -> int:
        """
        체크포인트에서 학습 상태 복원.

        Returns:
            복원된 에포크 번호
        """
        checkpoint = torch.load(path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scaler.load_state_dict(checkpoint["scaler_state_dict"])
        self.best_val_loss = checkpoint["best_val_loss"]
        self.history = checkpoint.get("history", self.history)

        if self.ema is not None and "ema_state_dict" in checkpoint:
            self.ema.load_state_dict(checkpoint["ema_state_dict"])

        epoch = checkpoint["epoch"]
        logger.info(f"Checkpoint loaded: epoch={epoch+1}, val_loss={checkpoint['val_loss']:.6f}")
        return epoch

    # -------------------------------------------------------------------
    #  전체 학습 루프
    # -------------------------------------------------------------------

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        resume_from: Optional[str] = None,
    ) -> Dict[str, list]:
        """
        전체 학습 파이프라인 실행.

        흐름:
            for each epoch:
                1. LR 업데이트 (Warmup + Cosine)
                2. train_epoch()
                3. validate()
                4. EMA 업데이트 (epoch >= ema_start_epoch)
                5. Early stopping 체크
                6. 체크포인트 저장

        Args:
            train_loader: 학습 DataLoader
            val_loader: 검증 DataLoader
            resume_from: 체크포인트 경로 (이어 학습)

        Returns:
            학습 히스토리 dict {"train_loss": [...], "val_loss": [...], "lr": [...]}
        """
        start_epoch = 0

        # ── 이어 학습 ──
        if resume_from and os.path.exists(resume_from):
            start_epoch = self.load_checkpoint(resume_from) + 1

        # ── EMA 초기화 ──
        self.ema = EMAModel(self.model, decay=self.config.ema_decay)

        logger.info("=" * 60)
        logger.info("Training started")
        logger.info(f"  Epochs    : {start_epoch + 1} -> {self.config.epochs}")
        logger.info(f"  LR        : {self.config.lr}")
        logger.info(f"  AMP       : {self.use_amp}")
        logger.info(f"  EMA decay : {self.config.ema_decay}")
        logger.info(f"  Patience  : {self.config.patience}")
        logger.info(f"  Device    : {self.device}")
        logger.info("=" * 60)

        total_start = time.time()

        for epoch in range(start_epoch, self.config.epochs):
            epoch_start = time.time()

            # ── 1) LR 업데이트 ──
            lr = get_lr(
                epoch, self.config.epochs, self.config.warmup_epochs,
                self.config.lr, self.config.min_lr,
            )
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = lr

            # ── 2) 학습 ──
            train_loss = self.train_epoch(train_loader, epoch)

            # ── 3) 검증 ──
            val_loss = self.validate(val_loader)

            # ── 히스토리 기록 ──
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["lr"].append(lr)

            epoch_time = time.time() - epoch_start

            # ── 로깅 ──
            logger.info(
                f"Epoch {epoch+1:>3}/{self.config.epochs} | "
                f"Train: {train_loss:.6f} | Val: {val_loss:.6f} | "
                f"LR: {lr:.2e} | Time: {epoch_time:.1f}s"
            )

            # ── 4) Best model 체크 & Early Stopping ──
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                logger.info(
                    f"  >> New best val_loss: {val_loss:.6f}"
                )
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.config.patience:
                    logger.info(
                        f"  >> Early stopping triggered "
                        f"(patience={self.config.patience})"
                    )
                    self.save_checkpoint(epoch, val_loss, is_best=False)
                    break

            # ── 5) 체크포인트 저장 ──
            if is_best or (epoch + 1) % self.config.save_every == 0:
                self.save_checkpoint(epoch, val_loss, is_best=is_best)

            # ── GPU 메모리 정리 ──
            if self.device.type == "cuda":
                torch.cuda.empty_cache()

        total_time = time.time() - total_start
        logger.info("=" * 60)
        logger.info(f"Training completed in {total_time / 60:.1f} minutes")
        logger.info(f"Best val_loss: {self.best_val_loss:.6f}")
        logger.info("=" * 60)

        return self.history


# ---------------------------------------------------------------------------
#  5. 메인 실행 (전체 파이프라인 통합 테스트)
# ---------------------------------------------------------------------------


def main():
    """
    data.py -> model.py -> scheduler.py -> train.py 전체 파이프라인 실행.

    순서:
        1. DiffusionDataPipeline으로 CSV 데이터 로드 & 전처리
        2. DataLoader 생성 (train / val / test)
        3. ConditionalDiffusionModel 생성
        4. DDIMScheduler 생성
        5. DiffusionTrainer로 학습 실행
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    # ── 1) 데이터 파이프라인 ──
    from diffusion.data import DiffusionDataPipeline, DiffusionDataConfig

    data_config = DiffusionDataConfig()
    pipeline = DiffusionDataPipeline(data_config)
    pipeline.prepare(force_rebuild=False)
    print(pipeline.summary())

    train_loader, val_loader, test_loader = pipeline.get_dataloaders()

    logger.info(
        f"DataLoaders ready: "
        f"train={len(train_loader.dataset):,} | "
        f"val={len(val_loader.dataset):,} | "
        f"test={len(test_loader.dataset):,}"
    )

    # ── 2) 모델 생성 ──
    model_config = ModelConfig(
        feature_dim=data_config.feature_dim,
        context_len=data_config.context_len,
        target_len=data_config.target_len,
        num_coins=data_config.num_coins,
    )
    model = ConditionalDiffusionModel(model_config)

    params = model.get_num_params()
    logger.info(f"Model parameters: {params}")

    # ── 3) 스케줄러 생성 ──
    scheduler_config = SchedulerConfig()
    scheduler = DDIMScheduler(scheduler_config)

    # ── 4) 학습 ──
    train_config = TrainConfig(
        epochs=100,
        lr=1e-4,
        save_dir="diffusion/checkpoints",
    )
    trainer = DiffusionTrainer(model, scheduler, train_config)

    history = trainer.fit(train_loader, val_loader)

    # ── 5) 학습 결과 요약 ──
    logger.info("\n[Training History]")
    for epoch_idx in range(len(history["train_loss"])):
        logger.info(
            f"  Epoch {epoch_idx+1:>3} | "
            f"Train: {history['train_loss'][epoch_idx]:.6f} | "
            f"Val: {history['val_loss'][epoch_idx]:.6f} | "
            f"LR: {history['lr'][epoch_idx]:.2e}"
        )


if __name__ == "__main__":
    main()
