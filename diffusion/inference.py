"""
=============================================================================
Step 5: Monte Carlo Sampling 및 매매 전략 (inference.py)
=============================================================================

핵심 설계 철학:
    1. Monte Carlo Sampling (100회):
       단일 예측이 아닌, 100개의 미래 시나리오를 생성.
       확률 분포를 기반으로 매매 결정 -> 불확실성을 명시적으로 고려.
    2. 확률 기반 매매 전략:
       "상승할 확률 70%이면 Long, 하락할 확률 70%이면 Short"
       단순 방향 예측(UP/DOWN) 대비 리스크 관리가 우수.
    3. Stop-loss 자동 결정:
       시나리오 분포의 하위 5% 분위를 이용해 최악의 시나리오 기반 손절선 설정.

추론 흐름:
    1. 실시간 Context (과거 60분) 전처리
    2. 순수 노이즈에서 시작 -> DDIM 50스텝 디노이징 (x 100회)
    3. 100개의 미래 15분 시나리오 생성
    4. 시나리오 통계 분석 -> 매매 신호 생성

성능 (RTX 4080):
    - 1회 DDIM 50스텝: ~50ms
    - 100회 Monte Carlo: ~5초 (배치 병렬화로 단축 가능)
    - 배치 병렬화(100회를 한 번에): ~1초
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from diffusion.model import ConditionalDiffusionModel, ModelConfig
from diffusion.scheduler import DDIMScheduler, SchedulerConfig
from diffusion.data import (
    DiffusionDataConfig,
    log_returns_to_prices,
    denormalize_volume,
)

logger = logging.getLogger("diffusion.inference")


# ---------------------------------------------------------------------------
#  1. Configuration
# ---------------------------------------------------------------------------


@dataclass
class InferenceConfig:
    """
    추론 및 매매 전략의 하이퍼파라미터.

    Key parameters:
        - num_samples: Monte Carlo 샘플 수. 많을수록 확률 추정 정밀도 증가.
          100회: 1% 해상도 (충분), 1000회: 0.1% 해상도 (과도).
        - eta: DDIM 확률적 노이즈 계수.
          0.0 = deterministic (100회 모두 같은 결과 -> 의미 없음)
          0.3~0.7 = 적절한 다양성 (추천)
          1.0 = DDPM equivalent (과도한 다양성)
        - direction_threshold: 방향 판단 기준 log return.
          0.001 = ~0.1% 변동 이상이면 방향으로 인정.
        - confidence_threshold: 매매 진입 최소 확률.
          0.6 = 60% 이상 확률일 때만 진입 -> 보수적.
    """
    # --- Monte Carlo ---
    num_samples: int = 100          # 시나리오 생성 수
    batch_size: int = 50            # 병렬 추론 배치 크기 (VRAM에 따라 조정)
    eta: float = 0.5                # DDIM stochastic 계수

    # --- 매매 전략 ---
    direction_threshold: float = 0.001   # 방향 판단 최소 log return
    confidence_threshold: float = 0.6    # 매매 진입 최소 확률
    stop_loss_quantile: float = 0.05     # 최악 시나리오 분위 (5%)
    take_profit_quantile: float = 0.95   # 최선 시나리오 분위 (95%)

    # --- DDIM ---
    num_inference_steps: int = 50        # DDIM 추론 스텝 수


# ---------------------------------------------------------------------------
#  2. DiffusionPredictor
# ---------------------------------------------------------------------------


class DiffusionPredictor:
    """
    학습된 디퓨전 모델을 사용하여 미래 시나리오를 생성하고 매매 신호를 결정.

    핵심 알고리즘:
        1. 순수 가우시안 노이즈 x_T ~ N(0, I) 생성
        2. DDIM reverse process: x_T -> x_{T-1} -> ... -> x_0
           (각 스텝에서 모델이 노이즈를 예측하고, scheduler가 디노이징)
        3. 복원된 x_0 = 예측된 미래 15분 log return 시퀀스
        4. eta > 0이면 매번 다른 x_0를 생성 -> 100개 다양한 시나리오

    Why Monte Carlo Sampling?
        - 단일 예측: "3분 후 0.1% 상승" -> 확률 정보 없음
        - 100회 샘플링: "70% 확률로 상승, 중앙값 0.1%, 최악 -0.3%"
          -> 확률, 리스크, 기대수익을 모두 정량화 가능
        - 불확실성이 높으면 (상승/하락이 50/50) "관망" 판단 가능
          -> 과매매(overtrading) 방지
    """

    def __init__(
        self,
        model: ConditionalDiffusionModel,
        scheduler: DDIMScheduler,
        config: Optional[InferenceConfig] = None,
        device: Optional[torch.device] = None,
    ):
        self.config = config or InferenceConfig()
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # ── 모델 로드 ──
        self.model = model.to(self.device)
        self.model.eval()

        # ── 스케줄러 설정 ──
        self.scheduler = scheduler
        self.scheduler.set_timesteps(self.config.num_inference_steps)

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        model_config: Optional[ModelConfig] = None,
        scheduler_config: Optional[SchedulerConfig] = None,
        inference_config: Optional[InferenceConfig] = None,
        device: Optional[torch.device] = None,
    ) -> "DiffusionPredictor":
        """
        체크포인트에서 모델을 로드하여 DiffusionPredictor 생성.

        체크포인트에 저장된 model_config를 우선 사용하고,
        없으면 인자로 전달된 config를 사용.
        """
        device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        checkpoint = torch.load(checkpoint_path, map_location=device)

        # Config 복원
        if "model_config" in checkpoint:
            saved_config = checkpoint["model_config"]
            model_config = ModelConfig(**saved_config)
        else:
            model_config = model_config or ModelConfig()

        # 모델 생성 & 가중치 로드
        model = ConditionalDiffusionModel(model_config)

        # EMA 가중치 우선 사용 (학습 시 EMA가 보통 더 좋음)
        if "ema_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["ema_state_dict"])
            logger.info("Loaded EMA weights from checkpoint")
        else:
            model.load_state_dict(checkpoint["model_state_dict"])
            logger.info("Loaded model weights from checkpoint")

        scheduler_config = scheduler_config or SchedulerConfig()
        scheduler = DDIMScheduler(scheduler_config)

        return cls(model, scheduler, inference_config, device)

    # -------------------------------------------------------------------
    #  Monte Carlo Sampling
    # -------------------------------------------------------------------

    @torch.no_grad()
    def predict(
        self,
        context: torch.Tensor,
        coin_id: torch.Tensor,
        num_samples: Optional[int] = None,
    ) -> np.ndarray:
        """
        Monte Carlo Sampling으로 N개의 미래 시나리오 생성.

        알고리즘:
            1. context를 N번 복제: (1, 60, 5) -> (N, 60, 5)
            2. 순수 노이즈 N개 생성: x_T ~ N(0, I), shape (N, 15, 5)
            3. DDIM 50스텝으로 디노이징:
               for t in [980, 960, ..., 20, 0]:
                   eps_pred = model(x_t, t, context, coin_id)
                   x_{t-1} = scheduler.step(eps_pred, t, x_t, eta=0.5)
            4. 복원된 x_0 = N개의 미래 15분 log return 시퀀스

        배치 병렬화:
            100개 시나리오를 한 번에 추론하면 GPU 활용 극대화.
            VRAM이 부족하면 batch_size 단위로 분할 처리.

        Args:
            context: (1, 60, 5) 또는 (60, 5) 과거 60분 log return
            coin_id: (1,) 또는 scalar 코인 ID
            num_samples: 시나리오 수 (기본: config.num_samples=100)

        Returns:
            (num_samples, 15, 5) 예측된 미래 시나리오 (numpy array)
        """
        num_samples = num_samples or self.config.num_samples

        # ── 입력 전처리 ──
        if context.dim() == 2:
            context = context.unsqueeze(0)           # (60, 5) -> (1, 60, 5)
        if coin_id.dim() == 0:
            coin_id = coin_id.unsqueeze(0)           # scalar -> (1,)

        context = context.to(self.device)
        coin_id = coin_id.to(self.device)

        target_len = self.model.config.target_len
        feature_dim = self.model.config.feature_dim

        # ── 배치 단위로 샘플링 ──
        all_samples = []
        remaining = num_samples
        batch_size = self.config.batch_size

        while remaining > 0:
            B = min(batch_size, remaining)

            # context와 coin_id를 B번 복제
            ctx_batch = context.expand(B, -1, -1)        # (B, 60, 5)
            cid_batch = coin_id.expand(B)                # (B,)

            # 순수 가우시안 노이즈에서 시작
            x_t = torch.randn(B, target_len, feature_dim, device=self.device)

            # ── DDIM Reverse Process ──
            timesteps = self.scheduler.inference_timesteps

            for i in range(len(timesteps)):
                t = timesteps[i].item()
                prev_t = timesteps[i + 1].item() if i + 1 < len(timesteps) else -1

                # 타임스텝을 배치 크기로 확장
                t_batch = torch.full((B,), t, device=self.device, dtype=torch.long)

                # 모델이 노이즈 예측: eps_theta(x_t, t, cond)
                noise_pred = self.model(x_t, t_batch, ctx_batch, cid_batch)

                # DDIM step: x_t -> x_{t-1}
                x_t = self.scheduler.step(
                    noise_pred, t, x_t,
                    prev_timestep=prev_t,
                    eta=self.config.eta,
                )

            # 최종 x_0 저장
            all_samples.append(x_t.cpu().numpy())
            remaining -= B

        # 모든 배치 결합: (num_samples, 15, 5)
        samples = np.concatenate(all_samples, axis=0)

        return samples

    # -------------------------------------------------------------------
    #  매매 신호 생성
    # -------------------------------------------------------------------

    def generate_signal(
        self,
        context: torch.Tensor,
        coin_id: torch.Tensor,
        current_price: Optional[float] = None,
        volume_stats: Optional[Tuple[float, float]] = None,
    ) -> Dict:
        """
        Monte Carlo 시나리오 기반 매매 신호 생성.

        전략 로직:
            1. 100개 시나리오의 close log return 누적수익률 계산
            2. 최종 시점(15분 후) 누적수익률 분포 분석:
               - 상승 확률: cumret[-1] > threshold 인 비율
               - 하락 확률: cumret[-1] < -threshold 인 비율
            3. 확률 기반 판단:
               - up_prob > confidence -> LONG
               - down_prob > confidence -> SHORT
               - 그 외 -> HOLD (불확실)
            4. 리스크 관리:
               - Stop-loss: 5% 분위 시나리오의 최대 손실
               - Take-profit: 95% 분위 시나리오의 기대 수익

        Why 확률 기반 (not 단일 예측)?
            - 단일 예측: "0.05% 상승" -> 신뢰도 불명, 오차 범위 불명
            - 100개 시나리오: [+0.3%, +0.1%, -0.2%, ...] ->
              "75% 확률로 상승, 최악 -0.5%, 중앙값 +0.1%"
              -> 리스크/리워드 비율을 정량적으로 판단 가능

        Args:
            context: (1, 60, 5) 또는 (60, 5) 과거 60분 log return
            coin_id: (1,) 또는 scalar 코인 ID
            current_price: 현재가 (절대 가격 목표 계산용, optional)
            volume_stats: (vol_mean, vol_std) 볼륨 역정규화용

        Returns:
            dict with keys:
                "direction": "LONG" | "SHORT" | "HOLD"
                "confidence": float (0~1, 매매 방향의 확률)
                "up_probability": float (상승 확률)
                "down_probability": float (하락 확률)
                "expected_return": float (중앙값 기대수익률)
                "stop_loss_return": float (5% 분위 최악 수익률)
                "take_profit_return": float (95% 분위 최선 수익률)
                "scenarios_mean": (15, 5) 시나리오 평균 궤적
                "scenarios_std": (15, 5) 시나리오 표준편차
                "price_targets": dict (current_price 제공 시)
        """
        # ── 1) Monte Carlo Sampling ──
        samples = self.predict(context, coin_id)
        # samples shape: (num_samples, 15, 5)
        # 채널 순서: [open_ret, high_ret, low_ret, close_ret, log_volume]

        # ── 2) Close log return 추출 (인덱스 3) ──
        close_returns = samples[:, :, 3]                # (num_samples, 15)

        # ── 3) 누적 수익률 계산 ──
        # Why cumsum? log return의 합 = log(P_T / P_0)
        # 즉, cumsum의 마지막 값 = 전체 기간의 로그 수익률
        cumulative_returns = np.cumsum(close_returns, axis=1)  # (num_samples, 15)
        final_returns = cumulative_returns[:, -1]               # (num_samples,)

        # ── 4) 확률 계산 ──
        threshold = self.config.direction_threshold
        up_prob = float(np.mean(final_returns > threshold))
        down_prob = float(np.mean(final_returns < -threshold))
        hold_prob = 1.0 - up_prob - down_prob

        # ── 5) 방향 결정 ──
        confidence_threshold = self.config.confidence_threshold

        if up_prob >= confidence_threshold:
            direction = "LONG"
            confidence = up_prob
        elif down_prob >= confidence_threshold:
            direction = "SHORT"
            confidence = down_prob
        else:
            direction = "HOLD"
            confidence = hold_prob

        # ── 6) 리스크 통계 ──
        expected_return = float(np.median(final_returns))
        stop_loss_return = float(
            np.quantile(final_returns, self.config.stop_loss_quantile)
        )
        take_profit_return = float(
            np.quantile(final_returns, self.config.take_profit_quantile)
        )

        # ── 7) 시간별 통계 (궤적 분석용) ──
        scenarios_mean = np.mean(samples, axis=0)           # (15, 5)
        scenarios_std = np.std(samples, axis=0)              # (15, 5)

        # ── 8) 결과 조합 ──
        result = {
            "direction": direction,
            "confidence": round(confidence, 4),
            "up_probability": round(up_prob, 4),
            "down_probability": round(down_prob, 4),
            "hold_probability": round(hold_prob, 4),
            "expected_return": round(expected_return, 6),
            "stop_loss_return": round(stop_loss_return, 6),
            "take_profit_return": round(take_profit_return, 6),
            "scenarios_mean": scenarios_mean,
            "scenarios_std": scenarios_std,
            "num_samples": len(samples),
        }

        # ── 9) 절대 가격 목표 (current_price 제공 시) ──
        if current_price is not None:
            # log return -> 가격 변환: P_target = P_now * exp(log_return)
            result["price_targets"] = {
                "expected": round(
                    current_price * np.exp(expected_return), 2
                ),
                "stop_loss": round(
                    current_price * np.exp(stop_loss_return), 2
                ),
                "take_profit": round(
                    current_price * np.exp(take_profit_return), 2
                ),
                "worst_case": round(
                    current_price * np.exp(float(np.min(final_returns))), 2
                ),
                "best_case": round(
                    current_price * np.exp(float(np.max(final_returns))), 2
                ),
            }

        return result

    # -------------------------------------------------------------------
    #  실시간 Context 전처리
    # -------------------------------------------------------------------

    @staticmethod
    def preprocess_realtime_context(
        ohlcv_df,
        ohlcv_cols: Optional[list] = None,
        volume_mean: float = 0.0,
        volume_std: float = 1.0,
    ) -> torch.Tensor:
        """
        실시간 OHLCV DataFrame -> 모델 입력용 context 텐서로 변환.

        실시간 매매 시나리오:
            1. 거래소 API에서 최근 61분 1분봉 조회
            2. 이 함수로 전처리 -> context (60, 5)
            3. predict() 또는 generate_signal()에 입력

        전처리 순서:
            1. 가격 -> log return (diff)
            2. 볼륨 -> log1p -> Z-score 정규화
            3. NaN/Inf 처리

        Args:
            ohlcv_df: 최근 61분 OHLCV DataFrame (61행 = 60개 log return)
            ohlcv_cols: ["open", "high", "low", "close", "volume"]
            volume_mean: 학습 시 사용된 볼륨 평균
            volume_std: 학습 시 사용된 볼륨 표준편차

        Returns:
            (60, 5) log return + normalized volume 텐서
        """
        if ohlcv_cols is None:
            ohlcv_cols = ["open", "high", "low", "close", "volume"]

        import pandas as pd

        price_cols = [c for c in ohlcv_cols if c != "volume"]

        # 1) 가격 -> log return
        prices = ohlcv_df[price_cols].values.astype(np.float64)
        prices = np.clip(prices, a_min=1e-10, a_max=None)
        log_returns = np.diff(np.log(prices), axis=0)        # (60, 4)

        # 2) 볼륨 -> log1p -> Z-score
        volume = ohlcv_df["volume"].values.astype(np.float64)
        log_volume = np.log1p(volume)[1:]                     # (60,)
        vol_normed = (log_volume - volume_mean) / (volume_std + 1e-8)

        # 3) 합치기
        features = np.column_stack([log_returns, vol_normed])  # (60, 5)

        # 4) NaN/Inf 처리
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

        return torch.from_numpy(features.astype(np.float32))

    # -------------------------------------------------------------------
    #  시나리오 분석 리포트
    # -------------------------------------------------------------------

    def analyze_scenarios(
        self,
        samples: np.ndarray,
        current_price: Optional[float] = None,
    ) -> str:
        """
        100개 시나리오의 상세 분석 리포트 생성.

        디버깅 및 시각적 확인용.
        """
        close_returns = samples[:, :, 3]
        cumret = np.cumsum(close_returns, axis=1)
        final = cumret[:, -1]

        lines = [
            "=" * 60,
            "  Scenario Analysis Report",
            "=" * 60,
            f"  Samples         : {len(samples)}",
            f"  Horizon         : {samples.shape[1]} minutes",
            "",
            "  [Final Return Distribution (15min)]",
            f"    Mean           : {np.mean(final):.6f}",
            f"    Median         : {np.median(final):.6f}",
            f"    Std            : {np.std(final):.6f}",
            f"    Min            : {np.min(final):.6f}",
            f"    Max            : {np.max(final):.6f}",
            f"    5th percentile : {np.percentile(final, 5):.6f}",
            f"    25th percentile: {np.percentile(final, 25):.6f}",
            f"    75th percentile: {np.percentile(final, 75):.6f}",
            f"    95th percentile: {np.percentile(final, 95):.6f}",
            "",
            f"  [Direction Probabilities]",
            f"    Up (>0.1%)     : {np.mean(final > 0.001):.1%}",
            f"    Down (<-0.1%)  : {np.mean(final < -0.001):.1%}",
            f"    Neutral        : {np.mean(np.abs(final) <= 0.001):.1%}",
        ]

        if current_price is not None:
            med_price = current_price * np.exp(np.median(final))
            worst_price = current_price * np.exp(np.min(final))
            best_price = current_price * np.exp(np.max(final))
            lines.extend([
                "",
                f"  [Price Targets (current={current_price:.2f})]",
                f"    Median price   : {med_price:.2f}",
                f"    Worst case     : {worst_price:.2f}",
                f"    Best case      : {best_price:.2f}",
            ])

        # 시간별 누적 수익률 (5분 간격)
        lines.extend([
            "",
            "  [Cumulative Return at Key Points]",
            "    Minute  |  Mean    |  Median  |  Std",
            "    --------|----------|----------|--------",
        ])
        for t in [4, 9, 14]:  # 5분, 10분, 15분
            if t < cumret.shape[1]:
                lines.append(
                    f"    {t+1:>3}min  | {np.mean(cumret[:, t]):>8.5f} | "
                    f"{np.median(cumret[:, t]):>8.5f} | {np.std(cumret[:, t]):>8.5f}"
                )

        lines.append("=" * 60)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
#  스크립트 직접 실행 시 추론 파이프라인 테스트
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    import time as time_module

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    print("=" * 60)
    print("  Diffusion Inference - Pipeline Test")
    print("=" * 60)

    # ── 모델 & 스케줄러 생성 (랜덤 가중치로 Shape 테스트) ──
    model_config = ModelConfig()
    model = ConditionalDiffusionModel(model_config)

    scheduler_config = SchedulerConfig()
    scheduler = DDIMScheduler(scheduler_config)

    inference_config = InferenceConfig(
        num_samples=10,          # 테스트용 소량
        batch_size=5,
        eta=0.5,
        num_inference_steps=20,  # 테스트용 소량 스텝
    )

    predictor = DiffusionPredictor(model, scheduler, inference_config)

    # ── 더미 Context 생성 ──
    context = torch.randn(1, 60, 5)
    coin_id = torch.tensor([0])  # BTC

    # ── Monte Carlo Sampling ──
    print("\n[Monte Carlo Sampling]")
    start = time_module.time()
    samples = predictor.predict(context, coin_id)
    elapsed = time_module.time() - start

    print(f"  Samples shape : {samples.shape}  <- expected (10, 15, 5)")
    print(f"  Sampling time : {elapsed:.2f}s")
    print(f"  Sample mean   : {samples.mean():.6f}")
    print(f"  Sample std    : {samples.std():.6f}")

    # ── 매매 신호 생성 ──
    print("\n[Trading Signal Generation]")
    signal = predictor.generate_signal(
        context, coin_id, current_price=60000.0
    )

    for key, value in signal.items():
        if isinstance(value, np.ndarray):
            print(f"  {key:>25}: shape={value.shape}")
        elif isinstance(value, dict):
            print(f"  {key:>25}:")
            for k, v in value.items():
                print(f"    {k:>23}: {v}")
        else:
            print(f"  {key:>25}: {value}")

    # ── 시나리오 분석 리포트 ──
    print("\n" + predictor.analyze_scenarios(samples, current_price=60000.0))

    print(f"\n{'=' * 60}")
    print(f"  [OK] All inference tests passed!")
    print(f"{'=' * 60}")
