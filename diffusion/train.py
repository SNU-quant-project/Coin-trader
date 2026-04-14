"""Self-Supervised 학습 루프: EMA, AMP, Cosine LR, Early Stopping."""

from __future__ import annotations
import os, copy, math, time, logging
from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast

from diffusion.model import ConditionalDiffusionModel
from diffusion.scheduler import DDIMScheduler

logger = logging.getLogger("diffusion.train")


@dataclass
class TrainConfig:
    epochs: int = 100
    lr: float = 1e-4
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    ema_decay: float = 0.9999
    ema_start_epoch: int = 5
    warmup_epochs: int = 5
    min_lr: float = 1e-6
    patience: int = 15
    save_dir: str = "diffusion/checkpoints"
    save_every: int = 10
    log_every: int = 100
    use_amp: bool = True
    device: str = "auto"


class EMAModel:
    def __init__(self, model, decay=0.9999):
        self.decay = decay
        self.ema_model = copy.deepcopy(model).eval().requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        for ema_p, mod_p in zip(self.ema_model.parameters(), model.parameters()):
            ema_p.data.mul_(self.decay).add_(mod_p.data, alpha=1 - self.decay)

    def state_dict(self): return self.ema_model.state_dict()
    def load_state_dict(self, d): self.ema_model.load_state_dict(d)


class DiffusionTrainer:
    def __init__(self, model: ConditionalDiffusionModel, scheduler: DDIMScheduler, config=None):
        self.config = config or TrainConfig()
        self.scheduler = scheduler
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if self.config.device == "auto" else torch.device(self.config.device)
        self.model = model.to(self.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.config.lr, weight_decay=self.config.weight_decay)
        self.criterion = nn.MSELoss()
        self.use_amp = self.config.use_amp and self.device.type == "cuda"
        self.scaler = GradScaler(enabled=self.use_amp)
        self.ema = None
        self.best_val_loss = float("inf")
        self.patience_counter = 0
        self.history = {"train_loss": [], "val_loss": [], "lr": []}

        if self.device.type == "cuda":
            logger.info(f"GPU: {torch.cuda.get_device_name(0)}")

    def _get_lr(self, epoch):
        c = self.config
        if epoch < c.warmup_epochs:
            return c.lr * (epoch + 1) / c.warmup_epochs
        prog = (epoch - c.warmup_epochs) / max(c.epochs - c.warmup_epochs, 1)
        return c.min_lr + 0.5 * (c.lr - c.min_lr) * (1 + math.cos(math.pi * prog))

    def _run_epoch(self, loader, is_train=True, epoch=0):
        model = self.model if is_train else (self.ema.ema_model if self.ema else self.model)
        model.train() if is_train else model.eval()
        total_loss, steps = 0.0, 0

        with torch.set_grad_enabled(is_train):
            for i, batch in enumerate(loader):
                ctx, tgt, cid = batch["context"].to(self.device), batch["target"].to(self.device), batch["coin_id"].to(self.device)
                t = torch.randint(0, self.scheduler.config.num_train_steps, (tgt.size(0),), device=self.device)
                noise = torch.randn_like(tgt)
                noisy_tgt = self.scheduler.add_noise(tgt, noise, t)

                with autocast(enabled=self.use_amp):
                    pred = model(noisy_tgt, t, ctx, cid)
                    loss = self.criterion(pred, noise)

                if is_train:
                    self.optimizer.zero_grad(set_to_none=True)
                    self.scaler.scale(loss).backward()
                    if self.config.grad_clip > 0:
                        self.scaler.unscale_(self.optimizer)
                        nn.utils.clip_grad_norm_(model.parameters(), self.config.grad_clip)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()

                    if self.ema and epoch >= self.config.ema_start_epoch:
                        self.ema.update(self.model)

                total_loss += loss.item()
                steps += 1

                if is_train and (i + 1) % self.config.log_every == 0:
                    logger.info(f"  Epoch {epoch+1} | Step {i+1}/{len(loader)} | Loss: {loss.item():.6f}")

        return total_loss / max(steps, 1)

    def save_checkpoint(self, epoch, val_loss, is_best=False):
        os.makedirs(self.config.save_dir, exist_ok=True)
        ckpt = {
            "epoch": epoch, "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(), "scaler_state_dict": self.scaler.state_dict(),
            "val_loss": val_loss, "best_val_loss": self.best_val_loss, "history": self.history,
            "model_config": self.model.config.__dict__
        }
        if self.ema: ckpt["ema_state_dict"] = self.ema.state_dict()
        
        path = os.path.join(self.config.save_dir, f"checkpoint_epoch{epoch+1}.pt")
        torch.save(ckpt, path)
        if is_best:
            best = os.path.join(self.config.save_dir, "best_model.pt")
            torch.save(ckpt, best)
            logger.info(f"  [BEST] Saved {best}")
        return path

    def load_checkpoint(self, path):
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.scaler.load_state_dict(ckpt["scaler_state_dict"])
        self.best_val_loss, self.history = ckpt["best_val_loss"], ckpt.get("history", self.history)
        if self.ema and "ema_state_dict" in ckpt: self.ema.load_state_dict(ckpt["ema_state_dict"])
        logger.info(f"Loaded {path} (epoch {ckpt['epoch']+1}, val_loss {ckpt['val_loss']:.6f})")
        return ckpt["epoch"]

    def fit(self, train_loader, val_loader, resume_from=None):
        start_epoch = self.load_checkpoint(resume_from) + 1 if resume_from and os.path.exists(resume_from) else 0
        self.ema = EMAModel(self.model, self.config.ema_decay)
        
        logger.info(f"Training: epochs={self.config.epochs}, lr={self.config.lr}, amp={self.use_amp}")
        total_start = time.time()

        for epoch in range(start_epoch, self.config.epochs):
            t0 = time.time()
            lr = self._get_lr(epoch)
            for pg in self.optimizer.param_groups: pg["lr"] = lr

            train_loss = self._run_epoch(train_loader, is_train=True, epoch=epoch)
            val_loss = self._run_epoch(val_loader, is_train=False)

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["lr"].append(lr)

            logger.info(f"Epoch {epoch+1:>3}/{self.config.epochs} | Train: {train_loss:.6f} | Val: {val_loss:.6f} | LR: {lr:.2e} | {time.time()-t0:.1f}s")

            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss, self.patience_counter = val_loss, 0
            else:
                self.patience_counter += 1

            if is_best or (epoch + 1) % self.config.save_every == 0:
                self.save_checkpoint(epoch, val_loss, is_best)

            if self.patience_counter >= self.config.patience:
                logger.info("Early stopping triggered.")
                break
            
            if self.device.type == "cuda": torch.cuda.empty_cache()

        logger.info(f"Done in {(time.time() - total_start)/60:.1f}m. Best val_loss: {self.best_val_loss:.6f}")
        return self.history


if __name__ == "__main__":
    from diffusion.data import DiffusionDataPipeline
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    pipe = DiffusionDataPipeline()
    pipe.prepare()
    tr_loader, va_loader, _ = pipe.get_dataloaders()
    
    from diffusion.model import ModelConfig
    model = ConditionalDiffusionModel(ModelConfig(context_len=60, target_len=15, num_coins=4))
    trainer = DiffusionTrainer(model, DDIMScheduler(), TrainConfig(epochs=2))
    trainer.fit(tr_loader, va_loader)
