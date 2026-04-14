"""DDIM Noise Scheduler: cosine beta schedule, forward/reverse diffusion."""

from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Optional
import numpy as np
import torch


@dataclass
class SchedulerConfig:
    num_train_steps: int = 1000
    schedule_type: str = "cosine"  # "cosine" or "linear"
    beta_start: float = 0.0001
    beta_end: float = 0.02
    clip_sample: bool = True
    clip_sample_range: float = 5.0
    num_inference_steps: int = 50


class DDIMScheduler:
    def __init__(self, config=None):
        self.config = config or SchedulerConfig()
        T = self.config.num_train_steps

        if self.config.schedule_type == "cosine":
            # Cosine schedule (Improved DDPM)
            s = 0.008
            t = torch.linspace(0, T, T + 1)
            f = torch.cos(((t / T) + s) / (1 + s) * math.pi * 0.5) ** 2
            ac = f / f[0]
            self.betas = torch.clamp(1 - ac[1:] / ac[:-1], 1e-4, 0.999)
        else:
            self.betas = torch.linspace(self.config.beta_start, self.config.beta_end, T)

        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, 0)
        self.sqrt_ac = torch.sqrt(self.alphas_cumprod)
        self.sqrt_1mac = torch.sqrt(1.0 - self.alphas_cumprod)
        self._inf_ts = None

    def add_noise(self, x0, noise, t):
        """Forward diffusion: x_t = sqrt(a_bar)*x0 + sqrt(1-a_bar)*noise."""
        dev = x0.device
        sa = self.sqrt_ac[t].to(dev)
        sma = self.sqrt_1mac[t].to(dev)
        while sa.dim() < x0.dim(): sa, sma = sa.unsqueeze(-1), sma.unsqueeze(-1)
        return sa * x0 + sma * noise

    def set_timesteps(self, n=None):
        n = n or self.config.num_inference_steps
        ratio = self.config.num_train_steps / n
        self._inf_ts = torch.from_numpy(np.round(np.arange(n) * ratio).astype(np.int64)[::-1].copy())

    @property
    def inference_timesteps(self):
        if self._inf_ts is None: self.set_timesteps()
        return self._inf_ts

    def step(self, noise_pred, t, sample, prev_t=None, eta=0.0):
        """DDIM reverse: x_t -> x_{t-1}."""
        dev = sample.device
        ab_t = self.alphas_cumprod[t].to(dev)
        ab_prev = self.alphas_cumprod[prev_t].to(dev) if prev_t is not None and prev_t >= 0 else torch.tensor(1.0, device=dev)

        # Predict x0
        x0 = (sample - torch.sqrt(1 - ab_t) * noise_pred) / torch.sqrt(ab_t)
        if self.config.clip_sample:
            x0 = x0.clamp(-self.config.clip_sample_range, self.config.clip_sample_range)

        # Sigma for stochastic sampling
        if eta > 0:
            var = (1 - ab_prev) / (1 - ab_t) * (1 - ab_t / ab_prev)
            sigma = eta * torch.sqrt(torch.clamp(var, min=1e-20))
        else:
            sigma = 0.0

        direction = torch.sqrt(torch.clamp(1 - ab_prev - sigma**2, min=0.0)) * noise_pred
        prev = torch.sqrt(ab_prev) * x0 + direction
        if eta > 0: prev = prev + sigma * torch.randn_like(sample)
        return prev


if __name__ == "__main__":
    s = DDIMScheduler()
    print(f"alpha_bar[0]={s.alphas_cumprod[0]:.4f}, alpha_bar[-1]={s.alphas_cumprod[-1]:.6f}")
    x0 = torch.randn(4, 15, 5)
    noise = torch.randn_like(x0)
    xt = s.add_noise(x0, noise, torch.tensor([500]*4))
    print(f"add_noise: {xt.shape}")
    s.set_timesteps(50)
    print(f"inference steps: {len(s.inference_timesteps)}, first={s.inference_timesteps[:3].tolist()}")
