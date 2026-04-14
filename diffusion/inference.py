"""Monte Carlo Sampling & 매매 전략 신호 생성."""

from __future__ import annotations
import logging
from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import numpy as np
import torch

from diffusion.model import ConditionalDiffusionModel, ModelConfig
from diffusion.scheduler import DDIMScheduler, SchedulerConfig

logger = logging.getLogger("diffusion.inference")


@dataclass
class InferenceConfig:
    num_samples: int = 100
    batch_size: int = 50
    eta: float = 0.5
    direction_threshold: float = 0.001
    confidence_threshold: float = 0.6
    stop_loss_quantile: float = 0.05
    take_profit_quantile: float = 0.95
    num_inference_steps: int = 50


class DiffusionPredictor:
    def __init__(self, model, scheduler, config=None, device=None):
        self.config = config or InferenceConfig()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device).eval()
        self.scheduler = scheduler
        self.scheduler.set_timesteps(self.config.num_inference_steps)

    @classmethod
    def from_checkpoint(cls, path, model_cfg=None, sched_cfg=None, inf_cfg=None, device=None):
        dev = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ckpt = torch.load(path, map_location=dev)
        mc = ModelConfig(**ckpt.get("model_config", {})) if "model_config" in ckpt else (model_cfg or ModelConfig())
        model = ConditionalDiffusionModel(mc)
        model.load_state_dict(ckpt.get("ema_state_dict", ckpt["model_state_dict"]))
        return cls(model, DDIMScheduler(sched_cfg or SchedulerConfig()), inf_cfg, dev)

    @torch.no_grad()
    def predict(self, context, coin_id, num_samples=None):
        N = num_samples or self.config.num_samples
        ctx = context.unsqueeze(0) if context.dim() == 2 else context
        cid = coin_id.unsqueeze(0) if coin_id.dim() == 0 else coin_id
        ctx, cid = ctx.to(self.device), cid.to(self.device)
        
        all_samples, rem = [], N
        while rem > 0:
            B = min(self.config.batch_size, rem)
            x_t = torch.randn(B, self.model.config.target_len, self.model.config.feature_dim, device=self.device)
            ctx_b, cid_b = ctx.expand(B, -1, -1), cid.expand(B)

            ts = self.scheduler.inference_timesteps
            for i, t in enumerate(ts):
                t_val = t.item()
                prev_t = ts[i+1].item() if i+1 < len(ts) else -1
                t_batch = torch.full((B,), t_val, device=self.device, dtype=torch.long)
                eps = self.model(x_t, t_batch, ctx_b, cid_b)
                x_t = self.scheduler.step(eps, t_val, x_t, prev_t, self.config.eta)
            
            all_samples.append(x_t.cpu().numpy())
            rem -= B
        return np.concatenate(all_samples, axis=0)

    def generate_signal(self, context, coin_id, current_price=None, volume_stats=None) -> Dict:
        samples = self.predict(context, coin_id)
        final_rets = np.cumsum(samples[:, :, 3], axis=1)[:, -1]
        
        t = self.config.direction_threshold
        p_up, p_dn = float(np.mean(final_rets > t)), float(np.mean(final_rets < -t))
        conf_t = self.config.confidence_threshold
        
        d = "LONG" if p_up >= conf_t else ("SHORT" if p_dn >= conf_t else "HOLD")
        c = p_up if d == "LONG" else (p_dn if d == "SHORT" else 1 - p_up - p_dn)
        
        res = {
            "direction": d, "confidence": round(c, 4),
            "up_probability": round(p_up, 4), "down_probability": round(p_dn, 4),
            "hold_probability": round(1 - p_up - p_dn, 4),
            "expected_return": round(float(np.median(final_rets)), 6),
            "stop_loss_return": round(float(np.quantile(final_rets, self.config.stop_loss_quantile)), 6),
            "take_profit_return": round(float(np.quantile(final_rets, self.config.take_profit_quantile)), 6),
            "scenarios_mean": np.mean(samples, axis=0),
            "scenarios_std": np.std(samples, axis=0),
            "num_samples": len(samples)
        }
        
        if current_price:
            res["price_targets"] = {
                k: round(current_price * np.exp(v), 2) for k, v in [
                    ("expected", res["expected_return"]), ("stop_loss", res["stop_loss_return"]),
                    ("take_profit", res["take_profit_return"]), ("worst_case", float(np.min(final_rets))),
                    ("best_case", float(np.max(final_rets)))
                ]
            }
        return res

    @staticmethod
    def preprocess_realtime_context(df, cols=None, v_mean=0., v_std=1.) -> torch.Tensor:
        cols = cols or ["open", "high", "low", "close", "volume"]
        prices = np.clip(df[[c for c in cols if c != "volume"]].values.astype(np.float64), 1e-10, None)
        rets = np.diff(np.log(prices), axis=0)
        v_norm = (np.log1p(df["volume"].values.astype(np.float64))[1:] - v_mean) / (v_std + 1e-8)
        return torch.from_numpy(np.nan_to_num(np.column_stack([rets, v_norm]), 0.)).float()

    def analyze_scenarios(self, samples, current_price=None):
        fim = np.cumsum(samples[:, :, 3], axis=1)[:, -1]
        lines = [f"=== Scenarios ({len(samples)}) | 15min Horizon ===",
                 f"Return  - Mean: {np.mean(fim):.4f} | Med: {np.median(fim):.4f} | Std: {np.std(fim):.4f}",
                 f"          Min : {np.min(fim):.4f} | Max: {np.max(fim):.4f}",
                 f"          5%  : {np.percentile(fim, 5):.4f} | 95%: {np.percentile(fim, 95):.4f}",
                 f"Prob    - Up  : {np.mean(fim > 0.001):.1%} | Dn: {np.mean(fim < -0.001):.1%}"]
        if current_price:
            lines.append(f"Prices  - Med: {current_price*np.exp(np.median(fim)):.2f} | Worst: {current_price*np.exp(np.min(fim)):.2f}")
        return "\n".join(lines)


if __name__ == "__main__":
    predictor = DiffusionPredictor(ConditionalDiffusionModel(), DDIMScheduler(), InferenceConfig(num_samples=10))
    sig = predictor.generate_signal(torch.randn(1, 60, 5), torch.tensor([0]), 60000.0)
    for k, v in sig.items():
        if isinstance(v, dict):
            print(f"{k}:"); [print(f"  {a}: {b}") for a, b in v.items()]
        elif not isinstance(v, np.ndarray):
            print(f"{k}: {v}")
