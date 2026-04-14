"""
1D Conditional Sequence Diffusion Model for Crypto Direction Prediction.

BTC, ETH, SOL, XRP 4개 코인의 1분봉 OHLCV 데이터를 기반으로
과거 60분 Context -> 미래 15분 Target을 예측하는
시퀀스 조건부 디퓨전 모델 파이프라인.

모듈 구성:
- data.py       : 데이터 전처리, 정규화, DataLoader 구축
- model.py      : Conditioning Encoder (Transformer) + 1D U-Net Denoising Network
- scheduler.py  : DDIM Noise Scheduler (Cosine beta schedule)
- train.py      : Self-supervised 학습 루프 (EMA, AMP, Early Stopping)
- inference.py  : Monte Carlo Sampling (100 scenarios) 및 매매 전략
"""

from diffusion.data import DiffusionDataConfig, DiffusionDataPipeline
from diffusion.model import ModelConfig, ConditionalDiffusionModel
from diffusion.scheduler import SchedulerConfig, DDIMScheduler
from diffusion.train import TrainConfig, DiffusionTrainer
from diffusion.inference import InferenceConfig, DiffusionPredictor

__all__ = [
    # Data
    "DiffusionDataConfig",
    "DiffusionDataPipeline",
    # Model
    "ModelConfig",
    "ConditionalDiffusionModel",
    # Scheduler
    "SchedulerConfig",
    "DDIMScheduler",
    # Training
    "TrainConfig",
    "DiffusionTrainer",
    # Inference
    "InferenceConfig",
    "DiffusionPredictor",
]
