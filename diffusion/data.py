"""데이터 전처리: CSV(1분봉) → 5분봉 리샘플 → Log Return → Memmap → DataLoader"""

from __future__ import annotations
import os, gc, logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset

logger = logging.getLogger("diffusion.data")


@dataclass
class DiffusionDataConfig:
    data_dir: str = "data/historical"
    cache_dir: str = "diffusion/cache"
    symbols: List[str] = field(default_factory=lambda: ["BTC", "ETH", "SOL", "XRP"])
    timeframe_minutes: int = 5
    context_len: int = 60       # 60봉 = 5시간 (5분봉 기준)
    target_len: int = 15        # 15봉 = 75분
    stride: int = 5
    feature_dim: int = 5
    ohlcv_cols: List[str] = field(default_factory=lambda: ["open", "high", "low", "close", "volume"])
    batch_size: int = 256
    num_workers: int = 4
    pin_memory: bool = True
    prefetch_factor: int = 2
    test_year: int = 2024
    train_val_ratio: float = 6 / 7  # pre-2024 중 train 비율 (6:1)
    volume_clip_quantile: float = 0.999

    @property
    def window_len(self) -> int: return self.context_len + self.target_len

    @property
    def num_coins(self) -> int: return len(self.symbols)


# ── 전처리 함수 ──────────────────────────────────────────────────────

def load_raw_csv(filepath: str, ohlcv_cols: List[str]) -> pd.DataFrame:
    """CSV 로드 + datetime 유지 + 시간순 정렬."""
    df = pd.read_csv(filepath, usecols=["timestamp", "datetime"] + ohlcv_cols)
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.drop(columns=["timestamp"])
    logger.info(f"  Loaded {filepath}: {len(df):,} rows")
    return df


def resample_to_5min(df: pd.DataFrame, ohlcv_cols: List[str]) -> pd.DataFrame:
    """1분봉 → 5분봉 리샘플링. OHLCV 규칙: O=first, H=max, L=min, C=last, V=sum."""
    agg = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    df_5m = df.set_index("datetime")[ohlcv_cols].resample("5min", label="left", closed="left").agg(agg).dropna().reset_index()
    logger.info(f"  Resampled to 5-min: {len(df_5m):,} candles")
    return df_5m


def compute_log_returns(df: pd.DataFrame, ohlcv_cols: List[str]) -> np.ndarray:
    """OHLCV → [open_ret, high_ret, low_ret, close_ret, log_volume] (T-1, 5)."""
    price_cols = [c for c in ohlcv_cols if c != "volume"]
    prices = np.clip(df[price_cols].values.astype(np.float64), 1e-10, None)
    log_returns = np.diff(np.log(prices), axis=0)
    log_volume = np.log1p(df["volume"].values.astype(np.float64))[1:]
    return np.column_stack([log_returns, log_volume]).astype(np.float32)


def normalize_volume(features: np.ndarray, clip_q: float = 0.999) -> Tuple[np.ndarray, float, float]:
    """Volume 컬럼(idx=4)만 Z-score 정규화. (mean, std) 반환."""
    vol = features[:, 4].copy()
    vol = np.clip(vol, None, np.quantile(vol, clip_q))
    mean, std = float(vol.mean()), float(vol.std() + 1e-8)
    features[:, 4] = (vol - mean) / std
    return features, mean, std


# ── Memmap 캐시 ──────────────────────────────────────────────────────

def build_memmap_cache(features, coin_id, cache_path, config, tag=""):
    """슬라이딩 윈도우 → context/target memmap 파일 저장."""
    T, window, stride = len(features), config.window_len, config.stride
    n_windows = max(0, (T - window) // stride + 1)
    if n_windows == 0:
        return "", "", 0

    os.makedirs(cache_path, exist_ok=True)
    prefix = f"coin{coin_id}_{tag}_" if tag else f"coin{coin_id}_"
    ctx_path = os.path.join(cache_path, f"{prefix}context.npy")
    tgt_path = os.path.join(cache_path, f"{prefix}target.npy")

    ctx = np.memmap(ctx_path, dtype=np.float32, mode="w+", shape=(n_windows, config.context_len, config.feature_dim))
    tgt = np.memmap(tgt_path, dtype=np.float32, mode="w+", shape=(n_windows, config.target_len, config.feature_dim))

    for i in range(n_windows):
        s = i * stride
        ctx[i] = features[s : s + config.context_len]
        tgt[i] = features[s + config.context_len : s + config.context_len + config.target_len]
    ctx.flush(); tgt.flush()
    del ctx, tgt

    logger.info(f"  coin{coin_id}[{tag}]: {n_windows:,} windows")
    return ctx_path, tgt_path, n_windows


# ── Dataset ──────────────────────────────────────────────────────────

class CoinDiffusionDataset(Dataset):
    """단일 코인 memmap 기반 Dataset. __getitem__ → {context, target, coin_id}."""
    def __init__(self, ctx_path, tgt_path, coin_id, n, ctx_len, tgt_len, feat_dim):
        super().__init__()
        self.coin_id, self.n = coin_id, n
        self.ctx = np.memmap(ctx_path, np.float32, "r", shape=(n, ctx_len, feat_dim))
        self.tgt = np.memmap(tgt_path, np.float32, "r", shape=(n, tgt_len, feat_dim))

    def __len__(self): return self.n

    def __getitem__(self, idx):
        return {
            "context": torch.from_numpy(self.ctx[idx].copy()),
            "target": torch.from_numpy(self.tgt[idx].copy()),
            "coin_id": torch.tensor(self.coin_id, dtype=torch.long),
        }


# ── 통합 파이프라인 ──────────────────────────────────────────────────

COIN_ID = {"BTC": 0, "ETH": 1, "SOL": 2, "XRP": 3}


class DiffusionDataPipeline:
    """CSV → 5분봉 → Log Return → 날짜 기반 분할 → Memmap → DataLoader."""

    def __init__(self, config=None):
        self.config = config or DiffusionDataConfig()
        self.volume_stats = {}
        self._ds = {"train": {}, "val": {}, "test": {}}

    def prepare(self, force_rebuild=False):
        cfg = self.config
        for symbol in cfg.symbols:
            cid = COIN_ID[symbol]
            cache_path = os.path.join(cfg.cache_dir, symbol)
            meta_path = os.path.join(cache_path, "meta.npz")

            if not force_rebuild and os.path.exists(meta_path):
                meta = np.load(meta_path)
                self.volume_stats[symbol] = (float(meta["vol_mean"]), float(meta["vol_std"]))
                self._make_datasets(symbol, cid, cache_path, int(meta["n_train"]), int(meta["n_val"]), int(meta["n_test"]))
                logger.info(f"[{symbol}] cached: train={int(meta['n_train']):,} val={int(meta['n_val']):,} test={int(meta['n_test']):,}")
                continue

            # 1) Load + resample
            csv_path = os.path.join(cfg.data_dir, f"{symbol}_USDT_1m.csv")
            if not os.path.exists(csv_path):
                logger.error(f"[{symbol}] CSV not found: {csv_path}"); continue
            logger.info(f"[{symbol}] Processing...")
            df = resample_to_5min(load_raw_csv(csv_path, cfg.ohlcv_cols), cfg.ohlcv_cols)

            # 2) Date split: pre-2024 → train+val, 2024 → test
            b1 = df[df["datetime"].dt.year >= cfg.test_year].index[0] if (df["datetime"].dt.year >= cfg.test_year).any() else len(df)
            b2 = df[df["datetime"].dt.year >= cfg.test_year + 1].index[0] if (df["datetime"].dt.year >= cfg.test_year + 1).any() else len(df)
            pre_df, test_df = df.iloc[:b1], df.iloc[b1:b2]
            t_end = int(len(pre_df) * cfg.train_val_ratio)
            train_df, val_df = pre_df.iloc[:t_end], pre_df.iloc[t_end:]
            logger.info(f"  Split: train={len(train_df):,} val={len(val_df):,} test={len(test_df):,}")
            del df, pre_df; gc.collect()

            # 3) Log returns + volume normalization (stats from train only)
            tr_feat = compute_log_returns(train_df, cfg.ohlcv_cols)
            va_feat = compute_log_returns(val_df, cfg.ohlcv_cols)
            te_feat = compute_log_returns(test_df, cfg.ohlcv_cols)
            del train_df, val_df, test_df; gc.collect()

            for f in [tr_feat, va_feat, te_feat]:
                f[:] = np.nan_to_num(f, nan=0., posinf=0., neginf=0.)

            tr_feat, vm, vs = normalize_volume(tr_feat, cfg.volume_clip_quantile)
            va_feat[:, 4] = (va_feat[:, 4] - vm) / vs
            te_feat[:, 4] = (te_feat[:, 4] - vm) / vs
            self.volume_stats[symbol] = (vm, vs)

            # 4) Build memmap caches
            _, _, n_tr = build_memmap_cache(tr_feat, cid, cache_path, cfg, "train")
            _, _, n_va = build_memmap_cache(va_feat, cid, cache_path, cfg, "val")
            _, _, n_te = build_memmap_cache(te_feat, cid, cache_path, cfg, "test")
            del tr_feat, va_feat, te_feat; gc.collect()

            np.savez(meta_path, n_train=n_tr, n_val=n_va, n_test=n_te, vol_mean=vm, vol_std=vs)
            self._make_datasets(symbol, cid, cache_path, n_tr, n_va, n_te)

    def _make_datasets(self, symbol, cid, path, n_tr, n_va, n_te):
        cfg = self.config
        for tag, n, store in [("train", n_tr, self._ds["train"]), ("val", n_va, self._ds["val"]), ("test", n_te, self._ds["test"])]:
            if n > 0:
                store[symbol] = CoinDiffusionDataset(
                    os.path.join(path, f"coin{cid}_{tag}_context.npy"),
                    os.path.join(path, f"coin{cid}_{tag}_target.npy"),
                    cid, n, cfg.context_len, cfg.target_len, cfg.feature_dim,
                )

    def get_dataloaders(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        kw = dict(batch_size=self.config.batch_size, num_workers=self.config.num_workers,
                  pin_memory=self.config.pin_memory, drop_last=False,
                  prefetch_factor=self.config.prefetch_factor if self.config.num_workers > 0 else None,
                  persistent_workers=self.config.num_workers > 0)
        mk = lambda d, shuf: DataLoader(ConcatDataset(list(d.values())), shuffle=shuf, **kw)
        return mk(self._ds["train"], True), mk(self._ds["val"], False), mk(self._ds["test"], False)

    def summary(self) -> str:
        lines = [f"=== Pipeline: {self.config.timeframe_minutes}min | ctx={self.config.context_len} | tgt={self.config.target_len} | test={self.config.test_year} ==="]
        for s in self.config.symbols:
            tr, va, te = len(self._ds["train"].get(s, [])), len(self._ds["val"].get(s, [])), len(self._ds["test"].get(s, []))
            lines.append(f"  {s}: train={tr:>8,} val={va:>8,} test={te:>8,}")
        return "\n".join(lines)


# ── 역변환 유틸 ──

def log_returns_to_prices(log_returns, initial_prices):
    return initial_prices[np.newaxis, :] * np.exp(np.cumsum(log_returns, axis=0))

def denormalize_volume(normalized_vol, vol_mean, vol_std):
    return np.clip(np.expm1(normalized_vol * vol_std + vol_mean), 0, None)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    pipe = DiffusionDataPipeline()
    pipe.prepare(force_rebuild=True)
    print(pipe.summary())
    tr, va, te = pipe.get_dataloaders()
    b = next(iter(tr))
    print(f"Batch: ctx={b['context'].shape} tgt={b['target'].shape} cid={b['coin_id'][:4]}")
