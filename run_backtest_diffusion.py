"""Diffusion 백테스트: Entry(up>=70%, RR>=1.5, Half-Kelly, t+1 Open) / Exit(SL/TP, 15봉, 반전)"""

import os, argparse, logging, time as _t
from dataclasses import dataclass
from typing import List, Optional
import numpy as np, pandas as pd, torch
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from tqdm import tqdm

from diffusion.data import load_raw_csv, resample_to_5min, DiffusionDataConfig, COIN_ID
from diffusion.inference import DiffusionPredictor, InferenceConfig
from diffusion.model import ConditionalDiffusionModel
from diffusion.scheduler import DDIMScheduler

logger = logging.getLogger("backtest")

@dataclass
class Trade:
    bar_in: int; p_in: float; tp: float; sl: float; alloc: float; kelly: float
    bar_out: int = 0; p_out: float = 0; reason: str = ""; pnl: float = 0

def half_kelly(p, tp_r, sl_r):
    b = abs(tp_r / sl_r) if abs(sl_r) > 1e-8 else 1
    return max(0, min((p * b - (1 - p)) / b, 1)) / 2

def run_backtest(symbol="BTC", checkpoint="diffusion/checkpoints/best_model.pt",
                 capital=100_000, fee=0.0005, entry_p=0.70, rev_p=0.70, min_rr=1.5,
                 max_hold=15, n_samples=20, n_steps=20, limit=0, use_random=False):

    # ── 1. Data ──
    dcfg = DiffusionDataConfig()
    df = resample_to_5min(load_raw_csv(
        os.path.join(dcfg.data_dir, f"{symbol}_USDT_1m.csv"), dcfg.ohlcv_cols), dcfg.ohlcv_cols)
    i0 = df[df["datetime"].dt.year >= 2024].index[0]
    i1 = df[df["datetime"].dt.year >= 2025].index[0] if (df["datetime"].dt.year >= 2025).any() else len(df)
    buf = max(0, i0 - 61)
    df = df.iloc[buf:i1].reset_index(drop=True)
    ts = i0 - buf  # test start idx

    meta = os.path.join(dcfg.cache_dir, symbol, "meta.npz")
    vm, vs = (float(np.load(meta)["vol_mean"]), float(np.load(meta)["vol_std"])) if os.path.exists(meta) else (0, 1)

    if limit > 0:
        df = df.iloc[:ts + limit + 1].reset_index(drop=True)

    # ── 2. Model ──
    icfg = InferenceConfig(num_samples=n_samples, num_inference_steps=n_steps, batch_size=n_samples)
    if not use_random and os.path.exists(checkpoint):
        pred = DiffusionPredictor.from_checkpoint(checkpoint, inference_config=icfg)
    else:
        logger.warning("Random weights mode")
        pred = DiffusionPredictor(ConditionalDiffusionModel(), DDIMScheduler(), icfg)

    cid = torch.tensor([COIN_ID[symbol]], dtype=torch.long)

    def signal(t):
        if t < 60: return None
        ctx = DiffusionPredictor.preprocess_realtime_context(df.iloc[t-60:t+1], v_mean=vm, v_std=vs)
        return pred.generate_signal(ctx, cid, float(df.iloc[t]["close"]))

    # ── 3. Loop ──
    equity, pos, pend_exit = capital, None, False
    trades: List[Trade] = []
    eq_curve = []
    n = len(df)

    for t in tqdm(range(ts, n - 1), desc="Backtest"):
        r = df.iloc[t]

        # Pending exit (signal reversal -> t+1 open)
        if pend_exit and pos:
            pos.bar_out, pos.p_out, pos.reason = t, float(r["open"]), "reversal"
            pos.pnl = pos.alloc * ((pos.p_out / pos.p_in) * (1 - fee)**2 - 1)
            equity += pos.pnl; trades.append(pos); pos = None; pend_exit = False

        # Exit checks
        if pos:
            bars = t - pos.bar_in
            ex_p, ex_r = 0, ""
            if r["low"] <= pos.sl and r["high"] >= pos.tp:
                ex_p, ex_r = pos.sl, "sl(both)"
            elif r["low"] <= pos.sl:
                ex_p, ex_r = pos.sl, "stop_loss"
            elif r["high"] >= pos.tp:
                ex_p, ex_r = pos.tp, "take_profit"
            elif bars >= max_hold:
                ex_p, ex_r = float(r["close"]), "timeout"
            else:
                sig = signal(t)
                if sig and sig["down_probability"] >= rev_p:
                    pend_exit = True

            if ex_p > 0:
                pos.bar_out, pos.p_out, pos.reason = t, ex_p, ex_r
                pos.pnl = pos.alloc * ((pos.p_out / pos.p_in) * (1 - fee)**2 - 1)
                equity += pos.pnl; trades.append(pos); pos = None

        # Entry check
        if pos is None and not pend_exit:
            sig = signal(t)
            if sig and sig["up_probability"] >= entry_p:
                tp_r, sl_r = sig["take_profit_return"], sig["stop_loss_return"]
                rr = abs(tp_r / sl_r) if abs(sl_r) > 1e-8 else 0
                if rr >= min_rr and sl_r < 0:
                    k = half_kelly(sig["up_probability"], tp_r, sl_r)
                    if k > 0.01:
                        ep = float(df.iloc[t + 1]["open"])
                        pos = Trade(t+1, ep, ep*np.exp(tp_r), ep*np.exp(sl_r), equity*k, k)

        eq_curve.append(equity)

    # Force close
    if pos:
        pos.bar_out, pos.p_out, pos.reason = n-1, float(df.iloc[-1]["close"]), "end"
        pos.pnl = pos.alloc * ((pos.p_out / pos.p_in) * (1 - fee)**2 - 1)
        equity += pos.pnl; trades.append(pos); eq_curve[-1] = equity

    # ── 4. Report ──
    eq = np.array(eq_curve)
    prices = df.iloc[ts:ts+len(eq)]["close"].values
    bh = (prices[-1] / prices[0] - 1) * 100
    ret = (eq[-1] / capital - 1) * 100
    peak = np.maximum.accumulate(eq)
    mdd = ((eq - peak) / peak).min() * 100
    dr = np.diff(eq) / eq[:-1]
    sharpe = dr.mean() / (dr.std() + 1e-10) * np.sqrt(105_120) if len(dr) > 1 else 0
    wins = sum(1 for t in trades if t.pnl > 0)
    wr = wins / len(trades) * 100 if trades else 0
    reasons = {}
    for t in trades: reasons[t.reason] = reasons.get(t.reason, 0) + 1

    print(f"\n{'='*55}")
    print(f"  {symbol} 2024 BACKTEST  |  {len(trades)} trades")
    print(f"{'='*55}")
    print(f"  Strategy : {ret:+.2f}%  |  B&H : {bh:+.2f}%")
    print(f"  MDD      : {mdd:.2f}%  |  Sharpe : {sharpe:.2f}")
    print(f"  Win Rate : {wr:.1f}%   |  Avg Ret: {np.mean([t.pnl/t.alloc*100 for t in trades]) if trades else 0:+.3f}%")
    print(f"  Exits    : {reasons}")
    print(f"{'='*55}\n")

    # ── 5. Plot ──
    bh_eq = capital * prices / prices[0]
    dates = pd.to_datetime(df.iloc[ts:ts+len(eq)]["datetime"].values)
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(14, 7), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    fig.suptitle(f"Diffusion vs Buy&Hold | {symbol} 2024", fontweight="bold")
    a1.plot(dates, eq, label=f"Strategy ({ret:+.1f}%)", color="#2196F3", lw=1.5)
    a1.plot(dates, bh_eq, label=f"B&H ({bh:+.1f}%)", color="#FF9800", lw=1.2, alpha=.8)
    a1.axhline(capital, color="gray", ls="--", alpha=.4); a1.legend(); a1.set_ylabel("Equity ($)"); a1.grid(alpha=.3)
    for tr in trades:
        i = tr.bar_in - ts
        if 0 <= i < len(dates): a1.axvline(dates[i], color="#4CAF50" if tr.pnl > 0 else "#F44336", alpha=.12, lw=.5)
    dd = (eq - peak) / peak * 100
    a2.fill_between(dates, dd, 0, color="#F44336", alpha=.3); a2.set_ylabel("DD (%)"); a2.grid(alpha=.3)
    plt.tight_layout()
    out = "diffusion/backtest_result.png"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    print(f"Chart -> {out}")
    return {"return": ret, "bh": bh, "mdd": mdd, "sharpe": sharpe, "wr": wr, "trades": len(trades)}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--symbol", default="BTC")
    p.add_argument("--checkpoint", default="diffusion/checkpoints/best_model.pt")
    p.add_argument("--random", action="store_true")
    p.add_argument("--samples", type=int, default=20)
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--capital", type=float, default=100_000)
    a = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    run_backtest(a.symbol, a.checkpoint, a.capital, use_random=a.random,
                 n_samples=a.samples, n_steps=a.steps, limit=a.limit)
