"""
캔들 OHLCV 기반 LSTM 분류기 — XGBoost 버전 비교용

Feature (XGBoost 버전과 동일 데이터, LSTM에 맞게 재구성):
  Layer 1 (시퀀스): 최근 10봉 × 5 feature → shape (10, 5)
    body_ratio, upper_ratio, lower_ratio, volume_ratio, close_pct
  Layer 2 (컨텍스트): 윈도우 요약 5개 → shape (5,)
    rsi14, adx14, funding_rate, vol_trend, price_range

Target:
  close[t+3] > open[t+1] → 1 (상승), 0 (하락)

방식:
  고정 chronological split (train 72% / val 8% / test 20%)
  walk-forward 없음 → XGBoost 대비 빠른 비교 가능

모델:
  LSTM(hidden=64, layers=2) → concat(Layer2) → FC(32) → Sigmoid
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib as mpl
mpl.rcParams["font.family"] = "Malgun Gothic"   # Windows 기본 한글 폰트
mpl.rcParams["axes.unicode_minus"] = False       # 마이너스 기호 깨짐 방지
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

from ma200_core import fetch_ohlcv, fetch_funding_rate


# =============================================================
# CONFIG
# =============================================================
SYMBOL           = "BTC/USDT:USDT"
TIMEFRAME        = "15m"
N_INPUT          = 10
N_OUTPUT         = 3
FETCH_CANDLES    = 70_080 + 500
FETCH_CANDLES_FR = 2_500
THRESHOLDS       = [0.60, 0.65, 0.70, 0.75]
SIGNAL_THRESHOLD = 0.6    # 시뮬레이션 롱 진입 최소 확률
FEE_RATE         = 0.0005   # 편도 수수료 0.05% (taker), 왕복 0.10%

TRAIN_RATIO      = 0.8     # 전체의 80% → train+val
VAL_RATIO        = 0.1     # train의 10% → val (나머지 90% → pure train)

# LSTM 하이퍼파라미터
HIDDEN_SIZE      = 64
NUM_LAYERS       = 2
DROPOUT          = 0.3
BATCH_SIZE       = 512
EPOCHS           = 50
LR               = 1e-3
PATIENCE         = 7       # early stopping patience

SEQ_FEAT_DIM     = 5       # Layer 1: per-candle feature 수
CTX_FEAT_DIM     = 5       # Layer 2: 윈도우 요약 feature 수

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# =============================================================
# STEP 1  데이터 수집
# =============================================================
def fetch_data():
    print("[1] 데이터 수집 중...")
    df    = fetch_ohlcv(SYMBOL, TIMEFRAME, total=FETCH_CANDLES)
    df_fr = fetch_funding_rate(SYMBOL, total=FETCH_CANDLES_FR)
    df    = df.iloc[:-1]

    cutoff = df.index[-1] - pd.DateOffset(years=2)
    warmup_hours = (N_INPUT + N_OUTPUT) * 15 / 60
    df = df[df.index >= cutoff - pd.Timedelta(hours=warmup_hours)]

    print(f"    15분봉: {len(df)}개  ({df.index[0]} ~ {df.index[-1]})")
    print(f"    펀딩비:  {len(df_fr)}개")
    print(f"    사용 디바이스: {DEVICE}")
    return df, df_fr


# =============================================================
# STEP 2  보조 지표 사전 계산 (XGBoost 버전과 동일)
# =============================================================
def compute_indicators(df: pd.DataFrame, df_fr: pd.DataFrame) -> pd.DataFrame:
    print("[2] 보조 지표 계산 중...")
    df = df.copy()

    body_top     = df[["open", "close"]].max(axis=1)
    body_bottom  = df[["open", "close"]].min(axis=1)
    upper_shadow = df["high"] - body_top
    lower_shadow = body_bottom - df["low"]

    prev_close = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"]  - prev_close).abs(),
    ], axis=1).max(axis=1)
    atr100 = tr.rolling(100).mean()

    df["body_ratio"]   = (df["close"] - df["open"]) / atr100
    df["upper_ratio"]  = upper_shadow / atr100
    df["lower_ratio"]  = lower_shadow / atr100
    df["close_pct"]    = df["close"].pct_change()
    df["volume_ratio"] = df["volume"] / df["volume"].rolling(20).mean()

    delta = df["close"].diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    rs    = gain / loss.replace(0, float("nan"))
    df["rsi14"] = 100 - 100 / (1 + rs)

    hi_diff  = df["high"].diff()
    lo_diff  = -df["low"].diff()
    plus_dm  = hi_diff.where((hi_diff > lo_diff) & (hi_diff > 0), 0.0)
    minus_dm = lo_diff.where((lo_diff > hi_diff) & (lo_diff > 0), 0.0)
    atr14    = tr.rolling(14).mean()
    plus_di  = 100 * plus_dm.rolling(14).mean() / atr14
    minus_di = 100 * minus_dm.rolling(14).mean() / atr14
    dx       = (100 * (plus_di - minus_di).abs()
                / (plus_di + minus_di).replace(0, float("nan")))
    df["adx14"] = dx.rolling(14).mean()

    df["funding_rate"] = (
        df_fr["fundingRate"]
        .reindex(df.index, method="ffill")
        .fillna(0.0)
    )
    df["atr100"] = atr100
    return df


# =============================================================
# STEP 3  Feature 행렬 구성 (LSTM용 shape으로 분리)
# =============================================================
def build_feature_matrix(df: pd.DataFrame):
    print("[3] Feature 행렬 구성 중...")

    per_candle_cols = ["body_ratio", "upper_ratio", "lower_ratio",
                       "volume_ratio", "close_pct"]
    n = len(df)
    half = N_INPUT // 2

    future_close = df["close"].shift(-N_OUTPUT).values
    future_open  = df["open"].shift(-1).values

    seq_rows    = []   # (10, 5) per sample  → Layer 1
    ctx_rows    = []   # (5,)   per sample   → Layer 2
    targets     = []
    entry_opens = []   # open[i+1]  → 진입가
    exit_closes = []   # close[i+3] → 청산가
    timestamps  = []

    for i in range(N_INPUT - 1, n - N_OUTPUT):
        window = df.iloc[i - N_INPUT + 1 : i + 1]

        # Layer 1: 시퀀스 (10, 5)
        layer1 = window[per_candle_cols].values

        # Layer 2: 컨텍스트 (5,)
        atr = df["atr100"].iloc[i]
        if atr == 0 or np.isnan(atr):
            continue

        vol_recent  = window["volume"].iloc[half:].mean()
        vol_old     = window["volume"].iloc[:half].mean()
        vol_trend   = vol_recent / vol_old if vol_old > 0 else 1.0
        price_range = (window["high"].max() - window["low"].min()) / atr

        layer2 = np.array([
            df["rsi14"].iloc[i],
            df["adx14"].iloc[i],
            df["funding_rate"].iloc[i],
            vol_trend,
            price_range,
        ])

        if np.isnan(layer1).any() or np.isnan(layer2).any():
            continue

        fc, fo = future_close[i], future_open[i]
        if np.isnan(fc) or np.isnan(fo):
            continue

        seq_rows.append(layer1)
        ctx_rows.append(layer2)
        targets.append(int(fc > fo))
        entry_opens.append(float(fo))   # open[i+1]  진입가
        exit_closes.append(float(fc))   # close[i+3] 청산가
        timestamps.append(df.index[i])

    seq = np.array(seq_rows,    dtype=np.float32)    # (N, 10, 5)
    ctx = np.array(ctx_rows,    dtype=np.float32)    # (N, 5)
    y   = np.array(targets,     dtype=np.float32)    # (N,)
    eo  = np.array(entry_opens, dtype=np.float64)    # (N,)
    ec  = np.array(exit_closes, dtype=np.float64)    # (N,)

    print(f"    시퀀스: {seq.shape}  컨텍스트: {ctx.shape}")
    print(f"    Target 상승 비율: {y.mean():.1%}")
    return seq, ctx, y, timestamps, eo, ec


# =============================================================
# STEP 4  Train / Val / Test 분리 + 정규화
# =============================================================
def split_and_scale(seq, ctx, y, timestamps, eo, ec):
    print("[4] 데이터 분할 및 정규화 중...")
    n = len(y)

    n_trainval = int(n * TRAIN_RATIO)
    n_val      = int(n_trainval * VAL_RATIO)
    n_train    = n_trainval - n_val

    # chronological 분할
    seq_train, ctx_train, y_train = seq[:n_train],    ctx[:n_train],    y[:n_train]
    seq_val,   ctx_val,   y_val   = seq[n_train:n_trainval], ctx[n_train:n_trainval], y[n_train:n_trainval]
    seq_test,  ctx_test,  y_test  = seq[n_trainval:], ctx[n_trainval:], y[n_trainval:]
    eo_test = eo[n_trainval:]    # 테스트 구간 진입가
    ec_test = ec[n_trainval:]    # 테스트 구간 청산가
    ts_test = timestamps[n_trainval:]

    print(f"    Train: {n_train}개  Val: {n_val}개  Test: {len(y_test)}개")

    # 정규화: train 기준으로 fit, val/test에 transform
    # Layer 1: (N, 10, 5) → (N, 50) → scale → (N, 10, 5)
    seq_scaler = StandardScaler()
    seq_train_2d = seq_train.reshape(n_train, -1)
    seq_train = seq_scaler.fit_transform(seq_train_2d).reshape(n_train, N_INPUT, SEQ_FEAT_DIM)
    seq_val   = seq_scaler.transform(seq_val.reshape(n_val, -1)).reshape(n_val, N_INPUT, SEQ_FEAT_DIM)
    seq_test  = seq_scaler.transform(seq_test.reshape(len(y_test), -1)).reshape(len(y_test), N_INPUT, SEQ_FEAT_DIM)

    # Layer 2: (N, 5) → scale
    ctx_scaler = StandardScaler()
    ctx_train = ctx_scaler.fit_transform(ctx_train)
    ctx_val   = ctx_scaler.transform(ctx_val)
    ctx_test  = ctx_scaler.transform(ctx_test)

    splits = {
        "train": (seq_train, ctx_train, y_train),
        "val":   (seq_val,   ctx_val,   y_val),
        "test":  (seq_test,  ctx_test,  y_test),
    }
    return splits, ts_test, eo_test, ec_test


# =============================================================
# PyTorch Dataset / Model
# =============================================================
class CandleDataset(Dataset):
    def __init__(self, seq, ctx, targets):
        self.seq     = torch.tensor(seq,     dtype=torch.float32)
        self.ctx     = torch.tensor(ctx,     dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return self.seq[idx], self.ctx[idx], self.targets[idx]


class CandleLSTM(nn.Module):
    """
    LSTM으로 10봉 시퀀스 처리 후 윈도우 요약(ctx)와 concat → 이진 분류
    """
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size  = SEQ_FEAT_DIM,
            hidden_size = HIDDEN_SIZE,
            num_layers  = NUM_LAYERS,
            batch_first = True,
            dropout     = DROPOUT if NUM_LAYERS > 1 else 0.0,
        )
        self.classifier = nn.Sequential(
            nn.Linear(HIDDEN_SIZE + CTX_FEAT_DIM, 32),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, seq, ctx):
        # seq: (batch, 10, 5)
        # ctx: (batch, 5)
        _, (h_n, _) = self.lstm(seq)
        h = h_n[-1]                        # 마지막 레이어 hidden: (batch, 64)
        x = torch.cat([h, ctx], dim=1)     # (batch, 64+5=69)
        return self.classifier(x).squeeze(1)  # (batch,)


# =============================================================
# STEP 5  학습
# =============================================================
def train_model(splits: dict) -> CandleLSTM:
    print("[5] LSTM 학습 중...")
    print(f"    Hidden={HIDDEN_SIZE}  Layers={NUM_LAYERS}  "
          f"Dropout={DROPOUT}  LR={LR}  Epochs={EPOCHS}")

    seq_tr, ctx_tr, y_tr = splits["train"]
    seq_vl, ctx_vl, y_vl = splits["val"]

    train_loader = DataLoader(
        CandleDataset(seq_tr, ctx_tr, y_tr),
        batch_size=BATCH_SIZE, shuffle=True,
    )

    model     = CandleLSTM().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.BCELoss()

    # val 텐서 (GPU로)
    seq_vl_t = torch.tensor(seq_vl, dtype=torch.float32).to(DEVICE)
    ctx_vl_t = torch.tensor(ctx_vl, dtype=torch.float32).to(DEVICE)
    y_vl_t   = torch.tensor(y_vl,   dtype=torch.float32).to(DEVICE)

    best_val_loss = float("inf")
    best_weights  = None
    no_improve    = 0

    print(f"    {'Epoch':>5}  {'TrainLoss':>10}  {'ValLoss':>9}  {'ValAcc':>7}")
    print(f"    {'-'*38}")

    for epoch in range(1, EPOCHS + 1):
        # ── Train ──────────────────────────────────────────────
        model.train()
        train_loss = 0.0
        for seq_b, ctx_b, y_b in train_loader:
            seq_b, ctx_b, y_b = seq_b.to(DEVICE), ctx_b.to(DEVICE), y_b.to(DEVICE)
            optimizer.zero_grad()
            pred = model(seq_b, ctx_b)
            loss = criterion(pred, y_b)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(y_b)
        train_loss /= len(y_tr)

        # ── Validation ─────────────────────────────────────────
        model.eval()
        with torch.no_grad():
            val_pred = model(seq_vl_t, ctx_vl_t)
            val_loss = criterion(val_pred, y_vl_t).item()
            val_acc  = ((val_pred >= 0.5).float() == y_vl_t).float().mean().item()

        print(f"    {epoch:>5}  {train_loss:>10.4f}  {val_loss:>9.4f}  {val_acc:>7.1%}")

        # ── Early Stopping ─────────────────────────────────────
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights  = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve    = 0
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print(f"    Early stopping at epoch {epoch}")
                break

    # 최적 가중치 복원
    model.load_state_dict(best_weights)
    print(f"    학습 완료  |  Best val loss: {best_val_loss:.4f}")
    return model


# =============================================================
# STEP 6  테스트 평가
# =============================================================
def evaluate(model: CandleLSTM, splits: dict,
             ts_test: list, eo_test: np.ndarray, ec_test: np.ndarray) -> pd.DataFrame:
    print("[6] 테스트셋 평가 중...")
    seq_te, ctx_te, y_te = splits["test"]

    model.eval()
    with torch.no_grad():
        seq_t = torch.tensor(seq_te, dtype=torch.float32).to(DEVICE)
        ctx_t = torch.tensor(ctx_te, dtype=torch.float32).to(DEVICE)
        probs = model(seq_t, ctx_t).cpu().numpy()

    res = pd.DataFrame({
        "bar_time":   ts_test,
        "prob":       probs,
        "pred":       (probs >= 0.5).astype(int),
        "actual":     y_te.astype(int),
        "entry_open": eo_test,
        "exit_close": ec_test,
    })
    return res


# =============================================================
# STEP 7  결과 출력 (XGBoost 버전과 동일 형식)
# =============================================================
def print_report(res: pd.DataFrame):
    print("\n" + "=" * 62)
    print("  LSTM 캔들 백테스트 결과 (Test set)")
    print("=" * 62)

    total     = len(res)
    base_rate = res["actual"].mean()
    acc       = (res["pred"] == res["actual"]).mean()

    print(f"\n  전체 샘플    : {total}개")
    print(f"  Base rate    : {base_rate:.1%}  (항상 상승 예측 시 정확도)")
    print(f"\n  ── 전체 정확도 (threshold=0.50) ──")
    print(f"    정확도: {acc:.1%}")

    # ── 혼동 행렬 ─────────────────────────────────────────────
    tp = ((res["pred"] == 1) & (res["actual"] == 1)).sum()
    tn = ((res["pred"] == 0) & (res["actual"] == 0)).sum()
    fp = ((res["pred"] == 1) & (res["actual"] == 0)).sum()
    fn = ((res["pred"] == 0) & (res["actual"] == 1)).sum()
    prec_bull = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    prec_bear = tn / (tn + fn) if (tn + fn) > 0 else float("nan")

    print(f"\n  ── 혼동 행렬 ──")
    print(f"    예측 상승 → 실제 상승(TP): {tp:>5}  |  실제 하락(FP): {fp:>5}")
    print(f"    예측 하락 → 실제 상승(FN): {fn:>5}  |  실제 하락(TN): {tn:>5}")
    print(f"    상승 예측 precision: {prec_bull:.1%}  |  하락 예측 precision: {prec_bear:.1%}")

    # ── 확률 구간별 정확도 ────────────────────────────────────
    print(f"\n  ── 확률 구간별 정확도 ──")
    print(f"  {'구간':<22} {'샘플':>6}  {'정확도':>7}  {'비율':>6}")
    print(f"  {'-' * 46}")
    for thr in THRESHOLDS:
        sub_bull = res[res["prob"] >= thr]
        if len(sub_bull) > 0:
            a = (sub_bull["actual"] == 1).mean()
            print(f"  prob >= {thr:.2f} (상승예측)  {len(sub_bull):>6}  {a:>7.1%}  "
                  f"{len(sub_bull)/total:>6.1%}")
        sub_bear = res[res["prob"] <= 1 - thr]
        if len(sub_bear) > 0:
            a = (sub_bear["actual"] == 0).mean()
            print(f"  prob <= {1-thr:.2f} (하락예측)  {len(sub_bear):>6}  {a:>7.1%}  "
                  f"{len(sub_bear)/total:>6.1%}")

    # ── 확률 분포 히스토그램 ──────────────────────────────────
    print(f"\n  ── 예측 확률 분포 ──")
    bins   = [0.0, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 1.01]
    labels = ["<0.40", "0.40~0.45", "0.45~0.50", "0.50~0.55",
              "0.55~0.60", "0.60~0.65", "0.65~0.70", "0.70~0.75", ">=0.75"]
    res["prob_bin"] = pd.cut(res["prob"], bins=bins, labels=labels, right=False)
    for label in labels:
        sub = res[res["prob_bin"] == label]
        if len(sub) == 0:
            continue
        preds = (sub["prob"] >= 0.5).astype(int)
        acc_b = (preds == sub["actual"]).mean()
        bar   = "#" * int(len(sub) / total * 80)
        print(f"  {str(label):<12}  {len(sub):>5}개  정확도={acc_b:.1%}  {bar}")

    print("\n" + "=" * 62)


# =============================================================
# STEP 8  시뮬레이션 + 수익률 그래프
# =============================================================
def simulate_and_plot(res: pd.DataFrame,
                      threshold: float = SIGNAL_THRESHOLD,
                      fee: float = FEE_RATE):
    """
    prob >= threshold 일 때 롱 진입 → N_OUTPUT봉 후 청산
    수수료: fee(편도) × 2 (왕복)
    """
    sig = res[res["prob"] >= threshold].copy().reset_index(drop=True)

    if len(sig) == 0:
        print(f"  신호 없음 (threshold={threshold})")
        return

    # ── 수익률 계산 ───────────────────────────────────────────
    sig["raw_pnl"] = (sig["exit_close"] - sig["entry_open"]) / sig["entry_open"]
    sig["net_pnl"] = sig["raw_pnl"] - fee * 2
    sig["cum_ret"] = (1 + sig["net_pnl"]).cumprod() - 1

    # ── Drawdown ──────────────────────────────────────────────
    equity      = 1 + sig["cum_ret"]
    rolling_max = equity.cummax()
    sig["dd"]   = (equity - rolling_max) / rolling_max

    # ── BTC Buy & Hold (테스트 기간) ──────────────────────────
    res_s     = res.sort_values("bar_time").reset_index(drop=True)
    bnh_start = res_s["entry_open"].iloc[0]
    bnh_total = (res_s["exit_close"].iloc[-1] - bnh_start) / bnh_start
    bnh_curve = (res_s["entry_open"] - bnh_start) / bnh_start

    # ── 통계 출력 ─────────────────────────────────────────────
    n_trades  = len(sig)
    win_rate  = (sig["net_pnl"] > 0).mean()
    avg_pnl   = sig["net_pnl"].mean()
    total_ret = sig["cum_ret"].iloc[-1]
    max_dd    = sig["dd"].min()

    print("\n" + "=" * 58)
    print(f"  시뮬레이션 결과  (prob≥{threshold}, 수수료 {fee*100:.2f}%×2)")
    print("=" * 58)
    print(f"  총 거래       : {n_trades}회")
    print(f"  승률          : {win_rate:.1%}")
    print(f"  평균 거래 수익: {avg_pnl:.3%}")
    print(f"  누적 수익률   : {total_ret:.2%}")
    print(f"  최대 낙폭     : {max_dd:.2%}")
    print(f"  BTC Buy&Hold  : {bnh_total:.2%}  (동일 기간)")
    print("=" * 58)

    # ── 그래프 ───────────────────────────────────────────────
    times     = pd.to_datetime(sig["bar_time"])
    bnh_times = pd.to_datetime(res_s["bar_time"])

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(14, 8),
        gridspec_kw={"height_ratios": [3, 1]},
        sharex=False,
    )

    ax1.plot(times, sig["cum_ret"] * 100,
             color="steelblue", linewidth=1.5,
             label=f"전략 (prob≥{threshold})  최종: {total_ret:.1%}")
    ax1.plot(bnh_times, bnh_curve * 100,
             color="orange", linewidth=1.2, linestyle="--",
             label=f"BTC Buy&Hold  최종: {bnh_total:.1%}")
    ax1.axhline(0, color="gray", linewidth=0.6, linestyle=":")
    ax1.set_ylabel("누적 수익률 (%)", fontsize=11)
    ax1.set_title(
        f"LSTM 롱 전략  |  진입: prob≥{threshold}  |  "
        f"청산: {N_OUTPUT}봉 후  |  수수료: {fee*100:.2f}%×2  |  "
        f"거래: {n_trades}회  승률: {win_rate:.1%}",
        fontsize=11,
    )
    ax1.legend(loc="upper left", fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=30, ha="right")

    ax2.fill_between(times, sig["dd"] * 100, 0,
                     color="crimson", alpha=0.45,
                     label=f"낙폭  최대: {max_dd:.1%}")
    ax2.axhline(0, color="gray", linewidth=0.6)
    ax2.set_ylabel("낙폭 (%)", fontsize=11)
    ax2.set_xlabel("시간", fontsize=11)
    ax2.legend(loc="lower left", fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=30, ha="right")

    fig.tight_layout()
    out_path = "lstm_long_simulation.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"  그래프 저장: {out_path}")
    plt.show()


# =============================================================
# MAIN
# =============================================================
if __name__ == "__main__":
    df_raw, df_fr              = fetch_data()
    df                         = compute_indicators(df_raw, df_fr)
    seq, ctx, y, times, eo, ec = build_feature_matrix(df)
    splits, ts_test, eo_test, ec_test = split_and_scale(seq, ctx, y, times, eo, ec)
    model                      = train_model(splits)
    results                    = evaluate(model, splits, ts_test, eo_test, ec_test)
    print_report(results)
    simulate_and_plot(results)
