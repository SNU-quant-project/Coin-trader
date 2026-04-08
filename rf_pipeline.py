"""
Random Forest Trading Strategy Pipeline
BTC/USDT 1-minute candles
"""

import os
import json
import time
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import classification_report

warnings.filterwarnings('ignore')

# ── 설정 ─────────────────────────────────────────────────────────
DATA_PATH    = "/Users/bigdohun/Desktop/Coin-trader/data/historical/BTC_USDT_1m.csv"
OUTPUT_DIR   = "./output"
FORWARD_WIN   = 240      # 레이블 기준: N봉 뒤 수익률 (240 = 4시간)
BUY_THRESH    = 0.003    # 4시간 뒤 +0.7% 초과 → BUY
SELL_THRESH   = -0.003   # 4시간 뒤 -0.7% 미만 → SELL
ANNUAL_BARS   = 525600   # 1분봉 기준 연간 봉 수
TRAIN_START   = "2019-01-01"   # 학습/검증 시작
TRAIN_END     = "2023-12-31"   # 학습/검증 종료
TEST_START    = "2024-01-01"   # 테스트 시작
TEST_END      = "2024-12-31"   # 테스트 종료
N_SPLITS      = 3        # TimeSeriesSplit fold 수
N_ESTIMATORS  = 100      # 트리 수
TREE_BATCH    = 5       # 진행률 표시 단위 (트리 몇 개마다 업데이트)
PROB_THRESH_BUY  = 0.38  # 롱 진입 확률 임계값
PROB_THRESH_SELL = 0.40  # 숏 진입 확률 임계값
MIN_HOLD_BARS = 240     # 최소 포지션 유지 시간 (봉 수, 240 = 4시간)

RF_PARAMS = dict(
    n_estimators=N_ESTIMATORS,
    max_depth=6,
    max_features='sqrt',
    min_samples_leaf=20,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1,
)


def fit_with_progress(model_params: dict, X, y, sample_weight=None, desc="학습 중") -> RandomForestClassifier:
    """warm_start를 이용해 TREE_BATCH 단위로 트리를 쌓으며 진행률 표시."""
    params = {**model_params, 'n_estimators': TREE_BATCH, 'warm_start': True}
    clf = RandomForestClassifier(**params)

    batches = N_ESTIMATORS // TREE_BATCH
    remainder = N_ESTIMATORS % TREE_BATCH

    with tqdm(total=N_ESTIMATORS, desc=desc, unit="trees", ncols=70, colour='green') as pbar:
        for _ in range(batches):
            clf.fit(X, y, sample_weight=sample_weight)
            clf.n_estimators += TREE_BATCH
            pbar.update(TREE_BATCH)
        if remainder:
            clf.n_estimators = N_ESTIMATORS
            clf.fit(X, y, sample_weight=sample_weight)
            pbar.update(remainder)

    clf.warm_start = False
    return clf

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ════════════════════════════════════════════════════════════════
# Step 1: 데이터 로드 & 검증
# ════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  Step 1. 데이터 로드 & 검증")
print("=" * 60)

try:
    df = pd.read_csv(DATA_PATH)
except Exception as e:
    print(f"❌ 파일 읽기 실패: {e}")
    raise

# 컬럼 확인 (timestamp / datetime 컬럼 처리)
if 'datetime' in df.columns:
    df['open_time'] = pd.to_datetime(df['datetime'])
elif 'timestamp' in df.columns:
    df['open_time'] = pd.to_datetime(df['timestamp'], unit='ms')
else:
    raise ValueError("시간 컬럼(datetime 또는 timestamp)이 없습니다.")

required = ['open', 'high', 'low', 'close', 'volume']
missing = [c for c in required if c not in df.columns]
if missing:
    raise ValueError(f"필수 컬럼 없음: {missing}")

for col in required:
    df[col] = pd.to_numeric(df[col], errors='coerce')

df = df.sort_values('open_time').reset_index(drop=True)

# 날짜 범위 필터링 (2019-01-01 ~ 2024-12-31)
before = len(df)
df = df[
    (df['open_time'] >= pd.Timestamp(TRAIN_START)) &
    (df['open_time'] <= pd.Timestamp(TEST_END))
].reset_index(drop=True)
print(f"  데이터 필터  : {TRAIN_START} ~ {TEST_END} ({before:,}행 → {len(df):,}행)")

# 중복 제거
dup_count = df.duplicated(subset=['open_time']).sum()
df = df.drop_duplicates(subset=['open_time']).reset_index(drop=True)

# 결측값 처리: 3개 이하 갭은 forward-fill, 초과는 제거
null_before = df[required].isnull().sum().sum()
df[required] = df[required].fillna(method='ffill', limit=3)
df = df.dropna(subset=required).reset_index(drop=True)

print(f"  shape        : {df.shape}")
print(f"  기간         : {df['open_time'].iloc[0]} ~ {df['open_time'].iloc[-1]}")
print(f"  중복 제거    : {dup_count}개")
print(f"  결측값 처리  : {null_before}개")
print(f"\n  처음 5행:")
print(df[['open_time'] + required].head().to_string(index=False))
print(f"\n  데이터 타입:\n{df[required].dtypes.to_string()}")

if len(df) < 5000:
    print("⚠️  Dataset may be too small for reliable results")


# ════════════════════════════════════════════════════════════════
# Step 2: 피처 엔지니어링 (look-ahead bias 없음)
# ════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  Step 2. 피처 엔지니어링")
print("=" * 60)

close  = df['close']
high   = df['high']
low    = df['low']
open_  = df['open']
volume = df['volume']


def rsi(series, period=14):
    delta = series.diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    rs    = gain / (loss + 1e-10)
    return 100 - (100 / (1 + rs))


def atr(high, low, close, period=14):
    hl = high - low
    hc = (high - close.shift(1)).abs()
    lc = (low  - close.shift(1)).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def adx(high, low, close, period=14):
    up   = high.diff()
    down = -low.diff()
    plus_dm  = up.where((up > down) & (up > 0), 0.0)
    minus_dm = down.where((down > up) & (down > 0), 0.0)
    atr_s    = atr(high, low, close, period)
    plus_di  = 100 * plus_dm.rolling(period).mean()  / (atr_s + 1e-10)
    minus_di = 100 * minus_dm.rolling(period).mean() / (atr_s + 1e-10)
    dx       = (100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-10))
    return dx.rolling(period).mean()


feat = pd.DataFrame(index=df.index)

# ── [A] 기존 피처 리팩토링 ────────────────────────────────────────

# RSI
feat['rsi_14']   = rsi(close, 14)
feat['rsi_slope'] = feat['rsi_14'].diff(5).shift(1)

# MACD hist 만 유지 (macd/signal 제거)
ema12 = close.ewm(span=12, adjust=False).mean()
ema26 = close.ewm(span=26, adjust=False).mean()
macd_line   = ema12 - ema26
signal_line = macd_line.ewm(span=9, adjust=False).mean()
feat['macd_hist'] = (macd_line - signal_line).shift(1)

# 볼린저 밴드
ma20     = close.rolling(20).mean()
std20    = close.rolling(20).std()
bb_upper = ma20 + 2 * std20
bb_lower = ma20 - 2 * std20
feat['bb_pct']   = (close - bb_lower) / (bb_upper - bb_lower + 1e-10)
feat['bb_width'] = ((bb_upper - bb_lower) / (ma20 + 1e-10)).shift(1)

# EMA gap (raw ema9/ema21 제거, 비율로 대체)
ema9  = close.ewm(span=9,  adjust=False).mean()
ema21 = close.ewm(span=21, adjust=False).mean()
feat['ema_gap_pct'] = ((ema9 - ema21) / (ema21 + 1e-10)).shift(1)

# ATR 상대화 (절대값 제거)
atr14 = atr(high, low, close, 14)
feat['atr_ratio']    = (atr14 / (close + 1e-10)).shift(1)
feat['atr_ma_ratio'] = (atr14 / (atr14.rolling(50).mean() + 1e-10)).shift(1)
feat['atr_change']   = atr14.pct_change(10).shift(1)

# ADX
feat['adx_14'] = adx(high, low, close, 14).shift(1)

# 거래량 비율
feat['vol_ratio']     = (volume / (volume.rolling(20).mean() + 1e-10)).shift(1)
feat['vol_std_ratio'] = (volume / (volume.rolling(20).std()  + 1e-10)).shift(1)

# ── [B] 모멘텀 확장 ───────────────────────────────────────────────
feat['return_1']  = close.pct_change(1).shift(1)
feat['return_5']  = close.pct_change(5).shift(1)
feat['return_10'] = close.pct_change(10).shift(1)
feat['return_30'] = close.pct_change(30).shift(1)
feat['return_60'] = close.pct_change(60).shift(1)

# ── [C] 변동성 고급 피처 ─────────────────────────────────────────

# Garman-Klass 변동성
gk = np.sqrt(
    0.5 * np.log((high / low).clip(1e-10))**2 -
    (2 * np.log(2) - 1) * np.log((close / open_.clip(1e-10)).clip(1e-10))**2
)
feat['gk_vol_ratio'] = (gk / (gk.rolling(50).mean() + 1e-10)).shift(1)

# 실현 변동성 비율
rv_short = close.pct_change().rolling(10).std()
rv_long  = close.pct_change().rolling(60).std()
feat['rv_ratio'] = (rv_short / (rv_long + 1e-9)).shift(1)

# True Range 가속도
tr_s = pd.concat([
    high - low,
    (high - close.shift(1)).abs(),
    (low  - close.shift(1)).abs()
], axis=1).max(axis=1)
feat['tr_accel'] = (tr_s.diff(5) / (tr_s.shift(5) + 1e-9)).shift(1)

# ── [F] 가격 위치 및 지지/저항 ───────────────────────────────────
high20_max = high.rolling(20).max()
low20_min  = low.rolling(20).min()

feat['price_position_20'] = (
    (close - low20_min) / (high20_max - low20_min + 1e-9)
).shift(1)
feat['dist_from_high_20'] = ((high20_max - close) / (close + 1e-9)).shift(1)
feat['dist_from_low_20']  = ((close - low20_min)  / (close + 1e-9)).shift(1)

# VWAP 편차
vwap = (close * volume).rolling(60).sum() / (volume.rolling(60).sum() + 1e-9)
feat['vwap_deviation'] = ((close - vwap) / (vwap + 1e-9)).shift(1)

# 가격 Z-score
feat['price_zscore_50'] = (
    (close - close.rolling(50).mean()) / (close.rolling(50).std() + 1e-9)
).shift(1)

# ── [G] 캔들 구조 피처 ───────────────────────────────────────────
hl_range = (high - low).clip(1e-9)
feat['candle_body']   = ((close - open_).abs() / hl_range).shift(1)
feat['upper_shadow']  = ((high - pd.concat([open_, close], axis=1).max(axis=1)) / hl_range).shift(1)
feat['lower_shadow']  = ((pd.concat([open_, close], axis=1).min(axis=1) - low) / hl_range).shift(1)
feat['wickedness']    = (feat['upper_shadow'] + feat['lower_shadow'])
feat['gap_pct']       = ((open_ - close.shift(1)) / (close.shift(1) + 1e-9)).shift(1)

# ── [H] 거래량 고급 피처 ─────────────────────────────────────────

# OBV slope
price_dir = np.sign(close.diff())
obv = (volume * price_dir).cumsum()
feat['obv_slope'] = (obv.diff(10) / (volume.rolling(10).mean() + 1e-9)).shift(1)

# ── [D] 시장 미시구조 ─────────────────────────────────────────────

# Amihud 비유동성
feat['amihud'] = (
    close.pct_change().abs() / (close * volume + 1e-9)
).rolling(20).mean().shift(1)

# 오더 플로우 불균형
price_change = close.diff()
buy_vol  = volume.where(price_change > 0, 0)
sell_vol = volume.where(price_change < 0, 0)
feat['order_flow_imbalance'] = (
    (buy_vol.rolling(20).sum() - sell_vol.rolling(20).sum()) /
    (volume.rolling(20).sum() + 1e-9)
).shift(1)

# Kyle's Lambda
feat['kyle_lambda_ratio'] = (
    close.diff().abs() / (volume + 1e-9) /
    ((close.diff().abs() / (volume + 1e-9)).rolling(50).mean() + 1e-9)
).shift(1)

# ── [J] 시간 피처 ────────────────────────────────────────────────
open_time = df['open_time']
hour = open_time.dt.hour
dow  = open_time.dt.dayofweek

feat['hour_sin'] = np.sin(2 * np.pi * hour / 24)
feat['hour_cos'] = np.cos(2 * np.pi * hour / 24)
feat['dow_sin']  = np.sin(2 * np.pi * dow / 7)
feat['dow_cos']  = np.cos(2 * np.pi * dow / 7)
feat['near_us_open']   = ((hour >= 13) & (hour <= 15)).astype(int)
feat['near_asia_open'] = ((hour >= 0)  & (hour <= 2)).astype(int)

# ── [E] 통계적 특성 (hurst 제외) ──────────────────────────────────
ret = close.pct_change()
feat['return_skew_20'] = ret.rolling(20).skew().shift(1)
feat['return_kurt_20'] = ret.rolling(20).kurt().shift(1)

# ── [I] 다이버전스 신호 ───────────────────────────────────────────
price_higher = close > close.shift(14)
rsi_series   = feat['rsi_14']
rsi_lower    = rsi_series < rsi_series.shift(14)
feat['bearish_divergence'] = (price_higher & rsi_lower).astype(int).shift(1)
feat['bullish_divergence'] = (~price_higher & ~rsi_lower).astype(int).shift(1)

FEATURE_COLS = [
    # 모멘텀
    'return_1', 'return_5', 'return_10', 'return_30', 'return_60',
    # 트렌드
    'ema_gap_pct', 'macd_hist', 'adx_14',
    # 변동성
    'atr_ratio', 'atr_ma_ratio', 'atr_change',
    'gk_vol_ratio', 'rv_ratio', 'tr_accel', 'bb_pct', 'bb_width',
    # 미시구조
    'order_flow_imbalance', 'amihud', 'kyle_lambda_ratio',
    # 가격 위치
    'rsi_14', 'rsi_slope', 'price_position_20',
    'dist_from_high_20', 'dist_from_low_20', 'vwap_deviation', 'price_zscore_50',
    # 통계
    'return_skew_20', 'return_kurt_20',
    # 캔들
    'candle_body', 'upper_shadow', 'lower_shadow', 'wickedness', 'gap_pct',
    # 거래량
    'vol_ratio', 'vol_std_ratio', 'obv_slope',
    # 시간
    'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos', 'near_us_open', 'near_asia_open',
    # 다이버전스
    'bearish_divergence', 'bullish_divergence',
]

df = pd.concat([df, feat], axis=1)
df[FEATURE_COLS] = df[FEATURE_COLS].replace([np.inf, -np.inf], np.nan)
df = df.dropna(subset=FEATURE_COLS).reset_index(drop=True)

print(f"  피처 수      : {len(FEATURE_COLS)}")
print(f"  피처 목록    : {FEATURE_COLS}")
print(f"  NaN 제거 후  : {df.shape[0]:,}행")


# ════════════════════════════════════════════════════════════════
# Step 3: 레이블 생성
# ════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  Step 3. 레이블 생성")
print("=" * 60)

close_s = df['close']
forward_return = close_s.shift(-FORWARD_WIN) / close_s - 1

label = pd.Series(0, index=df.index)
label[forward_return >  BUY_THRESH]  =  1
label[forward_return <  SELL_THRESH] = -1

# 마지막 forward_window행은 유효한 레이블 없음 → 제거
df = df.iloc[:-FORWARD_WIN].copy()
label = label.iloc[:-FORWARD_WIN].copy()
df['label'] = label.values

dist = df['label'].value_counts().sort_index()
total = len(df)
print(f"  레이블 분포:")
for k, v in dist.items():
    name = {1: 'BUY(+1)', -1: 'SELL(-1)', 0: 'HOLD(0)'}[k]
    print(f"    {name:10s}: {v:>8,}  ({v/total*100:.1f}%)")
print(f"  총 샘플      : {total:,}")


# ── 작업 1: 레이블 임계값 탐색 ───────────────────────────────────
def scan_thresholds(close_series, forward_win, thresholds):
    fr = close_series.shift(-forward_win) / close_series - 1
    fr = fr.iloc[:-forward_win]
    n  = len(fr)
    print(f"\n  [임계값 탐색] forward_win={forward_win}봉")
    for t in thresholds:
        buy  = (fr >  t).sum()
        sell = (fr < -t).sum()
        hold = n - buy - sell
        print(f"  threshold={t:.3f} | BUY={buy/n*100:5.1f}%  HOLD={hold/n*100:5.1f}%  SELL={sell/n*100:5.1f}%")

scan_thresholds(df['close'], FORWARD_WIN, [0.001, 0.002, 0.003, 0.005, 0.008])


# ── 작업 2: sample_weight 생성 ───────────────────────────────────
def make_sample_weight(labels, future_returns):
    classes, counts = np.unique(labels, return_counts=True)
    n_total   = len(labels)
    n_classes = len(classes)
    base_w = {c: n_total / (n_classes * cnt) for c, cnt in zip(classes, counts)}

    weights = np.zeros(n_total, dtype=float)
    for i, (lbl, ret) in enumerate(zip(labels, future_returns)):
        bw = base_w[lbl]
        if lbl == 1 or lbl == -1:
            weights[i] = bw * (1 + abs(ret) * 10)
        else:  # HOLD
            weights[i] = bw * 0.5

    print(f"\n  [sample_weight 분포]")
    for c, name in [(-1, 'SELL'), (0, 'HOLD'), (1, 'BUY')]:
        mask = labels == c
        if mask.sum() == 0:
            continue
        w = weights[mask]
        print(f"    {name:4s} | 평균={w.mean():.4f}  최대={w.max():.4f}")
    return weights

# forward_return을 sample_weight 계산용으로 train 구간에서만 사용
forward_return_trimmed = forward_return.iloc[:-FORWARD_WIN].values
sample_weight_all = make_sample_weight(df['label'].values, forward_return_trimmed)


# ════════════════════════════════════════════════════════════════
# Step 4: TimeSeriesSplit 교차 검증
# ════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  Step 4. TimeSeriesSplit 교차 검증")
print("=" * 60)

X = df[FEATURE_COLS].values
y = df['label'].values

# 날짜 기반으로 train/test 분리
test_mask      = df['open_time'] >= pd.Timestamp(TEST_START)
split_idx      = int(test_mask.idxmax())  # 2024-01-01 첫 인덱스
X_train_final  = X[:split_idx]
y_train_final  = y[:split_idx]
sw_train_final = sample_weight_all[:split_idx]
X_test         = X[split_idx:]
y_test         = y[split_idx:]
df_test        = df.iloc[split_idx:].copy()

print(f"  Train/Val : {TRAIN_START} ~ {TRAIN_END}  ({split_idx:,}행)")
print(f"  Test      : {TEST_START} ~ {TEST_END}  ({len(X_test):,}행)")

# 교차 검증은 train 구간(2019~2023)에만 적용
tscv = TimeSeriesSplit(n_splits=N_SPLITS)
fold_scores = []
splits = list(tscv.split(X_train_final))

print(f"\n  총 {N_SPLITS}개 Fold 교차 검증 시작 (train 구간 내)\n")

for fold, (train_idx, test_idx) in enumerate(splits, 1):
    if len(train_idx) < 100 or len(test_idx) < 100:
        print(f"  ⚠️  Fold {fold}: 샘플 부족 — 건너뜀")
        continue

    X_tr, X_te = X_train_final[train_idx], X_train_final[test_idx]
    y_tr, y_te = y_train_final[train_idx], y_train_final[test_idx]
    sw_tr      = sw_train_final[train_idx]

    t0 = time.time()
    model_cv = fit_with_progress(
        RF_PARAMS, X_tr, y_tr, sw_tr,
        desc=f"  Fold {fold}/{N_SPLITS} (train {len(train_idx):,}행)"
    )
    elapsed = time.time() - t0

    y_pred = model_cv.predict(X_te)
    print(f"  → 완료 {elapsed:.1f}s | val {len(test_idx):,}행")
    print(classification_report(y_te, y_pred, target_names=['SELL(-1)', 'HOLD(0)', 'BUY(+1)'],
                                 labels=[-1, 0, 1], zero_division=0))
    fold_scores.append(fold)

print(f"  완료된 Fold: {len(fold_scores)}/{N_SPLITS}")


# ════════════════════════════════════════════════════════════════
# Step 4 (후반): 최종 모델 학습 (2019~2023 전체)
# ════════════════════════════════════════════════════════════════
print(f"\n  최종 모델 학습: {split_idx:,}행 ({TRAIN_START} ~ {TRAIN_END})")
t0 = time.time()
final_model = fit_with_progress(RF_PARAMS, X_train_final, y_train_final, sw_train_final, desc="  최종 모델")
print(f"  학습 완료 ({time.time() - t0:.1f}s) ✓")


# ════════════════════════════════════════════════════════════════
# Step 5: 피처 중요도
# ════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  Step 5. 피처 중요도")
print("=" * 60)

importances = pd.Series(final_model.feature_importances_, index=FEATURE_COLS).sort_values()

fig, ax = plt.subplots(figsize=(8, 6))
importances.plot(kind='barh', ax=ax, color='steelblue')
ax.set_title('Feature Importances (Random Forest)')
ax.set_xlabel('Importance')
ax.grid(axis='x', alpha=0.3)
plt.tight_layout()
try:
    fig.savefig(os.path.join(OUTPUT_DIR, 'feature_importance.png'), dpi=150)
    print(f"  저장: {OUTPUT_DIR}/feature_importance.png")
except Exception as e:
    print(f"  ❌ 저장 실패: {e}")
plt.close()

print(f"\n  Top 5 중요 피처:")
for name, val in importances.sort_values(ascending=False).head(5).items():
    print(f"    {name:<15s}: {val:.4f}")


# ════════════════════════════════════════════════════════════════
# Step 6: 백테스팅 (테스트 구간 20%만 사용)
# ════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  Step 6. 백테스팅")
print("=" * 60)

# ── 확률 기반 순차 포지션 시뮬레이션 ───────────────────────────
proba   = final_model.predict_proba(X_test)
classes = list(final_model.classes_)
idx_buy  = classes.index(1)
idx_sell = classes.index(-1)

pos_arr   = np.zeros(len(df_test), dtype=float)
cur_pos   = 0
bars_held = MIN_HOLD_BARS  # 시작 시 바로 진입 가능

for i in range(len(df_test)):
    pos_arr[i] = cur_pos  # 현재 봉에서의 포지션 기록

    # 현금 상태이거나 최소 보유 시간을 채웠으면 재평가
    if cur_pos == 0 or bars_held >= MIN_HOLD_BARS:
        p_buy  = proba[i, idx_buy]
        p_sell = proba[i, idx_sell]

        if p_buy >= PROB_THRESH_BUY:
            desired = 1
        elif p_sell >= PROB_THRESH_SELL:
            desired = -1
        else:
            desired = 0

        if desired != cur_pos:
            cur_pos   = desired
            bars_held = 0

    bars_held += 1

position = pd.Series(pos_arr, index=df_test.index)

close_test   = df_test['close']
close_return = close_test.pct_change().fillna(0)

# ── 수수료 계산 (롱↔숏 전환은 2배) ─────────────────────────────
pos_prev   = position.shift(1).fillna(0)
changed    = position != pos_prev
flipped    = ((position == 1) & (pos_prev == -1)) | ((position == -1) & (pos_prev == 1))
transaction_cost = changed.astype(float) * 0.001 + flipped.astype(float) * 0.001
strategy_return  = position * close_return - transaction_cost

# ── 누적 수익률 ──────────────────────────────────────────────────
cum_strategy = (1 + strategy_return).cumprod()
cum_bnh      = (1 + close_return).cumprod()

# ── 성과 지표 ────────────────────────────────────────────────────
total_return_strat = cum_strategy.iloc[-1] - 1
total_return_bnh   = cum_bnh.iloc[-1] - 1

std_r  = strategy_return.std()
sharpe = (strategy_return.mean() / std_r * np.sqrt(ANNUAL_BARS)) if std_r > 0 else 0.0

rolling_max = cum_strategy.cummax()
mdd         = ((cum_strategy - rolling_max) / rolling_max).min()

# 진입 시점 (포지션이 바뀌는 봉)
long_entry  = df_test['open_time'][(position == 1)  & (pos_prev != 1)]
short_entry = df_test['open_time'][(position == -1) & (pos_prev != -1)]
exit_entry  = df_test['open_time'][(position == 0)  & (pos_prev != 0)]

n_trades    = int(changed.sum())

# 트레이드 단위 win rate: 진입~청산 구간의 누적 수익이 플러스인 비율
trade_results = []
in_trade = False
trade_return = 0.0
for i in range(len(position)):
    cur = position.iloc[i]
    prev = pos_prev.iloc[i]
    # 진입
    if not in_trade and cur != 0:
        in_trade = True
        trade_return = 0.0
    # 포지션 유지 중 수익 누적
    if in_trade:
        trade_return += strategy_return.iloc[i]
    # 청산 (포지션이 0으로 바뀌거나 반전)
    if in_trade and (cur == 0 or (cur != 0 and cur != prev and prev != 0)):
        trade_results.append(trade_return)
        in_trade = False
        trade_return = 0.0
        # 반전 즉시 새 포지션 시작
        if cur != 0:
            in_trade = True
            trade_return = 0.0
# 마지막 열린 트레이드 처리
if in_trade:
    trade_results.append(trade_return)

win_rate = (sum(1 for r in trade_results if r > 0) / len(trade_results)) if trade_results else 0.0

print(f"  진입 횟수 (롱): {len(long_entry)}")
print(f"  진입 횟수 (숏): {len(short_entry)}")
print(f"  청산 횟수     : {len(exit_entry)}")

# ── 차트 (상단: 가격 + 진입 마커 / 하단: 누적 수익률) ────────────
times = df_test['open_time'].values

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9),
                                gridspec_kw={'height_ratios': [2, 1]}, sharex=True)

# 상단 — BTC 가격 & 진입 마커
ax1.plot(times, close_test.values, color='#aaaaaa', linewidth=0.8, label='BTC Price')
if len(long_entry):
    le_prices = close_test[long_entry.index].values
    ax1.scatter(long_entry.values, le_prices,
                marker='^', color='#2ecc71', s=80, zorder=5, label='Long Entry')
if len(short_entry):
    se_prices = close_test[short_entry.index].values
    ax1.scatter(short_entry.values, se_prices,
                marker='v', color='#e74c3c', s=80, zorder=5, label='Short Entry')
if len(exit_entry):
    ee_prices = close_test[exit_entry.index].values
    ax1.scatter(exit_entry.values, ee_prices,
                marker='o', color='#95a5a6', s=40, zorder=4, label='Exit')
ax1.set_ylabel('Price (USDT)')
ax1.set_title(f'BTC/USDT — Entries & Exits  (buy≥{PROB_THRESH_BUY}, sell≥{PROB_THRESH_SELL}, hold≥{MIN_HOLD_BARS}bars)')
ax1.legend(loc='upper left', fontsize=8)
ax1.grid(alpha=0.2)

# 하단 — 누적 수익률
ax2.plot(times, cum_strategy.values, label='Strategy', color='#2ecc71', linewidth=1.2)
ax2.plot(times, cum_bnh.values,      label='Buy & Hold', color='#3498db', linewidth=1.2, alpha=0.7)
ax2.axhline(1.0, color='gray', linewidth=0.8, linestyle='--')
ax2.set_ylabel('Cumulative Return')
ax2.set_xlabel('Date')
ax2.legend(loc='upper left', fontsize=8)
ax2.grid(alpha=0.2)

plt.tight_layout()
try:
    fig.savefig(os.path.join(OUTPUT_DIR, 'backtest_result.png'), dpi=150)
    print(f"  저장: {OUTPUT_DIR}/backtest_result.png")
except Exception as e:
    print(f"  ❌ 저장 실패: {e}")
plt.close()

if sharpe < 0:
    print("⚠️  Strategy underperforms — review features and labels.")


# ════════════════════════════════════════════════════════════════
# Step 7: 결과 저장
# ════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  Step 7. 결과 저장")
print("=" * 60)

# 모델 저장
try:
    joblib.dump(final_model, os.path.join(OUTPUT_DIR, 'rf_model.pkl'))
    print(f"  저장: {OUTPUT_DIR}/rf_model.pkl")
except Exception as e:
    print(f"  ❌ 모델 저장 실패: {e}")

# 지표 저장
metrics = {
    'period_start'      : str(df_test['open_time'].iloc[0])[:10],
    'period_end'        : str(df_test['open_time'].iloc[-1])[:10],
    'total_trades'      : len(trade_results),
    'win_rate'          : round(float(win_rate) * 100, 2),
    'sharpe_ratio'      : round(float(sharpe), 4),
    'max_drawdown_pct'  : round(float(mdd) * 100, 2),
    'strategy_return_pct': round(float(total_return_strat) * 100, 2),
    'buyhold_return_pct': round(float(total_return_bnh) * 100, 2),
}
try:
    with open(os.path.join(OUTPUT_DIR, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"  저장: {OUTPUT_DIR}/metrics.json")
except Exception as e:
    print(f"  ❌ metrics 저장 실패: {e}")

# 예측값 저장
try:
    pred_df = pd.DataFrame({
        'datetime'   : df_test['open_time'].values,
        'close'      : close_test.values,
        'position'   : position.values,
        'p_buy'      : proba[:, idx_buy],
        'p_sell'     : proba[:, idx_sell],
        'return'     : strategy_return.values,
    })
    pred_df.to_csv(os.path.join(OUTPUT_DIR, 'predictions.csv'), index=False)
    print(f"  저장: {OUTPUT_DIR}/predictions.csv")
except Exception as e:
    print(f"  ❌ predictions 저장 실패: {e}")


# ════════════════════════════════════════════════════════════════
# 최종 요약
# ════════════════════════════════════════════════════════════════
print("\n=== BACKTEST SUMMARY ===")
print(f"Period           : {metrics['period_start']} ~ {metrics['period_end']}")
print(f"Total Trades     : {metrics['total_trades']:,}")
print(f"Win Rate         : {metrics['win_rate']}%")
print(f"Sharpe Ratio     : {metrics['sharpe_ratio']:.2f}")
print(f"Max Drawdown     : {metrics['max_drawdown_pct']:.1f}%")
print(f"Strategy Return  : {metrics['strategy_return_pct']:+.1f}%")
print(f"Buy & Hold Return: {metrics['buyhold_return_pct']:+.1f}%")
print("========================\n")
