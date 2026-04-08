"""
CatBoost Trading Pipeline
BTC/USDT 1분봉 → 1시간봉 리샘플 | 2019-2023 train | 2024 test
6시간봉 후 수익률 예측 | 최소 3시간 포지션 유지 | MA200 레짐 필터
"""

import os
import json
import time
import datetime
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from catboost import CatBoostClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import classification_report, balanced_accuracy_score, f1_score

warnings.filterwarnings('ignore')

# ════════════════════════════════════════════════════════════════
# 설정값
# ════════════════════════════════════════════════════════════════
DATA_PATH     = "/Users/bigdohun/Desktop/Coin-trader/data/historical/BTC_USDT_1m.csv"
OUTPUT_DIR    = "./results"
TRAIN_START   = "2019-01-01"
TRAIN_END     = "2023-12-31"
TEST_START    = "2024-01-01"
TEST_END      = "2024-12-31"

FORWARD_BARS  = 6        # 6시간봉 후 수익률 예측
THRESHOLD     = 0.02    # ±1.5% (시간봉 기준 노이즈 필터)
LABEL_MAP     = {-1: 0, 0: 1, 1: 2}
LABEL_MAP_INV = {0: -1, 1: 0, 2: 1}

N_SPLITS      = 3
ES_VAL_RATIO  = 0.2
TARGET_TRADES = 700      # 연간 목표 거래 횟수
MIN_HOLD_BARS = 6        # 최소 6시간 포지션 유지 (청산 후 재진입은 즉시 가능)
FEE_RATE      = 0.001
ANNUAL_BARS   = 8760     # 시간봉 기준 연간 봉 수 (24 × 365)

# 롱/숏 임계값을 독립적으로 스캔 → 각각 TARGET_TRADES//2 목표
RANDOM_STATE  = 42

CAT_PARAMS = dict(
    iterations            = 1000,
    depth                 = 6,
    learning_rate         = 0.05,
    l2_leaf_reg           = 3.0,
    bootstrap_type        = 'Bernoulli',
    subsample             = 0.8,
    colsample_bylevel     = 0.8,
    min_data_in_leaf      = 20,
    loss_function         = 'MultiClass',
    eval_metric           = 'Accuracy',
    boosting_type         = 'Ordered',
    early_stopping_rounds = 100,
    random_seed           = RANDOM_STATE,
    thread_count          = -1,
    verbose               = 100,
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ════════════════════════════════════════════════════════════════
# 유틸
# ════════════════════════════════════════════════════════════════
def fmt_sec(sec):
    sec = max(0, int(sec))
    if sec >= 3600:
        return f"{sec//3600}시간 {(sec%3600)//60}분 {sec%60}초"
    elif sec >= 60:
        return f"{sec//60}분 {sec%60}초"
    return f"{sec}초"


def now():
    return datetime.datetime.now().strftime('%H:%M:%S')


class ETA:
    """단계 수를 기반으로 예상 잔여 시간 계산."""
    def __init__(self, total):
        self.total   = total
        self.done    = 0
        self.start   = time.time()

    def step(self):
        self.done += 1

    def remaining(self):
        if self.done == 0:
            return 0
        elapsed = time.time() - self.start
        return elapsed / self.done * (self.total - self.done)

    def elapsed(self):
        return time.time() - self.start


# ════════════════════════════════════════════════════════════════
# 피처 엔지니어링
# ════════════════════════════════════════════════════════════════
def _rsi(series, period=14):
    delta = series.diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    return 100 - (100 / (1 + gain / (loss + 1e-10)))


def _atr(high, low, close, period=14):
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low  - close.shift(1)).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def _adx(high, low, close, period=14):
    up, down   = high.diff(), -low.diff()
    plus_dm    = up.where((up > down) & (up > 0), 0.0)
    minus_dm   = down.where((down > up) & (down > 0), 0.0)
    atr_s      = _atr(high, low, close, period)
    plus_di    = 100 * plus_dm.rolling(period).mean()  / (atr_s + 1e-10)
    minus_di   = 100 * minus_dm.rolling(period).mean() / (atr_s + 1e-10)
    dx         = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-10)
    return dx.rolling(period).mean()


def make_features(df):
    close, high, low, open_, volume = (
        df['close'], df['high'], df['low'], df['open'], df['volume']
    )
    feat = pd.DataFrame(index=df.index)

    # 모멘텀 (시간봉 기준: 1h, 6h, 12h, 24h, 48h)
    for n in [1, 6, 12, 24, 48]:
        feat[f'return_{n}'] = close.pct_change(n).shift(1)

    # 트렌드
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd  = ema12 - ema26
    feat['macd_hist']   = (macd - macd.ewm(span=9, adjust=False).mean()).shift(1)
    ema9  = close.ewm(span=9,  adjust=False).mean()
    ema21 = close.ewm(span=21, adjust=False).mean()
    feat['ema_gap_pct'] = ((ema9 - ema21) / (ema21 + 1e-10)).shift(1)
    feat['adx_14']      = _adx(high, low, close, 14).shift(1)

    # 변동성
    atr14 = _atr(high, low, close, 14)
    feat['atr_ratio']    = (atr14 / (close + 1e-10)).shift(1)
    feat['atr_ma_ratio'] = (atr14 / (atr14.rolling(50).mean() + 1e-10)).shift(1)
    ma20  = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    feat['bb_width']     = ((4 * std20) / (ma20 + 1e-10)).shift(1)
    gk = np.sqrt(
        0.5 * np.log((high / low).clip(1e-10))**2 -
        (2 * np.log(2) - 1) * np.log((close / open_.clip(1e-10)).clip(1e-10))**2
    )
    feat['gk_vol_ratio'] = (gk / (gk.rolling(50).mean() + 1e-10)).shift(1)

    # 미시구조
    feat['amihud'] = (
        close.pct_change().abs() / (close * volume + 1e-9)
    ).rolling(20).mean().shift(1)
    feat['kyle_lambda_ratio'] = (
        close.diff().abs() / (volume + 1e-9) /
        ((close.diff().abs() / (volume + 1e-9)).rolling(50).mean() + 1e-9)
    ).shift(1)

    # 가격 위치
    feat['rsi_14']            = _rsi(close, 14)
    high20 = high.rolling(20).max()
    low20  = low.rolling(20).min()
    feat['price_position_20'] = ((close - low20)  / (high20 - low20 + 1e-9)).shift(1)
    feat['dist_from_high_20'] = ((high20 - close) / (close + 1e-9)).shift(1)
    feat['dist_from_low_20']  = ((close - low20)  / (close + 1e-9)).shift(1)
    vwap = (close * volume).rolling(60).sum() / (volume.rolling(60).sum() + 1e-9)
    feat['vwap_deviation']    = ((close - vwap) / (vwap + 1e-9)).shift(1)
    feat['price_zscore_50']   = (
        (close - close.rolling(50).mean()) / (close.rolling(50).std() + 1e-9)
    ).shift(1)

    # 통계
    feat['return_kurt_20'] = close.pct_change().rolling(20).kurt().shift(1)

    # 캔들
    hl = (high - low).clip(1e-9)
    feat['candle_body'] = ((close - open_).abs() / hl).shift(1)
    upper = (high - pd.concat([open_, close], axis=1).max(axis=1)) / hl
    lower = (pd.concat([open_, close], axis=1).min(axis=1) - low) / hl
    feat['wickedness']  = (upper + lower).shift(1)

    # 시간
    hour = df['open_time'].dt.hour
    dow  = df['open_time'].dt.dayofweek
    feat['hour_sin'] = np.sin(2 * np.pi * hour / 24)
    feat['hour_cos'] = np.cos(2 * np.pi * hour / 24)
    feat['dow_sin']  = np.sin(2 * np.pi * dow  / 7)

    return feat


FEATURE_COLS = [
    'return_1', 'return_6', 'return_12', 'return_24', 'return_48',
    'ema_gap_pct', 'macd_hist', 'adx_14',
    'atr_ratio', 'atr_ma_ratio', 'gk_vol_ratio', 'bb_width',
    'amihud', 'kyle_lambda_ratio',
    'rsi_14', 'price_position_20', 'dist_from_high_20', 'dist_from_low_20',
    'vwap_deviation', 'price_zscore_50',
    'return_kurt_20',
    'candle_body', 'wickedness',
    'hour_sin', 'hour_cos', 'dow_sin',
]


def make_sample_weight(labels, future_returns):
    classes, counts = np.unique(labels, return_counts=True)
    n_total  = len(labels)
    n_cls    = len(classes)
    base_w   = {c: n_total / (n_cls * cnt) for c, cnt in zip(classes, counts)}
    weights  = np.array([
        base_w[lbl] * (1 + abs(ret) * 10) if lbl != 0 else base_w[lbl]
        for lbl, ret in zip(labels, future_returns)
    ], dtype=float)
    return weights


def simulate_positions(proba, thresh_buy, thresh_sell, min_hold, regime=None):
    """proba: (N,3), 컬럼 순서 [SELL(0), HOLD(1), BUY(2)]
    regime: array of int, 1=상승장(MA200 위), -1=하락장(MA200 아래), None=필터 없음
      - 상승장(1):  SELL 신호 무시 → 롱 온리
      - 하락장(-1): BUY  신호 무시 → 숏 온리
    롱↔숏 직접 전환 허용 (수수료는 run_backtest에서 2×FEE_RATE 부과)
    """
    pos_arr   = np.zeros(len(proba), dtype=float)
    cur_pos   = 0
    bars_held = min_hold
    for i in range(len(proba)):
        pos_arr[i] = cur_pos
        if cur_pos == 0 or bars_held >= min_hold:
            p_buy  = proba[i, 2]
            p_sell = proba[i, 0]
            # MA200 레짐 필터
            if regime is not None:
                if regime[i] == 1:    # 상승장: 숏 신호 무시
                    p_sell = 0.0
                elif regime[i] == -1: # 하락장: 롱 신호 무시
                    p_buy  = 0.0
            if p_buy >= thresh_buy:
                desired = 1
            elif p_sell >= thresh_sell:
                desired = -1
            else:
                desired = 0
            if desired != cur_pos:
                cur_pos   = desired
                bars_held = 0
        bars_held += 1
    pos      = pd.Series(pos_arr)
    n_trades = int((pos != pos.shift(1).fillna(0)).sum())
    return pos_arr, n_trades


def _count_long_short(proba, thresh_buy, thresh_sell, min_hold, regime=None):
    """롱/숏 진입 횟수를 개별로 반환."""
    pos_arr, _ = simulate_positions(proba, thresh_buy, thresh_sell, min_hold, regime)
    pos  = pd.Series(pos_arr)
    prev = pos.shift(1).fillna(0)
    n_long  = int(((pos == 1)  & (prev != 1)).sum())
    n_short = int(((pos == -1) & (prev != -1)).sum())
    return n_long, n_short


def run_backtest(proba, df_test, regime=None):
    thresholds  = np.arange(0.20, 0.80, 0.01)
    target_each = TARGET_TRADES // 2   # 롱/숏 각각 목표

    # 롱 임계값 독립 스캔 (숏은 0.99 고정 → 롱만 카운트)
    buy_scan  = [(round(t, 2), _count_long_short(proba, round(t, 2), 0.99, MIN_HOLD_BARS, regime)[0])
                 for t in thresholds]
    # 숏 임계값 독립 스캔 (롱은 0.99 고정 → 숏만 카운트)
    sell_scan = [(round(t, 2), _count_long_short(proba, 0.99, round(t, 2), MIN_HOLD_BARS, regime)[1])
                 for t in thresholds]

    thresh_buy,  best_long  = min(buy_scan,  key=lambda x: abs(x[1] - target_each))
    thresh_sell, best_short = min(sell_scan, key=lambda x: abs(x[1] - target_each))

    print(f"  목표 거래 (각 방향): {target_each}회")
    print(f"\n  {'thresh':>7} | {'롱':>6} || {'thresh':>7} | {'숏':>6}")
    for (tb, nl), (ts, ns) in zip(buy_scan, sell_scan):
        lb = " ◀" if tb == thresh_buy  else ""
        ls = " ◀" if ts == thresh_sell else ""
        print(f"  {tb:>7.2f} | {nl:>6}{lb:<3}   {ts:>7.2f} | {ns:>6}{ls}")
    print(f"\n  최적 롱 임계값  : {thresh_buy:.2f}  (→ 롱 {best_long}회 목표)")
    print(f"  최적 숏 임계값  : {thresh_sell:.2f}  (→ 숏 {best_short}회 목표)")

    # 최적 임계값으로 최종 시뮬레이션
    pos_arr, final_n = simulate_positions(proba, thresh_buy, thresh_sell, MIN_HOLD_BARS, regime)
    n_long_final, n_short_final = _count_long_short(proba, thresh_buy, thresh_sell, MIN_HOLD_BARS, regime)
    print(f"  최종 거래 횟수  : 롱 {n_long_final}회 + 숏 {n_short_final}회 = {n_long_final+n_short_final}회")

    position    = pd.Series(pos_arr, index=df_test.index)
    close_test  = df_test['close']
    close_ret   = close_test.pct_change().fillna(0)
    pos_prev    = position.shift(1).fillna(0)
    changed     = position != pos_prev
    flipped     = ((position == 1) & (pos_prev == -1)) | ((position == -1) & (pos_prev == 1))
    tx_cost     = changed.astype(float) * FEE_RATE + flipped.astype(float) * FEE_RATE
    strat_ret   = position * close_ret - tx_cost
    cum_strat   = (1 + strat_ret).cumprod()
    cum_bnh     = (1 + close_ret).cumprod()

    total_ret   = cum_strat.iloc[-1] - 1
    bnh_ret     = cum_bnh.iloc[-1] - 1
    std_r       = strat_ret.std()
    sharpe      = (strat_ret.mean() / std_r * np.sqrt(ANNUAL_BARS)) if std_r > 0 else 0.0
    mdd         = ((cum_strat - cum_strat.cummax()) / cum_strat.cummax()).min()
    n_trades    = int(changed.sum())

    # 트레이드 단위 win rate
    trade_results, in_trade, trade_return = [], False, 0.0
    for i in range(len(position)):
        cur, prev = position.iloc[i], pos_prev.iloc[i]
        if not in_trade and cur != 0:
            in_trade, trade_return = True, 0.0
        if in_trade:
            trade_return += strat_ret.iloc[i]
        if in_trade and (cur == 0 or (cur != 0 and cur != prev and prev != 0)):
            trade_results.append(trade_return)
            in_trade, trade_return = False, 0.0
            if cur != 0:
                in_trade, trade_return = True, 0.0
    if in_trade:
        trade_results.append(trade_return)
    win_rate = (sum(1 for r in trade_results if r > 0) / len(trade_results)) if trade_results else 0.0

    long_entries  = (position == 1)  & (pos_prev != 1)
    short_entries = (position == -1) & (pos_prev != -1)
    exits         = (position == 0)  & (pos_prev != 0)

    return {
        'total_return' : round(float(total_ret) * 100, 2),
        'bnh_return'   : round(float(bnh_ret) * 100, 2),
        'sharpe'       : round(float(sharpe), 4),
        'mdd'          : round(float(mdd) * 100, 2),
        'n_trades'     : n_trades,
        'win_rate'     : round(win_rate * 100, 2),
        'thresh_buy'   : thresh_buy,
        'thresh_sell'  : thresh_sell,
        'long_entries' : int(long_entries.sum()),
        'short_entries': int(short_entries.sum()),
        'exits'        : int(exits.sum()),
        'pct_long'     : round(float((position == 1).mean()  * 100), 1),
        'pct_short'    : round(float((position == -1).mean() * 100), 1),
        'pct_flat'     : round(float((position == 0).mean()  * 100), 1),
    }, position, cum_strat, cum_bnh, long_entries, short_entries, exits


# ════════════════════════════════════════════════════════════════
# Step 1. 데이터 로드 & 피처 엔지니어링
# ════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  Step 1. 데이터 로드 & 피처 엔지니어링")
print(f"  시작: {now()}")
print("=" * 60)

t_step = time.time()
df = pd.read_csv(DATA_PATH)
if 'datetime' in df.columns:
    df['open_time'] = pd.to_datetime(df['datetime'])
elif 'timestamp' in df.columns:
    df['open_time'] = pd.to_datetime(df['timestamp'], unit='ms')
else:
    raise ValueError("시간 컬럼 없음")

for col in ['open', 'high', 'low', 'close', 'volume']:
    df[col] = pd.to_numeric(df[col], errors='coerce')

df = df.sort_values('open_time').reset_index(drop=True)
df = df[(df['open_time'] >= pd.Timestamp(TRAIN_START)) &
        (df['open_time'] <= pd.Timestamp(TEST_END))].reset_index(drop=True)
df = df.drop_duplicates(subset=['open_time']).reset_index(drop=True)
df[['open','high','low','close','volume']] = (
    df[['open','high','low','close','volume']].fillna(method='ffill', limit=3)
)
df = df.dropna(subset=['open','high','low','close','volume']).reset_index(drop=True)

# 1분봉 → 1시간봉 리샘플링
print(f"  1분봉 shape: {df.shape}")
df = df.set_index('open_time')
df = df[['open','high','low','close','volume']].resample('1h').agg({
    'open'  : 'first',
    'high'  : 'max',
    'low'   : 'min',
    'close' : 'last',
    'volume': 'sum',
}).dropna().reset_index()
print(f"  1시간봉 shape: {df.shape}")
print(f"  기간   : {df['open_time'].iloc[0]} ~ {df['open_time'].iloc[-1]}")

feat = make_features(df)
df   = pd.concat([df, feat], axis=1)
df[FEATURE_COLS] = df[FEATURE_COLS].replace([np.inf, -np.inf], np.nan)
df   = df.dropna(subset=FEATURE_COLS).reset_index(drop=True)

# MA200 레짐 계산 (현재 봉 종가 vs 200시간 이동평균)
# 1: 상승장(close > MA200) → 숏 신호 무시
# -1: 하락장(close < MA200) → 롱 신호 무시
df['ma200']  = df['close'].rolling(200).mean()
df['regime'] = np.where(df['close'] > df['ma200'], 1, -1)

bull_pct = (df['regime'] == 1).mean() * 100
print(f"  피처   : {len(FEATURE_COLS)}개 | NaN 제거 후 {len(df):,}행")
print(f"  레짐   : 상승장 {bull_pct:.1f}%  |  하락장 {100-bull_pct:.1f}%")
print(f"  완료   : {fmt_sec(time.time() - t_step)}")


# ════════════════════════════════════════════════════════════════
# Step 2. 레이블 & sample_weight 생성
# ════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  Step 2. 레이블 생성")
print("=" * 60)

close_s        = df['close']
forward_return = close_s.shift(-FORWARD_BARS) / close_s - 1
label          = pd.Series(0, index=df.index)
label[forward_return >  THRESHOLD] =  1
label[forward_return < -THRESHOLD] = -1

df                     = df.iloc[:-FORWARD_BARS].copy()
label                  = label.iloc[:-FORWARD_BARS].copy()
forward_return_trimmed = forward_return.iloc[:-FORWARD_BARS].values
df['label']            = label.values

dist  = df['label'].value_counts().sort_index()
total = len(df)
for k, v in dist.items():
    name = {1: 'BUY(+1)', -1: 'SELL(-1)', 0: 'HOLD(0)'}[k]
    print(f"  {name:10s}: {v:>8,}  ({v/total*100:.1f}%)")
print(f"  총 샘플  : {total:,}")

sample_weight_all = make_sample_weight(df['label'].values, forward_return_trimmed)


# ════════════════════════════════════════════════════════════════
# Step 3. 데이터 분할
# ════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  Step 3. 데이터 분할")
print("=" * 60)

X = df[FEATURE_COLS].values
y = df['label'].values

test_mask      = df['open_time'] >= pd.Timestamp(TEST_START)
split_idx      = int(test_mask.idxmax())
X_train_final  = X[:split_idx]
y_train_final  = y[:split_idx]
sw_train_final = sample_weight_all[:split_idx]
X_test         = X[split_idx:]
y_test         = y[split_idx:]
df_test        = df.iloc[split_idx:].copy()
regime_test    = df['regime'].iloc[split_idx:].values   # MA200 레짐 (테스트셋)

y_train_idx = np.array([LABEL_MAP[v] for v in y_train_final])
y_test_idx  = np.array([LABEL_MAP[v] for v in y_test])

n_val        = int(len(X_train_final) * ES_VAL_RATIO)
X_tr_es      = X_train_final[:-n_val]
y_tr_es_idx  = y_train_idx[:-n_val]
sw_tr_es     = sw_train_final[:-n_val]
X_val_es     = X_train_final[-n_val:]
y_val_es_idx = y_train_idx[-n_val:]

bull_test_pct = (regime_test == 1).mean() * 100
print(f"  Train  : {TRAIN_START} ~ {TRAIN_END}  ({split_idx:,}행)")
print(f"  Test   : {TEST_START} ~ {TEST_END}  ({len(X_test):,}행)")
print(f"  ES val : train 뒤 {n_val:,}행 ({ES_VAL_RATIO*100:.0f}%)")
print(f"  테스트 레짐: 상승장 {bull_test_pct:.1f}%  |  하락장 {100-bull_test_pct:.1f}%")


# ════════════════════════════════════════════════════════════════
# Step 4. TimeSeriesSplit 교차검증
# ════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  Step 4. TimeSeriesSplit 교차검증")
print(f"  시작: {now()}")
print("=" * 60)

tscv          = TimeSeriesSplit(n_splits=N_SPLITS)
splits        = list(tscv.split(X_train_final))
cv_bal_accs   = []
cv_f1s        = []
cv_best_iters = []
eta_cv        = ETA(total=N_SPLITS)

for fold, (tr_idx, te_idx) in enumerate(splits, 1):
    if len(tr_idx) < 100 or len(te_idx) < 100:
        continue

    X_tr, X_te   = X_train_final[tr_idx], X_train_final[te_idx]
    y_tr_idx_f   = y_train_idx[tr_idx]
    y_te_raw     = y_train_final[te_idx]
    sw_tr_f      = sw_train_final[tr_idx]

    n_v          = int(len(X_tr) * ES_VAL_RATIO)
    X_tv         = X_tr[-n_v:]
    y_tv_idx     = y_tr_idx_f[-n_v:]
    X_tr_f       = X_tr[:-n_v]
    y_tr_f_idx   = y_tr_idx_f[:-n_v]
    sw_tr_f2     = sw_tr_f[:-n_v]

    remaining = fmt_sec(eta_cv.remaining())
    print(f"\n  ── Fold {fold}/{N_SPLITS}  │  경과 {fmt_sec(eta_cv.elapsed())}  │  예상 잔여 {remaining}")
    print(f"     train {len(X_tr_f):,}행 / val(ES) {n_v:,}행 / eval {len(X_te):,}행")

    t0      = time.time()
    cat_cv  = CatBoostClassifier(**CAT_PARAMS)
    cat_cv.fit(X_tr_f, y_tr_f_idx, sample_weight=sw_tr_f2,
               eval_set=(X_tv, y_tv_idx))

    elapsed = time.time() - t0
    eta_cv.step()

    y_pred_idx = cat_cv.predict(X_te).flatten().astype(int)
    y_pred     = np.array([LABEL_MAP_INV[v] for v in y_pred_idx])

    bal_acc = balanced_accuracy_score(y_te_raw, y_pred)
    f1      = f1_score(y_te_raw, y_pred, average='macro', zero_division=0)
    cv_bal_accs.append(bal_acc)
    cv_f1s.append(f1)
    cv_best_iters.append(cat_cv.best_iteration_)

    print(f"  완료: {fmt_sec(elapsed)}  │  예상 잔여 {fmt_sec(eta_cv.remaining())}")
    print(classification_report(y_te_raw, y_pred,
          target_names=['SELL(-1)', 'HOLD(0)', 'BUY(+1)'],
          labels=[-1, 0, 1], zero_division=0))

print(f"\n  CV 전체 소요: {fmt_sec(eta_cv.elapsed())}")
print(f"  bal_acc: {np.mean(cv_bal_accs):.4f} ± {np.std(cv_bal_accs):.4f}")
print(f"  f1_macro: {np.mean(cv_f1s):.4f} ± {np.std(cv_f1s):.4f}")

# CV best_iteration 중 최댓값을 최종 모델 n_estimators로 사용
# (ES val이 훈련 분포와 달라 조기 종료가 너무 일찍 발동되는 문제 방지)
final_n_iters = max(max(cv_best_iters, default=300), 300)
print(f"  CV best iterations: {cv_best_iters}  →  최종 모델 n_estimators: {final_n_iters}")


# ════════════════════════════════════════════════════════════════
# Step 5. 최종 모델 학습
# ════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  Step 5. 최종 모델 학습")
print(f"  시작: {now()}")
print("=" * 60)

# 전체 훈련 데이터로 고정 iteration 학습 (조기 종료 없음)
final_params = {k: v for k, v in CAT_PARAMS.items()
                if k not in ('early_stopping_rounds',)}
final_params['iterations'] = final_n_iters
final_params['verbose']    = 100

t0        = time.time()
cat_model = CatBoostClassifier(**final_params)
cat_model.fit(X_train_final, y_train_idx, sample_weight=sample_weight_all[:split_idx])
train_elapsed = time.time() - t0
print(f"  완료: {fmt_sec(train_elapsed)}  │  {now()}")


# ════════════════════════════════════════════════════════════════
# Step 6. 테스트셋 평가
# ════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  Step 6. 테스트셋 평가")
print("=" * 60)

y_pred_test_idx = cat_model.predict(X_test).flatten().astype(int)
y_pred_test     = np.array([LABEL_MAP_INV[v] for v in y_pred_test_idx])

test_bal_acc = balanced_accuracy_score(y_test, y_pred_test)
test_f1      = f1_score(y_test, y_pred_test, average='macro', zero_division=0)

print(classification_report(y_test, y_pred_test,
      target_names=['SELL(-1)', 'HOLD(0)', 'BUY(+1)'],
      labels=[-1, 0, 1], zero_division=0))
print(f"  balanced_accuracy : {test_bal_acc:.4f}")
print(f"  f1_macro          : {test_f1:.4f}")


# ════════════════════════════════════════════════════════════════
# Step 7. 백테스팅
# ════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  Step 7. 백테스팅")
print("=" * 60)

proba = cat_model.predict_proba(X_test)   # (N,3): [SELL, HOLD, BUY]
bt, position, cum_strat, cum_bnh, long_e, short_e, exits = run_backtest(proba, df_test, regime_test)

print(f"\n  롱 진입: {bt['long_entries']}회  │  숏 진입: {bt['short_entries']}회  │  청산: {bt['exits']}회")

print(f"\n=== BACKTEST SUMMARY ===")
print(f"Period          : {TEST_START} ~ {TEST_END}")
print(f"Thresh Buy/Sell : {bt['thresh_buy']:.2f} / {bt['thresh_sell']:.2f}")
print(f"Total Trades    : {bt['n_trades']}")
print(f"Win Rate        : {bt['win_rate']}%")
print(f"Sharpe Ratio    : {bt['sharpe']:.2f}")
print(f"Max Drawdown    : {bt['mdd']:.1f}%")
print(f"Strategy Return : {bt['total_return']:+.1f}%")
print(f"Buy&Hold Return : {bt['bnh_return']:+.1f}%")
print(f"--- 포지션 보유 비율 ---")
print(f"롱  보유: {bt['pct_long']:>5.1f}%")
print(f"숏  보유: {bt['pct_short']:>5.1f}%")
print(f"청산(flat): {bt['pct_flat']:>5.1f}%")
print("========================")


# ════════════════════════════════════════════════════════════════
# Step 8. 결과 저장
# ════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  Step 8. 결과 저장")
print("=" * 60)

# 모델
cat_model.save_model(os.path.join(OUTPUT_DIR, 'catboost_model.cbm'))
print(f"  저장: {OUTPUT_DIR}/catboost_model.cbm")

# metrics
metrics = {
    'train_period'    : f"{TRAIN_START} ~ {TRAIN_END}",
    'test_period'     : f"{TEST_START} ~ {TEST_END}",
    'cv_bal_acc_mean' : round(float(np.mean(cv_bal_accs)), 4),
    'cv_bal_acc_std'  : round(float(np.std(cv_bal_accs)), 4),
    'cv_f1_mean'      : round(float(np.mean(cv_f1s)), 4),
    'test_bal_acc'    : round(float(test_bal_acc), 4),
    'test_f1_macro'   : round(float(test_f1), 4),
    'train_sec'       : round(train_elapsed, 1),
    **bt,
}
with open(os.path.join(OUTPUT_DIR, 'catboost_metrics.json'), 'w') as f:
    json.dump(metrics, f, indent=2, ensure_ascii=False)
print(f"  저장: {OUTPUT_DIR}/catboost_metrics.json")

# predictions
pred_df = pd.DataFrame({
    'datetime' : df_test['open_time'].values,
    'close'    : df_test['close'].values,
    'regime'   : regime_test,
    'position' : position.values,
    'p_sell'   : proba[:, 0],
    'p_hold'   : proba[:, 1],
    'p_buy'    : proba[:, 2],
})
pred_df.to_csv(os.path.join(OUTPUT_DIR, 'catboost_predictions.csv'), index=False)
print(f"  저장: {OUTPUT_DIR}/catboost_predictions.csv")

# feature importance
imp = pd.Series(cat_model.get_feature_importance(), index=FEATURE_COLS).sort_values()
fig, ax = plt.subplots(figsize=(8, 8))
imp.plot(kind='barh', ax=ax, color='mediumpurple')
ax.set_title('CatBoost Feature Importance')
ax.set_xlabel('Importance')
ax.grid(axis='x', alpha=0.3)
plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, 'catboost_feature_importance.png'), dpi=150)
plt.close()
print(f"  저장: {OUTPUT_DIR}/catboost_feature_importance.png")

# 백테스트 차트
times      = df_test['open_time'].values
close_vals = df_test['close'].values
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9),
                                gridspec_kw={'height_ratios': [2, 1]}, sharex=True)
ax1.plot(times, close_vals, color='#aaaaaa', linewidth=0.8, label='BTC Price')
if long_e.sum():
    ax1.scatter(df_test['open_time'][long_e].values,
                df_test['close'][long_e].values,
                marker='^', color='#2ecc71', s=80, zorder=5, label='Long Entry')
if short_e.sum():
    ax1.scatter(df_test['open_time'][short_e].values,
                df_test['close'][short_e].values,
                marker='v', color='#e74c3c', s=80, zorder=5, label='Short Entry')
if exits.sum():
    ax1.scatter(df_test['open_time'][exits].values,
                df_test['close'][exits].values,
                marker='x', color='#f39c12', s=60, linewidths=1.5,
                zorder=5, label='Exit')
ax1.set_ylabel('Price (USDT)')
ax1.set_title(f'CatBoost — Entries & Exits  (buy≥{bt["thresh_buy"]:.2f}, sell≥{bt["thresh_sell"]:.2f}, hold≥{MIN_HOLD_BARS}bars)')
ax1.legend(loc='upper left', fontsize=8)
ax1.grid(alpha=0.2)

ax2.plot(times, cum_strat.values, label='CatBoost', color='#9b59b6', linewidth=1.2)
ax2.plot(times, cum_bnh.values,   label='Buy & Hold', color='#3498db',
         linewidth=1.2, linestyle='--', alpha=0.8)
ax2.axhline(1.0, color='gray', linewidth=0.8, linestyle=':')
ax2.set_ylabel('Cumulative Return')
ax2.set_xlabel('Date')
ax2.legend(loc='upper left', fontsize=8)
ax2.grid(alpha=0.2)
plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, 'catboost_backtest.png'), dpi=150)
plt.close()
print(f"  저장: {OUTPUT_DIR}/catboost_backtest.png")

print(f"\n  모든 결과 저장 완료 ({now()})")
