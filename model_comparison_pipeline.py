"""
Model Comparison Pipeline: RF vs XGBoost vs CatBoost
BTC/USDT 1-minute candles | 2019-2023 train | 2024 test
"""

import os
import json
import time
import datetime
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (classification_report, balanced_accuracy_score,
                             f1_score)
import xgboost as xgb
from catboost import CatBoostClassifier

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

FORWARD_BARS  = 240        # 4시간
THRESHOLD     = 0.004      # ±0.4%
LABEL_MAP     = {-1: 0, 0: 1, 1: 2}   # XGB/CatBoost 0-indexed
LABEL_MAP_INV = {0: -1, 1: 0, 2: 1}   # 역변환

N_SPLITS      = 3
ES_VAL_RATIO  = 0.2        # early stopping용 val 비율
TARGET_TRADES = 500
MIN_HOLD_BARS = 240
FEE_RATE      = 0.001
ANNUAL_BARS   = 525600
RANDOM_STATE  = 42

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ════════════════════════════════════════════════════════════════
# 모델 정의
# ════════════════════════════════════════════════════════════════
RF_PARAMS = dict(
    n_estimators     = 300,
    max_depth        = 8,
    max_features     = 'sqrt',
    min_samples_leaf = 20,
    random_state     = RANDOM_STATE,
    n_jobs           = -1,
)

XGB_PARAMS = dict(
    n_estimators          = 1000,
    max_depth             = 5,
    learning_rate         = 0.05,
    subsample             = 0.8,
    colsample_bytree      = 0.8,
    min_child_weight      = 20,
    reg_alpha             = 0.1,
    reg_lambda            = 1.0,
    objective             = 'multi:softprob',
    num_class             = 3,
    eval_metric           = 'mlogloss',
    early_stopping_rounds = 50,
    tree_method           = 'hist',
    random_state          = RANDOM_STATE,
    n_jobs                = -1,
)

CAT_PARAMS = dict(
    iterations            = 1000,
    depth                 = 6,
    learning_rate         = 0.05,
    l2_leaf_reg           = 3.0,
    bootstrap_type        = 'Bernoulli',  # Ordered boosting에서 subsample 사용 시 필요
    subsample             = 0.8,
    colsample_bylevel     = 0.8,
    min_data_in_leaf      = 20,
    loss_function         = 'MultiClass',
    eval_metric           = 'TotalF1',
    boosting_type         = 'Ordered',
    early_stopping_rounds = 50,
    random_seed           = RANDOM_STATE,
    thread_count          = -1,
    verbose               = 100,
)


# ════════════════════════════════════════════════════════════════
# 헬퍼 함수
# ════════════════════════════════════════════════════════════════
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
    up       = high.diff()
    down     = -low.diff()
    plus_dm  = up.where((up > down) & (up > 0), 0.0)
    minus_dm = down.where((down > up) & (down > 0), 0.0)
    atr_s    = atr(high, low, close, period)
    plus_di  = 100 * plus_dm.rolling(period).mean()  / (atr_s + 1e-10)
    minus_di = 100 * minus_dm.rolling(period).mean() / (atr_s + 1e-10)
    dx       = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-10)
    return dx.rolling(period).mean()


def make_features(df):
    close  = df['close']
    high   = df['high']
    low    = df['low']
    open_  = df['open']
    volume = df['volume']

    feat = pd.DataFrame(index=df.index)

    # 모멘텀
    for n in [1, 5, 10, 30, 60]:
        feat[f'return_{n}'] = close.pct_change(n).shift(1)

    # 트렌드
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd_line   = ema12 - ema26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    feat['macd_hist']   = (macd_line - signal_line).shift(1)

    ema9  = close.ewm(span=9,  adjust=False).mean()
    ema21 = close.ewm(span=21, adjust=False).mean()
    feat['ema_gap_pct'] = ((ema9 - ema21) / (ema21 + 1e-10)).shift(1)

    feat['adx_14'] = adx(high, low, close, 14).shift(1)

    # 변동성
    atr14 = atr(high, low, close, 14)
    feat['atr_ratio']    = (atr14 / (close + 1e-10)).shift(1)
    feat['atr_ma_ratio'] = (atr14 / (atr14.rolling(50).mean() + 1e-10)).shift(1)

    ma20     = close.rolling(20).mean()
    std20    = close.rolling(20).std()
    bb_upper = ma20 + 2 * std20
    bb_lower = ma20 - 2 * std20
    feat['bb_width'] = ((bb_upper - bb_lower) / (ma20 + 1e-10)).shift(1)

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
    feat['rsi_14'] = rsi(close, 14)

    high20_max = high.rolling(20).max()
    low20_min  = low.rolling(20).min()
    feat['price_position_20'] = ((close - low20_min) / (high20_max - low20_min + 1e-9)).shift(1)
    feat['dist_from_high_20'] = ((high20_max - close) / (close + 1e-9)).shift(1)
    feat['dist_from_low_20']  = ((close - low20_min)  / (close + 1e-9)).shift(1)

    vwap = (close * volume).rolling(60).sum() / (volume.rolling(60).sum() + 1e-9)
    feat['vwap_deviation']  = ((close - vwap) / (vwap + 1e-9)).shift(1)
    feat['price_zscore_50'] = ((close - close.rolling(50).mean()) / (close.rolling(50).std() + 1e-9)).shift(1)

    # 통계
    feat['return_kurt_20'] = close.pct_change().rolling(20).kurt().shift(1)

    # 캔들
    hl_range = (high - low).clip(1e-9)
    feat['candle_body'] = ((close - open_).abs() / hl_range).shift(1)
    upper_shadow = (high - pd.concat([open_, close], axis=1).max(axis=1)) / hl_range
    lower_shadow = (pd.concat([open_, close], axis=1).min(axis=1) - low)  / hl_range
    feat['wickedness'] = (upper_shadow + lower_shadow).shift(1)

    # 시간
    hour = df['open_time'].dt.hour
    dow  = df['open_time'].dt.dayofweek
    feat['hour_sin'] = np.sin(2 * np.pi * hour / 24)
    feat['hour_cos'] = np.cos(2 * np.pi * hour / 24)
    feat['dow_sin']  = np.sin(2 * np.pi * dow / 7)

    return feat


FEATURE_COLS = [
    'return_1', 'return_5', 'return_10', 'return_30', 'return_60',
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
    n_total   = len(labels)
    n_classes = len(classes)
    base_w    = {c: n_total / (n_classes * cnt) for c, cnt in zip(classes, counts)}
    weights   = np.zeros(n_total, dtype=float)
    for i, (lbl, ret) in enumerate(zip(labels, future_returns)):
        bw = base_w[lbl]
        weights[i] = bw * (1 + abs(ret) * 10) if lbl != 0 else bw
    return weights


def simulate_positions(proba, idx_buy, idx_sell, thresh_buy, thresh_sell, min_hold):
    pos_arr   = np.zeros(len(proba), dtype=float)
    cur_pos   = 0
    bars_held = min_hold
    for i in range(len(proba)):
        pos_arr[i] = cur_pos
        if cur_pos == 0 or bars_held >= min_hold:
            p_buy  = proba[i, idx_buy]
            p_sell = proba[i, idx_sell]
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


def run_backtest(proba_raw, df_test, classes_list, model_name):
    """proba_raw: shape (N, 3), classes_list: [-1, 0, 1] 또는 [0, 1, 2]"""
    # 클래스 인덱스 통일 (-1/0/1 기준)
    if -1 in classes_list:
        idx_buy  = classes_list.index(1)
        idx_sell = classes_list.index(-1)
    else:
        # XGB/CatBoost: 0→SELL, 1→HOLD, 2→BUY
        idx_buy  = 2
        idx_sell = 0

    # 목표 거래 횟수 기반 자동 threshold 탐색
    thresholds   = np.arange(0.25, 0.75, 0.01)
    scan_results = [(round(t, 2), simulate_positions(proba_raw, idx_buy, idx_sell, t, t, MIN_HOLD_BARS)[1])
                    for t in thresholds]
    best_thresh, best_n = min(scan_results, key=lambda x: abs(x[1] - TARGET_TRADES))
    print(f"  [{model_name}] 선택 임계값: {best_thresh:.2f} (거래 {best_n}회)")

    pos_arr, _ = simulate_positions(proba_raw, idx_buy, idx_sell,
                                    best_thresh, best_thresh, MIN_HOLD_BARS)
    position     = pd.Series(pos_arr, index=df_test.index)
    close_test   = df_test['close']
    close_return = close_test.pct_change().fillna(0)

    pos_prev   = position.shift(1).fillna(0)
    changed    = position != pos_prev
    flipped    = ((position == 1) & (pos_prev == -1)) | ((position == -1) & (pos_prev == 1))
    tx_cost    = changed.astype(float) * FEE_RATE + flipped.astype(float) * FEE_RATE
    strat_ret  = position * close_return - tx_cost

    cum_strat  = (1 + strat_ret).cumprod()
    cum_bnh    = (1 + close_return).cumprod()

    total_ret  = cum_strat.iloc[-1] - 1
    bnh_ret    = cum_bnh.iloc[-1] - 1
    std_r      = strat_ret.std()
    sharpe     = (strat_ret.mean() / std_r * np.sqrt(ANNUAL_BARS)) if std_r > 0 else 0.0
    mdd        = ((cum_strat - cum_strat.cummax()) / cum_strat.cummax()).min()
    n_trades   = int(changed.sum())

    # win rate (트레이드 단위)
    trade_results = []
    in_trade = False
    trade_return = 0.0
    for i in range(len(position)):
        cur  = position.iloc[i]
        prev = pos_prev.iloc[i]
        if not in_trade and cur != 0:
            in_trade = True
            trade_return = 0.0
        if in_trade:
            trade_return += strat_ret.iloc[i]
        if in_trade and (cur == 0 or (cur != 0 and cur != prev and prev != 0)):
            trade_results.append(trade_return)
            in_trade = False
            trade_return = 0.0
            if cur != 0:
                in_trade = True
                trade_return = 0.0
    if in_trade:
        trade_results.append(trade_return)
    win_rate = (sum(1 for r in trade_results if r > 0) / len(trade_results)) if trade_results else 0.0

    return {
        'total_return'  : round(float(total_ret) * 100, 2),
        'bnh_return'    : round(float(bnh_ret) * 100, 2),
        'sharpe'        : round(float(sharpe), 4),
        'mdd'           : round(float(mdd) * 100, 2),
        'n_trades'      : n_trades,
        'win_rate'      : round(win_rate * 100, 2),
        'thresh'        : best_thresh,
    }, position, cum_strat, cum_bnh


# ════════════════════════════════════════════════════════════════
# Step 1. 데이터 로드 & 피처 엔지니어링
# ════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  Step 1. 데이터 로드 & 피처 엔지니어링")
print("=" * 60)

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
df[['open','high','low','close','volume']] = df[['open','high','low','close','volume']].fillna(method='ffill', limit=3)
df = df.dropna(subset=['open','high','low','close','volume']).reset_index(drop=True)

print(f"  shape  : {df.shape}")
print(f"  기간   : {df['open_time'].iloc[0]} ~ {df['open_time'].iloc[-1]}")

feat = make_features(df)
df   = pd.concat([df, feat], axis=1)
df[FEATURE_COLS] = df[FEATURE_COLS].replace([np.inf, -np.inf], np.nan)
df   = df.dropna(subset=FEATURE_COLS).reset_index(drop=True)
print(f"  피처   : {len(FEATURE_COLS)}개 | NaN 제거 후 {len(df):,}행")


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

df   = df.iloc[:-FORWARD_BARS].copy()
label = label.iloc[:-FORWARD_BARS].copy()
forward_return_trimmed = forward_return.iloc[:-FORWARD_BARS].values
df['label'] = label.values

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

# XGB/CatBoost용 0-indexed 레이블
y_train_idx = np.array([LABEL_MAP[v] for v in y_train_final])
y_test_idx  = np.array([LABEL_MAP[v] for v in y_test])

# early stopping용 val 분리
n_val         = int(len(X_train_final) * ES_VAL_RATIO)
X_tr_es       = X_train_final[:-n_val]
y_tr_es_raw   = y_train_final[:-n_val]
y_tr_es_idx   = y_train_idx[:-n_val]
sw_tr_es      = sw_train_final[:-n_val]
X_val_es      = X_train_final[-n_val:]
y_val_es_idx  = y_train_idx[-n_val:]

print(f"  Train    : {TRAIN_START} ~ {TRAIN_END}  ({split_idx:,}행)")
print(f"  Test     : {TEST_START} ~ {TEST_END}  ({len(X_test):,}행)")
print(f"  ES val   : train 뒤 {n_val:,}행 ({ES_VAL_RATIO*100:.0f}%)")


# ════════════════════════════════════════════════════════════════
# Step 4. TimeSeriesSplit 교차검증 (3개 모델 공통 fold)
# ════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  Step 4. TimeSeriesSplit 교차검증")
print("=" * 60)

tscv   = TimeSeriesSplit(n_splits=N_SPLITS)
splits = list(tscv.split(X_train_final))

cv_results  = {name: {'bal_acc': [], 'f1_macro': [], 'elapsed': []} for name in ['RF', 'XGBoost', 'CatBoost']}
cv_start    = time.time()
total_steps = N_SPLITS * 3   # fold × 모델 수
done_steps  = 0


def fmt_sec(sec):
    sec = int(sec)
    return f"{sec//60}분 {sec%60}초" if sec >= 60 else f"{sec}초"


for fold, (tr_idx, te_idx) in enumerate(splits, 1):
    if len(tr_idx) < 100 or len(te_idx) < 100:
        continue

    X_tr, X_te = X_train_final[tr_idx], X_train_final[te_idx]
    y_tr_raw   = y_train_final[tr_idx]
    y_tr_idx_f = y_train_idx[tr_idx]
    y_te_raw   = y_train_final[te_idx]
    sw_tr_f    = sw_train_final[tr_idx]

    # early stopping val: fold 내 뒤 20%
    n_v             = int(len(X_tr) * ES_VAL_RATIO)
    X_tv, y_tv_idx  = X_tr[-n_v:], y_tr_idx_f[-n_v:]
    X_tr_f          = X_tr[:-n_v]
    y_tr_f_idx      = y_tr_idx_f[:-n_v]
    sw_tr_f2        = sw_tr_f[:-n_v]

    elapsed_total = time.time() - cv_start
    eta = (elapsed_total / done_steps * (total_steps - done_steps)) if done_steps > 0 else 0
    print(f"\n  ── Fold {fold}/{N_SPLITS} │ 경과 {fmt_sec(elapsed_total)} │ 예상 잔여 {fmt_sec(eta)}")
    print(f"     train {len(tr_idx):,}행 / val {len(te_idx):,}행")

    # RF
    t0 = time.time()
    rf_cv = RandomForestClassifier(**RF_PARAMS)
    rf_cv.fit(X_tr, y_tr_raw, sample_weight=sw_tr_f)
    rf_elapsed = time.time() - t0
    y_pred_rf = rf_cv.predict(X_te)
    cv_results['RF']['bal_acc'].append(balanced_accuracy_score(y_te_raw, y_pred_rf))
    cv_results['RF']['f1_macro'].append(f1_score(y_te_raw, y_pred_rf, average='macro', zero_division=0))
    cv_results['RF']['elapsed'].append(rf_elapsed)
    done_steps += 1
    eta = (time.time() - cv_start) / done_steps * (total_steps - done_steps)
    print(f"  RF       완료 ({fmt_sec(rf_elapsed)}) │ 예상 잔여 {fmt_sec(eta)}")
    print(classification_report(y_te_raw, y_pred_rf,
          target_names=['SELL(-1)','HOLD(0)','BUY(+1)'], labels=[-1,0,1], zero_division=0))

    # XGBoost
    t0 = time.time()
    xgb_cv = xgb.XGBClassifier(**XGB_PARAMS)
    xgb_cv.fit(X_tr_f, y_tr_f_idx, sample_weight=sw_tr_f2,
               eval_set=[(X_tv, y_tv_idx)], verbose=False)
    xgb_elapsed = time.time() - t0
    y_pred_xgb_idx = xgb_cv.predict(X_te)
    y_pred_xgb = np.array([LABEL_MAP_INV[v] for v in y_pred_xgb_idx])
    cv_results['XGBoost']['bal_acc'].append(balanced_accuracy_score(y_te_raw, y_pred_xgb))
    cv_results['XGBoost']['f1_macro'].append(f1_score(y_te_raw, y_pred_xgb, average='macro', zero_division=0))
    cv_results['XGBoost']['elapsed'].append(xgb_elapsed)
    done_steps += 1
    eta = (time.time() - cv_start) / done_steps * (total_steps - done_steps)
    print(f"  XGBoost  완료 ({fmt_sec(xgb_elapsed)}) │ best iter {xgb_cv.best_iteration} │ 예상 잔여 {fmt_sec(eta)}")
    print(classification_report(y_te_raw, y_pred_xgb,
          target_names=['SELL(-1)','HOLD(0)','BUY(+1)'], labels=[-1,0,1], zero_division=0))

    # CatBoost
    t0 = time.time()
    cat_cv = CatBoostClassifier(**CAT_PARAMS)
    cat_cv.fit(X_tr_f, y_tr_f_idx, sample_weight=sw_tr_f2,
               eval_set=(X_tv, y_tv_idx))
    cat_elapsed = time.time() - t0
    y_pred_cat_idx = cat_cv.predict(X_te).flatten().astype(int)
    y_pred_cat = np.array([LABEL_MAP_INV[v] for v in y_pred_cat_idx])
    cv_results['CatBoost']['bal_acc'].append(balanced_accuracy_score(y_te_raw, y_pred_cat))
    cv_results['CatBoost']['f1_macro'].append(f1_score(y_te_raw, y_pred_cat, average='macro', zero_division=0))
    cv_results['CatBoost']['elapsed'].append(cat_elapsed)
    done_steps += 1
    eta = (time.time() - cv_start) / done_steps * (total_steps - done_steps)
    print(f"  CatBoost 완료 ({fmt_sec(cat_elapsed)}) │ 예상 잔여 {fmt_sec(eta)}")
    print(classification_report(y_te_raw, y_pred_cat,
          target_names=['SELL(-1)','HOLD(0)','BUY(+1)'], labels=[-1,0,1], zero_division=0))

print(f"\n  CV 전체 소요: {fmt_sec(time.time() - cv_start)}")
print("\n  [CV 요약]")
for name, scores in cv_results.items():
    print(f"  {name:<10s} | bal_acc={np.mean(scores['bal_acc']):.4f} "
          f"| f1_macro={np.mean(scores['f1_macro']):.4f} "
          f"| 평균 {fmt_sec(np.mean(scores['elapsed']))}/fold")


# ════════════════════════════════════════════════════════════════
# Step 5. 최종 모델 학습
# ════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  Step 5. 최종 모델 학습")
print("=" * 60)

all_models  = {}
train_times = {}

# RF
print(f"\n  [RF] 학습 시작 {datetime.datetime.now().strftime('%H:%M:%S')}")
t0 = time.time()
# CV 기반 예상 학습 시간 계산 (데이터 크기 비율로 스케일)
cv_rf_mean  = np.mean(cv_results['RF']['elapsed'])
cv_xgb_mean = np.mean(cv_results['XGBoost']['elapsed'])
cv_cat_mean = np.mean(cv_results['CatBoost']['elapsed'])
# 최종 학습은 fold 크기(train의 2/3)보다 크므로 비율 보정
final_scale = len(X_train_final) / (len(splits[-1][0]) * (1 - ES_VAL_RATIO))
print(f"  CV 기반 예상 학습 시간 → RF: {fmt_sec(cv_rf_mean * final_scale)} │ "
      f"XGB: {fmt_sec(cv_xgb_mean * 2)} │ CatBoost: {fmt_sec(cv_cat_mean * 2)}")

final_start = time.time()

rf_model = RandomForestClassifier(**RF_PARAMS)
rf_model.fit(X_train_final, y_train_final, sample_weight=sw_train_final)
train_times['RF'] = round(time.time() - t0, 1)
all_models['RF']  = rf_model
elapsed_rf = time.time() - final_start
print(f"  [RF] 완료 ({fmt_sec(elapsed_rf)}) {datetime.datetime.now().strftime('%H:%M:%S')}")

# XGBoost
print(f"\n  [XGBoost] 학습 시작 {datetime.datetime.now().strftime('%H:%M:%S')}")
t0 = time.time()
xgb_model = xgb.XGBClassifier(**XGB_PARAMS)
xgb_model.fit(X_tr_es, y_tr_es_idx, sample_weight=sw_tr_es,
              eval_set=[(X_val_es, y_val_es_idx)], verbose=100)
elapsed_xgb = time.time() - t0
train_times['XGBoost'] = round(elapsed_xgb, 1)
all_models['XGBoost']  = xgb_model
print(f"  [XGBoost] 완료 ({fmt_sec(elapsed_xgb)}) │ best iter: {xgb_model.best_iteration} "
      f"│ {datetime.datetime.now().strftime('%H:%M:%S')}")

# CatBoost
print(f"\n  [CatBoost] 학습 시작 {datetime.datetime.now().strftime('%H:%M:%S')}")
t0 = time.time()
cat_model = CatBoostClassifier(**CAT_PARAMS)
cat_model.fit(X_tr_es, y_tr_es_idx, sample_weight=sw_tr_es,
              eval_set=(X_val_es, y_val_es_idx))
elapsed_cat = time.time() - t0
train_times['CatBoost'] = round(elapsed_cat, 1)
all_models['CatBoost']  = cat_model
print(f"  [CatBoost] 완료 ({fmt_sec(elapsed_cat)}) {datetime.datetime.now().strftime('%H:%M:%S')}")


# ════════════════════════════════════════════════════════════════
# Step 6. 테스트셋 평가
# ════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  Step 6. 테스트셋 평가")
print("=" * 60)

test_metrics = {}

# RF
y_pred_rf_test = rf_model.predict(X_test)
test_metrics['RF'] = {
    'bal_acc' : round(balanced_accuracy_score(y_test, y_pred_rf_test), 4),
    'f1_macro': round(f1_score(y_test, y_pred_rf_test, average='macro', zero_division=0), 4),
}
print("\n  [RF]")
print(classification_report(y_test, y_pred_rf_test,
      target_names=['SELL(-1)','HOLD(0)','BUY(+1)'], labels=[-1,0,1], zero_division=0))

# XGBoost
y_pred_xgb_test_idx = xgb_model.predict(X_test)
y_pred_xgb_test     = np.array([LABEL_MAP_INV[v] for v in y_pred_xgb_test_idx])
test_metrics['XGBoost'] = {
    'bal_acc' : round(balanced_accuracy_score(y_test, y_pred_xgb_test), 4),
    'f1_macro': round(f1_score(y_test, y_pred_xgb_test, average='macro', zero_division=0), 4),
}
print("\n  [XGBoost]")
print(classification_report(y_test, y_pred_xgb_test,
      target_names=['SELL(-1)','HOLD(0)','BUY(+1)'], labels=[-1,0,1], zero_division=0))

# CatBoost
y_pred_cat_test_idx = cat_model.predict(X_test).flatten().astype(int)
y_pred_cat_test     = np.array([LABEL_MAP_INV[v] for v in y_pred_cat_test_idx])
test_metrics['CatBoost'] = {
    'bal_acc' : round(balanced_accuracy_score(y_test, y_pred_cat_test), 4),
    'f1_macro': round(f1_score(y_test, y_pred_cat_test, average='macro', zero_division=0), 4),
}
print("\n  [CatBoost]")
print(classification_report(y_test, y_pred_cat_test,
      target_names=['SELL(-1)','HOLD(0)','BUY(+1)'], labels=[-1,0,1], zero_division=0))


# ════════════════════════════════════════════════════════════════
# Step 7. 백테스팅
# ════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  Step 7. 백테스팅")
print("=" * 60)

backtest_results = {}
positions_dict   = {}
cum_strats       = {}
cum_bnh_ref      = None

# RF
rf_proba = rf_model.predict_proba(X_test)
rf_classes = list(rf_model.classes_)
bt_rf, pos_rf, cum_rf, cum_bnh = run_backtest(rf_proba, df_test, rf_classes, 'RF')
backtest_results['RF'] = bt_rf
positions_dict['RF']   = pos_rf
cum_strats['RF']       = cum_rf
cum_bnh_ref            = cum_bnh

# XGBoost
xgb_proba = xgb_model.predict_proba(X_test)
bt_xgb, pos_xgb, cum_xgb, _ = run_backtest(xgb_proba, df_test, [0,1,2], 'XGBoost')
backtest_results['XGBoost'] = bt_xgb
positions_dict['XGBoost']   = pos_xgb
cum_strats['XGBoost']       = cum_xgb

# CatBoost
cat_proba = cat_model.predict_proba(X_test)
bt_cat, pos_cat, cum_cat, _ = run_backtest(cat_proba, df_test, [0,1,2], 'CatBoost')
backtest_results['CatBoost'] = bt_cat
positions_dict['CatBoost']   = pos_cat
cum_strats['CatBoost']       = cum_cat

print(f"\n  [백테스트 요약]")
for name, bt in backtest_results.items():
    print(f"  {name:<10s} | Return={bt['total_return']:+.1f}%  "
          f"Sharpe={bt['sharpe']:.2f}  MDD={bt['mdd']:.1f}%  "
          f"Trades={bt['n_trades']}  WinRate={bt['win_rate']:.1f}%")
print(f"  Buy&Hold   | Return={bt_rf['bnh_return']:+.1f}%")


# ════════════════════════════════════════════════════════════════
# Step 8. 결과 저장
# ════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  Step 8. 결과 저장")
print("=" * 60)

# 모델 저장
joblib.dump(rf_model,  os.path.join(OUTPUT_DIR, 'rf_model.pkl'))
xgb_model.save_model(  os.path.join(OUTPUT_DIR, 'xgb_model.json'))
cat_model.save_model(  os.path.join(OUTPUT_DIR, 'catboost_model.cbm'))
print(f"  모델 저장 완료")

# comparison_metrics.json
full_metrics = {}
for name in ['RF', 'XGBoost', 'CatBoost']:
    full_metrics[name] = {
        'cv_bal_acc_mean' : round(np.mean(cv_results[name]['bal_acc']), 4),
        'cv_bal_acc_std'  : round(np.std(cv_results[name]['bal_acc']), 4),
        'cv_f1_mean'      : round(np.mean(cv_results[name]['f1_macro']), 4),
        'test_bal_acc'    : test_metrics[name]['bal_acc'],
        'test_f1_macro'   : test_metrics[name]['f1_macro'],
        'train_sec'       : train_times[name],
        **backtest_results[name],
    }
with open(os.path.join(OUTPUT_DIR, 'comparison_metrics.json'), 'w') as f:
    json.dump(full_metrics, f, indent=2)
print(f"  저장: {OUTPUT_DIR}/comparison_metrics.json")

# comparison_summary.csv
summary_rows = []
for name, m in full_metrics.items():
    summary_rows.append({
        'model'          : name,
        'cv_bal_acc_mean': m['cv_bal_acc_mean'],
        'cv_f1_mean'     : m['cv_f1_mean'],
        'test_bal_acc'   : m['test_bal_acc'],
        'test_f1_macro'  : m['test_f1_macro'],
        'total_return'   : m['total_return'],
        'sharpe'         : m['sharpe'],
        'mdd'            : m['mdd'],
        'n_trades'       : m['n_trades'],
        'win_rate'       : m['win_rate'],
        'train_sec'      : m['train_sec'],
    })
summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(os.path.join(OUTPUT_DIR, 'comparison_summary.csv'), index=False)
print(f"  저장: {OUTPUT_DIR}/comparison_summary.csv")

# 누적 수익률 차트
times = df_test['open_time'].values
fig, ax = plt.subplots(figsize=(14, 6))
colors = {'RF': '#2ecc71', 'XGBoost': '#e67e22', 'CatBoost': '#9b59b6'}
for name, cum in cum_strats.items():
    ax.plot(times, cum.values, label=name, color=colors[name], linewidth=1.2)
ax.plot(times, cum_bnh_ref.values, label='Buy & Hold', color='#3498db',
        linewidth=1.2, linestyle='--', alpha=0.8)
ax.axhline(1.0, color='gray', linewidth=0.8, linestyle=':')
ax.set_title('Cumulative Return: RF vs XGBoost vs CatBoost (2024)')
ax.set_xlabel('Date')
ax.set_ylabel('Cumulative Return')
ax.legend()
ax.grid(alpha=0.2)
plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, 'cumulative_return_comparison.png'), dpi=150)
plt.close()
print(f"  저장: {OUTPUT_DIR}/cumulative_return_comparison.png")

# feature importance 비교 차트 (상위 15개)
fig, axes = plt.subplots(1, 3, figsize=(18, 7))
top_n = 15

imp_data = {
    'RF'      : pd.Series(rf_model.feature_importances_, index=FEATURE_COLS),
    'XGBoost' : pd.Series(xgb_model.feature_importances_, index=FEATURE_COLS),
    'CatBoost': pd.Series(cat_model.get_feature_importance(), index=FEATURE_COLS),
}
imp_colors = {'RF': 'steelblue', 'XGBoost': 'darkorange', 'CatBoost': 'mediumpurple'}

for ax, (name, imp) in zip(axes, imp_data.items()):
    top = imp.sort_values(ascending=False).head(top_n).sort_values()
    ax.barh(top.index, top.values, color=imp_colors[name])
    ax.set_title(f'{name} Feature Importance (Top {top_n})')
    ax.set_xlabel('Importance')
    ax.grid(axis='x', alpha=0.3)

fig.suptitle('Feature Importance Comparison', fontsize=14)
plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, 'feature_importance_comparison.png'), dpi=150)
plt.close()
print(f"  저장: {OUTPUT_DIR}/feature_importance_comparison.png")


# ════════════════════════════════════════════════════════════════
# 최종 요약 출력
# ════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  COMPARISON SUMMARY")
print("=" * 60)
print(summary_df.to_string(index=False))
print("=" * 60)
