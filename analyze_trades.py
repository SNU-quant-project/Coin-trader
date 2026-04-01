import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import warnings
warnings.filterwarnings('ignore')

print("데이터 분석 준비 중...")
df = pd.read_csv('data/BTC_USDT_1m.csv', usecols=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
df.set_index('datetime', inplace=True)
df_15m = df.resample('15min').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'}).dropna()

hl_diff = (df_15m['high'] - df_15m['low']) + 1e-8
open_close_max = df_15m[['open', 'close']].max(axis=1)
open_close_min = df_15m[['open', 'close']].min(axis=1)
df_15m['body_ratio'] = (df_15m['close'] - df_15m['open']).abs() / hl_diff
df_15m['upper_ratio'] = (df_15m['high'] - open_close_max) / hl_diff
df_15m['lower_ratio'] = (open_close_min - df_15m['low']) / hl_diff
df_15m['up_down'] = (df_15m['close'] > df_15m['open']).astype(float)
df_15m['box_length'] = (df_15m['close'] - df_15m['open']).abs() / (df_15m['open'] + 1e-8)
df_15m['volume_ratio'] = df_15m['volume'] / (df_15m['volume'].rolling(20).mean() + 1e-8)
df_15m['target'] = (df_15m['close'].shift(-1) > df_15m['open'].shift(-1)).astype(float)
df_15m.dropna(inplace=True)

features = ['body_ratio', 'upper_ratio', 'lower_ratio', 'up_down', 'box_length', 'volume_ratio']
df_train = df_15m.loc[:"2022-12-31"].copy()
df_test = df_15m.loc["2023-07-01":"2024-04-01"].copy()

mean_vals = df_train[features].mean()
std_vals = df_train[features].std() + 1e-8
df_test[features] = (df_test[features] - mean_vals) / std_vals

SEQ_LEN = 10
X_test, idx_test = [], []
data_feat = df_test[features].values
indices = df_test.index

for i in range(len(df_test) - SEQ_LEN):
    X_test.append(data_feat[i : i + SEQ_LEN])
    idx_test.append(indices[i + SEQ_LEN])
X_test = np.array(X_test, dtype=np.float32)

class LSTMAlpha(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=6, hidden_size=64, num_layers=2, batch_first=True, dropout=0.3)
        self.fc = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.3), nn.Linear(32, 1), nn.Sigmoid())
    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        return self.fc(h_n[-1]).squeeze(1)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = LSTMAlpha().to(device)
model.load_state_dict(torch.load('lstm_best.pth', map_location=device))
model.eval()

probs = []
with torch.no_grad():
    for i in range(0, len(X_test), 512):
        bx = torch.tensor(X_test[i:i+512]).to(device)
        probs.extend(model(bx).cpu().numpy())

# ==========================================
# 매매 분석 (Trade Analysis)
# ==========================================
print("\n[알파 적중률 및 수익 분석]")
THRESHOLD_L = 0.65
THRESHOLD_S = 0.35
FEE = 0.001 # 편도 0.1%

res = pd.DataFrame({'datetime': idx_test, 'prob': probs})
df_test_prices = df_test.loc[idx_test].copy()
res['actual'] = df_test_prices['target'].values # 다음 봉 상승 여부 (1=상승, 0=하락)
res['open'] = df_test_prices['open'].values
res['close'] = df_test_prices['close'].values
res['next_open'] = df_test_prices['open'].shift(-1).values
res['next_close'] = df_test_prices['close'].shift(-1).values
res.dropna(inplace=True)

# 1. 단순 방향(단일 캔들) 예측 적중률
pred_long = res[res['prob'] >= THRESHOLD_L]
pred_short = res[res['prob'] <= THRESHOLD_S]

long_acc = (pred_long['actual'] == 1).mean()
short_acc = (pred_short['actual'] == 0).mean()

print(f"1보 전진 단순 분류 적중률:")
print(f" - Long 시그널 ({len(pred_long)}개): {long_acc*100:.2f}% 맞힘")
print(f" - Short 시그널 ({len(pred_short)}개): {short_acc*100:.2f}% 맞힘")

# 2. 거래 관점(Trade-level) 수익률 분석 (Holding 로직 포함)
trades_log = []

position = 0 
entry_price = 0.0

for i in range(len(res)):
    prob = res.iloc[i]['prob']
    next_open = res.iloc[i]['next_open']
    
    if position == 0:
        if prob >= THRESHOLD_L:
            position = 1
            entry_price = next_open
        elif prob <= THRESHOLD_S:
            position = -1
            entry_price = next_open
    else:
        if position == 1:
            if prob >= THRESHOLD_L:
                pass # 유지
            else:
                ret_gross = (next_open - entry_price) / entry_price
                trades_log.append({'type': 'LONG', 'gross_ret': ret_gross})
                position = 0
        elif position == -1:
            if prob <= THRESHOLD_S:
                pass # 유지
            else:
                ret_gross = (entry_price - next_open) / entry_price
                trades_log.append({'type': 'SHORT', 'gross_ret': ret_gross})
                position = 0

df_trades = pd.DataFrame(trades_log)

if len(df_trades) > 0:
    df_trades['net_ret'] = df_trades['gross_ret'] - (FEE * 2) # 왕복 수수료 0.2% 차감
    
    avg_gross_ret = df_trades['gross_ret'].mean() * 100 
    avg_net_ret = df_trades['net_ret'].mean() * 100 
    win_rate_gross = (df_trades['gross_ret'] > 0).mean() * 100
    win_rate_net = (df_trades['net_ret'] > 0).mean() * 100
    
    print("\n트레이딩 건별 상세 수익 통계:")
    print(f" - 총 완료된 거래 수: {len(df_trades)}건")
    print(f" - 1회 거래당 평균 Gross 수익금 (수수료 제외 전): {avg_gross_ret:.4f}%")
    print(f" - 1회 거래당 평균 Net 수익금 (수수료 제외 후): {avg_net_ret:.4f}%")
    print(f" - Gross 승률 (원금이 조금이라도 오른 거래): {win_rate_gross:.2f}%")
    print(f" - Net 승률 (수수료 차감 후에도 이득인 거래): {win_rate_net:.2f}%")
