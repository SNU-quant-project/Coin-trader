import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

print("데이터 및 저장된 모델(lstm_best.pth) 로딩 중...")
df = pd.read_csv('data/BTC_USDT_1m.csv', usecols=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
df.set_index('datetime', inplace=True)

df_15m = df.resample('15min').agg({
    'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
}).dropna()

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

print("추론 진행 중...")
probs = []
with torch.no_grad():
    for i in range(0, len(X_test), 512):
        bx = torch.tensor(X_test[i:i+512]).to(device)
        probs.extend(model(bx).cpu().numpy())

# ==========================================
# 백테스트: 유지(Holding) 로직 포함
# ==========================================
print("\n[백테스트 시뮬레이션 - 0.65/0.35 적용]")
THRESHOLD_L = 0.65
THRESHOLD_S = 0.35

res = pd.DataFrame({'datetime': idx_test, 'prob': probs})
df_test_prices = df_test.loc[idx_test].copy()
res['open'] = df_test_prices['open'].values
res['close'] = df_test_prices['close'].values
res['next_open'] = df_test_prices['open'].shift(-1).values
res['next_close'] = df_test_prices['close'].shift(-1).values
res.dropna(inplace=True)

capital = 10000.0
position = 0 
FEE = 0.001 # 0.1% 편도 수수료
entry_price = 0.0
trades = 0

equity_curve = [capital]

# Buy and Hold 비교용
bnh_capital = 10000.0
bnh_entry = res.iloc[0]['next_open']
bnh_curve = [bnh_capital]

for i in range(len(res)):
    prob = res.iloc[i]['prob']
    next_open = res.iloc[i]['next_open']
    
    # BnH 계산
    bnh_ret = (next_open - bnh_entry) / bnh_entry
    bnh_curve.append(bnh_capital * (1 + bnh_ret))
    
    if position == 0:
        if prob >= THRESHOLD_L:
            position = 1
            entry_price = next_open
            capital *= (1 - FEE) 
            trades += 1
        elif prob <= THRESHOLD_S:
            position = -1
            entry_price = next_open
            capital *= (1 - FEE) 
            trades += 1
    else:
        if position == 1:
            if prob >= THRESHOLD_L:
                pass # 유지
            else:
                ret = (next_open - entry_price) / entry_price
                capital *= (1 + ret)
                capital *= (1 - FEE) 
                position = 0
                
        elif position == -1:
            if prob <= THRESHOLD_S:
                pass # 유지
            else:
                ret = (entry_price - next_open) / entry_price
                capital *= (1 + ret)
                capital *= (1 - FEE) 
                position = 0
                
    equity_curve.append(capital)

print(f"=====================================")
print(f"최종 자본금: ${capital:.2f} (초기: $10000)")
print(f"총 매매 횟수: {trades}회")
print(f"=====================================")

plt.figure(figsize=(12, 6))
plt.plot(res['datetime'], equity_curve[1:], label=f"모델 전략 (자본금 ${capital:.0f})")
plt.plot(res['datetime'], bnh_curve[1:], label=f"단순 보유 (자본금 ${bnh_curve[-1]:.0f})", linestyle='dashed', alpha=0.7)
plt.title(f"예측 알고리즘 성과 (진입 확신도 롱:{THRESHOLD_L}/숏:{THRESHOLD_S})")
plt.xlabel("Date")
plt.ylabel("Equity (USD)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig("backtest_equity_v2.png", bbox_inches='tight')
print("[!] 갱신된 백테스트 결과가 'backtest_equity_v2.png'로 저장되었습니다.")
