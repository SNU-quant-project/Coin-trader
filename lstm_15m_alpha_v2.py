import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import gc
import warnings
warnings.filterwarnings('ignore')

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

torch.manual_seed(42)
np.random.seed(42)

# ==========================================
# 설정 및 파라미터 
# ==========================================
SEQ_LEN = 10
BATCH_SIZE = 512
EPOCHS = 15
LR = 0.001
HIDDEN_SIZE = 64
NUM_LAYERS = 2
DROPOUT = 0.3
TARGET_PCT = 0.002 # 0.2% 이상 타겟팅
THRESHOLD_L = 0.65 # 강한 확신
THRESHOLD_S = 0.65
FEE = 0.001        # 0.1% 편도

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[{device}] 디바이스로 학습을 준비합니다.")

# ==========================================
# 데이터 처리
# ==========================================
print("데이터 로딩 중...")
df = pd.read_csv('data/BTC_USDT_1m.csv', usecols=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
df.set_index('datetime', inplace=True)
df.sort_index(inplace=True)

df_15m = df.resample('15min').agg({
    'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
}).dropna()

# 피처 추출
hl_diff = (df_15m['high'] - df_15m['low']) + 1e-8
open_close_max = df_15m[['open', 'close']].max(axis=1)
open_close_min = df_15m[['open', 'close']].min(axis=1)
df_15m['body_ratio'] = (df_15m['close'] - df_15m['open']).abs() / hl_diff
df_15m['upper_ratio'] = (df_15m['high'] - open_close_max) / hl_diff
df_15m['lower_ratio'] = (open_close_min - df_15m['low']) / hl_diff
df_15m['up_down'] = (df_15m['close'] > df_15m['open']).astype(float)
df_15m['box_length'] = (df_15m['close'] - df_15m['open']).abs() / (df_15m['open'] + 1e-8)
df_15m['volume_ratio'] = df_15m['volume'] / (df_15m['volume'].rolling(20).mean() + 1e-8)

# [핵심] 다중 타겟 추출
# 다음 봉의 종가-시가 차이가 시가 대비 0.2% 이상인가?
next_ret = (df_15m['close'].shift(-1) - df_15m['open'].shift(-1)) / (df_15m['open'].shift(-1) + 1e-8)

df_15m['target_long'] = (next_ret >= TARGET_PCT).astype(float)
df_15m['target_short'] = (next_ret <= -TARGET_PCT).astype(float)
df_15m.dropna(inplace=True)

features = ['body_ratio', 'upper_ratio', 'lower_ratio', 'up_down', 'box_length', 'volume_ratio']

# ==========================================
# Train / Val / Test 스플릿
# ==========================================
df_train = df_15m.loc[:"2022-12-31"].copy()
df_val = df_15m.loc["2023-01-01":"2023-06-30"].copy()
df_test = df_15m.loc["2023-07-01":"2024-04-01"].copy()

mean_vals = df_train[features].mean()
std_vals = df_train[features].std() + 1e-8
df_train[features] = (df_train[features] - mean_vals) / std_vals
df_val[features] = (df_val[features] - mean_vals) / std_vals
df_test[features] = (df_test[features] - mean_vals) / std_vals

def build_sequences(df_sub):
    X, y, idx = [], [], []
    data_f = df_sub[features].values
    data_l = df_sub['target_long'].values
    data_s = df_sub['target_short'].values
    indices = df_sub.index
    
    for i in range(len(df_sub) - SEQ_LEN):
        X.append(data_f[i:i+SEQ_LEN])
        y.append([data_l[i+SEQ_LEN-1], data_s[i+SEQ_LEN-1]])
        idx.append(indices[i+SEQ_LEN])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32), idx

X_trn, y_trn, _ = build_sequences(df_train)
X_val, y_val, idx_val = build_sequences(df_val)
X_tst, y_tst, idx_tst = build_sequences(df_test)

print(f"Target 분포 | Train L:{y_trn[:,0].mean():.3f} S:{y_trn[:,1].mean():.3f} | Test L:{y_tst[:,0].mean():.3f} S:{y_tst[:,1].mean():.3f}")

class SeqData(Dataset):
    def __init__(self, x, y):
        self.x, self.y = torch.tensor(x), torch.tensor(y)
    def __len__(self): return len(self.x)
    def __getitem__(self, idx): return self.x[idx], self.y[idx]

train_ld = DataLoader(SeqData(X_trn, y_trn), batch_size=BATCH_SIZE, shuffle=True)
val_ld = DataLoader(SeqData(X_val, y_val), batch_size=BATCH_SIZE, shuffle=False)
test_ld = DataLoader(SeqData(X_tst, y_tst), batch_size=BATCH_SIZE, shuffle=False)

del df, df_15m
gc.collect()

# ==========================================
# 모델 선언 (FC Layer Output=2)
# ==========================================
class MultiLabelLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=len(features), hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS, batch_first=True, dropout=DROPOUT)
        self.fc = nn.Sequential(nn.Linear(HIDDEN_SIZE, 32), nn.ReLU(), nn.Dropout(DROPOUT), nn.Linear(32, 2), nn.Sigmoid())
    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        return self.fc(h_n[-1])

model = MultiLabelLSTM().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
criterion = nn.BCELoss() # 이진 분류 2개를 독립적으로 판별 (Sigmoid 결과이므로)

# ==========================================
# 학습 진행
# ==========================================
print("\n[학습 시작 (다중 타겟)]")
best_vloss = float('inf')

for ep in range(1, EPOCHS + 1):
    model.train()
    t_loss = 0
    for bx, by in train_ld:
        bx, by = bx.to(device), by.to(device)
        optimizer.zero_grad()
        out = model(bx)
        loss = criterion(out, by)
        loss.backward()
        optimizer.step()
        t_loss += loss.item() * len(bx)
    t_loss /= len(train_ld.dataset)
    
    model.eval()
    v_loss = 0
    with torch.no_grad():
        for bx, by in val_ld:
            bx, by = bx.to(device), by.to(device)
            out = model(bx)
            loss = criterion(out, by)
            v_loss += loss.item() * len(bx)
    v_loss /= len(val_ld.dataset)
    print(f"Epoch {ep:2d} | Train Loss: {t_loss:.4f} | Val Loss: {v_loss:.4f}")
    if v_loss < best_vloss:
        best_vloss = v_loss
        torch.save(model.state_dict(), 'lstm_multi_best.pth')

# ==========================================
# 검증(Validation) 및 히스토그램
# ==========================================
model.load_state_dict(torch.load('lstm_multi_best.pth', map_location=device))
model.eval()

tst_out = []
with torch.no_grad():
    for bx, _ in test_ld:
        bx = bx.to(device)
        tst_out.extend(model(bx).cpu().numpy())
tst_out = np.array(tst_out) # shape: (N, 2)

plt.figure(figsize=(14, 5))
plt.subplot(1, 2, 1)
plt.hist(tst_out[:, 0], bins=50, color='blue', alpha=0.7)
plt.title("Long (>=0.2%) 확률 분포")
plt.subplot(1, 2, 2)
plt.hist(tst_out[:, 1], bins=50, color='red', alpha=0.7)
plt.title("Short (<=-0.2%) 확률 분포")
plt.savefig('v3_histogram.png', bbox_inches='tight')

# ==========================================
# 백테스트 시뮬레이션
# ==========================================
print("\n[백테스트: 진입 롱/숏 > 0.65, 타겟 0.2%, 모델 유지]")

res = pd.DataFrame({'datetime': idx_tst, 'prob_long': tst_out[:, 0], 'prob_short': tst_out[:, 1]})
prc = df_test.loc[idx_tst].copy()
res['actual_long'] = y_tst[:,0]
res['actual_short'] = y_tst[:,1]
res['next_open'] = prc['open'].shift(-1).values
res['next_close'] = prc['close'].shift(-1).values
res.dropna(inplace=True)

capital = 10000.0
position = 0 
entry_price = 0.0
trades = 0

curve = [capital]
trades_log = []

for i in range(len(res)):
    pl = res.iloc[i]['prob_long']
    ps = res.iloc[i]['prob_short']
    n_o = res.iloc[i]['next_open']
    
    if position == 0:
        if pl >= THRESHOLD_L and pl > ps:
            position = 1
            entry_price = n_o
            capital *= (1 - FEE)
            trades += 1
        elif ps >= THRESHOLD_S and ps > pl:
            position = -1
            entry_price = n_o
            capital *= (1 - FEE)
            trades += 1
    else:
        if position == 1:
            if pl >= THRESHOLD_L and pl > ps:
                pass # 유지
            else:
                ret = (n_o - entry_price) / entry_price
                trades_log.append(ret)
                capital *= (1 + ret) * (1 - FEE)
                position = 0
        elif position == -1:
            if ps >= THRESHOLD_S and ps > pl:
                pass # 유지
            else:
                ret = (entry_price - n_o) / entry_price
                trades_log.append(ret)
                capital *= (1 + ret) * (1 - FEE)
                position = 0
                
    curve.append(capital)

res_trade = pd.DataFrame({'gross': trades_log})
res_trade['net'] = res_trade['gross'] - (FEE*2)

print("\n--- 파이널 지표 ---")
print(f"최종 자산: ${capital:.2f} (초기 10,000)")
print(f"총 거래 횟수: {trades}회")

if len(res_trade) > 0:
    print(f"거래 1회당 평균 Gross: {res_trade['gross'].mean()*100:.3f}%")
    print(f"거래 1회당 평균 Net: {res_trade['net'].mean()*100:.3f}%")
    print(f"Net(+수수료) 승률: {(res_trade['net'] > 0).mean()*100:.1f}%")

bnh_entry = res.iloc[0]['next_open']
bnh_curve = [10000 * (1 + (n_o - bnh_entry) / bnh_entry) for n_o in res['next_open']]

plt.figure(figsize=(12, 6))
plt.plot(res['datetime'], curve[1:], label=f"0.2% 타겟 모델 (${capital:.0f})")
plt.plot(res['datetime'], bnh_curve, label=f"Buy&Hold (${bnh_curve[-1]:.0f})", ls='--', alpha=0.6)
plt.title(f"2채널 타겟(0.2% 이상) 변동 모델 백테스트 성과 (롱숏 기준:{THRESHOLD_L})")
plt.legend()
plt.savefig("v3_equity.png", bbox_inches='tight')
print("[!] 시각화 파일 'v3_equity.png' 저장 완료.")
