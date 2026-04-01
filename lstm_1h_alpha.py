import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
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
SEQ_LEN    = 5       # 직전 5개 봉 참고
BATCH_SIZE = 256
EPOCHS     = 20
LR         = 0.001
HIDDEN     = 64
N_LAYERS   = 2
DROPOUT    = 0.3
TARGET_PCT = 0.002   # 강한 상승/하락 기준: 0.2%
THRESHOLD  = 0.50    # 진입 확신 임계값 (히스토그램 보고 조정)
FEE        = 0.001   # 0.1% 편도

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[{device}] 디바이스 준비 완료")

# ==========================================
# 데이터 로드 및 1시간봉 변환
# ==========================================
print("데이터 로딩 중...")
df = pd.read_csv('data/BTC_USDT_1m.csv',
                 usecols=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
df.set_index('datetime', inplace=True)
df.sort_index(inplace=True)

print("1시간봉 리샘플링 중...")
df_1h = df.resample('1h').agg({
    'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
}).dropna()

del df
gc.collect()

# ==========================================
# 피처 계산 (캔들 형태 기반 6개)
# ==========================================
hl_diff = (df_1h['high'] - df_1h['low']) + 1e-8
oc_max  = df_1h[['open', 'close']].max(axis=1)
oc_min  = df_1h[['open', 'close']].min(axis=1)

df_1h['body_ratio']   = (df_1h['close'] - df_1h['open']).abs() / hl_diff
df_1h['upper_ratio']  = (df_1h['high'] - oc_max) / hl_diff
df_1h['lower_ratio']  = (oc_min - df_1h['low']) / hl_diff
df_1h['up_down']      = (df_1h['close'] > df_1h['open']).astype(float)
df_1h['box_length']   = (df_1h['close'] - df_1h['open']).abs() / (df_1h['open'] + 1e-8)
df_1h['volume_ratio'] = df_1h['volume'] / (df_1h['volume'].rolling(20).mean() + 1e-8)

# ==========================================
# 3진 분류 타겟
#   0: 강한 하락 (다음 봉 수익률 <= -0.2%)
#   1: 횡보     (-0.2% < 수익률 < +0.2%)
#   2: 강한 상승 (다음 봉 수익률 >= +0.2%)
# ==========================================
next_ret = (df_1h['close'].shift(-1) - df_1h['open'].shift(-1)) / (df_1h['open'].shift(-1) + 1e-8)
df_1h['target'] = np.select(
    [next_ret <= -TARGET_PCT, next_ret >= TARGET_PCT],
    [0, 2],
    default=1
).astype(np.int64)

df_1h.dropna(inplace=True)

features = ['body_ratio', 'upper_ratio', 'lower_ratio', 'up_down', 'box_length', 'volume_ratio']

n0 = (df_1h['target'] == 0).mean()
n1 = (df_1h['target'] == 1).mean()
n2 = (df_1h['target'] == 2).mean()
print(f"1시간봉 데이터: {len(df_1h)}개")
print(f"타겟 분포 | 강한하락(0): {n0:.3f} | 횡보(1): {n1:.3f} | 강한상승(2): {n2:.3f}")

# ==========================================
# Train / Val / Test 스플릿
# ==========================================
df_train = df_1h.loc[:"2022-12-31"].copy()
df_val   = df_1h.loc["2023-01-01":"2023-06-30"].copy()
df_test  = df_1h.loc["2023-07-01":"2024-04-01"].copy()

# Train 기준으로 정규화 (data leakage 방지)
mean_v = df_train[features].mean()
std_v  = df_train[features].std() + 1e-8
df_train[features] = (df_train[features] - mean_v) / std_v
df_val[features]   = (df_val[features]   - mean_v) / std_v
df_test[features]  = (df_test[features]  - mean_v) / std_v

def build_sequences(df_sub):
    X, y, idx = [], [], []
    feat = df_sub[features].values
    tgt  = df_sub['target'].values
    ts   = df_sub.index
    for i in range(len(df_sub) - SEQ_LEN):
        X.append(feat[i : i + SEQ_LEN])
        y.append(tgt[i + SEQ_LEN - 1])
        idx.append(ts[i + SEQ_LEN])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64), idx

X_trn, y_trn, _       = build_sequences(df_train)
X_val, y_val, idx_val = build_sequences(df_val)
X_tst, y_tst, idx_tst = build_sequences(df_test)

print(f"샘플 수 | Train: {len(X_trn)} | Val: {len(X_val)} | Test: {len(X_tst)}")

class SeqData(Dataset):
    def __init__(self, x, y):
        self.x = torch.tensor(x)
        self.y = torch.tensor(y)
    def __len__(self): return len(self.x)
    def __getitem__(self, i): return self.x[i], self.y[i]

train_ld = DataLoader(SeqData(X_trn, y_trn), batch_size=BATCH_SIZE, shuffle=True)
val_ld   = DataLoader(SeqData(X_val, y_val), batch_size=BATCH_SIZE, shuffle=False)
test_ld  = DataLoader(SeqData(X_tst, y_tst), batch_size=BATCH_SIZE, shuffle=False)

# ==========================================
# 모델 (3진 분류, CrossEntropyLoss)
# ==========================================
class LSTMClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=len(features), hidden_size=HIDDEN,
            num_layers=N_LAYERS, batch_first=True, dropout=DROPOUT
        )
        self.fc = nn.Sequential(
            nn.Linear(HIDDEN, 32),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(32, 3)   # logits for 3 classes
        )

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        return self.fc(h_n[-1])

model     = LSTMClassifier().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# ==========================================
# 학습
# ==========================================
print("\n[학습 시작 - 3진 분류]")
best_vloss = float('inf')

for ep in range(1, EPOCHS + 1):
    model.train()
    t_loss = 0
    for bx, by in train_ld:
        bx, by = bx.to(device), by.to(device)
        optimizer.zero_grad()
        loss = criterion(model(bx), by)
        loss.backward()
        optimizer.step()
        t_loss += loss.item() * len(bx)
    t_loss /= len(train_ld.dataset)

    model.eval()
    v_loss, correct = 0, 0
    with torch.no_grad():
        for bx, by in val_ld:
            bx, by = bx.to(device), by.to(device)
            out = model(bx)
            v_loss  += criterion(out, by).item() * len(bx)
            correct += (out.argmax(1) == by).sum().item()
    v_loss /= len(val_ld.dataset)
    v_acc   = correct / len(val_ld.dataset)

    print(f"Epoch {ep:2d} | Train Loss: {t_loss:.4f} | Val Loss: {v_loss:.4f} | Val Acc: {v_acc*100:.2f}%")
    if v_loss < best_vloss:
        best_vloss = v_loss
        torch.save(model.state_dict(), 'lstm_1h_best.pth')

# ==========================================
# 테스트 추론
# ==========================================
print("\n[테스트셋 추론]")
model.load_state_dict(torch.load('lstm_1h_best.pth', map_location=device))
model.eval()

tst_probs = []
with torch.no_grad():
    for bx, _ in test_ld:
        bx = bx.to(device)
        probs = F.softmax(model(bx), dim=1)
        tst_probs.extend(probs.cpu().numpy())
tst_probs = np.array(tst_probs)  # shape: (N, 3)

# 테스트 정확도
tst_preds = tst_probs.argmax(axis=1)
tst_acc   = (tst_preds == y_tst).mean()
print(f"Test Accuracy: {tst_acc*100:.2f}%")
for cls, name in [(0, '강한하락'), (1, '횡보'), (2, '강한상승')]:
    mask = y_tst == cls
    if mask.sum() > 0:
        cls_acc = (tst_preds[mask] == cls).mean()
        print(f"  {name}({cls}) 클래스 정확도: {cls_acc*100:.2f}% ({mask.sum()}개)")

# ==========================================
# 확률 분포 히스토그램
# ==========================================
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for i, (name, color) in enumerate([('강한 하락(0)', 'red'), ('횡보(1)', 'gray'), ('강한 상승(2)', 'blue')]):
    axes[i].hist(tst_probs[:, i], bins=50, color=color, alpha=0.7, edgecolor='black')
    axes[i].axvline(THRESHOLD, color='black', linestyle='--', linewidth=1, label=f'임계값 {THRESHOLD}')
    axes[i].set_title(f"{name} 예측 확률 분포")
    axes[i].set_xlabel("예측 확률")
    axes[i].set_ylabel("빈도")
    axes[i].legend()
plt.tight_layout()
plt.savefig('1h_histogram.png', bbox_inches='tight')
print("[!] 히스토그램 저장: 1h_histogram.png")

# ==========================================
# 백테스트 (1시간 후 청산)
# ==========================================
print(f"\n[백테스트: 임계값 {THRESHOLD}, 1봉(1시간) 후 청산]")

res = pd.DataFrame({
    'datetime':  idx_tst,
    'prob_down': tst_probs[:, 0],
    'prob_side': tst_probs[:, 1],
    'prob_up':   tst_probs[:, 2],
    'actual':    y_tst
})
prc = df_test.loc[idx_tst].copy()
res['next_open']  = prc['open'].shift(-1).values
res['next_close'] = prc['close'].shift(-1).values
res.dropna(inplace=True)

capital    = 10000.0
trades     = 0
wins       = 0
trades_log = []
curve      = [capital]

bnh_entry = res.iloc[0]['next_open']
bnh_curve = [10000 * (1 + (no - bnh_entry) / bnh_entry) for no in res['next_open']]

for _, row in res.iterrows():
    p_up   = row['prob_up']
    p_down = row['prob_down']
    n_o    = row['next_open']
    n_c    = row['next_close']

    # 강한 상승/하락 확신이 임계값 이상이고 서로 중 더 높을 때 진입
    if p_up >= THRESHOLD and p_up > p_down:
        ret = (n_c - n_o) / n_o
        capital *= (1 + ret) * (1 - FEE) ** 2
        trades += 1
        wins   += int(ret > 0)
        trades_log.append({'type': 'LONG',  'ret': ret})

    elif p_down >= THRESHOLD and p_down > p_up:
        ret = (n_o - n_c) / n_o
        capital *= (1 + ret) * (1 - FEE) ** 2
        trades += 1
        wins   += int(ret > 0)
        trades_log.append({'type': 'SHORT', 'ret': ret})

    curve.append(capital)

print(f"\n--- 백테스트 결과 ---")
print(f"최종 자산: ${capital:.2f}  (초기 $10,000)")
print(f"Buy&Hold: ${bnh_curve[-1]:.2f}")
print(f"총 거래:  {trades}회")

if trades > 0:
    print(f"승률:     {wins/trades*100:.1f}%")
    df_trades = pd.DataFrame(trades_log)
    print(f"평균 수익(gross): {df_trades['ret'].mean()*100:.3f}%")
    for t in ['LONG', 'SHORT']:
        sub = df_trades[df_trades['type'] == t]
        if len(sub) > 0:
            print(f"  {t} ({len(sub)}회) 평균: {sub['ret'].mean()*100:.3f}% | 승률: {(sub['ret']>0).mean()*100:.1f}%")

plt.figure(figsize=(12, 6))
plt.plot(res['datetime'], curve[1:], label=f"모델 전략 (${capital:.0f})")
plt.plot(res['datetime'], bnh_curve, label=f"Buy&Hold (${bnh_curve[-1]:.0f})", ls='--', alpha=0.6)
plt.title(f"1시간봉 LSTM 3진분류 백테스트 (임계값: {THRESHOLD})")
plt.xlabel("날짜")
plt.ylabel("자산 (USD)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('1h_equity.png', bbox_inches='tight')
print("[!] 백테스트 결과 저장: 1h_equity.png")
