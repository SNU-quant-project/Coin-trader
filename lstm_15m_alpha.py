import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import os
import gc

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

# 시드 고정
torch.manual_seed(42)
np.random.seed(42)

# ==========================================
# 1. 하이퍼파라미터 및 설정
# ==========================================
DATA_PATH = "data/BTC_USDT_1m.csv"
SEQ_LEN = 10
BATCH_SIZE = 512
EPOCHS = 15
LR = 0.001
HIDDEN_SIZE = 64
NUM_LAYERS = 2
DROPOUT = 0.3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"[{DEVICE}] 디바이스로 학습을 준비합니다.")

# ==========================================
# 2. 데이터 로드 및 15분봉 변환
# ==========================================
print("데이터 로딩 시작...")
# 전체 데이터를 읽되, 필요한 컬럼만
df = pd.read_csv(DATA_PATH, usecols=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
df.set_index('datetime', inplace=True)
df.sort_index(inplace=True)

print("15분봉 리샘플링 중...")
df_15m = df.resample('15min').agg({
    'open': 'first',
    'high': 'max',
    'low': 'min',
    'close': 'last',
    'volume': 'sum'
}).dropna()

# ==========================================
# 3. 피처 계산 및 정규화
# ==========================================
print("피처(Feature) 계산 중...")

hl_diff = (df_15m['high'] - df_15m['low']) + 1e-8
open_close_max = df_15m[['open', 'close']].max(axis=1)
open_close_min = df_15m[['open', 'close']].min(axis=1)

# 1. 상자가 차지하는 비율
df_15m['body_ratio'] = (df_15m['close'] - df_15m['open']).abs() / hl_diff
# 2. 위쪽 꼬리의 비율
df_15m['upper_ratio'] = (df_15m['high'] - open_close_max) / hl_diff
# 3. 아래쪽 꼬리의 비율
df_15m['lower_ratio'] = (open_close_min - df_15m['low']) / hl_diff
# 4. 상승/하락 여부 (1=Up, 0=Down)
df_15m['up_down'] = (df_15m['close'] > df_15m['open']).astype(float)
# 5. 상자의 길이 (수익률 정규화)
df_15m['box_length'] = (df_15m['close'] - df_15m['open']).abs() / (df_15m['open'] + 1e-8)
# 6. 거래량 (이동평균 대비 비율)
df_15m['volume_ratio'] = df_15m['volume'] / (df_15m['volume'].rolling(20).mean() + 1e-8)

# Target: 다음 1개 15분봉이 양봉인지 여부 (t+1 close > t+1 open => 1)
df_15m['target'] = (df_15m['close'].shift(-1) > df_15m['open'].shift(-1)).astype(float)

df_15m.dropna(inplace=True) # rolling(20) 밑 shift(-1) 결측치 제거

features = ['body_ratio', 'upper_ratio', 'lower_ratio', 'up_down', 'box_length', 'volume_ratio']
print(f"제작 완료 데이터 크기 (15분봉): {df_15m.shape}")

# 메모리 해제
del df
gc.collect()

# ==========================================
# 4. 시퀀스 데이터 생성 및 기간 분할
# ==========================================
print("시퀀스 데이터 생성 및 Train/Val/Test 분할 중...")
# Train: ~ 2022-12-31
# Val  : 2023-01-01 ~ 2023-06-30
# Test : 2023-07-01 ~

df_train = df_15m.loc[:"2022-12-31"].copy()
df_val = df_15m.loc["2023-01-01":"2023-06-30"].copy()
df_test = df_15m.loc["2023-07-01":"2024-04-01"].copy()

# 입력 텐서는 분포 차이가 클 수 있으니 mean/std 정규화를 해줍니다 (학습 안정성)
mean_vals = df_train[features].mean()
std_vals = df_train[features].std() + 1e-8

df_train[features] = (df_train[features] - mean_vals) / std_vals
df_val[features] = (df_val[features] - mean_vals) / std_vals
df_test[features] = (df_test[features] - mean_vals) / std_vals

def create_sequences(df_sub):
    X, y, idx_arr = [], [], []
    data_feat = df_sub[features].values
    data_target = df_sub['target'].values
    indices = df_sub.index
    
    for i in range(len(df_sub) - SEQ_LEN):
        X.append(data_feat[i : i + SEQ_LEN])
        y.append(data_target[i + SEQ_LEN - 1]) # 끝나는 봉에 맞춘 타겟
        idx_arr.append(indices[i + SEQ_LEN]) # 실제 진입 타임스탬프
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32), idx_arr

X_train, y_train, _ = create_sequences(df_train)
X_val, y_val, idx_val = create_sequences(df_val)
X_test, y_test, idx_test = create_sequences(df_test)

class SeqDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X)
        self.y = torch.tensor(y)
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.y[i]

train_loader = DataLoader(SeqDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(SeqDataset(X_val, y_val), batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(SeqDataset(X_test, y_test), batch_size=BATCH_SIZE, shuffle=False)

# ==========================================
# 5. 모델 구성
# ==========================================
class LSTMAlpha(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=len(features), hidden_size=HIDDEN_SIZE, 
                            num_layers=NUM_LAYERS, batch_first=True, dropout=DROPOUT)
        self.fc = nn.Sequential(
            nn.Linear(HIDDEN_SIZE, 32),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        out = self.fc(h_n[-1])
        return out.squeeze(1)

model = LSTMAlpha().to(DEVICE)
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# ==========================================
# 6. 추론 및 확률 분포(Histogram) 출력
# ==========================================
def get_predictions(data_loader):
    model.eval()
    all_probs = []
    with torch.no_grad():
        for bx, _ in data_loader:
            bx = bx.to(DEVICE)
            probs = model(bx).cpu().numpy()
            all_probs.extend(probs)
    return np.array(all_probs)

def plot_histogram(probs, name="Validation"):
    plt.figure(figsize=(10, 5))
    plt.hist(probs, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
    plt.title(f"{name} Set - 예측 확률 분포 (Sigmoid Output)")
    plt.xlabel("상승 예측 확률 (Probability of Up)")
    plt.ylabel("빈도 (Frequency)")
    plt.axvline(0.5, color='red', linestyle='dashed', linewidth=1)
    
    file_name = f"histogram_{name.lower()}.png"
    plt.savefig(file_name, bbox_inches='tight')
    print(f"\n[!] 히스토그램 시각화가 '{file_name}' 파일로 저장되었습니다!")
    plt.close()

# ==========================================
# 학습 수행
# ==========================================
print("\n[학습 시작]")
best_val_loss = float('inf')

for epoch in range(1, EPOCHS + 1):
    model.train()
    train_loss = 0
    for bx, by in train_loader:
        bx, by = bx.to(DEVICE), by.to(DEVICE)
        optimizer.zero_grad()
        out = model(bx)
        loss = criterion(out, by)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * len(bx)
        
    train_loss /= len(train_loader.dataset)
    
    # Val Test
    model.eval()
    val_loss = 0
    correct = 0
    with torch.no_grad():
        for bx, by in val_loader:
            bx, by = bx.to(DEVICE), by.to(DEVICE)
            out = model(bx)
            loss = criterion(out, by)
            val_loss += loss.item() * len(bx)
            
            preds = (out >= 0.5).float()
            correct += (preds == by).sum().item()
            
    val_loss /= len(val_loader.dataset)
    val_acc = correct / len(val_loader.dataset)
    
    print(f"Epoch [{epoch}/{EPOCHS}] Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}%")
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), "lstm_best.pth")

print("\n학습 완료! 검증셋(Val)의 확률 분포 히스토그램을 추출합니다...")
model.load_state_dict(torch.load("lstm_best.pth"))
val_probs = get_predictions(val_loader)
plot_histogram(val_probs, "Validation")

# ==========================================
# 7. 백테스트: 유지(Holding) 로직 포함
# ==========================================
print("\n[백테스트 시뮬레이션 - 유지 로직 적용]")
test_probs = get_predictions(test_loader)

# 임계값 임의 설정 (히스토그램 보고 추후 변경 가능)
THRESHOLD_L = 0.55 # 롱 (55% 이상 확신)
THRESHOLD_S = 0.45 # 숏 (45% 이하 확신)

# 테스트 결과 통합
res = pd.DataFrame({'datetime': idx_test, 'prob': test_probs, 'actual': y_test})
# 가격 데이터 매핑 (다음 봉 진입가 및 청산가 계산을 위해 원본 df_15m 데이터 결합)
# 시그널 발생 시각이 `datetime` 이므로, 진입은 해당 봉 다음(오픈), 모델 예측 타겟은 원래 다음 15분.
# df_15m와 시간 동기화
df_test_prices = df_test.loc[idx_test].copy()
res['open'] = df_test_prices['open'].values
res['close'] = df_test_prices['close'].values
# 다음 봉의 open, close
res['next_open'] = df_test_prices['open'].shift(-1).values
res['next_close'] = df_test_prices['close'].shift(-1).values
res.dropna(inplace=True)

capital = 10000.0
position = 0 # 1 (Long), -1 (Short), 0 (Neutral)
FEE = 0.001 # 0.1% 수수료
entry_price = 0.0

equity_curve = [capital]

# 반복문을 통한 시한별 시뮬레이션
for i in range(len(res)):
    prob = res.iloc[i]['prob']
    next_open = res.iloc[i]['next_open']
    next_close = res.iloc[i]['next_close']
    
    if position == 0:
        # 1. 포지션이 0일 때 진입 조건 검사
        if prob >= THRESHOLD_L:
            position = 1
            entry_price = next_open
            capital *= (1 - FEE) # 진입 수수료
        elif prob <= THRESHOLD_S:
            position = -1
            entry_price = next_open
            capital *= (1 - FEE) # 진입 수수료
    else:
        # 2. 포지션이 있을 때: "청산할지 유지할지 결단"
        if position == 1:
            # 롱 포지션인데, 이번 예측도 롱 강한 확신이면 홀딩(스킵)
            if prob >= THRESHOLD_L:
                pass # 유지
            else:
                # 확신이 떨어졌거나 숏이면 청산
                ret = (next_open - entry_price) / entry_price
                capital *= (1 + ret)
                capital *= (1 - FEE) # 청산 수수료
                position = 0
                
        elif position == -1:
            # 숏 포지션인데, 이번 예측도 숏 강한 확신이면 홀딩(스킵)
            if prob <= THRESHOLD_S:
                pass # 유지
            else:
                # 청산
                ret = (entry_price - next_open) / entry_price
                capital *= (1 + ret)
                capital *= (1 - FEE) # 청산 수수료
                position = 0
                
    equity_curve.append(capital)

print(f"최종 자본금: ${capital:.2f}")

plt.figure(figsize=(12, 6))
plt.plot(res['datetime'], equity_curve[1:], label=f"전략 수익금 (Long > {THRESHOLD_L}, Short < {THRESHOLD_S})")
plt.title("LSTM 15분봉 예측 알파 성과")
plt.xlabel("시간")
plt.ylabel("자산(USD)")
plt.legend()
plt.savefig("backtest_equity.png", bbox_inches='tight')
print("[!] 백테스트 자산 커브가 'backtest_equity.png'로 저장되었습니다.")
