# Coin Trader

바이낸스 암호화폐 데이터를 수집하고 자동매매 전략을 개발하기 위한 프레임워크입니다.

## 현재 상태

- [x] 과거 데이터 수집 (BTC, ETH, XRP, SOL / 1분봉 / 2019-01-01 ~)
- [x] 백테스터 (look-ahead bias 제거, 수수료 반영, 롱/숏)
- [x] MA 교차 전략
- [x] 백테스트 결과 시각화 (인터랙티브 HTML)
- [x] **LSTM 알파 모델 (15분봉 캔들 패턴 기반)**
- [ ] 자동매매 봇
- [ ] 대시보드

## 프로젝트 구조

```
Coin-trader/
├── common/             # 공통 모듈 (거래소 연결, 로거, 유틸리티)
├── config/             # 설정 파일 (settings.yaml)
├── data/               # 데이터 수집 및 저장 모듈
│   ├── historical/     # CSV 데이터 저장 위치 (gitignore)
│   ├── downloader.py   # 과거 데이터 다운로더
│   └── storage.py      # 데이터 저장/로드
├── strategy/
│   ├── base.py         # 전략 베이스 클래스
│   └── ma_cross.py     # 이동평균 교차 전략
├── backtester/
│   ├── engine.py       # 백테스트 엔진 (바 단위 시뮬레이션)
│   ├── portfolio.py    # 포지션 및 자산 관리
│   ├── report.py       # 성과 지표 출력
│   └── visualizer.py   # 인터랙티브 차트 생성
├── bot/                # 자동매매 봇 (개발 예정)
├── dashboard/          # 대시보드 (개발 예정)
├── tests/              # 연결 및 기능 테스트
├── download_all.py     # 데이터 다운로드 스크립트
├── update_all.py       # 데이터 증분 업데이트 스크립트
├── check_all_data.py   # 저장된 데이터 현황 확인
└── run_backtest.py     # 백테스트 실행 스크립트
```

## 설치

### 1. 저장소 클론

```bash
git clone <repository-url>
cd Coin-trader
```

### 2. 패키지 설치

```bash
pip install -r requirements.txt
```

### 3. 환경 변수 설정

```bash
cp .env.example .env
```

`.env` 파일을 열어 바이낸스 API 키를 입력합니다.
데이터 다운로드는 API 키 없이도 가능합니다.

## 데이터 수집

### 연결 테스트

```bash
python tests/test_connection.py
```

### 데이터 다운로드 (권장)

수집된 데이터를 구글 드라이브에서 직접 받을 수 있습니다.

**[구글 드라이브에서 다운로드](https://drive.google.com/drive/folders/1CU-CUAQYkk_GmXJ4YbSFZzR5yY_k5iVT?usp=drive_link)**

다운받은 CSV 파일들을 `data/historical/` 폴더에 넣으면 됩니다.

```
data/historical/
├── BTC_USDT_1m.csv
├── ETH_USDT_1m.csv
├── XRP_USDT_1m.csv
└── SOL_USDT_1m.csv
```

### 직접 다운로드 (선택)

구글 드라이브 대신 직접 바이낸스에서 받으려면 `download_all.py`에서 원하는 코인의 주석을 해제하고 실행합니다.

```bash
python download_all.py
```

코인별로 순서대로 실행하도록 주석 처리되어 있습니다. (BTC 기준 약 2시간 30분 소요)

### 데이터 증분 업데이트

이미 다운받은 데이터에서 최신 데이터만 추가로 받습니다.

```bash
python update_all.py
```

### 데이터 현황 확인

```bash
python check_all_data.py
```

## 백테스트

### 실행

```bash
python run_backtest.py
```

결과 지표(수익률, MDD, 샤프 지수 등)가 출력되고 `backtest_result.html`이 생성됩니다.
브라우저에서 열면 가격 차트, 이동평균선, 거래 시점, 자산 곡선을 인터랙티브하게 확인할 수 있습니다.

### 주요 파라미터 (`run_backtest.py`)

```python
strategy = MACrossStrategy(
    fast_period=10,      # 단기 이동평균 기간
    slow_period=60,      # 장기 이동평균 기간
    min_diff_pct=0.3,    # MA 간격이 가격의 N% 이상일 때만 교차 인정 (노이즈 필터)
)
engine = Engine(
    strategy,
    initial_capital=10_000.0,  # 초기 자산 (USDT)
    fee_rate=0.001,             # 수수료 (0.1%)
    cooldown=1440,              # 거래 후 최소 대기 바 수 (1440 = 1일)
)
```

### 백테스터 설계 원칙

- **look-ahead bias 제거**: 바 i의 시그널 → 바 i+1 시가에 체결
- **수수료 반영**: 진입/청산 시 각각 0.1% 적용
- **강제 청산**: 자산이 0 이하가 되면 즉시 청산 (음수 자산 방지)
- **롱/숏 모두 지원**

## LSTM 알파 모델

BTC/USDT 1분봉 데이터를 15분봉으로 리샘플하여 캔들 패턴 기반 LSTM 모델로 다음 봉의 방향을 예측합니다.

### 모델 구조

```
입력: 캔들 피처 6개 × 시퀀스 10봉
  → LSTM (Hidden 64, 2층, Dropout 0.3)
  → FC (64 → 32 → ReLU → Dropout → 출력)
  → Sigmoid
```

### 입력 피처 (6개)

| 피처 | 설명 |
|------|------|
| `body_ratio` | 캔들 몸통 길이 / 전체 고저 범위 |
| `upper_ratio` | 윗꼬리 길이 / 전체 고저 범위 |
| `lower_ratio` | 아랫꼬리 길이 / 전체 고저 범위 |
| `up_down` | 양봉(1) / 음봉(0) 여부 |
| `box_length` | 몸통 길이 / 시가 (수익률 기준 정규화) |
| `volume_ratio` | 거래량 / 20봉 이동평균 거래량 |

### 학습 / 검증 / 테스트 기간 분할

| 구간 | 기간 |
|------|------|
| Train | ~ 2022-12-31 |
| Validation | 2023-01-01 ~ 2023-06-30 |
| Test | 2023-07-01 ~ 2024-04-01 |

정규화는 Train 셋의 mean/std로만 계산하여 데이터 누수(data leakage) 방지

### 버전별 타겟 설계

**V1 (`lstm_15m_alpha.py`) — 단일 이진 분류**
- 타겟: 다음 봉 종가 > 다음 봉 시가 → 양봉(1) / 음봉(0)
- 저장 모델: `lstm_best.pth`
- 임계값: 롱 ≥ 0.55, 숏 ≤ 0.45 (기본) / 0.65/0.35 (강한 확신)

**V2 (`lstm_15m_alpha_v2.py`) — 다중 타겟 이진 분류**
- 타겟 롱: 다음 봉 수익률 ≥ +0.2%
- 타겟 숏: 다음 봉 수익률 ≤ -0.2%
- 출력: Sigmoid 2채널 (롱 확률, 숏 확률) 동시 예측
- 저장 모델: `lstm_multi_best.pth`
- 임계값: 롱/숏 모두 ≥ 0.65, 반대 채널보다 높을 때만 진입

### 백테스트 로직 (포지션 유지 포함)

```
포지션 없을 때:
  확신도 ≥ 임계값 → 진입 (다음 봉 시가에 체결)

포지션 있을 때:
  같은 방향 확신 유지 → 홀딩 (수수료 없이 유지)
  확신 소멸 또는 반대 시그널 → 청산 (다음 봉 시가에 체결)

수수료: 0.1% 편도 (진입/청산 각각 적용)
```

### 관련 스크립트

| 파일 | 설명 |
|------|------|
| `lstm_15m_alpha.py` | V1 모델 학습 + 히스토그램 + 백테스트 |
| `lstm_15m_alpha_v2.py` | V2 다중 타겟 모델 학습 + 백테스트 |
| `run_simulation.py` | V1 모델 재추론 + Buy&Hold 비교 백테스트 |
| `analyze_trades.py` | V1 모델 거래 단위 승률/수익률 상세 분석 |
| `check_dist.py` | 피처 분포 확인 |

### 결과 이미지

| 파일 | 내용 |
|------|------|
| `histogram_validation.png` | V1 검증셋 예측 확률 분포 |
| `backtest_equity.png` | V1 백테스트 자산 곡선 |
| `backtest_equity_v2.png` | V1 강한 확신(0.65/0.35) + Buy&Hold 비교 |
| `v3_histogram.png` | V2 롱/숏 확률 분포 (2채널) |
| `v3_equity.png` | V2 백테스트 자산 곡선 + Buy&Hold 비교 |

### 실행 방법

```bash
# V1 모델 학습
python lstm_15m_alpha.py

# V2 다중 타겟 모델 학습
python lstm_15m_alpha_v2.py

# V1 모델로 시뮬레이션 (저장된 가중치 사용)
python run_simulation.py

# 거래 단위 수익 분석
python analyze_trades.py
```

> **데이터 경로**: `data/BTC_USDT_1m.csv` (구글 드라이브에서 다운로드 후 배치)

---

## 새로운 전략 추가

`strategy/base.py`의 `BaseStrategy`를 상속하여 `generate_signals()`를 구현합니다.

```python
from strategy.base import BaseStrategy
import numpy as np
import pandas as pd

class MyStrategy(BaseStrategy):
    def generate_signals(self, df: pd.DataFrame) -> np.ndarray:
        # 반환: np.ndarray (int8), 1=롱, -1=숏, 0=중립
        # 주의: rolling() 등 과거 데이터만 사용 (look-ahead bias 금지)
        ...
```

## 설정

`config/settings.yaml`에서 거래소, 대상 코인, 저장 경로 등을 변경할 수 있습니다.

## 주의사항

- `data/historical/` 디렉토리의 CSV 파일은 용량이 크므로 git에서 제외됩니다.
- `.env` 파일은 절대 git에 올리지 마세요.
