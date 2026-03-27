# Coin Trader

바이낸스 암호화폐 데이터를 수집하고 자동매매 전략을 개발하기 위한 프레임워크입니다.

## 현재 상태

- [x] 과거 데이터 수집 (BTC, ETH, XRP, SOL / 1분봉 / 2019-01-01 ~)
- [ ] 전략 개발
- [ ] 백테스터
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
├── strategy/           # 매매 전략 (개발 예정)
├── backtester/         # 백테스터 (개발 예정)
├── bot/                # 자동매매 봇 (개발 예정)
├── dashboard/          # 대시보드 (개발 예정)
├── tests/              # 연결 및 기능 테스트
├── download_all.py     # 데이터 다운로드 실행 스크립트
└── check_all_data.py   # 저장된 데이터 현황 확인
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

### 데이터 다운로드

`download_all.py`에서 원하는 코인의 주석을 해제하고 실행합니다.

```bash
python download_all.py
```

코인별로 순서대로 실행하도록 주석 처리되어 있습니다.
(BTC 기준 약 2시간 30분 소요)

### 데이터 현황 확인

```bash
python check_all_data.py
```

## 설정

`config/settings.yaml`에서 거래소, 대상 코인, 저장 경로 등을 변경할 수 있습니다.

## 주의사항

- `data/historical/` 디렉토리의 CSV 파일은 용량이 크므로 git에서 제외됩니다.
- `.env` 파일은 절대 git에 올리지 마세요.
