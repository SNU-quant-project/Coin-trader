"""
Donchian Channel Breakout 백테스트 실행 스크립트

실행 방법:
    python run_backtest.py

데이터 흐름:
    1. BTC/USDT 1분봉 CSV 로드
    2. 4시간봉으로 리샘플링 (노이즈 제거 + 적절한 거래 빈도)
    3. 테스트 기간 필터링 (2024년)
    4. Donchian 전략으로 시그널 생성 → 엔진에서 시뮬레이션
    5. 성과 리포트 출력 + 인터랙티브 차트 생성

주요 설정:
    - 타임프레임: 4시간봉
    - 진입 채널: 120봉 (= 20일) 최고가 돌파 시 롱
    - 청산 채널:  60봉 (= 10일) 최저가 이탈 시 청산
    - 수수료: 0.1% (바이낸스 기본)
    - 쿨다운: 6봉 (= 24시간, 잦은 whipsaw 방지)
"""

import pandas as pd
from strategy.donchian_breakout import DonchianBreakoutStrategy
from backtester.engine import Engine
from backtester.report import Report
from backtester.visualizer import plot


# ══════════════════════════════════════════════════════════════
#  1. 설정
# ══════════════════════════════════════════════════════════════

# 데이터 경로 (구글 드라이브에서 다운로드한 1분봉 CSV)
DATA_PATH = "data/historical/BTC_USDT_1m.csv"

# 리샘플링 타임프레임
TIMEFRAME = "4h"

# 테스트 기간
TEST_START = "2025-01-01"
TEST_END   = "2026-03-25"

# 전략 파라미터
# 4시간봉 기준: 하루 = 6봉, 20일 = 120봉, 10일 = 60봉
ENTRY_PERIOD = 120   # 진입 채널 기간 (20일)
EXIT_PERIOD  = 60    # 청산 채널 기간 (10일)

# 엔진 파라미터
INITIAL_CAPITAL = 10_000.0   # 초기 자산 (USDT)
FEE_RATE = 0.001             # 수수료 (0.1%)
COOLDOWN = 6                 # 거래 후 최소 대기 봉 수 (6봉 = 24시간)


# ══════════════════════════════════════════════════════════════
#  2. 데이터 로드 + 리샘플링
# ══════════════════════════════════════════════════════════════

print(f"데이터 로드 중: {DATA_PATH}")
df_raw = pd.read_csv(DATA_PATH, parse_dates=['datetime'])

# 워밍업 구간 확보: 테스트 시작일보다 entry_period만큼 앞선 데이터 포함
# 4시간봉 120봉 = 약 20일 → 여유있게 30일 앞부터 로드
warmup_start = pd.Timestamp(TEST_START) - pd.Timedelta(days=30)
df_raw = df_raw[df_raw['datetime'] >= warmup_start].copy()

print(f"원본 1분봉: {len(df_raw):,}행")
print(f"기간: {df_raw['datetime'].iloc[0]} ~ {df_raw['datetime'].iloc[-1]}")

# 4시간봉으로 리샘플링
df_raw = df_raw.set_index('datetime')
df_4h = df_raw.resample(TIMEFRAME).agg({
    'open':   'first',
    'high':   'max',
    'low':    'min',
    'close':  'last',
    'volume': 'sum',
}).dropna().reset_index()

print(f"리샘플링 완료 ({TIMEFRAME}봉): {len(df_4h):,}행")
# 테스트 종료일 이후 데이터 제거
df_4h = df_4h[df_4h['datetime'] <= TEST_END].copy()
print(f"테스트 기간 적용 후: {len(df_4h):,}행 (~{TEST_END})")


# ══════════════════════════════════════════════════════════════
#  3. 전략 + 엔진 설정
# ══════════════════════════════════════════════════════════════

strategy = DonchianBreakoutStrategy(
    entry_period=ENTRY_PERIOD,
    exit_period=EXIT_PERIOD,
)

engine = Engine(
    strategy,
    initial_capital=INITIAL_CAPITAL,
    fee_rate=FEE_RATE,
    cooldown=COOLDOWN,
)


# ══════════════════════════════════════════════════════════════
#  4. 백테스트 실행
# ══════════════════════════════════════════════════════════════

print(f"\n{'='*60}")
print(f"  전략: Donchian Channel Breakout (롱 온리)")
print(f"  진입 채널: {ENTRY_PERIOD}봉 ({ENTRY_PERIOD // 6}일) 최고가 돌파")
print(f"  청산 채널: {EXIT_PERIOD}봉 ({EXIT_PERIOD // 6}일) 최저가 이탈")
print(f"  타임프레임: {TIMEFRAME}")
print(f"  테스트 기간: {TEST_START} ~ {TEST_END}")
print(f"  초기 자산: {INITIAL_CAPITAL:,.0f} USDT")
print(f"  수수료: {FEE_RATE * 100:.1f}%")
print(f"  쿨다운: {COOLDOWN}봉 ({COOLDOWN * 4}시간)")
print(f"{'='*60}\n")

result = engine.run(df_4h)


# ══════════════════════════════════════════════════════════════
#  5. 리포트 출력
# ══════════════════════════════════════════════════════════════

report = Report(result, timeframe=TIMEFRAME)
stats = report.summary()


# ══════════════════════════════════════════════════════════════
#  6. 시각화
# ══════════════════════════════════════════════════════════════

output_path = "backtest_result.html"
plot(result, df_4h, strategy, output_path=output_path)
print(f"\n브라우저에서 {output_path} 파일을 열어 차트를 확인하세요.")
