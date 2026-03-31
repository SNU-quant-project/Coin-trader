from data.storage import Storage
from strategy.ma_cross import MACrossStrategy
from backtester.engine import Engine
from backtester.report import Report
from backtester.visualizer import plot

storage = Storage()
df = storage.load_candles("BTC/USDT", "1m")

strategy = MACrossStrategy(
    fast_period=10,
    slow_period=60,
    min_diff_pct=0.3,   # MA 간격이 가격의 0.1% 이상일 때만 교차 인정
)
engine = Engine(
    strategy,
    initial_capital=10_000.0,
    fee_rate=0.001,
    cooldown=1440,        # 거래 후 최소 240분(4시간) 대기
)

result = engine.run(df)
report = Report(result, timeframe="1m")
report.summary()

plot(result, df, strategy, output_path="backtest_result.html")
