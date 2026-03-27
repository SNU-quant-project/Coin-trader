from data.storage import Storage
import pandas as pd

storage = Storage()
stats = storage.stats()

print("=" * 60)
print("  전체 데이터 현황")
print("=" * 60)

total_records = 0
total_size = 0

for symbol, info in stats["symbols"].items():
    df = storage.load_candles(symbol, "1m")
    if df is None:
        continue

    # 커버리지 계산
    total_minutes = (df["timestamp"].iloc[-1] - df["timestamp"].iloc[0]) / 60000
    gaps = df["timestamp"].diff()
    missing = gaps[gaps > 60000].sum() / 60000
    coverage = ((total_minutes - missing) / total_minutes) * 100

    print(f"\n  {symbol}")
    print(f"    데이터 수: {info['records']:,}개")
    print(f"    기간: {info['first']} ~ {info['last']}")
    print(f"    파일 크기: {info['size_mb']}MB")
    print(f"    커버리지: {coverage:.2f}%")

    total_records += info["records"]
    total_size += info["size_mb"]

print(f"\n{'=' * 60}")
print(f"  합계: {stats['total_symbols']}개 코인, {total_records:,}개 캔들, {total_size:.1f}MB")
print("=" * 60)