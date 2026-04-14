import pandas as pd

df = pd.read_csv("data/historical/BTC_USDT_1m.csv", usecols=["datetime"])
df["datetime"] = pd.to_datetime(df["datetime"])
idx_2024 = df[df["datetime"] >= "2024-01-01"].index[0]
idx_2025 = df[df["datetime"] >= "2025-01-01"].index[0]
total = len(df)

print(f"Total rows: {total:,}")
print(f"2024 starts at index: {idx_2024:,}")
print(f"2025 starts at index: {idx_2025:,}")
print(f"Pre-2024: {idx_2024:,} rows")
print(f"Year 2024: {idx_2025 - idx_2024:,} rows")
print(f"First date: {df.iloc[0]['datetime']}")
print(f"First 2024: {df.iloc[idx_2024]['datetime']}")
print(f"First 2025: {df.iloc[idx_2025]['datetime']}")
print(f"Last date:  {df.iloc[-1]['datetime']}")

# 5분봉 변환 시 예상 크기
pre_2024_5m = idx_2024 // 5
y2024_5m = (idx_2025 - idx_2024) // 5
print(f"\n5-min candles:")
print(f"  Pre-2024: ~{pre_2024_5m:,}")
print(f"  Year 2024: ~{y2024_5m:,}")
