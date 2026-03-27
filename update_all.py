"""
증분 업데이트 스크립트
마지막 저장된 시점 이후의 데이터만 다운로드해서 CSV에 이어붙입니다.
"""

from data.downloader import Downloader
from data.storage import Storage
from common.utils import timestamp_to_str, now_str

SYMBOLS = [
    ("BTC/USDT", "1m"),
    ("ETH/USDT", "1m"),
    ("XRP/USDT", "1m"),
    ("SOL/USDT", "1m"),
]

def update():
    dl = Downloader()
    storage = Storage()

    print("=" * 60)
    print(f"  증분 업데이트 시작: {now_str()}")
    print("=" * 60)

    for symbol, timeframe in SYMBOLS:
        print(f"\n  [{symbol}] 확인 중...")

        df = storage.load_candles(symbol, timeframe)

        if df is None:
            print(f"  [{symbol}] 기존 데이터 없음 → 전체 다운로드 필요 (download_all.py 실행)")
            continue

        last_ts = int(df["timestamp"].iloc[-1])
        last_str = timestamp_to_str(last_ts)
        print(f"  [{symbol}] 마지막 데이터: {last_str}")

        # 마지막 캔들 다음 1분부터 다운로드
        start_ts = last_ts + 60000
        start_str = timestamp_to_str(start_ts)

        before_count = len(df)
        dl.download(symbol, timeframe, start=start_str)
        after_count = storage.count_candles(symbol, timeframe)

        added = after_count - before_count
        print(f"  [{symbol}] 업데이트 완료: +{added:,}개 추가 (총 {after_count:,}개)")

    print(f"\n{'=' * 60}")
    print(f"  업데이트 완료: {now_str()}")
    print("=" * 60)


if __name__ == "__main__":
    update()
