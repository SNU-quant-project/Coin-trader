"""
과거 데이터 다운로드 모듈
"""

import time
from datetime import datetime
from common.exchange import Exchange
from common.utils import str_to_timestamp, timestamp_to_str, now_timestamp, now_str
from common.logger import get_logger
from data.storage import Storage

logger = get_logger("downloader")


class Downloader:

    def __init__(self):
        self.exchange = Exchange("binance")
        self.storage = Storage()
        self.max_candles = 1000
        self.sleep_time = 0.2

    def download(self, symbol, timeframe="1m", start="2019-01-01", end=None):
        logger.info(f"{symbol} {timeframe} 다운로드 시작: {start} ~")

        since = str_to_timestamp(start)
        end_ts = str_to_timestamp(end) if end else now_timestamp()

        all_candles = []
        total_expected = self._estimate_candles(since, end_ts, timeframe)
        request_count = 0

        while since < end_ts:
            try:
                candles = self.exchange.get_ohlcv(
                    symbol, timeframe, since=since, limit=self.max_candles
                )

                if not candles:
                    break

                all_candles.extend(candles)
                since = candles[-1][0] + 1
                request_count += 1

                from_str = timestamp_to_str(candles[0][0])
                to_str = timestamp_to_str(candles[-1][0])
                print(f"    {from_str} ~ {to_str} ({len(candles):,}개) 다운로드 완료")

                if len(all_candles) >= 10000:
                    self.storage.save_candles(symbol, all_candles, timeframe)
                    all_candles = []
                    saved_total = self.storage.count_candles(symbol, timeframe)
                    self._print_progress(saved_total, total_expected, symbol)

                time.sleep(self.sleep_time)

            except Exception as e:
                logger.error(f"다운로드 에러: {e}, 5초 후 재시도...")
                time.sleep(5)

        if all_candles:
            self.storage.save_candles(symbol, all_candles, timeframe)
            saved_total = self.storage.count_candles(symbol, timeframe)
            self._print_progress(saved_total, total_expected, symbol)

        logger.info(f"{symbol} 다운로드 완료! (API 호출 {request_count}회)")

    def download_batch(self, symbols, timeframe="1m", start="2019-01-01", end=None):
        logger.info(f"{len(symbols)}개 코인 배치 다운로드 시작")
        for i, symbol in enumerate(symbols):
            logger.info(f"[{i+1}/{len(symbols)}] {symbol} 시작")
            try:
                self.download(symbol, timeframe, start, end)
            except Exception as e:
                logger.error(f"{symbol} 다운로드 실패: {e}")
            logger.info(f"[{i+1}/{len(symbols)}] {symbol} 완료")

    def verify(self, symbol, timeframe="1m"):
        df = self.storage.load_candles(symbol, timeframe)
        if df is None:
            return {"error": "데이터 없음"}

        total = len(df)
        duplicates = df.duplicated(subset=["timestamp"]).sum()
        first = df["datetime"].iloc[0]
        last = df["datetime"].iloc[-1]

        return {
            "symbol": symbol,
            "timeframe": timeframe,
            "total_candles": total,
            "duplicates": duplicates,
            "first": str(first),
            "last": str(last),
        }

    def _estimate_candles(self, start_ts, end_ts, timeframe):
        intervals = {
            "1m": 60000, "5m": 300000, "15m": 900000,
            "1h": 3600000, "4h": 14400000, "1d": 86400000,
        }
        interval_ms = intervals.get(timeframe, 60000)
        return int((end_ts - start_ts) / interval_ms)

    def _print_progress(self, current, total, symbol):
        if total <= 0:
            return
        pct = min(current / total * 100, 100)
        bar_len = 30
        filled = int(bar_len * pct / 100)
        bar = "█" * filled + "░" * (bar_len - filled)
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"  [{symbol}] {bar} {pct:.1f}% ({current:,}/{total:,}) [{now}]")
