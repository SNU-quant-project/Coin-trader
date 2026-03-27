"""
데이터 저장 모듈
"""

import os
import pandas as pd
from common.logger import get_logger
from common.utils import timestamp_to_str

logger = get_logger("storage")


class Storage:

    def __init__(self, base_path="data/historical"):
        self.base_path = base_path
        os.makedirs(base_path, exist_ok=True)

    def save_candles(self, symbol, candles, timeframe="1m"):
        if not candles:
            return

        df = pd.DataFrame(candles, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms")

        filename = self._symbol_to_filename(symbol, timeframe)
        filepath = os.path.join(self.base_path, filename)

        if os.path.exists(filepath):
            existing = pd.read_csv(filepath)
            df_combined = pd.concat([existing, df], ignore_index=True)
            df_combined = df_combined.drop_duplicates(subset=["timestamp"], keep="last").sort_values("timestamp")
            df_combined.to_csv(filepath, index=False)
            logger.info(f"{symbol} 데이터 추가 저장: {len(candles)}개 (총 {len(df_combined)}개)")
        else:
            df.to_csv(filepath, index=False)
            logger.info(f"{symbol} 데이터 새로 저장: {len(candles)}개")

    def load_candles(self, symbol, timeframe="1m", start=None, end=None):
        filename = self._symbol_to_filename(symbol, timeframe)
        filepath = os.path.join(self.base_path, filename)

        if not os.path.exists(filepath):
            logger.warning(f"데이터 없음: {filepath}")
            return None

        df = pd.read_csv(filepath)
        df["datetime"] = pd.to_datetime(df["datetime"])

        if start:
            df = df[df["datetime"] >= start]
        if end:
            df = df[df["datetime"] <= end]

        logger.info(f"{symbol} 데이터 로드: {len(df)}개")
        return df

    def list_symbols(self):
        symbols = []
        for f in os.listdir(self.base_path):
            if f.endswith(".csv"):
                name = f.replace(".csv", "").rsplit("_", 1)[0]
                symbol = name.replace("_", "/")
                symbols.append(symbol)
        return list(set(symbols))

    def count_candles(self, symbol, timeframe="1m"):
        filename = self._symbol_to_filename(symbol, timeframe)
        filepath = os.path.join(self.base_path, filename)
        if not os.path.exists(filepath):
            return 0
        with open(filepath) as f:
            return sum(1 for _ in f) - 1  # 헤더 제외

    def has_data(self, symbol, timeframe="1m"):
        filename = self._symbol_to_filename(symbol, timeframe)
        return os.path.exists(os.path.join(self.base_path, filename))

    def stats(self):
        result = {"total_symbols": 0, "total_records": 0, "symbols": {}}
        for f in os.listdir(self.base_path):
            if not f.endswith(".csv"):
                continue
            filepath = os.path.join(self.base_path, f)
            df = pd.read_csv(filepath)
            name = f.replace(".csv", "").rsplit("_", 1)[0].replace("_", "/")
            result["symbols"][name] = {
                "records": len(df),
                "first": df["datetime"].iloc[0] if len(df) > 0 else None,
                "last": df["datetime"].iloc[-1] if len(df) > 0 else None,
                "size_mb": round(os.path.getsize(filepath) / 1024 / 1024, 2),
            }
            result["total_records"] += len(df)
        result["total_symbols"] = len(result["symbols"])
        return result

    def _symbol_to_filename(self, symbol, timeframe="1m"):
        return f"{symbol.replace('/', '_')}_{timeframe}.csv"
