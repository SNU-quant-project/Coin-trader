"""
설정 파일 로딩 모듈
"""

import os
import yaml
from dotenv import load_dotenv


class Config:

    def __init__(self, config_path="config/settings.yaml"):
        load_dotenv()
        self._raw = self._load_yaml(config_path)

        exchange = self._raw.get("exchange", {})
        self.exchange_name = exchange.get("name", "binance")
        self.exchange_type = exchange.get("type", "spot")
        self.testnet = exchange.get("testnet", False)

        prefix = self.exchange_name.upper()
        self.api_key = os.getenv(f"{prefix}_API_KEY", "")
        self.secret_key = os.getenv(f"{prefix}_SECRET_KEY", "")

        self.symbols = self._raw.get("symbols", ["BTC/USDT"])

        collector = self._raw.get("collector", {})
        self.collect_interval = collector.get("interval", 60)
        self.batch_size = collector.get("batch_size", 100)
        self.retry_count = collector.get("retry_count", 3)
        self.retry_delay = collector.get("retry_delay", 5)

        storage = self._raw.get("storage", {})
        self.data_path = storage.get("base_path", "data/historical")
        self.storage_format = storage.get("format", "csv")
        self.split_by = storage.get("split_by", "month")

        downloader = self._raw.get("downloader", {})
        self.default_timeframe = downloader.get("default_timeframe", "1m")
        self.max_candles = downloader.get("max_candles_per_request", 1000)
        self.download_sleep = downloader.get("sleep_between_requests", 0.5)

        logging_conf = self._raw.get("logging", {})
        self.log_level = logging_conf.get("level", "INFO")
        self.log_console = logging_conf.get("console", True)
        self.log_file = logging_conf.get("file", True)
        self.log_dir = logging_conf.get("log_dir", "logs")

    def _load_yaml(self, path):
        if not os.path.exists(path):
            print(f"설정 파일을 찾을 수 없습니다: {path}")
            return {}
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return data if data else {}

    def has_api_keys(self):
        return bool(self.api_key) and bool(self.secret_key)

    def validate(self):
        errors = []
        valid_exchanges = ["binance", "upbit", "bithumb"]
        if self.exchange_name not in valid_exchanges:
            errors.append(f"알 수 없는 거래소: {self.exchange_name}")
        if not self.symbols:
            errors.append("거래 대상 코인이 없습니다")
        for symbol in self.symbols:
            if "/" not in symbol:
                errors.append(f"잘못된 심볼: {symbol}")
        return errors

    def summary(self):
        print("=" * 50)
        print("  현재 설정")
        print("=" * 50)
        print(f"  거래소: {self.exchange_name}")
        print(f"  API 키: {'설정됨' if self.has_api_keys() else '없음'}")
        print(f"  대상 코인: {", ".join(self.symbols)}")
        print(f"  데이터 경로: {self.data_path}")
        print("=" * 50)
