"""
거래소 API 연결 모듈 (공개 모드 - API 키 불필요)
"""

import ccxt
from common.logger import get_logger

logger = get_logger("exchange")


class Exchange:

    def __init__(self, exchange_name="binance"):
        self.exchange_name = exchange_name
        exchange_class = getattr(ccxt, exchange_name)
        self.exchange = exchange_class({
            "options": {"defaultType": "spot"},
        })
        logger.info(f"{exchange_name} connected (public mode)")

    def get_price(self, symbol="BTC/USDT"):
        ticker = self.exchange.fetch_ticker(symbol)
        return ticker["last"]

    def get_ohlcv(self, symbol="BTC/USDT", timeframe="1h", since=None, limit=1000):
        ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=limit)
        return ohlcv

    def test_connection(self):
        try:
            price = self.get_price("BTC/USDT")
            logger.info(f"Connection OK! BTC: ${price:,.2f}")
            return True
        except Exception as e:
            logger.error(f"Connection failed: {e}")
            return False