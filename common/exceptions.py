"""
커스텀 에러 정의
"""

class CryptoTradingError(Exception):
    pass

class ExchangeError(CryptoTradingError):
    pass

class ExchangeConnectionError(ExchangeError):
    pass

class ExchangeAuthError(ExchangeError):
    pass

class ExchangeRateLimitError(ExchangeError):
    pass

class OrderError(ExchangeError):
    pass

class DataError(CryptoTradingError):
    pass

class DataNotFoundError(DataError):
    pass

class DataValidationError(DataError):
    pass

class StorageError(DataError):
    pass

class ConfigError(CryptoTradingError):
    pass

class StrategyError(CryptoTradingError):
    pass

class BotError(CryptoTradingError):
    pass
