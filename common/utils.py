"""
공통 유틸리티 함수
"""

from datetime import datetime


def timestamp_to_str(timestamp_ms):
    dt = datetime.fromtimestamp(timestamp_ms / 1000)
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def str_to_timestamp(date_str):
    if " " in date_str:
        dt = datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S")
    else:
        dt = datetime.strptime(date_str, "%Y-%m-%d")
    return int(dt.timestamp() * 1000)


def now_str():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def now_timestamp():
    return int(datetime.now().timestamp() * 1000)


def format_price(price, currency="$"):
    if price >= 1:
        return f"{currency}{price:,.2f}"
    else:
        return f"{currency}{price:,.4f}"


def format_volume(volume):
    if volume >= 1_000_000:
        return f"{volume / 1_000_000:.2f}M"
    elif volume >= 1_000:
        return f"{volume / 1_000:.2f}K"
    else:
        return f"{volume:.2f}"


def calc_change(old_price, new_price):
    if old_price == 0:
        return 0.0
    return round(((new_price - old_price) / old_price) * 100, 2)


def symbol_to_exchange(symbol):
    return symbol.replace("/", "")


def symbol_from_exchange(symbol):
    if "USDT" in symbol:
        base = symbol.replace("USDT", "")
        return f"{base}/USDT"
    return symbol


def get_base_currency(symbol):
    return symbol.split("/")[0]


def candle_to_dict(candle):
    return {
        "timestamp": timestamp_to_str(candle[0]),
        "timestamp_ms": candle[0],
        "open": candle[1],
        "high": candle[2],
        "low": candle[3],
        "close": candle[4],
        "volume": candle[5],
    }


def candles_to_dicts(candles):
    return [candle_to_dict(c) for c in candles]
