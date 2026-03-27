from common.exchange import Exchange

ex = Exchange("binance")
price = ex.get_price("BTC/USDT")
print(f"BTC price: {price}")