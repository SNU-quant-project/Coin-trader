from data.downloader import Downloader

dl = Downloader()
dl.download("BTC/USDT", "1m", start="2025-03-01", end="2025-03-25")