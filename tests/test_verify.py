from data.downloader import Downloader
from data.storage import Storage

dl = Downloader()
storage = Storage()

# 다운받은 데이터 검증
result = dl.verify("BTC/USDT", "1m")
print(result)

# 저장소 전체 현황
stats = storage.stats()
print(stats)