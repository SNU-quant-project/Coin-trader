from data.downloader import Downloader

dl = Downloader()

print("다운로드 시작!")
# 1번째: BTC (약 30분)
dl.download("BTC/USDT", "1m", start="2019-01-01")

# 2번째: BTC 끝나면 아래 주석(#) 지우고 다시 실행
dl.download("ETH/USDT", "1m", start="2019-01-01")

# 3번째
dl.download("XRP/USDT", "1m", start="2019-01-01")

# 4번째 (SOL은 2020년 8월 바이낸스 상장)
dl.download("SOL/USDT", "1m", start="2020-08-01")

print("다운로드 완료!")