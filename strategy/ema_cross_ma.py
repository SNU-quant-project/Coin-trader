"""
EMA 크로스오버 전략 + MA 레짐 필터 (롱 온리)

핵심 아이디어:
  - 단기 EMA가 장기 EMA를 위로 교차 → 상승 모멘텀 시작 → 롱 진입
  - 단기 EMA가 장기 EMA를 아래로 교차 → 모멘텀 종료 → 청산
  - Donchian(돌파)보다 신호가 빠르고 자주 발생

Donchian과의 차이:
  - Donchian: "N일 최고가 돌파" → 큰 움직임에만 반응, 거래 적음
  - EMA 크로스: "평균선 교차" → 작은 추세 전환에도 반응, 거래 많음
  - 둘 다 추세추종이지만 민감도가 다름

파라미터:
  - fast_ema: 단기 EMA 기간 (기본 10봉 = 약 1.7일)
  - slow_ema: 장기 EMA 기간 (기본 30봉 = 5일)
  - ma_filter: 레짐 필터 기간 (기본 600봉 = 100일 MA)

왜 100일 MA인가 (200일이 아닌 이유):
  - EMA 크로스는 더 민감한 전략이므로, 필터도 약간 빠르게 반응하는 게 좋음
  - 200일 MA는 너무 느려서 상승 초기의 좋은 크로스 신호를 놓칠 수 있음
  - 100일 MA는 중기 추세를 충분히 반영하면서도 반응성이 적절함

look-ahead bias 방지:
  - EMA는 과거 데이터의 가중평균이므로 미래 정보 사용 없음
  - 시그널 i는 바 i 종가 기준 → 바 i+1 시가에 체결 (엔진 자동 처리)
"""

import numpy as np
import pandas as pd
from strategy.base import BaseStrategy


class EMACrossMAFilter(BaseStrategy):

    def __init__(
        self,
        fast_ema: int = 10,     # 단기 EMA 기간 (4시간봉 기준 ~1.7일)
        slow_ema: int = 30,     # 장기 EMA 기간 (4시간봉 기준 5일)
        ma_filter: int = 600,   # 레짐 필터: 100일 MA (4시간봉 × 6 × 100)
    ):
        self.fast_ema = fast_ema
        self.slow_ema = slow_ema
        self.ma_filter = ma_filter

    def generate_signals(self, df: pd.DataFrame) -> np.ndarray:
        """
        EMA 크로스오버 + MA 필터 시그널 생성

        진입 조건 (2가지 동시):
          1. 단기 EMA > 장기 EMA (골든 크로스)
          2. 종가 > 100일 MA (상승 추세 확인)

        청산 조건:
          - 단기 EMA < 장기 EMA (데드 크로스)

        반환: np.ndarray (int8), 1=롱, 0=현금
        """
        close = df['close']
        n = len(df)

        # ── EMA 계산 ────────────────────────────────────────
        # EMA는 지수이동평균: 최근 데이터에 더 큰 가중치
        # → SMA보다 가격 변화에 빠르게 반응
        fast = close.ewm(span=self.fast_ema, adjust=False).mean()
        slow = close.ewm(span=self.slow_ema, adjust=False).mean()

        # ── 레짐 필터 (100일 MA) ─────────────────────────────
        ma = close.rolling(self.ma_filter).mean()

        # numpy 변환
        fast_arr = fast.to_numpy(dtype=np.float64)
        slow_arr = slow.to_numpy(dtype=np.float64)
        ma_arr = ma.to_numpy(dtype=np.float64)
        close_arr = close.to_numpy(dtype=np.float64)

        # ── 시그널 생성 ─────────────────────────────────────
        signals = np.zeros(n, dtype=np.int8)
        position = 0

        for i in range(n):
            if np.isnan(ma_arr[i]):
                signals[i] = 0
                continue

            if position == 0:
                # 진입: 골든 크로스 + MA 필터
                if fast_arr[i] > slow_arr[i] and close_arr[i] > ma_arr[i]:
                    position = 1
            elif position == 1:
                # 청산: 데드 크로스 (MA 필터 무관)
                if fast_arr[i] < slow_arr[i]:
                    position = 0

            signals[i] = position

        return signals
