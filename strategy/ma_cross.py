"""
이동평균 교차 전략 (MA Crossover)

단기 MA가 장기 MA를 위로 교차 → 롱
단기 MA가 장기 MA를 아래로 교차 → 숏
"""

import numpy as np
import pandas as pd
from strategy.base import BaseStrategy


class MACrossStrategy(BaseStrategy):

    def __init__(self, fast_period: int = 20, slow_period: int = 60, min_diff_pct: float = 0.0):
        """
        fast_period  : 단기 MA 기간
        slow_period  : 장기 MA 기간
        min_diff_pct : 교차 시 MA 간격이 가격 대비 최소 몇 % 이상이어야 진입
                       (노이즈 교차 필터링. 예: 0.1 = 0.1%)
        """
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.min_diff_pct = min_diff_pct

    def generate_signals(self, df: pd.DataFrame) -> np.ndarray:
        close = df['close']

        # rolling().mean()은 인덱스 i까지의 데이터만 사용 → look-ahead bias 없음
        fast_ma = close.rolling(self.fast_period).mean()
        slow_ma = close.rolling(self.slow_period).mean()

        diff = fast_ma - slow_ma
        prev_diff = diff.shift(1)

        # MA 간격이 가격 대비 min_diff_pct% 이상인 경우만 유효한 교차로 인정
        diff_pct = (diff.abs() / close) * 100
        strong_enough = diff_pct >= self.min_diff_pct

        n = len(df)
        signals = np.zeros(n, dtype=np.int8)

        # 골든크로스: 이전에는 단기 < 장기, 현재는 단기 > 장기 → 롱
        golden = (prev_diff < 0) & (diff > 0) & strong_enough
        # 데드크로스: 이전에는 단기 > 장기, 현재는 단기 < 장기 → 숏
        dead = (prev_diff > 0) & (diff < 0) & strong_enough

        signals[golden.values] = 1
        signals[dead.values] = -1

        # MA 계산 불가 구간(워밍업)은 NaN으로 마킹 후 제외
        signals[:self.slow_period] = 0

        # 교차 사이 구간은 이전 시그널 유지 (pandas ffill로 벡터화)
        s = pd.Series(signals, dtype=float)
        s[s == 0] = np.nan
        s.iloc[:self.slow_period] = np.nan
        s = s.ffill().fillna(0)

        return s.to_numpy(dtype=np.int8)
