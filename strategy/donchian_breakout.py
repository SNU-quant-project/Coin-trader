"""
Donchian Channel Breakout 전략 (롱 온리)

핵심 아이디어:
  - 최근 N봉 최고가를 돌파하면 강한 상승 추세 시작 → 롱 진입
  - 최근 M봉 최저가를 이탈하면 추세 종료 → 포지션 청산
  - BTC의 장기 우상향 편향을 활용하여 숏은 하지 않음

파라미터:
  - entry_period: 진입 채널 기간 (기본 120봉 = 4시간봉 기준 20일)
  - exit_period:  청산 채널 기간 (기본 60봉 = 4시간봉 기준 10일)

진입/청산 채널의 기간이 다른 이유:
  - 진입 채널(긴 기간): 느리게 반응 → 가짜 돌파(whipsaw) 필터링
  - 청산 채널(짧은 기간): 빠르게 반응 → 수익 보호, 손실 조기 차단
  - "천천히 들어가고, 빠르게 나온다" = 추세추종의 핵심 원칙

look-ahead bias 방지:
  - 채널 계산 시 shift(1)을 사용하여 현재 봉 데이터 제외
  - 시그널 i는 바 i 종가 기준 → 바 i+1 시가에 체결 (엔진이 자동 처리)
"""

import numpy as np
import pandas as pd
from strategy.base import BaseStrategy


class DonchianBreakoutStrategy(BaseStrategy):

    def __init__(self, entry_period: int = 120, exit_period: int = 60):
        """
        Args:
            entry_period: 진입 채널 기간 (rolling window 크기)
                          4시간봉 기준: 120 = 20일, 90 = 15일
            exit_period:  청산 채널 기간
                          4시간봉 기준: 60 = 10일, 30 = 5일
        """
        self.entry_period = entry_period
        self.exit_period = exit_period

    def generate_signals(self, df: pd.DataFrame) -> np.ndarray:
        """
        Donchian Channel 기반 롱 온리 시그널 생성

        반환: np.ndarray (int8, shape=(N,))
            1 = 롱 보유
            0 = 현금 보유 (포지션 없음)
           -1 = 사용하지 않음 (숏 없음)
        """
        high = df['high']
        low = df['low']
        close = df['close']
        n = len(df)

        # ── 채널 계산 ────────────────────────────────────────────
        # shift(1): 직전 봉까지의 데이터만 사용 (현재 봉 제외)
        # → 현재 봉의 고가가 채널에 포함되면 "종가 > 채널"이 사소하게 참이 됨
        upper_channel = high.rolling(self.entry_period).max().shift(1)
        lower_channel = low.rolling(self.exit_period).min().shift(1)

        # numpy로 변환 (루프 성능 향상)
        close_arr = close.to_numpy(dtype=np.float64)
        upper_arr = upper_channel.to_numpy(dtype=np.float64)
        lower_arr = lower_channel.to_numpy(dtype=np.float64)

        # ── 시그널 생성 (상태 머신) ──────────────────────────────
        # 상태: 0 = 현금, 1 = 롱
        # 전환 조건이 상태 의존적이므로 순차 루프 사용
        # (4시간봉 기준 연간 ~2,190봉이라 성능 문제 없음)
        signals = np.zeros(n, dtype=np.int8)
        position = 0  # 현재 상태

        for i in range(n):
            # 워밍업 구간: 채널이 아직 계산되지 않은 구간
            if np.isnan(upper_arr[i]) or np.isnan(lower_arr[i]):
                signals[i] = 0
                continue

            if position == 0:
                # 현금 상태에서 종가가 상단 채널을 돌파하면 롱 진입
                if close_arr[i] > upper_arr[i]:
                    position = 1
            elif position == 1:
                # 롱 상태에서 종가가 하단 채널을 이탈하면 청산
                if close_arr[i] < lower_arr[i]:
                    position = 0

            signals[i] = position

        return signals
