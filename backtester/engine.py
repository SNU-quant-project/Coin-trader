"""
백테스트 엔진

look-ahead bias 제거 핵심:
- signals[i]: 바 i 종가 기준으로 계산된 시그널
- 체결 시점: 바 i+1의 시가 (open[i+1])
"""

import numpy as np
import pandas as pd
from backtester.portfolio import Portfolio


class Engine:

    def __init__(self, strategy, initial_capital: float = 10_000.0, fee_rate: float = 0.001, cooldown: int = 0):
        """
        cooldown: 거래 체결 후 다음 거래까지 최소 대기 바 수
                  (1분봉 기준: 60 = 1시간, 1440 = 1일)
        """
        self.strategy = strategy
        self.initial_capital = initial_capital
        self.fee_rate = fee_rate
        self.cooldown = cooldown

    def run(self, df: pd.DataFrame) -> dict:
        df = df.reset_index(drop=True)

        # 전체 시그널 한번에 계산 (벡터화)
        signals = self.strategy.generate_signals(df)

        # numpy 배열로 추출 (루프 내 pandas 접근 제거)
        open_arr  = df['open'].to_numpy(dtype=np.float64)
        close_arr = df['close'].to_numpy(dtype=np.float64)
        dt_arr    = df['datetime'].values

        portfolio = Portfolio(self.initial_capital, self.fee_rate)
        n = len(df)
        equity_arr = np.empty(n, dtype=np.float64)
        pending = None
        cooldown_remaining = 0

        print(f"시뮬레이션 시작: {n:,}개 바 (쿨다운: {self.cooldown}바)")
        for i in range(n):
            if i % 500_000 == 0 and i > 0:
                print(f"  진행 중... {i:,} / {n:,} ({i/n*100:.0f}%)")

            # 1. 이전 바 시그널 → 현재 바 시가에 체결
            if pending is not None:
                # pending = (시그널값, 시그널 생성 시점의 국면 정보)
                sig_to_execute, regime_info = pending
                portfolio.execute(sig_to_execute, open_arr[i], dt_arr[i], regime_info=regime_info)
                pending = None
                cooldown_remaining = self.cooldown

            # 2. 현재 바 종가 기준 자산 평가 기록
            current_equity = portfolio.get_equity(close_arr[i])

            # 3. 강제 청산: 자산이 0 이하면 현재 종가에 즉시 청산 (파산)
            if current_equity <= 0 and portfolio.position != 0:
                portfolio.execute(0, close_arr[i], dt_arr[i])
                current_equity = max(portfolio.cash, 0.0)
                pending = None
                cooldown_remaining = 0

            equity_arr[i] = current_equity

            # 파산 이후에는 루프를 계속하되 더 이상 거래하지 않음
            if equity_arr[i] <= 0:
                equity_arr[i:] = 0.0
                break

            # 4. 쿨다운 중이면 시그널 무시
            if cooldown_remaining > 0:
                cooldown_remaining -= 1
                continue

            # 5. 시그널 변경 시 다음 바에 체결 예약
            # 국면 정보를 시그널 생성 시점(봉 i)에 함께 저장해둠
            # → 체결 시점(봉 i+1)에서 읽으면 1봉 늦은 국면이 찍히는 버그 방지
            sig = int(signals[i])
            if sig != portfolio.position:
                regime_info = None
                if hasattr(self.strategy, '_combined_action_arr'):
                    regime_info = {
                        'action':     int(self.strategy._combined_action_arr[i]),
                        'adx_regime': int(self.strategy._adx_regime_arr[i]),
                        'r2_regime':  int(self.strategy._r2_regime_arr[i]),
                    }
                pending = (sig, regime_info)

        # 마지막 포지션 종가에 청산
        if portfolio.position != 0:
            portfolio.execute(0, close_arr[n - 1], dt_arr[n - 1])

        return {
            'equity_curve': equity_arr,
            'trades': portfolio.trades,
            'signals': signals,
            'initial_capital': self.initial_capital,
            'start': df['datetime'].iloc[0],
            'end': df['datetime'].iloc[-1],
        }
