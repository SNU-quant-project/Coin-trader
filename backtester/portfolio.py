"""
포트폴리오 상태 관리

수수료 계산:
- 진입: cash * fee_rate
- 청산 (롱): coin_qty * exit_price * fee_rate
- 청산 (숏): coin_qty * exit_price * fee_rate
"""


class Portfolio:

    def __init__(self, initial_capital: float, fee_rate: float = 0.001):
        self.initial_capital = initial_capital
        self.fee_rate = fee_rate

        self.cash = initial_capital
        self.position = 0            # 0: 없음, 1: 롱, -1: 숏
        self.coin_qty = 0.0
        self.entry_price = 0.0
        self.capital_at_entry = 0.0  # 진입 후 실제 투입 자본 (수수료 차감 후)
        self._entry_cash = 0.0       # 진입 직전 현금 (P&L 계산용)
        self._entry_dt = None        # 진입 시각

        self.trades = []

    def execute(self, signal: int, price: float, dt=None):
        """시그널에 따라 포지션 변경. 같은 방향이면 무시."""
        if signal == self.position:
            return

        if self.position != 0:
            self._close(price, dt)

        if signal != 0:
            self._open(signal, price, dt)

    def _open(self, signal: int, price: float, dt=None):
        entry_fee = self.cash * self.fee_rate
        self.capital_at_entry = self.cash - entry_fee
        self.coin_qty = self.capital_at_entry / price
        self._entry_cash = self.cash
        self._entry_dt = dt
        self.position = signal
        self.entry_price = price
        self.cash = 0.0

    def _close(self, price: float, dt=None):
        if self.position == 1:
            gross = self.coin_qty * price
            close_fee = gross * self.fee_rate
            final_equity = gross - close_fee
        else:  # 숏
            pnl = self.coin_qty * (self.entry_price - price)
            close_fee = self.coin_qty * price * self.fee_rate
            final_equity = self.capital_at_entry + pnl - close_fee

        # 파산 시 최대 손실은 진입 자본 전체 (-100%)
        final_equity = max(final_equity, 0.0)

        self.trades.append({
            'direction': self.position,
            'entry_dt': self._entry_dt,
            'exit_dt': dt,
            'entry_price': self.entry_price,
            'exit_price': price,
            'coin_qty': self.coin_qty,
            'pnl': final_equity - self._entry_cash,
            'pnl_pct': (final_equity / self._entry_cash - 1) * 100 if self._entry_cash > 0 else -100.0,
        })

        self.cash = final_equity
        self.position = 0
        self.coin_qty = 0.0
        self.entry_price = 0.0
        self.capital_at_entry = 0.0
        self._entry_dt = None

    def get_equity(self, current_price: float) -> float:
        """현재 가격 기준 총 자산 (mark-to-market, 청산 수수료 미포함)"""
        if self.position == 0:
            return self.cash
        elif self.position == 1:
            return self.coin_qty * current_price
        else:  # 숏
            pnl = self.coin_qty * (self.entry_price - current_price)
            return max(self.capital_at_entry + pnl, 0.0)
