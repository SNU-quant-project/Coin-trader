"""
백테스트 결과 리포트
"""

import numpy as np
import pandas as pd

ANNUAL_BARS = {
    '1m':  525_600,
    '5m':  105_120,
    '15m':  35_040,
    '1h':    8_760,
    '4h':    2_190,
    '1d':      365,
}


class Report:

    def __init__(self, result: dict, timeframe: str = '1m'):
        self.equity_curve = result['equity_curve']
        self.trades = result['trades']
        self.initial_capital = result['initial_capital']
        self.start = pd.Timestamp(result['start'])
        self.end = pd.Timestamp(result['end'])
        self.timeframe = timeframe

    def summary(self) -> dict:
        stats = self._compute()
        self._print(stats)
        return stats

    def _compute(self) -> dict:
        equity = self.equity_curve
        initial = self.initial_capital
        final = float(equity[-1])

        # 수익률
        total_return = (final / initial - 1) * 100

        # 연환산 수익률
        years = (self.end - self.start).days / 365.25
        annual_return = ((final / initial) ** (1 / years) - 1) * 100 if years > 0 else 0.0

        # MDD
        running_max = np.maximum.accumulate(equity)
        drawdown = (equity - running_max) / running_max
        mdd = float(drawdown.min()) * 100

        # 샤프 지수 (연환산, 무위험수익률 = 0)
        returns = np.diff(equity) / equity[:-1]
        annual_bars = ANNUAL_BARS.get(self.timeframe, 525_600)
        sharpe = float(returns.mean() / returns.std() * np.sqrt(annual_bars)) if returns.std() > 0 else 0.0

        # 거래 통계
        total_trades = len(self.trades)
        if total_trades > 0:
            wins = sum(1 for t in self.trades if t['pnl'] > 0)
            win_rate = wins / total_trades * 100
            avg_pnl_pct = sum(t['pnl_pct'] for t in self.trades) / total_trades
        else:
            win_rate = 0.0
            avg_pnl_pct = 0.0

        return {
            'start': self.start.strftime('%Y-%m-%d'),
            'end': self.end.strftime('%Y-%m-%d'),
            'days': (self.end - self.start).days,
            'initial_capital': initial,
            'final_equity': final,
            'total_return': total_return,
            'annual_return': annual_return,
            'mdd': mdd,
            'sharpe': sharpe,
            'total_trades': total_trades,
            'win_rate': win_rate,
            'avg_pnl_pct': avg_pnl_pct,
        }

    def _print(self, s: dict):
        print("=" * 60)
        print("  백테스트 결과")
        print("=" * 60)
        print(f"  기간        : {s['start']} ~ {s['end']} ({s['days']:,}일)")
        print(f"  초기 자산   : {s['initial_capital']:>18,.2f} USDT")
        print(f"  최종 자산   : {s['final_equity']:>18,.2f} USDT")
        print("-" * 60)
        print(f"  총 수익률   : {s['total_return']:>+17.2f} %")
        print(f"  연환산 수익 : {s['annual_return']:>+17.2f} %")
        print(f"  MDD         : {s['mdd']:>+17.2f} %")
        print(f"  샤프 지수   : {s['sharpe']:>18.2f}")
        print("-" * 60)
        print(f"  총 거래 수  : {s['total_trades']:>17,} 회")
        print(f"  승률        : {s['win_rate']:>17.2f} %")
        print(f"  평균 수익   : {s['avg_pnl_pct']:>+17.2f} %")
        print("=" * 60)

        # 거래 내역 상세 출력
        if self.trades:
            # 국면 값 → 한글 레이블
            _regime_lbl = {1: '상승', 0: '횡보', -1: '하락'}
            _action_lbl = {1: 'LONG', -1: 'SHORT', 0: 'CASH', 2: 'COUNTER', 3: 'HOLD'}

            print("\n  [거래 내역]")
            print(f"  {'#':>3}  {'방향':4}  {'진입 시각':^13}  {'청산 시각':^13}  {'진입가':^8}  {'청산가':^8}  {'손익':>4}  {'국면대응전략':^1}  {'ADX':^6}  {'R²':>6}")
            #print(f"  {'#':>3}  {'방향':4}  {'진입 시각':<18}  {'청산 시각':<11}  {'진입가':>4}  {'청산가':>5}  {'손익':>8}  {'국면대응전략':<10}  {'ADX':^6}  {'R²':^6}")

            print("  " + "-" * 115)
            for i, t in enumerate(self.trades, 1):
                direction = '롱 ' if t['direction'] == 1 else '숏 '
                entry_dt  = str(pd.Timestamp(t['entry_dt']))[:19] if t['entry_dt'] is not None else '-'
                exit_dt   = str(pd.Timestamp(t['exit_dt']))[:19]  if t['exit_dt']  is not None else '-'

                ri = t.get('regime_info') or {}
                action_str = _action_lbl.get(ri.get('action'), '-')
                adx_str    = _regime_lbl.get(ri.get('adx_regime'), '-')
                r2_str     = _regime_lbl.get(ri.get('r2_regime'), '-')

                print(f"  {i:>3}  {direction}  {entry_dt:<19}  {exit_dt:<19}  {t['entry_price']:>10,.2f}  {t['exit_price']:>10,.2f}  {t['pnl_pct']:>+7.2f}%  {action_str:<10}  {adx_str:^6}  {r2_str:^6}")
