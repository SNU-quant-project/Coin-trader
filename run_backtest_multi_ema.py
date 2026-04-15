"""
멀티 에셋 EMA 크로스오버 백테스트

Donchian(10d/5d)과 비교 포인트:
  - Donchian: "N일 신고가 돌파" → 큰 움직임에만 반응 (연 13회)
  - EMA 크로스: "평균선 교차" → 작은 추세 전환에도 반응 (연 20~30회 예상)

실행: python run_backtest_multi_ema.py
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from strategy.ema_cross_ma import EMACrossMAFilter
from backtester.engine import Engine
from backtester.report import Report


# ══════════════════════════════════════════════════════════════
#  설정
# ══════════════════════════════════════════════════════════════

ASSETS = {
    'BTC': 'data/historical/BTC_USDT_1m.csv',
    'ETH': 'data/historical/ETH_USDT_1m.csv',
    'XRP': 'data/historical/XRP_USDT_1m.csv',
    'SOL': 'data/historical/SOL_USDT_1m.csv',
}

TIMEFRAME = "4h"
TEST_START = "2019-01-01"
TEST_END   = "2026-03-25"

# EMA 크로스 파라미터
FAST_EMA = 10      # 단기 EMA (~1.7일)
SLOW_EMA = 30      # 장기 EMA (5일)
MA_FILTER = 600    # 100일 MA 필터

# 엔진
TOTAL_CAPITAL = 10_000.0
CAPITAL_PER_ASSET = TOTAL_CAPITAL / len(ASSETS)
FEE_RATE = 0.001
COOLDOWN = 2  # 8시간 (EMA 크로스는 더 민감하므로 쿨다운도 짧게)


# ══════════════════════════════════════════════════════════════
#  데이터 로드 + 리샘플링
# ══════════════════════════════════════════════════════════════

def load_and_resample(csv_path, asset_name):
    print(f"  [{asset_name}] 로드 중: {csv_path}")
    df_raw = pd.read_csv(csv_path, parse_dates=['datetime'])
    warmup_start = pd.Timestamp(TEST_START) - pd.Timedelta(days=120)
    df_raw = df_raw[df_raw['datetime'] >= warmup_start].copy()
    df_raw = df_raw.set_index('datetime')
    df_4h = df_raw.resample(TIMEFRAME).agg({
        'open': 'first', 'high': 'max', 'low': 'min',
        'close': 'last', 'volume': 'sum',
    }).dropna().reset_index()
    df_4h = df_4h[df_4h['datetime'] <= TEST_END].copy()
    print(f"  [{asset_name}] {len(df_4h):,}봉")
    return df_4h


# ══════════════════════════════════════════════════════════════
#  실행
# ══════════════════════════════════════════════════════════════

print(f"\n{'='*70}")
print(f"  멀티 에셋 EMA 크로스오버 (EMA{FAST_EMA}/{SLOW_EMA} + MA{MA_FILTER//6})")
print(f"  종목: {', '.join(ASSETS.keys())}")
print(f"  자본 배분: {TOTAL_CAPITAL:,.0f} USDT → {CAPITAL_PER_ASSET:,.0f} USDT/종목")
print(f"  테스트 기간: {TEST_START} ~ {TEST_END}")
print(f"{'='*70}\n")

results = {}
all_trades = []

for asset_name, csv_path in ASSETS.items():
    print(f"\n── {asset_name}/USDT ──────────────────────────────────")
    try:
        df = load_and_resample(csv_path, asset_name)
    except FileNotFoundError:
        print(f"  [{asset_name}] 파일 없음. 건너뜀.")
        continue

    strategy = EMACrossMAFilter(
        fast_ema=FAST_EMA, slow_ema=SLOW_EMA, ma_filter=MA_FILTER,
    )
    engine = Engine(strategy, initial_capital=CAPITAL_PER_ASSET,
                    fee_rate=FEE_RATE, cooldown=COOLDOWN)
    result = engine.run(df)
    report = Report(result, timeframe=TIMEFRAME)
    stats = report.summary()

    results[asset_name] = {
        'result': result, 'stats': stats, 'df': df,
        'equity': result['equity_curve'],
    }
    for t in result['trades']:
        t['asset'] = asset_name
        all_trades.append(t)


# ══════════════════════════════════════════════════════════════
#  포트폴리오 합산
# ══════════════════════════════════════════════════════════════

test_start_ts = pd.Timestamp(TEST_START)
equity_dfs = {}
for name, r in results.items():
    df = r['df'].copy()
    df['equity'] = r['equity']
    df = df[df['datetime'] >= test_start_ts].copy()
    equity_dfs[name] = df.set_index('datetime')['equity']

combined = pd.DataFrame(equity_dfs).ffill()
combined['total'] = combined.sum(axis=1)

total_equity = combined['total'].values
initial = CAPITAL_PER_ASSET * len(results)
final = total_equity[-1]
total_return = (final / initial - 1) * 100

running_max = np.maximum.accumulate(total_equity)
drawdown = (total_equity - running_max) / running_max
mdd = float(drawdown.min()) * 100

returns = np.diff(total_equity) / total_equity[:-1]
sharpe = float(returns.mean() / returns.std() * np.sqrt(2190)) if returns.std() > 0 else 0

total_trades = len(all_trades)
wins = sum(1 for t in all_trades if t['pnl'] > 0)
win_rate = wins / total_trades * 100 if total_trades > 0 else 0
avg_pnl = sum(t['pnl_pct'] for t in all_trades) / total_trades if total_trades > 0 else 0

print(f"\n\n{'='*70}")
print(f"  포트폴리오 합산 결과 (균등 배분)")
print(f"{'='*70}")
print(f"  초기 자산   : {initial:>18,.2f} USDT")
print(f"  최종 자산   : {final:>18,.2f} USDT")
print(f"-" * 70)
print(f"  총 수익률   : {total_return:>+17.2f} %")
print(f"  MDD         : {mdd:>+17.2f} %")
print(f"  샤프 지수   : {sharpe:>18.2f}")
print(f"-" * 70)
print(f"  총 거래 수  : {total_trades:>17,} 회")
print(f"  승률        : {win_rate:>17.2f} %")
print(f"  평균 수익   : {avg_pnl:>+17.2f} %")
print(f"{'='*70}")

print(f"\n\n{'='*70}")
print(f"  종목별 성과 비교")
print(f"{'='*70}")
print(f"  {'종목':<6} {'수익률':>10} {'Sharpe':>10} {'MDD':>10} {'거래수':>8} {'승률':>8}")
print(f"  {'-'*56}")
for name, r in results.items():
    s = r['stats']
    print(f"  {name:<6} {s['total_return']:>+9.2f}% {s['sharpe']:>10.2f} {s['mdd']:>+9.2f}% {s['total_trades']:>7}회 {s['win_rate']:>7.1f}%")
print(f"  {'-'*56}")
print(f"  {'합산':<6} {total_return:>+9.2f}% {sharpe:>10.2f} {mdd:>+9.2f}% {total_trades:>7}회 {win_rate:>7.1f}%")

# 거래 내역
print(f"\n\n  [전체 거래 내역 — 시간순]")
print(f"  {'#':>3}  {'종목':<5} {'진입 시각':<19}  {'청산 시각':<19}  {'진입가':>12}  {'청산가':>12}  {'손익':>8}")
print(f"  " + "-" * 90)
sorted_trades = sorted(all_trades, key=lambda t: t['entry_dt'] if t['entry_dt'] is not None else pd.Timestamp.max)
for i, t in enumerate(sorted_trades, 1):
    entry_dt = str(pd.Timestamp(t['entry_dt']))[:19] if t['entry_dt'] is not None else '-'
    exit_dt = str(pd.Timestamp(t['exit_dt']))[:19] if t['exit_dt'] is not None else '-'
    print(f"  {i:>3}  {t.get('asset','?'):<5} {entry_dt:<19}  {exit_dt:<19}  {t['entry_price']:>12,.2f}  {t['exit_price']:>12,.2f}  {t['pnl_pct']:>+7.2f}%")
print(f"\n  총 {len(sorted_trades)}건 거래 완료")


# ══════════════════════════════════════════════════════════════
#  시각화
# ══════════════════════════════════════════════════════════════

print(f"\n차트 생성 중...")

# Buy & Hold 계산
bnh_dfs = {}
for name, r in results.items():
    df = r['df'].copy()
    df = df[df['datetime'] >= test_start_ts].copy()
    start_price = df['close'].iloc[0]
    df['bnh'] = df['close'] / start_price * CAPITAL_PER_ASSET
    bnh_dfs[name] = df.set_index('datetime')['bnh']
bnh_combined = pd.DataFrame(bnh_dfs).ffill()
bnh_combined['total'] = bnh_combined.sum(axis=1)

datetimes = combined.index
COLORS = {'BTC': '#F7931A', 'ETH': '#627EEA', 'XRP': '#23292F', 'SOL': '#9945FF'}

fig = make_subplots(
    rows=2, cols=1, shared_xaxes=True,
    row_heights=[0.55, 0.45], vertical_spacing=0.08,
    subplot_titles=("포트폴리오 (EMA 크로스 전략 vs Buy&Hold)", "종목별 자산 곡선")
)

fig.add_trace(go.Scatter(
    x=datetimes, y=combined['total'], name='EMA 크로스 포트폴리오',
    line=dict(color='#2ecc71', width=2), fill='tozeroy',
    fillcolor='rgba(46,204,113,0.08)',
    hovertemplate='%{x}<br>전략: %{y:,.0f} USDT<extra></extra>'
), row=1, col=1)

fig.add_trace(go.Scatter(
    x=datetimes, y=bnh_combined['total'], name='Buy & Hold (균등배분)',
    line=dict(color='#888888', width=1.5, dash='dash'),
    hovertemplate='%{x}<br>B&H: %{y:,.0f} USDT<extra></extra>'
), row=1, col=1)

fig.add_hline(y=TOTAL_CAPITAL, line_dash="dot", line_color="rgba(255,255,255,0.3)",
              annotation_text=f"초기 {TOTAL_CAPITAL:,.0f}", annotation_position="bottom right", row=1, col=1)

# 종목별
for name in results.keys():
    if name in combined.columns:
        fig.add_trace(go.Scatter(
            x=datetimes, y=combined[name], name=f'{name} 전략',
            line=dict(color=COLORS.get(name, '#888'), width=1.5),
            hovertemplate=f'{name}<br>%{{x}}<br>%{{y:,.0f}} USDT<extra></extra>'
        ), row=2, col=1)
    if name in bnh_combined.columns:
        fig.add_trace(go.Scatter(
            x=datetimes, y=bnh_combined[name], name=f'{name} B&H',
            line=dict(color=COLORS.get(name, '#888'), width=1, dash='dot'),
            showlegend=False,
            hovertemplate=f'{name} B&H<br>%{{x}}<br>%{{y:,.0f}} USDT<extra></extra>'
        ), row=2, col=1)

fig.add_hline(y=CAPITAL_PER_ASSET, line_dash="dot", line_color="rgba(255,255,255,0.2)", row=2, col=1)

fig.update_layout(
    title=f'멀티에셋 EMA 크로스 — {", ".join(results.keys())} (EMA{FAST_EMA}/{SLOW_EMA}+MA{MA_FILTER//6})',
    template='plotly_dark', height=950, hovermode='x unified',
    legend=dict(orientation='h', y=1.02, x=0, font=dict(size=11)), dragmode='zoom',
)
fig.update_xaxes(showspikes=True, spikemode='across', fixedrange=False, rangeslider=dict(visible=False))
fig.update_yaxes(fixedrange=False)
fig.update_layout(xaxis2=dict(rangeslider=dict(visible=True, thickness=0.04), type='date'))

output_path = "backtest_multi_ema.html"
fig.write_html(output_path, config={'scrollZoom': True, 'displayModeBar': True})
print(f"차트 저장 완료: {output_path}")
