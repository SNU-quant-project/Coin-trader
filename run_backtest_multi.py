"""
멀티 에셋 Donchian Breakout 백테스트

아이디어:
  - 동일한 Donchian 10d/5d + MA200 전략을 BTC, ETH, XRP, SOL 4종목에 동시 적용
  - 자본을 4등분(각 2,500 USDT)하여 독립 운용
  - 각 종목의 돌파 시점이 다르므로 거래 빈도 4배 + 분산 효과

기대 효과:
  - 거래 빈도: 약 13회/년(BTC) × 4 = 50회+/년
  - MDD 감소: 종목 간 상관관계가 1 미만이므로 손실이 분산됨
  - 통계적 신뢰도 향상: 거래 수 증가로 백테스트 결과 안정화

실행: python run_backtest_multi.py
"""

import numpy as np
import pandas as pd
from strategy.donchian_short_ma import DonchianShortMAFilter
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
TEST_START = "2024-01-01"
TEST_END   = "2024-12-31"

# 전략 파라미터 (모든 종목에 동일 적용)
ENTRY_PERIOD = 60       # 10일
EXIT_PERIOD = 30        # 5일
MA_FILTER_PERIOD = 1200 # 200일

# 자본 배분
TOTAL_CAPITAL = 10_000.0
CAPITAL_PER_ASSET = TOTAL_CAPITAL / len(ASSETS)  # 2,500 USDT씩

FEE_RATE = 0.001
COOLDOWN = 3


# ══════════════════════════════════════════════════════════════
#  데이터 로드 + 리샘플링 함수
# ══════════════════════════════════════════════════════════════

def load_and_resample(csv_path, asset_name):
    """1분봉 CSV를 로드하고 4시간봉으로 리샘플링"""
    print(f"  [{asset_name}] 로드 중: {csv_path}")
    df_raw = pd.read_csv(csv_path, parse_dates=['datetime'])

    # MA200 워밍업: 210일 전부터
    warmup_start = pd.Timestamp(TEST_START) - pd.Timedelta(days=210)
    df_raw = df_raw[df_raw['datetime'] >= warmup_start].copy()

    # 리샘플링
    df_raw = df_raw.set_index('datetime')
    df_4h = df_raw.resample(TIMEFRAME).agg({
        'open': 'first', 'high': 'max', 'low': 'min',
        'close': 'last', 'volume': 'sum',
    }).dropna().reset_index()

    # 테스트 종료일 필터
    df_4h = df_4h[df_4h['datetime'] <= TEST_END].copy()
    print(f"  [{asset_name}] {len(df_4h):,}봉 ({df_4h['datetime'].iloc[0].date()} ~ {df_4h['datetime'].iloc[-1].date()})")
    return df_4h


# ══════════════════════════════════════════════════════════════
#  개별 종목 백테스트 실행
# ══════════════════════════════════════════════════════════════

print(f"\n{'='*70}")
print(f"  멀티 에셋 Donchian Breakout (10d/5d + MA200)")
print(f"  종목: {', '.join(ASSETS.keys())}")
print(f"  자본 배분: {TOTAL_CAPITAL:,.0f} USDT → {CAPITAL_PER_ASSET:,.0f} USDT/종목")
print(f"  테스트 기간: {TEST_START} ~ {TEST_END}")
print(f"{'='*70}\n")

# 각 종목별 결과 저장
results = {}
all_trades = []

for asset_name, csv_path in ASSETS.items():
    print(f"\n── {asset_name}/USDT ──────────────────────────────────")

    # 데이터 로드
    try:
        df = load_and_resample(csv_path, asset_name)
    except FileNotFoundError:
        print(f"  [{asset_name}] 파일을 찾을 수 없습니다. 건너뜁니다.")
        continue

    # 전략 + 엔진 (종목마다 새 인스턴스)
    strategy = DonchianShortMAFilter(
        entry_period=ENTRY_PERIOD,
        exit_period=EXIT_PERIOD,
        ma_filter_period=MA_FILTER_PERIOD,
    )
    engine = Engine(
        strategy,
        initial_capital=CAPITAL_PER_ASSET,
        fee_rate=FEE_RATE,
        cooldown=COOLDOWN,
    )

    # 백테스트 실행
    result = engine.run(df)

    # 개별 리포트
    report = Report(result, timeframe=TIMEFRAME)
    stats = report.summary()

    # 결과 저장
    results[asset_name] = {
        'result': result,
        'stats': stats,
        'df': df,
        'equity': result['equity_curve'],
    }

    # 거래 내역에 종목명 추가
    for t in result['trades']:
        t['asset'] = asset_name
        all_trades.append(t)


# ══════════════════════════════════════════════════════════════
#  포트폴리오 합산 결과
# ══════════════════════════════════════════════════════════════

print(f"\n\n{'='*70}")
print(f"  포트폴리오 합산 결과 (균등 배분)")
print(f"{'='*70}")

# 각 종목 equity curve를 날짜 기준으로 정렬하여 합산
# (모든 종목이 같은 4시간봉 타임스탬프를 공유하므로 단순 합산 가능)
# 테스트 기간 내 데이터만 사용
test_start_ts = pd.Timestamp(TEST_START)

# 각 종목의 equity를 DataFrame으로 만들어서 합산
equity_dfs = {}
for name, r in results.items():
    df = r['df'].copy()
    df['equity'] = r['equity']
    # 테스트 기간만
    df = df[df['datetime'] >= test_start_ts].copy()
    equity_dfs[name] = df.set_index('datetime')['equity']

# 합산 (같은 날짜에 없는 값은 forward fill)
combined = pd.DataFrame(equity_dfs)
combined = combined.fillna(method='ffill')
combined['total'] = combined.sum(axis=1)

# 합산 통계
total_equity = combined['total'].values
initial = CAPITAL_PER_ASSET * len(results)
final = total_equity[-1]
total_return = (final / initial - 1) * 100

# MDD
running_max = np.maximum.accumulate(total_equity)
drawdown = (total_equity - running_max) / running_max
mdd = float(drawdown.min()) * 100

# Sharpe
returns = np.diff(total_equity) / total_equity[:-1]
annual_bars = 2190  # 4시간봉 연간
sharpe = float(returns.mean() / returns.std() * np.sqrt(annual_bars)) if returns.std() > 0 else 0

# 총 거래 수
total_trades = len(all_trades)
wins = sum(1 for t in all_trades if t['pnl'] > 0)
win_rate = wins / total_trades * 100 if total_trades > 0 else 0
avg_pnl = sum(t['pnl_pct'] for t in all_trades) / total_trades if total_trades > 0 else 0

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


# ══════════════════════════════════════════════════════════════
#  종목별 요약 비교
# ══════════════════════════════════════════════════════════════

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
print(f"{'='*70}")


# ══════════════════════════════════════════════════════════════
#  전체 거래 내역 (종목별)
# ══════════════════════════════════════════════════════════════

print(f"\n\n  [전체 거래 내역 — 시간순]")
print(f"  {'#':>3}  {'종목':<5} {'진입 시각':<19}  {'청산 시각':<19}  {'진입가':>12}  {'청산가':>12}  {'손익':>8}")
print(f"  " + "-" * 90)

# 시간순 정렬
sorted_trades = sorted(all_trades, key=lambda t: t['entry_dt'] if t['entry_dt'] is not None else pd.Timestamp.max)

for i, t in enumerate(sorted_trades, 1):
    asset = t.get('asset', '?')
    entry_dt = str(pd.Timestamp(t['entry_dt']))[:19] if t['entry_dt'] is not None else '-'
    exit_dt = str(pd.Timestamp(t['exit_dt']))[:19] if t['exit_dt'] is not None else '-'
    print(f"  {i:>3}  {asset:<5} {entry_dt:<19}  {exit_dt:<19}  {t['entry_price']:>12,.2f}  {t['exit_price']:>12,.2f}  {t['pnl_pct']:>+7.2f}%")

print(f"\n  총 {len(sorted_trades)}건 거래 완료")


# ══════════════════════════════════════════════════════════════
#  시각화: 포트폴리오 equity curve + Buy&Hold 비교
# ══════════════════════════════════════════════════════════════

import plotly.graph_objects as go
from plotly.subplots import make_subplots

print(f"\n차트 생성 중...")

# ── Buy & Hold 계산 (4종목 균등배분) ─────────────────────────
# 각 종목의 종가를 테스트 시작 시점 대비 수익률로 환산 후 합산
bnh_dfs = {}
for name, r in results.items():
    df = r['df'].copy()
    df = df[df['datetime'] >= test_start_ts].copy()
    # 시작가 대비 수익률 × 배분 자본
    start_price = df['close'].iloc[0]
    df['bnh_equity'] = df['close'] / start_price * CAPITAL_PER_ASSET
    bnh_dfs[name] = df.set_index('datetime')['bnh_equity']

bnh_combined = pd.DataFrame(bnh_dfs)
bnh_combined = bnh_combined.fillna(method='ffill')
bnh_combined['total'] = bnh_combined.sum(axis=1)

# ── 날짜 인덱스 통일 ────────────────────────────────────────
datetimes = combined.index

# ── 종목별 색상 ──────────────────────────────────────────────
COLORS = {
    'BTC': '#F7931A',   # 비트코인 오렌지
    'ETH': '#627EEA',   # 이더리움 퍼플
    'XRP': '#23292F',   # 리플 다크
    'SOL': '#9945FF',   # 솔라나 퍼플
}
COLORS_LIGHT = {
    'BTC': 'rgba(247,147,26,0.15)',
    'ETH': 'rgba(98,126,234,0.15)',
    'XRP': 'rgba(35,41,47,0.15)',
    'SOL': 'rgba(153,69,255,0.15)',
}

# ── 서브플롯: 상단 = 포트폴리오 비교, 하단 = 종목별 equity ──
fig = make_subplots(
    rows=2, cols=1,
    shared_xaxes=True,
    row_heights=[0.55, 0.45],
    vertical_spacing=0.08,
    subplot_titles=(
        "포트폴리오 자산 곡선 (전략 vs Buy&Hold)",
        "종목별 자산 곡선"
    )
)

# ── 상단: 포트폴리오 합산 ────────────────────────────────────

# 전략 포트폴리오
fig.add_trace(go.Scatter(
    x=datetimes, y=combined['total'],
    name='전략 포트폴리오',
    line=dict(color='#2ecc71', width=2),
    fill='tozeroy', fillcolor='rgba(46,204,113,0.08)',
    hovertemplate='%{x}<br>전략: %{y:,.0f} USDT<extra></extra>'
), row=1, col=1)

# Buy & Hold 포트폴리오
fig.add_trace(go.Scatter(
    x=datetimes, y=bnh_combined['total'],
    name='Buy & Hold (균등배분)',
    line=dict(color='#888888', width=1.5, dash='dash'),
    hovertemplate='%{x}<br>B&H: %{y:,.0f} USDT<extra></extra>'
), row=1, col=1)

# 기준선 (초기 자본)
fig.add_hline(
    y=TOTAL_CAPITAL, line_dash="dot", line_color="rgba(255,255,255,0.3)",
    annotation_text=f"초기 자본 {TOTAL_CAPITAL:,.0f}",
    annotation_position="bottom right",
    row=1, col=1
)

# 거래 마커 (상단 차트에 종목별 색상으로 표시)
for name, r in results.items():
    trades = r['result']['trades']
    if not trades:
        continue

    # 진입 마커 — equity 기준 y좌표는 포트폴리오 합산으로
    entry_times = [pd.Timestamp(t['entry_dt']) for t in trades if t['entry_dt'] is not None]
    entry_y = []
    for dt in entry_times:
        # 해당 시점의 포트폴리오 합산 equity 찾기
        idx = combined.index.searchsorted(dt)
        if idx < len(combined):
            entry_y.append(combined['total'].iloc[min(idx, len(combined)-1)])
        else:
            entry_y.append(combined['total'].iloc[-1])

    if entry_times:
        fig.add_trace(go.Scatter(
            x=entry_times, y=entry_y,
            mode='markers', name=f'{name} 진입',
            marker=dict(symbol='triangle-up', size=8,
                       color='rgba(0,0,0,0)',
                       line=dict(color=COLORS.get(name, '#888'), width=1.5)),
            hovertemplate=f'{name} 진입<br>%{{x}}<extra></extra>'
        ), row=1, col=1)


# ── 하단: 종목별 equity curve ────────────────────────────────
for name in results.keys():
    if name in combined.columns:
        fig.add_trace(go.Scatter(
            x=datetimes, y=combined[name],
            name=f'{name} 전략',
            line=dict(color=COLORS.get(name, '#888'), width=1.5),
            hovertemplate=f'{name}<br>%{{x}}<br>%{{y:,.0f}} USDT<extra></extra>'
        ), row=2, col=1)

    # 종목별 Buy & Hold
    if name in bnh_combined.columns:
        fig.add_trace(go.Scatter(
            x=datetimes, y=bnh_combined[name],
            name=f'{name} B&H',
            line=dict(color=COLORS.get(name, '#888'), width=1, dash='dot'),
            showlegend=False,
            hovertemplate=f'{name} B&H<br>%{{x}}<br>%{{y:,.0f}} USDT<extra></extra>'
        ), row=2, col=1)

# 하단 기준선
fig.add_hline(
    y=CAPITAL_PER_ASSET, line_dash="dot", line_color="rgba(255,255,255,0.2)",
    row=2, col=1
)


# ── 레이아웃 ──────────────────────────────────────────────────
fig.update_layout(
    title=f'멀티에셋 Donchian Breakout — {", ".join(results.keys())} (균등배분)',
    template='plotly_dark',
    height=950,
    hovermode='x unified',
    legend=dict(orientation='h', y=1.02, x=0, font=dict(size=11)),
    dragmode='zoom',
)

fig.update_xaxes(
    showspikes=True, spikemode='across', fixedrange=False,
    rangeslider=dict(visible=False),
)
fig.update_yaxes(fixedrange=False)

# 하단 x축에만 슬라이더
fig.update_layout(
    xaxis2=dict(
        rangeslider=dict(visible=True, thickness=0.04),
        type='date',
    )
)

output_path = "backtest_multi.html"
fig.write_html(output_path, config={
    'scrollZoom': True,
    'displayModeBar': True,
})
print(f"차트 저장 완료: {output_path}")
print(f"브라우저에서 {output_path} 파일을 열어 차트를 확인하세요.")