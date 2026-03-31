"""
백테스트 결과 시각화 (인터랙티브 HTML)

차트 구성:
  1. 가격 + MA선 + 매수/매도 시점
  2. 자산 곡선 (equity curve)

조작법:
  - 드래그: 구간 선택 확대
  - 더블클릭: 전체 보기로 복귀
  - 상단 툴바 → '축 고정 해제' 후 각 축 독립 스크롤 가능
  - shift + 드래그: y축 이동
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def plot(result: dict, df: pd.DataFrame, strategy, output_path: str = "backtest_result.html"):
    df = df.reset_index(drop=True)
    df['datetime'] = pd.to_datetime(df['datetime'])
    signals = result['signals']
    equity_curve = result['equity_curve']
    trades = result['trades']

    # ── 1시간봉으로 다운샘플링 (차트 렌더링 성능) ─────────────
    df['_equity'] = equity_curve
    df_plot = df.set_index('datetime').resample('1D').agg({
        'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last',
        '_equity': 'last',
    }).dropna().reset_index()

    # 일봉 기준 MA (원본 분봉 파라미터와 별개로 일봉에서 보기 좋은 값 사용)
    fast_ma_plot = df_plot['close'].rolling(10).mean()
    slow_ma_plot = df_plot['close'].rolling(30).mean()
    datetimes    = df_plot['datetime']
    equity_plot  = df_plot['_equity']

    # ── 거래 시점 (entry_dt 기준 직접 매칭) ───────────────────
    long_entries  = [t for t in trades if t['direction'] ==  1]
    short_entries = [t for t in trades if t['direction'] == -1]

    def match_trade_times(entries):
        times, prices = [], []
        for t in entries:
            if t['entry_dt'] is None:
                continue
            dt = pd.Timestamp(t['entry_dt'])
            times.append(dt)
            prices.append(float(t['entry_price']))
        return times, prices

    long_times,  long_prices  = match_trade_times(long_entries)
    short_times, short_prices = match_trade_times(short_entries)

    # ── 서브플롯 ──────────────────────────────────────────────
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[0.65, 0.35],
        vertical_spacing=0.06,
        subplot_titles=("가격 & 이동평균", "자산 곡선 (Equity Curve)")
    )

    # 가격
    fig.add_trace(go.Scatter(
        x=datetimes, y=df_plot['close'],
        name='종가', line=dict(color='#888888', width=1),
        hovertemplate='%{x}<br>종가: %{y:,.2f}<extra></extra>'
    ), row=1, col=1)

    # MA 선
    fig.add_trace(go.Scatter(
        x=datetimes, y=fast_ma_plot,
        name=f'MA{strategy.fast_period}',
        line=dict(color='#f5a623', width=1.5),
        hovertemplate=f'MA{strategy.fast_period}: %{{y:,.2f}}<extra></extra>'
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=datetimes, y=slow_ma_plot,
        name=f'MA{strategy.slow_period}',
        line=dict(color='#4a90e2', width=1.5),
        hovertemplate=f'MA{strategy.slow_period}: %{{y:,.2f}}<extra></extra>'
    ), row=1, col=1)

    # 롱 진입 마커 (크기 줄이고 외곽선으로 표시)
    if long_times:
        fig.add_trace(go.Scatter(
            x=long_times, y=long_prices,
            mode='markers', name='롱 진입',
            marker=dict(
                symbol='triangle-up',
                size=7,
                color='rgba(0,0,0,0)',
                line=dict(color='#2ecc71', width=1.5),
            ),
            hovertemplate='롱 진입<br>%{x}<br>%{y:,.2f}<extra></extra>'
        ), row=1, col=1)

    # 숏 진입 마커
    if short_times:
        fig.add_trace(go.Scatter(
            x=short_times, y=short_prices,
            mode='markers', name='숏 진입',
            marker=dict(
                symbol='triangle-down',
                size=7,
                color='rgba(0,0,0,0)',
                line=dict(color='#e74c3c', width=1.5),
            ),
            hovertemplate='숏 진입<br>%{x}<br>%{y:,.2f}<extra></extra>'
        ), row=1, col=1)

    # 자산 곡선
    fig.add_trace(go.Scatter(
        x=datetimes, y=equity_plot,
        name='자산', line=dict(color='#2ecc71', width=1.5),
        fill='tozeroy', fillcolor='rgba(46,204,113,0.07)',
        hovertemplate='%{x}<br>자산: %{y:,.2f} USDT<extra></extra>'
    ), row=2, col=1)

    # ── 레이아웃 ──────────────────────────────────────────────
    fig.update_layout(
        title=f'백테스트 결과 — BTC/USDT (MA{strategy.fast_period}/MA{strategy.slow_period})',
        template='plotly_dark',
        height=900,
        hovermode='x unified',
        legend=dict(orientation='h', y=1.03, x=0, font=dict(size=12)),
        dragmode='zoom',
        # 툴바에 축 독립 줌 버튼 추가
        modebar=dict(
            add=['v1hovermode', 'togglespikelines'],
        ),
    )

    # x축: 범위 슬라이더 + 독립 확대 가능
    fig.update_xaxes(
        rangeslider=dict(visible=False),
        showspikes=True,
        spikemode='across',
        fixedrange=False,
    )

    # y축: 독립 확대 가능
    fig.update_yaxes(fixedrange=False)

    # 하단 x축에만 슬라이더 표시
    fig.update_layout(
        xaxis2=dict(
            rangeslider=dict(visible=True, thickness=0.04),
            type='date',
        )
    )

    fig.write_html(output_path, config={
        'scrollZoom': True,        # 마우스 휠로 확대/축소
        'displayModeBar': True,
        'modeBarButtonsToAdd': ['drawline', 'eraseshape'],
    })
    print(f"차트 저장 완료: {output_path}")
