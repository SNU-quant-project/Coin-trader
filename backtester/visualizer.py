"""
백테스트 결과 시각화 (인터랙티브 HTML)

차트 구성:
  1. 가격 + Donchian 채널 (상단/하단) + 매수/매도 시점
  2. 자산 곡선 (equity curve)

조작법:
  - 드래그: 구간 선택 확대
  - 더블클릭: 전체 보기로 복귀
  - 상단 툴바 → '축 고정 해제' 후 각 축 독립 스크롤 가능
  - shift + 드래그: y축 이동

전략 호환:
  - DonchianBreakoutStrategy: 진입/청산 채널 표시
  - MACrossStrategy: 이동평균선 표시 (기존 호환)
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

    # ── 전략 타입 감지 ────────────────────────────────────────
    is_donchian = hasattr(strategy, 'entry_period') and hasattr(strategy, 'exit_period')
    is_ma_cross = hasattr(strategy, 'fast_period') and hasattr(strategy, 'slow_period')
    is_bollinger = hasattr(strategy, 'bb_period') and hasattr(strategy, 'bb_std')
    is_ema_cross = hasattr(strategy, 'fast_ema') and hasattr(strategy, 'slow_ema')

    # ── 다운샘플링 (차트 렌더링 성능) ─────────────────────────
    # 4시간봉 이하면 일봉으로 리샘플, 일봉 이상이면 그대로 사용
    df['_equity'] = equity_curve

    # 데이터 포인트가 1000개 이하면 리샘플링 불필요
    if len(df) > 1000:
        df_plot = df.set_index('datetime').resample('1D').agg({
            'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last',
            '_equity': 'last',
        }).dropna().reset_index()
    else:
        df_plot = df.copy()

    datetimes   = df_plot['datetime']
    equity_plot = df_plot['_equity']

    # ── 전략별 보조선 계산 ────────────────────────────────────
    if is_donchian:
        # 차트용 Donchian 채널 (일봉 기준으로 재계산)
        # 원본 전략의 기간을 일수로 환산
        entry_days = strategy.entry_period // 6  # 4시간봉 → 일수
        exit_days = strategy.exit_period // 6

        upper_line = df_plot['high'].rolling(entry_days).max().shift(1)
        lower_line = df_plot['low'].rolling(exit_days).min().shift(1)

        has_ma_filter = hasattr(strategy, 'ma_filter_period')
        if has_ma_filter:
            ma_days = strategy.ma_filter_period // 6
            title_suffix = f"Donchian({entry_days}d/{exit_days}d)+MA{ma_days}"
        else:
            title_suffix = f"Donchian({entry_days}d/{exit_days}d)"
        line1_name = f"진입 채널 ({entry_days}일 최고)"
        line2_name = f"청산 채널 ({exit_days}일 최저)"
    elif is_bollinger:
        # 볼린저밴드 (차트용)
        bb_ma = df_plot['close'].rolling(strategy.bb_period).mean()
        bb_std_val = df_plot['close'].rolling(strategy.bb_period).std()
        upper_line = bb_ma + strategy.bb_std * bb_std_val
        lower_line = bb_ma - strategy.bb_std * bb_std_val

        title_suffix = f"BB({strategy.bb_period},{strategy.bb_std}σ)+RSI({strategy.rsi_period})"
        line1_name = f"BB 상단 ({strategy.bb_std}σ)"
        line2_name = f"BB 하단 ({strategy.bb_std}σ)"
    elif is_ema_cross:
        # EMA 크로스오버 전략
        upper_line = df_plot['close'].ewm(span=strategy.fast_ema, adjust=False).mean()
        lower_line = df_plot['close'].ewm(span=strategy.slow_ema, adjust=False).mean()
        title_suffix = f"EMA({strategy.fast_ema}/{strategy.slow_ema})+MA{strategy.ma_filter // 6}"
        line1_name = f"EMA {strategy.fast_ema} (단기)"
        line2_name = f"EMA {strategy.slow_ema} (장기)"
    elif is_ma_cross:
        # 기존 MA Cross 호환
        upper_line = df_plot['close'].rolling(10).mean()
        lower_line = df_plot['close'].rolling(30).mean()
        title_suffix = f"MA{strategy.fast_period}/MA{strategy.slow_period}"
        line1_name = f"MA{strategy.fast_period}"
        line2_name = f"MA{strategy.slow_period}"
    else:
        upper_line = None
        lower_line = None
        title_suffix = "Custom Strategy"
        line1_name = ""
        line2_name = ""

    # ── 거래 시점 (entry_dt 기준) ─────────────────────────────
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

    # 청산 시점도 표시 (Donchian 전략에서 유용)
    exit_times, exit_prices = [], []
    for t in trades:
        if t['exit_dt'] is None:
            continue
        exit_times.append(pd.Timestamp(t['exit_dt']))
        exit_prices.append(float(t['exit_price']))

    # ── 서브플롯 ──────────────────────────────────────────────
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[0.65, 0.35],
        vertical_spacing=0.06,
        subplot_titles=("가격 & 채널", "자산 곡선 (Equity Curve)")
    )

    # 가격 (종가)
    fig.add_trace(go.Scatter(
        x=datetimes, y=df_plot['close'],
        name='종가', line=dict(color='#888888', width=1),
        hovertemplate='%{x}<br>종가: %{y:,.2f}<extra></extra>'
    ), row=1, col=1)

    # 보조선 (채널 또는 MA)
    if upper_line is not None:
        fig.add_trace(go.Scatter(
            x=datetimes, y=upper_line,
            name=line1_name,
            line=dict(color='#4a90e2', width=1.2, dash='dot'),
            hovertemplate=f'{line1_name}: %{{y:,.2f}}<extra></extra>'
        ), row=1, col=1)

    if lower_line is not None:
        fig.add_trace(go.Scatter(
            x=datetimes, y=lower_line,
            name=line2_name,
            line=dict(color='#e74c3c', width=1.2, dash='dot'),
            hovertemplate=f'{line2_name}: %{{y:,.2f}}<extra></extra>'
        ), row=1, col=1)

    # 볼린저밴드 중간선 (청산 목표)
    if is_bollinger:
        bb_mid = df_plot['close'].rolling(strategy.bb_period).mean()
        fig.add_trace(go.Scatter(
            x=datetimes, y=bb_mid,
            name=f'BB 중간선 ({strategy.bb_period}MA)',
            line=dict(color='#f5a623', width=1.0, dash='dash'),
            hovertemplate=f'BB 중간선: %{{y:,.2f}}<extra></extra>'
        ), row=1, col=1)

    # 200일 MA 필터 선 (Donchian + MA 필터 전략)
    if is_donchian and hasattr(strategy, 'ma_filter_period'):
        ma_days = strategy.ma_filter_period // 6
        ma_line = df_plot['close'].rolling(ma_days).mean()
        fig.add_trace(go.Scatter(
            x=datetimes, y=ma_line,
            name=f'{ma_days}일 MA (레짐 필터)',
            line=dict(color='#f5a623', width=1.5, dash='dash'),
            hovertemplate=f'{ma_days}일 MA: %{{y:,.2f}}<extra></extra>'
        ), row=1, col=1)

    # 100일 MA 필터 선 (EMA 크로스 전략)
    if is_ema_cross and hasattr(strategy, 'ma_filter'):
        ma_days = strategy.ma_filter // 6
        ma_line = df_plot['close'].rolling(ma_days).mean()
        fig.add_trace(go.Scatter(
            x=datetimes, y=ma_line,
            name=f'{ma_days}일 MA (레짐 필터)',
            line=dict(color='#f5a623', width=1.5, dash='dash'),
            hovertemplate=f'{ma_days}일 MA: %{{y:,.2f}}<extra></extra>'
        ), row=1, col=1)

    # Donchian 채널 사이 영역 채우기
    if (is_donchian or is_bollinger) and upper_line is not None and lower_line is not None:
        fig.add_trace(go.Scatter(
            x=datetimes, y=upper_line,
            mode='lines', line=dict(width=0),
            showlegend=False, hoverinfo='skip',
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=datetimes, y=lower_line,
            mode='lines', line=dict(width=0),
            fill='tonexty', fillcolor='rgba(74, 144, 226, 0.06)',
            showlegend=False, hoverinfo='skip',
        ), row=1, col=1)

    # 롱 진입 마커
    if long_times:
        fig.add_trace(go.Scatter(
            x=long_times, y=long_prices,
            mode='markers', name='롱 진입',
            marker=dict(
                symbol='triangle-up',
                size=9,
                color='rgba(0,0,0,0)',
                line=dict(color='#2ecc71', width=2),
            ),
            hovertemplate='롱 진입<br>%{x}<br>%{y:,.2f}<extra></extra>'
        ), row=1, col=1)

    # 숏 진입 마커 (롱 온리에서는 없지만, 범용성을 위해 유지)
    if short_times:
        fig.add_trace(go.Scatter(
            x=short_times, y=short_prices,
            mode='markers', name='숏 진입',
            marker=dict(
                symbol='triangle-down',
                size=9,
                color='rgba(0,0,0,0)',
                line=dict(color='#e74c3c', width=2),
            ),
            hovertemplate='숏 진입<br>%{x}<br>%{y:,.2f}<extra></extra>'
        ), row=1, col=1)

    # 청산 마커 (× 표시)
    if exit_times:
        fig.add_trace(go.Scatter(
            x=exit_times, y=exit_prices,
            mode='markers', name='청산',
            marker=dict(
                symbol='x',
                size=7,
                color='#f39c12',
                line=dict(width=1.5),
            ),
            hovertemplate='청산<br>%{x}<br>%{y:,.2f}<extra></extra>'
        ), row=1, col=1)

    # 자산 곡선
    fig.add_trace(go.Scatter(
        x=datetimes, y=equity_plot,
        name='자산', line=dict(color='#2ecc71', width=1.5),
        fill='tozeroy', fillcolor='rgba(46,204,113,0.07)',
        hovertemplate='%{x}<br>자산: %{y:,.2f} USDT<extra></extra>'
    ), row=2, col=1)

    # Buy & Hold 비교선 (초기 자본 기준)
    if len(df_plot) > 0:
        initial_price = df_plot['close'].iloc[0]
        initial_capital = result.get('initial_capital', 10_000.0)
        bnh = df_plot['close'] / initial_price * initial_capital
        fig.add_trace(go.Scatter(
            x=datetimes, y=bnh,
            name='Buy & Hold', line=dict(color='#888888', width=1, dash='dash'),
            hovertemplate='%{x}<br>B&H: %{y:,.2f} USDT<extra></extra>'
        ), row=2, col=1)

    # ── 레이아웃 ──────────────────────────────────────────────
    fig.update_layout(
        title=f'백테스트 결과 — BTC/USDT ({title_suffix})',
        template='plotly_dark',
        height=900,
        hovermode='x unified',
        legend=dict(orientation='h', y=1.03, x=0, font=dict(size=12)),
        dragmode='zoom',
        modebar=dict(
            add=['v1hovermode', 'togglespikelines'],
        ),
    )

    fig.update_xaxes(
        rangeslider=dict(visible=False),
        showspikes=True,
        spikemode='across',
        fixedrange=False,
    )
    fig.update_yaxes(fixedrange=False)

    fig.update_layout(
        xaxis2=dict(
            rangeslider=dict(visible=True, thickness=0.04),
            type='date',
        )
    )

    fig.write_html(output_path, config={
        'scrollZoom': True,
        'displayModeBar': True,
        'modeBarButtonsToAdd': ['drawline', 'eraseshape'],
    })
    print(f"차트 저장 완료: {output_path}")
