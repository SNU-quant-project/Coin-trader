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

try:
    from strategy.filters.adx_and_Rsquare import (
        REGIME_TABLE, LONG, SHORT, CASH, COUNTER, HOLD
    )
    _REGIME_IMPORT_OK = True
except ImportError:
    _REGIME_IMPORT_OK = False


def plot(result: dict, df: pd.DataFrame, strategy, output_path: str = "backtest_result.html", stats: dict = None):
    df = df.reset_index(drop=True)
    df['datetime'] = pd.to_datetime(df['datetime'])
    signals = result['signals']
    equity_curve = result['equity_curve']
    trades = result['trades']

    # ── 전략 타입 감지 ────────────────────────────────────────
    is_donchian = hasattr(strategy, 'entry_period') and hasattr(strategy, 'exit_period')
    is_ma_cross = hasattr(strategy, 'fast_period') and hasattr(strategy, 'slow_period')

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

        title_suffix = f"Donchian({entry_days}d/{exit_days}d)"
        line1_name = f"진입 채널 ({entry_days}일 최고)"
        line2_name = f"청산 채널 ({exit_days}일 최저)"
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

    # Donchian 채널 사이 영역 채우기
    if is_donchian and upper_line is not None and lower_line is not None:
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

    # ── 국면 음영 (regime_filter가 있을 때만 실행) ────────────
    if hasattr(strategy, 'regime_filter') and strategy.regime_filter is not None:
        regime = strategy.regime_filter.get_regime(df)

        # df → df_plot 기준으로 맞추기 (다운샘플된 날짜 기준)
        regime_series = pd.Series(regime.values, index=df['datetime'])
        regime_plot = regime_series.reindex(df_plot['datetime'], method='ffill').fillna(0)

        # 연속된 같은 국면 구간으로 묶기
        # [(국면값, 시작날짜, 끝날짜), ...]
        segments = []
        prev_val = None
        seg_start = None
        dates = df_plot['datetime'].values
        vals  = regime_plot.values

        for i, (dt, val) in enumerate(zip(dates, vals)):
            if val != prev_val:
                if prev_val is not None:
                    segments.append((prev_val, seg_start, dates[i - 1]))
                seg_start = dt
                prev_val  = val
        if prev_val is not None:
            segments.append((prev_val, seg_start, dates[-1]))

        # 국면별 색상
        regime_colors = {
             1: 'rgba(46,  204, 113, 0.20)',   # 상승 → 초록
             0: 'rgba(160, 160, 160, 0.10)',   # 횡보 → 회색
            -1: 'rgba(231,  76,  60, 0.20)',   # 하락 → 빨강
        }
        regime_labels = {
             1: '상승 국면',
             0: '횡보 국면',
            -1: '하락 국면',
        }

        # 이미 범례에 추가된 국면 추적 (중복 방지)
        legend_added = set()

        for val, x0, x1 in segments:
            color = regime_colors.get(int(val), 'rgba(160,160,160,0.10)')
            label = regime_labels.get(int(val), '')
            show_legend = int(val) not in legend_added

            # numpy datetime64 → 문자열로 변환 (Plotly 호환성)
            x0_str = str(pd.Timestamp(x0))
            x1_str = str(pd.Timestamp(x1))

            # 가격 차트(row=1)와 자산 곡선(row=2) 모두에 음영 추가
            fig.add_vrect(
                x0=x0_str, x1=x1_str,
                fillcolor=color,
                layer='below',
                line_width=0,
                row='all', col=1,
                # 범례는 더미 scatter로 별도 추가 (vrect는 범례 미지원)
            )

            if show_legend:
                legend_added.add(int(val))

        # 범례용 더미 trace 추가 (실제 데이터 없음, 색상 표시 목적)
        for val in [1, 0, -1]:
            if val in legend_added:
                color_solid = {
                     1: 'rgba(46,  204, 113, 0.4)',
                     0: 'rgba(160, 160, 160, 0.4)',
                    -1: 'rgba(231,  76,  60, 0.4)',
                }[val]
                fig.add_trace(go.Scatter(
                    x=[None], y=[None],
                    mode='markers',
                    marker=dict(size=10, color=color_solid, symbol='square'),
                    name=regime_labels[val],
                    showlegend=True,
                ), row=1, col=1)

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

    chart_html = fig.to_html(full_html=False, config={
        'scrollZoom': True,
        'displayModeBar': True,
        'modeBarButtonsToAdd': ['drawline', 'eraseshape'],
    })

    # ── 성과 요약 표 HTML ─────────────────────────────────────
    table_html = ''
    if stats is not None:
        s = stats

        def _color(val, positive_green=True):
            if val > 0:
                return '#2ecc71' if positive_green else '#e74c3c'
            elif val < 0:
                return '#e74c3c' if positive_green else '#2ecc71'
            return '#cccccc'

        def _row(label, value, color=None):
            color_style = f"color:{color};" if color else ''
            return (
                f"<tr>"
                f"<td style='padding:6px 20px 6px 12px; color:#aaaaaa;'>{label}</td>"
                f"<td style='padding:6px 12px 6px 0; text-align:right; font-weight:bold; {color_style}'>{value}</td>"
                f"</tr>"
            )

        table_html = f"""
        <div style="font-family: 'Helvetica Neue', Arial, sans-serif; margin: 24px auto; max-width: 500px;">
          <h2 style="color:#eeeeee; margin-bottom:12px; font-size:16px; letter-spacing:1px;">
            ▎ 백테스트 결과 요약
          </h2>
          <table style="width:100%; border-collapse:collapse; background:#1e1e1e; border-radius:8px; overflow:hidden; font-size:14px;">
            <tbody>
              <tr style="background:#2a2a2a;">
                <td colspan="2" style="padding:8px 12px; color:#888888; font-size:12px; letter-spacing:1px;">기간 및 자산</td>
              </tr>
              {_row('기간', f"{s['start']} ~ {s['end']} ({s['days']:,}일)")}
              {_row('초기 자산', f"{s['initial_capital']:,.2f} USDT")}
              {_row('최종 자산', f"{s['final_equity']:,.2f} USDT", _color(s['total_return']))}
              <tr style="background:#2a2a2a;">
                <td colspan="2" style="padding:8px 12px; color:#888888; font-size:12px; letter-spacing:1px;">수익 지표</td>
              </tr>
              {_row('총 수익률', f"{s['total_return']:+.2f}%", _color(s['total_return']))}
              {_row('연환산 수익', f"{s['annual_return']:+.2f}%", _color(s['annual_return']))}
              {_row('MDD', f"{s['mdd']:+.2f}%", '#e74c3c')}
              {_row('샤프 지수', f"{s['sharpe']:.2f}", _color(s['sharpe']))}
              <tr style="background:#2a2a2a;">
                <td colspan="2" style="padding:8px 12px; color:#888888; font-size:12px; letter-spacing:1px;">거래 통계</td>
              </tr>
              {_row('총 거래 수', f"{s['total_trades']:,}회")}
              {_row('승률', f"{s['win_rate']:.2f}%", _color(s['win_rate'] - 50))}
              {_row('평균 손익', f"{s['avg_pnl_pct']:+.2f}%", _color(s['avg_pnl_pct']))}
            </tbody>
          </table>
        </div>
        """

    # ── 국면 분류 테이블 HTML ─────────────────────────────────
    regime_table_html = ''
    if _REGIME_IMPORT_OK and hasattr(strategy, 'regime_filter') and strategy.regime_filter is not None:
        rf = strategy.regime_filter

        # 액션 → 표시 텍스트 및 색상
        action_style = {
            LONG:    ('<b>LONG</b>',    '#2ecc71', '#1a3a28'),
            SHORT:   ('<b>SHORT</b>',   '#e74c3c', '#3a1a1a'),
            CASH:    ('<b>CASH</b>',    '#f39c12', '#3a2e1a'),
            COUNTER: ('<b>COUNTER</b>', '#9b59b6', '#2a1a3a'),
            HOLD:    ('<b>HOLD</b>',    '#888888', '#222222'),
        }
        # 국면 값 → 레이블
        regime_label = {1: '상승 ↑', 0: '횡보 →', -1: '하락 ↓'}
        regime_header_color = {1: '#2ecc71', 0: '#888888', -1: '#e74c3c'}

        # 파라미터 읽기
        adx_period    = rf.adx_filter.period
        adx_threshold = rf.adx_filter.threshold
        r2_period     = rf.r2_filter.period
        r2_threshold  = rf.r2_filter.r2_threshold

        # 테이블 행 생성 (ADX 국면 순서: 상승/횡보/하락)
        rows_html = ''
        for adx_val in [1, 0, -1]:
            adx_lbl   = regime_label[adx_val]
            adx_color = regime_header_color[adx_val]
            for r2_val in [1, 0, -1]:
                r2_lbl    = regime_label[r2_val]
                r2_color  = regime_header_color[r2_val]
                action    = REGIME_TABLE.get((adx_val, r2_val), CASH)
                text, fg, bg = action_style.get(action, ('<b>?</b>', '#cccccc', '#222'))
                rows_html += (
                    f"<tr>"
                    f"<td style='padding:6px 10px; color:{adx_color}; text-align:center;'>{adx_lbl}</td>"
                    f"<td style='padding:6px 10px; color:{r2_color}; text-align:center;'>{r2_lbl}</td>"
                    f"<td style='padding:6px 14px; text-align:center; color:{fg}; background:{bg}; border-radius:4px;'>{text}</td>"
                    f"</tr>"
                )

        regime_table_html = f"""
        <div style="font-family: 'Helvetica Neue', Arial, sans-serif; margin: 24px 0 24px 32px;">
          <h2 style="color:#eeeeee; margin-bottom:12px; font-size:16px; letter-spacing:1px;">
            ▎ 국면 분류 테이블
          </h2>
          <div style="display:flex; gap:16px; align-items:flex-start;">

            <!-- 9가지 케이스 테이블 -->
            <table style="border-collapse:collapse; background:#1e1e1e; border-radius:8px; overflow:hidden; font-size:13px;">
              <thead>
                <tr style="background:#2a2a2a;">
                  <th style="padding:8px 10px; color:#888888; font-weight:normal;">ADX</th>
                  <th style="padding:8px 10px; color:#888888; font-weight:normal;">R²</th>
                  <th style="padding:8px 14px; color:#888888; font-weight:normal;">액션</th>
                </tr>
              </thead>
              <tbody>
                {rows_html}
              </tbody>
            </table>

            <!-- 파라미터 박스 -->
            <table style="border-collapse:collapse; background:#1e1e1e; border-radius:8px; overflow:hidden; font-size:13px; align-self:flex-start;">
              <thead>
                <tr style="background:#2a2a2a;">
                  <th colspan="2" style="padding:8px 12px; color:#888888; font-weight:normal; text-align:left;">파라미터</th>
                </tr>
              </thead>
              <tbody>
                <tr><td style="padding:5px 16px 5px 12px; color:#aaaaaa;">ADX 기간</td>
                    <td style="padding:5px 12px 5px 0; color:#eeeeee; text-align:right; font-weight:bold;">{adx_period} 봉</td></tr>
                <tr><td style="padding:5px 16px 5px 12px; color:#aaaaaa;">ADX 임계값</td>
                    <td style="padding:5px 12px 5px 0; color:#eeeeee; text-align:right; font-weight:bold;">{adx_threshold}</td></tr>
                <tr style="background:#2a2a2a;"><td colspan="2" style="padding:6px 12px; color:#888888; font-size:11px;"> </td></tr>
                <tr><td style="padding:5px 16px 5px 12px; color:#aaaaaa;">R² 기간</td>
                    <td style="padding:5px 12px 5px 0; color:#eeeeee; text-align:right; font-weight:bold;">{r2_period} 봉</td></tr>
                <tr><td style="padding:5px 16px 5px 12px; color:#aaaaaa;">R² 임계값</td>
                    <td style="padding:5px 12px 5px 0; color:#eeeeee; text-align:right; font-weight:bold;">{r2_threshold}</td></tr>
              </tbody>
            </table>

          </div>
        </div>
        """

    full_html = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>백테스트 결과</title>
  <style>
    body {{ background-color: #111111; color: #eeeeee; margin: 0; padding: 16px; }}
    tr:hover {{ background: #2e2e2e !important; }}
  </style>
</head>
<body>
  <div style="display:flex; flex-wrap:wrap; align-items:flex-start;">
    {table_html}
    {regime_table_html}
  </div>
  {chart_html}
</body>
</html>"""

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(full_html)
    print(f"차트 저장 완료: {output_path}")
