"""
ADX (Average Directional Index) 기반 시장 국면 판단 필터

핵심 아이디어:
  - ADX는 추세의 방향이 아닌 '강도'만 측정하는 지표
  - ADX <  threshold → 횡보장
  - ADX >= threshold, +DI > -DI → 상승 추세
  - ADX >= threshold, -DI > +DI → 하락 추세

Lookahead bias 방지:
  - 모든 계산에 shift(1) 적용
  - i번째 봉의 국면 판단에 i-1번째 봉까지의 데이터만 사용
  - 즉, 오늘 캔들이 닫히기 전에는 오늘 국면을 알 수 없음

파라미터:
  - period:    ADX 계산 기간 (기본 14봉, Wilder 원전 기준)
  - threshold: 추세/횡보 경계값 (기본 25, 일반적으로 통용되는 기준)
"""

import numpy as np
import pandas as pd


class ADXFilter:

    def __init__(self, period: int = 14, threshold: float = 25.0):
        """
        Args:
            period:    ADX 스무딩 기간 (기본 14봉)
            threshold: 추세/횡보 경계값 (기본 25)
                       ADX >= threshold → 추세장
                       ADX <  threshold → 횡보장
        """
        self.period = period
        self.threshold = threshold

    def _compute_raw(self, df: pd.DataFrame):
        """
        ADX, +DI, -DI를 모두 계산하여 반환 (shift(1) 적용 완료)
        내부 전용 메서드 — compute()와 get_regime() 모두 이걸 사용

        Returns:
            (adx, plus_di, minus_di): 각각 pd.Series, shift(1) 적용됨
        """
        high  = df['high']
        low   = df['low']
        close = df['close']

        # ── 1. True Range (TR) 계산 ───────────────────────────────
        # TR = max(당일고가-당일저가, |당일고가-전일종가|, |당일저가-전일종가|)
        # 전일 종가 대비 갭을 반영하기 위해 세 가지 중 최댓값 사용
        prev_close = close.shift(1)
        tr = pd.concat([
            high - low,
            (high - prev_close).abs(),
            (low  - prev_close).abs(),
        ], axis=1).max(axis=1)

        # ── 2. Directional Movement (DM) 계산 ────────────────────
        # +DM: 상승 압력 (오늘 고가가 어제 고가보다 얼마나 높아졌는가)
        # -DM: 하락 압력 (오늘 저가가 어제 저가보다 얼마나 낮아졌는가)
        up   = high - high.shift(1)   # 고가 상승폭
        down = low.shift(1) - low     # 저가 하락폭

        # 조건: 반대 방향보다 클 때만 유효, 음수는 0으로 처리
        plus_dm  = np.where((up > down) & (up > 0),   up,   0.0)
        minus_dm = np.where((down > up) & (down > 0), down, 0.0)

        plus_dm  = pd.Series(plus_dm,  index=df.index)
        minus_dm = pd.Series(minus_dm, index=df.index)

        # ── 3. Wilder 스무딩 (지수 이동평균 방식) ─────────────────
        # Wilder는 단순 이동평균 대신 자신만의 스무딩 방식을 사용
        # pandas ewm으로 근사: alpha = 1/period
        alpha = 1.0 / self.period

        atr      = tr.ewm(alpha=alpha, adjust=False).mean()
        plus_di  = 100 * plus_dm.ewm(alpha=alpha, adjust=False).mean() / atr
        minus_di = 100 * minus_dm.ewm(alpha=alpha, adjust=False).mean() / atr

        # ── 4. DX → ADX 계산 ─────────────────────────────────────
        # DX  = |+DI - (-DI)| / |+DI + (-DI)| * 100
        # ADX = DX의 Wilder 스무딩
        dx_denom = (plus_di + minus_di).replace(0, np.nan)  # 0으로 나누기 방지
        dx  = 100 * (plus_di - minus_di).abs() / dx_denom
        adx = dx.ewm(alpha=alpha, adjust=False).mean()

        # ── 5. Lookahead bias 방지: shift(1) 적용 ─────────────────
        # i번째 봉의 매매 결정에 i번째 봉 데이터가 쓰이지 않도록
        # 한 봉 뒤로 밀어서 반환
        return adx.shift(1), plus_di.shift(1), minus_di.shift(1)

    def compute(self, df: pd.DataFrame) -> pd.Series:
        """
        ADX 값만 반환 (shift(1) 적용 완료)

        Returns:
            pd.Series: 각 봉의 ADX 값 (0~100)
                       - i번째 값은 i-1번째 봉까지의 데이터로 계산됨
                       - 워밍업 구간(초반 period*2 봉)은 NaN
        """
        adx, _, _ = self._compute_raw(df)
        return adx

    def is_trending(self, df: pd.DataFrame) -> np.ndarray:
        """
        각 봉이 추세장인지 횡보장인지 판단하여 반환
        donchian_breakout.py의 매매 로직에서 사용

        Returns:
            np.ndarray (bool):
                True  → 추세장 (ADX >= threshold)
                False → 횡보장 (ADX <  threshold 또는 NaN 워밍업 구간)
        """
        adx = self.compute(df)

        # NaN(워밍업 구간)은 안전하게 횡보로 처리
        # → 데이터 부족 구간에서 불필요한 매매 방지
        is_trending = adx >= self.threshold
        is_trending = is_trending.fillna(False)

        return is_trending.to_numpy(dtype=bool)

    def get_regime(self, df: pd.DataFrame) -> pd.Series:
        """
        각 봉의 시장 국면을 상승/횡보/하락으로 구분하여 반환
        주로 시각화(visualizer.py)에서 음영 처리에 사용

        판단 기준:
          ADX <  threshold            → 횡보 (0)
          ADX >= threshold, +DI > -DI → 상승 (1)
          ADX >= threshold, -DI > +DI → 하락 (-1)

        Returns:
            pd.Series (int8):
                 1 → 상승 추세
                 0 → 횡보
                -1 → 하락 추세
                NaN인 워밍업 구간은 0(횡보)으로 처리
        """
        adx, plus_di, minus_di = self._compute_raw(df)

        # 기본값: 횡보(0)
        regime = pd.Series(0, index=df.index, dtype='int8')

        # 추세 강도 충분 → 방향으로 구분
        is_trend = adx >= self.threshold

        # 상승: 추세 있고 +DI가 -DI보다 큼
        regime[is_trend & (plus_di > minus_di)] = 1

        # 하락: 추세 있고 -DI가 +DI보다 큼
        regime[is_trend & (minus_di > plus_di)] = -1

        # NaN 구간(워밍업)은 0(횡보)으로 처리
        regime[adx.isna()] = 0

        return regime
