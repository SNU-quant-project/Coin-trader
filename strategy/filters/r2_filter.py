"""
선형회귀 R² (결정계수) 기반 시장 국면 판단 필터

핵심 아이디어:
  - 최근 N봉 종가에 직선(선형회귀)을 피팅했을 때 얼마나 잘 맞는가를 R²로 측정
  - R² ≈ 1.0 → 강한 추세 (가격이 직선에 가깝게 움직임)
  - R² ≈ 0.0 → 횡보장  (가격이 직선과 무관하게 지그재그로 움직임)
  - 기울기 방향(양수/음수)으로 상승 추세 / 하락 추세를 구분

수학적 근거:
  - 단순 선형회귀에서 R² = 상관계수(corr)²
  - pandas의 rolling().corr()로 효율적으로 계산 가능
  - 상관계수의 부호 = 회귀선의 기울기 방향

ADXFilter와 동일한 인터페이스:
  - compute()     → R² 값 반환
  - is_trending() → 추세장 여부 (bool 배열)
  - get_regime()  → 상승(1) / 횡보(0) / 하락(-1) 분류

Lookahead bias 방지:
  - 모든 계산에 shift(1) 적용
  - i번째 봉의 국면 판단에 i-1번째 봉까지의 데이터만 사용

파라미터:
  - period:        회귀 계산 기간 (봉 수, 기본 40봉)
                   4시간봉 기준: 20봉 ≈ 3.3일 / 40봉 ≈ 6.7일 / 60봉 ≈ 10일
  - r2_threshold:  추세/횡보 경계값 (기본 0.65, 범위: 0.0 ~ 1.0)
                   높을수록 더 확실한 추세만 인정 (엄격)
                   낮을수록 더 많은 구간을 추세로 인정 (느슨)
"""

import numpy as np
import pandas as pd


# ════════════════════════════════════════════════════════════════
#  파라미터 설정  ← 여기서 수정하세요
# ════════════════════════════════════════════════════════════════

R2_PERIOD     = 40    # 선형회귀를 피팅할 봉 수 (4시간봉 기준 40봉 ≈ 약 6.7일)
                      # 추천 범위: 20 ~ 60봉
                      # 줄이면 → 반응 빠름, 노이즈 많음
                      # 늘리면 → 반응 느림, 안정적

R2_THRESHOLD  = 0.65  # 추세/횡보 경계값 (0.0 ~ 1.0)
                      # 추천 범위: 0.5 ~ 0.8
                      # 높이면 → 더 확실한 추세만 진입 (거래 횟수 감소)
                      # 낮추면 → 더 많은 구간에서 진입 (거래 횟수 증가)

# ════════════════════════════════════════════════════════════════


class R2Filter:

    def __init__(self, period: int = R2_PERIOD, r2_threshold: float = R2_THRESHOLD):
        """
        Args:
            period:       선형회귀를 피팅할 봉 수
                          - 너무 짧으면(< 20) 노이즈에 민감해짐
                          - 너무 길면(> 80) 반응이 느려 국면 전환을 늦게 감지
            r2_threshold: 추세/횡보 경계값
                          - R² >= threshold → 추세장으로 판단
                          - R² <  threshold → 횡보장으로 판단
        """
        self.period       = period
        self.r2_threshold = r2_threshold

    def _compute_raw(self, df: pd.DataFrame):
        """
        R²와 기울기 방향을 계산하여 반환 (shift(1) 적용 완료)

        [계산 원리]
        단순 선형회귀(y = a + bx)에서:
          R² = corr(x, y)²       ← x와 y의 상관계수를 제곱하면 R²와 동일
          기울기 방향 = corr의 부호  ← 양수면 상승, 음수면 하락

        따라서 rolling().corr()만으로 R²와 기울기 방향 모두 구할 수 있음

        Returns:
            r2         (pd.Series): 각 봉의 R² 값 (0.0 ~ 1.0), shift(1) 적용됨
            slope_sign (pd.Series): 기울기 방향 (+1 상승 / -1 하락), shift(1) 적용됨
        """
        close = df['close']

        # x = 0, 1, 2, ..., N-1 형태의 단조증가 수열 (시간 축 역할)
        # 상관계수는 x의 스케일에 무관하므로 단순 정수열로 충분
        x = pd.Series(np.arange(len(df), dtype=np.float64), index=df.index)

        # rolling 구간 내 종가(y)와 시간(x)의 상관계수 계산
        rolling_corr = close.rolling(self.period).corr(x)

        # R² = 상관계수²  (0 ~ 1 사이 값, 1에 가까울수록 추세 명확)
        r2 = rolling_corr ** 2

        # 기울기 방향: 상관계수의 부호 (+1 상승 / 0 수평 / -1 하락)
        slope_sign = np.sign(rolling_corr)

        # Lookahead bias 방지: i번째 봉 판단에 i번째 봉 데이터 사용 금지
        return r2.shift(1), slope_sign.shift(1)

    def compute(self, df: pd.DataFrame) -> pd.Series:
        """
        R² 값만 반환 (shift(1) 적용 완료)

        Returns:
            pd.Series: 각 봉의 R² 값 (0.0 ~ 1.0)
                       - 1.0에 가까울수록 강한 추세
                       - 0.0에 가까울수록 횡보
                       - 초반 (period - 1)개 봉은 NaN (워밍업 구간)
        """
        r2, _ = self._compute_raw(df)
        return r2

    def is_trending(self, df: pd.DataFrame) -> np.ndarray:
        """
        각 봉이 추세장인지 횡보장인지 판단

        donchian_breakout.py의 매매 로직에서 ADXFilter 대신 또는 함께 사용

        Returns:
            np.ndarray (bool):
                True  → 추세장 (R² >= r2_threshold)
                False → 횡보장 (R² <  r2_threshold 또는 워밍업 NaN 구간)
        """
        r2, _ = self._compute_raw(df)

        is_trending = r2 >= self.r2_threshold

        # NaN 구간(워밍업)은 안전하게 횡보로 처리 → 불필요한 매매 방지
        is_trending = is_trending.fillna(False)

        return is_trending.to_numpy(dtype=bool)

    def get_regime(self, df: pd.DataFrame) -> pd.Series:
        """
        각 봉의 시장 국면을 상승 / 횡보 / 하락으로 구분

        주로 시각화(visualizer.py)에서 국면 음영 처리에 사용
        ADXFilter.get_regime()과 동일한 반환 형식

        판단 기준:
          R² <  threshold                  → 횡보 (0)
          R² >= threshold, 기울기 양수(+)  → 상승 추세 (1)
          R² >= threshold, 기울기 음수(-)  → 하락 추세 (-1)

        Returns:
            pd.Series (int8):
                 1 → 상승 추세
                 0 → 횡보
                -1 → 하락 추세
                NaN인 워밍업 구간은 0(횡보)으로 처리
        """
        r2, slope_sign = self._compute_raw(df)

        # 기본값: 횡보(0)
        regime = pd.Series(0, index=df.index, dtype='int8')

        is_trend = r2 >= self.r2_threshold

        # 추세 있고 기울기 양수 → 상승
        regime[is_trend & (slope_sign > 0)] = 1

        # 추세 있고 기울기 음수 → 하락
        regime[is_trend & (slope_sign < 0)] = -1

        # 워밍업 구간(NaN)은 횡보로 처리
        regime[r2.isna()] = 0

        return regime
