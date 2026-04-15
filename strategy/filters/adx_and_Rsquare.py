"""
ADX + R² 혼합 국면 분류기

핵심 아이디어:
  - ADX와 R² 두 지표의 국면 판단을 조합하여 9가지 케이스로 명시적 분류
  - 정량 분석 결과: ADX가 상승/하락 탐지 모두 R²보다 약 1~2일 빠름
                   R²는 '확인' 역할에 강점 (노이즈 없이 일관된 판단)

9가지 케이스 테이블:
  (ADX 국면, R² 국면) → 액션
  ──────────────────────────────────────────────
  ADX 상승 + R² 상승  → 롱  (두 지표 동의)
  ADX 상승 + R² 횡보  → 롱  (ADX가 빠르므로 ADX 우선)
  ADX 상승 + R² 하락  → 롱  (충돌 케이스, 전체 0.7%, ADX 우선)
  ADX 횡보 + R² 상승  → 현금 (확신 없음, 관망)
  ADX 횡보 + R² 횡보  → 반대매매 (둘 다 횡보 동의)
  ADX 횡보 + R² 하락  → 현금 (하락 신호 불충분, 관망)
  ADX 하락 + R² 상승  → 현금 (충돌 케이스, 전체 1.1%, 보수적)
  ADX 하락 + R² 횡보  → 현금 (하락 신호 불충분, 관망)
  ADX 하락 + R² 하락  → 숏  (두 지표 동의)

Lookahead bias:
  - ADXFilter, R2Filter 모두 내부적으로 shift(1) 적용됨
  - 이 클래스는 두 필터의 결과를 조합할 뿐, 추가 shift 불필요

인터페이스 (ADXFilter / R2Filter와 동일):
  - get_combined_action() → 4가지 액션 반환 (이 클래스 고유 메서드)
  - is_trending()         → LONG/SHORT 구간 여부
  - is_counter_ranging()  → COUNTER 구간 여부 (반대매매 구간 식별용)
  - get_regime()          → 시각화 호환 (-1 / 0 / 1)
"""

import numpy as np
import pandas as pd
from strategy.filters.adx import ADXFilter
from strategy.filters.r2_filter import R2Filter


# ════════════════════════════════════════════════════════════════
#  액션 상수 정의
#    - get_combined_action()의 반환값으로 사용됩니다
#    - 전략 코드에서 action == LONG 처럼 비교할 때 사용하세요
# ════════════════════════════════════════════════════════════════

LONG    =  1   # 롱 포지션 유지
SHORT   = -1   # 숏 포지션 유지
CASH    =  0   # 현금 보유 (현재 포지션 청산 후 현금으로 대기)
COUNTER =  2   # 반대매매 (현재 포지션의 반대 방향으로 진입)
HOLD    =  3   # 현재 포지션 그대로 유지 (아무것도 하지 않음)


# ════════════════════════════════════════════════════════════════
#  ADX 파라미터  ← 여기서 수정하세요
# ════════════════════════════════════════════════════════════════

ADX_PERIOD    = 12    # ADX 계산 기간 (봉 수, 기본 14봉)
                      # 줄이면 → 빠른 반응 / 늘리면 → 안정적
ADX_THRESHOLD = 25    # 추세/횡보 경계값 (기본 25)
                      # 높이면 → 더 강한 추세만 인정 / 낮추면 → 더 많은 구간을 추세로 인정


# ════════════════════════════════════════════════════════════════
#  R² 파라미터  ← 여기서 수정하세요
# ════════════════════════════════════════════════════════════════

R2_PERIOD     = 40    # 선형회귀 기간 (봉 수, 기본 40봉 ≈ 6.7일)
                      # 줄이면 → 빠른 반응, 노이즈 많음 / 늘리면 → 느리지만 안정적
R2_THRESHOLD  = 0.55  # R² 임계값 (기본 0.55, 범위 0.0 ~ 1.0)
                      # 높이면 → 더 확실한 추세만 인정 / 낮추면 → 더 많은 구간을 추세로 인정


# ════════════════════════════════════════════════════════════════
#  9가지 케이스 테이블  ← 여기서 각 케이스의 액션을 수정하세요
#
#  형식: (ADX 국면, R² 국면): 액션
#
#  ADX/R² 국면 값:
#      1  = 상승 추세
#      0  = 횡보
#     -1  = 하락 추세
#
#  액션 값 (위의 상수 사용):
#      LONG    = 롱 포지션
#      SHORT   = 숏 포지션
#      CASH    = 현금 (포지션 없음)
#      COUNTER = 반대매매
#
#  ※ 이 테이블을 수정하면 전략 동작이 즉시 바뀝니다
#    나머지 코드는 건드릴 필요가 없습니다
# ════════════════════════════════════════════════════════════════

REGIME_TABLE: dict[tuple[int, int], int] = {

    # ── ADX 상승 구간 (3가지) ──────────────────────────────────
    ( 1,  1): LONG,     # ADX 상승 + R² 상승  → 롱 (두 지표 모두 상승 동의)
    ( 1,  0): LONG,     # ADX 상승 + R² 횡보  → 롱 (ADX 탐지가 빠르므로 ADX 우선)
    ( 1, -1): LONG,     # ADX 상승 + R² 하락  → 롱 (충돌 케이스, 전체 0.7%, ADX 우선)

    # ── ADX 횡보 구간 (3가지) ──────────────────────────────────
    ( 0,  1): HOLD,     # ADX 횡보 + R² 상승  → 포지션 유지 (방향 확신 없음)
    ( 0,  0): COUNTER  ,  # ADX 횡보 + R² 횡보  → 반대매매 (둘 다 횡보 동의)
    ( 0, -1): HOLD,     # ADX 횡보 + R² 하락  → 포지션 유지 (하락 신호 불충분)

    # ── ADX 하락 구간 (3가지) ──────────────────────────────────
    (-1,  1): CASH,     # ADX 하락 + R² 상승  → 청산 후 현금 (충돌 케이스, 보수적)
    (-1,  0): SHORT,     # ADX 하락 + R² 횡보  → 숏 (하락 신호 불충분, 보수적)
    (-1, -1): SHORT,    # ADX 하락 + R² 하락  → 숏 (두 지표 모두 하락 동의)

}

# ════════════════════════════════════════════════════════════════


class ADXandR2Filter:
    """
    ADX + R² 혼합 국면 분류기

    사용 예시:
        combined = ADXandR2Filter()
        action   = combined.get_combined_action(df)  # 4가지 액션 반환
    """

    def __init__(
        self,
        adx_period: int   = ADX_PERIOD,
        adx_threshold: float = ADX_THRESHOLD,
        r2_period: int    = R2_PERIOD,
        r2_threshold: float = R2_THRESHOLD,
    ):
        """
        Args:
            adx_period:    ADX 계산 기간 (봉 수)
            adx_threshold: ADX 추세/횡보 경계값
            r2_period:     R² 선형회귀 기간 (봉 수)
            r2_threshold:  R² 추세/횡보 경계값
        """
        self.adx_filter = ADXFilter(period=adx_period, threshold=adx_threshold)
        self.r2_filter  = R2Filter(period=r2_period, r2_threshold=r2_threshold)

    def get_combined_action(self, df: pd.DataFrame) -> pd.Series:
        """
        REGIME_TABLE을 참조하여 각 봉의 액션을 반환합니다.

        [동작 방식]
        1. ADXFilter와 R2Filter에서 각각 국면값(-1/0/1)을 계산
        2. 두 국면값의 조합을 REGIME_TABLE에서 조회
        3. 해당 조합에 맞는 액션(LONG/SHORT/CASH/COUNTER)을 반환

        Returns:
            pd.Series (int8):
                LONG    ( 1): 롱 포지션 취할 것
                SHORT   (-1): 숏 포지션 취할 것
                CASH    ( 0): 현금 보유
                COUNTER ( 2): 반대매매 (현재 포지션의 반대)
        """
        # ADXFilter와 R2Filter는 내부적으로 shift(1) 처리됨
        adx_regime = self.adx_filter.get_regime(df).to_numpy(dtype=np.int8)
        r2_regime  = self.r2_filter.get_regime(df).to_numpy(dtype=np.int8)

        # 각 봉마다 (ADX국면, R²국면) 조합을 REGIME_TABLE에서 조회
        # 테이블에 없는 조합은 CASH(0)로 처리 (안전 기본값)
        action_values = np.array(
            [REGIME_TABLE.get((int(a), int(r)), CASH)
             for a, r in zip(adx_regime, r2_regime)],
            dtype=np.int8
        )

        return pd.Series(action_values, index=df.index, dtype='int8')

    def is_trending(self, df: pd.DataFrame) -> np.ndarray:
        """
        액션이 LONG 또는 SHORT인 봉을 True로 반환합니다.
        donchian_breakout.py의 is_trending() 역할을 대체합니다.

        Returns:
            np.ndarray (bool):
                True  → LONG 또는 SHORT (추세 진입 조건 충족)
                False → CASH 또는 COUNTER (추세 진입 안 함)
        """
        action = self.get_combined_action(df)
        return ((action == LONG) | (action == SHORT)).to_numpy(dtype=bool)

    def is_cash_out(self, df: pd.DataFrame) -> np.ndarray:
        """
        액션이 CASH인 봉을 True로 반환합니다.
        CASH 구간에서는 현재 포지션을 청산하고 현금으로 대기합니다.

        Returns:
            np.ndarray (bool):
                True  → CASH (포지션 청산 후 현금 보유)
                False → 나머지 모든 구간
        """
        action = self.get_combined_action(df)
        return (action == CASH).to_numpy(dtype=bool)

    def is_counter_ranging(self, df: pd.DataFrame) -> np.ndarray:
        """
        액션이 COUNTER인 봉을 True로 반환합니다.
        반대매매 구간(COUNTER)과 현금 구간(CASH)을 구분하기 위해 사용합니다.

        전략 코드에서 아래처럼 사용하세요:
            is_counter = regime_filter.is_counter_ranging(df)
            if not is_trending[i] and is_counter[i]:
                signals[i] = position * -1  # 반대매매

        Returns:
            np.ndarray (bool):
                True  → COUNTER (ADX 횡보 + R² 횡보, 반대매매 구간)
                False → 나머지 모든 구간
        """
        action = self.get_combined_action(df)
        return (action == COUNTER).to_numpy(dtype=bool)

    def get_regime(self, df: pd.DataFrame) -> pd.Series:
        """
        시각화(visualizer.py) 및 기존 코드 호환용 메서드.
        COUNTER(2)는 횡보(0)으로 변환하여 반환합니다.

        [변환 규칙]
            LONG    ( 1) → 상승( 1)
            SHORT   (-1) → 하락(-1)
            CASH    ( 0) → 횡보( 0)
            COUNTER ( 2) → 횡보( 0) ← 시각화 호환을 위해 0으로 변환

        Returns:
            pd.Series (int8):
                 1 → 상승 추세
                 0 → 횡보 / 현금 / 반대매매
                -1 → 하락 추세
        """
        action = self.get_combined_action(df)

        # COUNTER(2), HOLD(3)는 횡보(0)으로 매핑
        regime = action.copy()
        regime[regime == COUNTER] = 0
        regime[regime == HOLD]    = 0

        return regime.astype('int8')
