"""
Donchian Channel Breakout 전략 + ADX & R² 혼합 국면 필터
[Option B: base_position 분리 + 국면 적합성 검사]

[원본 donchian_adx_r2.py 대비 변경 사항]
  1. position → base_position 으로 변수명 변경 (generate_signals 내부 전체)
     - "추세 로직이 판단한 기반 방향"임을 명확히 표현
  2. COUNTER 종료 시 국면 적합성 검사 블록 추가
     - base_position이 새 국면과 충돌하면 0으로 초기화
     - 새 국면과 일치하면 초기화하지 않고 유지 (Option A와의 차이)

[Option A vs Option B 비교]
  ┌─────────────────────────────────────────────────────────┐
  │ 케이스: COUNTER 종료, base_position=-1(숏), 새 국면=SHORT│
  │  Option A: 무조건 0으로 초기화 → 재진입 지연 발생        │
  │  Option B: 국면 일치 → 초기화 안 함, 숏 유지             │
  ├─────────────────────────────────────────────────────────┤
  │ 케이스: COUNTER 종료, base_position=-1(숏), 새 국면=LONG │
  │  Option A: 무조건 0으로 초기화                           │
  │  Option B: 국면 충돌 → 0으로 초기화 (동일)               │
  └─────────────────────────────────────────────────────────┘

[국면 적합성 판단 기준]
  LONG 국면  (is_trending=True, is_bearish=False):
    base_position >= 0 (롱 또는 중립) → 적합, 유지
    base_position  < 0 (숏)           → 충돌, 0으로 초기화
  SHORT 국면 (is_trending=True, is_bearish=True):
    base_position <= 0 (숏 또는 중립) → 적합, 유지
    base_position  > 0 (롱)           → 충돌, 0으로 초기화
  비추세 국면 (HOLD/CASH/COUNTER):
    항상 0으로 초기화 (방향 불확실)

[국면별 매매 동작]
  ┌──────────────┬──────────────┬──────────────────────────────────────┐
  │ ADX          │ R²           │ 동작                                  │
  ├──────────────┼──────────────┼──────────────────────────────────────┤
  │ 상승          │ 상승/횡보/하락   │ 돈치안 상단 돌파 시 롱 진입           │
  │ 횡보          │ 횡보           │ 반대매매 (현재 포지션 × -1)           │
  │ 횡보          │ 상승 또는 하락   │ 현금 유지 (포지션 변경 없음)          │
  │ 하락          │ 횡보 또는 상승   │ 현금 유지 (포지션 변경 없음)          │
  │ 하락          │ 하락           │ 돈치안 하단 돌파 시 숏 진입           │
  └──────────────┴──────────────┴──────────────────────────────────────┘

look-ahead bias 방지:
  - 채널 계산: shift(1)으로 현재 봉 데이터 제외
  - 국면 판단: ADXandR2Filter 내부적으로 shift(1) 적용
  - 시그널 i는 바 i 종가 기준 → 바 i+1 시가에 체결 (엔진 자동 처리)
"""

import numpy as np
import pandas as pd
from strategy.base import BaseStrategy
from strategy.filters.adx_and_Rsquare import ADXandR2Filter


# ════════════════════════════════════════════════════════════════
#  전략 설정  ← 여기서만 수정하면 됩니다
# ════════════════════════════════════════════════════════════════

USE_SHORT = True   # True  = ADX 하락 + R² 하락 구간에서 숏 진입 허용
                   # False = 롱 온리 (숏 진입 비활성화)

SHORT_ENTRY_ADX_CAP = 40.0
# ADX가 이 값 이상이면 신규 숏 진입을 차단합니다.
#
# 아이디어: ADX가 40 이상으로 과열된 하락장 = 추세 소진 구간
#   → 이미 급락이 상당 부분 진행됐을 가능성이 높음
#   → 이 시점의 신규 숏 진입은 "고점 추격 숏"에 해당 → 손실 위험
#
# 미래 확장 포인트:
#   ADX 과열 구간에서 오히려 롱을 잡는 역추세 진입을 추가할 예정
#   아래 generate_signals()의 adx_overheated 분기를 참고하세요.

# ADX / R² 파라미터는 strategy/filters/adx_and_Rsquare.py 상단에서 수정하세요

COUNTER_LONG_EXIT_COEFF = 0.97
# COUNTER 구간에서 롱 포지션 청산 임계치 계수
# 조건: close < long_exit_ch * 이 값  → 롱 청산
# 1.0       = 기본 채널 그대로
# 1.0 초과  = 임계치가 높아짐 → 채널 이탈 전에 더 일찍 청산 (손실 축소)
# 1.0 미만  = 임계치가 낮아짐 → 채널 이탈 후에 더 늦게 청산 (손실 확대)

COUNTER_SHORT_EXIT_COEFF = 1.03
# COUNTER 구간에서 숏 포지션 청산 임계치 계수
# 조건: close > short_exit_ch * 이 값 → 숏 청산
# 1.0       = 기본 채널 그대로
# 1.0 미만  = 임계치가 낮아짐 → 채널 이탈 전에 더 일찍 청산 (손실 축소)
# 1.0 초과  = 임계치가 높아짐 → 채널 이탈 후에 더 늦게 청산 (손실 확대)

USE_COUNTER_TRADING = True
# True  = 반대매매 활성화 (COUNTER 구간에서 포지션 반전)
# False = 반대매매 비활성화 (COUNTER 구간을 HOLD처럼 처리)

USE_COUNTER_STOP_LOSS = True
# True  = 반대매매 중 손절 기능 활성화
# False = 손절 없이 기존 방식대로 (COUNTER 구간 종료 시 자동 복귀)

COUNTER_STOP_LOSS_PCT = 0.045
# 반대매매 손절 기준 비율 (USE_COUNTER_STOP_LOSS = True일 때만 적용)
# 0.02 = 반대매매 진입가 대비 2% 손실 시 청산 후 원래 포지션으로 복귀
# 손절 후에는 해당 COUNTER 구간이 끝날 때까지 재진입하지 않음
# ════════════════════════════════════════════════════════════════


class DonchianADXR2Strategy(BaseStrategy):

    def __init__(self, entry_period: int = 120, exit_period: int = 60):
        """
        Args:
            entry_period: 진입 채널 기간
                          4시간봉 기준: 120봉 = 20일, 80봉 = 13일
                          - 늘리면 → 더 강한 추세에만 진입 (거래 횟수 감소)
                          - 줄이면 → 더 빨리 진입 (거래 횟수 증가, whipsaw 위험)

            exit_period:  청산 채널 기간
                          4시간봉 기준: 60봉 = 10일, 40봉 = 6일
                          - 늘리면 → 포지션을 더 오래 유지
                          - 줄이면 → 더 빨리 청산 (손실도, 수익도 일찍 끊음)
        """
        self.entry_period             = entry_period
        self.exit_period              = exit_period
        self.use_short                = USE_SHORT
        self.counter_long_exit_coeff  = COUNTER_LONG_EXIT_COEFF
        self.counter_short_exit_coeff = COUNTER_SHORT_EXIT_COEFF
        self.use_counter_trading      = USE_COUNTER_TRADING
        self.use_counter_stop_loss    = USE_COUNTER_STOP_LOSS
        self.counter_stop_loss_pct    = COUNTER_STOP_LOSS_PCT
        self.regime_filter            = ADXandR2Filter()

    def generate_signals(self, df: pd.DataFrame) -> np.ndarray:
        """
        Donchian Channel + ADX & R² 혼합 필터 기반 시그널 생성

        반환: np.ndarray (int8, shape=(N,))
             1 = 롱 보유
             0 = 현금 보유
            -1 = 숏 보유 (USE_SHORT=True이고 ADX+R² 모두 하락일 때만)
        """
        high  = df['high']
        low   = df['low']
        close = df['close']
        n     = len(df)

        # ── 1. 돈치안 채널 계산 ───────────────────────────────────
        # shift(1): 현재 봉의 종가가 채널 계산에 포함되지 않도록 한 봉 뒤로 밀기
        #           → "지금 이 봉이 닫히기 전" 기준으로 채널을 계산하는 효과
        #
        # 롱 진입 채널: 최근 entry_period봉 중 최고가
        # 롱 청산 채널: 최근 exit_period봉 중 최저가
        # 숏 진입 채널: 최근 entry_period봉 중 최저가 (롱과 대칭)
        # 숏 청산 채널: 최근 exit_period봉 중 최고가  (롱과 대칭)
        long_entry_ch  = high.rolling(self.entry_period).max().shift(1)
        long_exit_ch   = low.rolling(self.exit_period).min().shift(1)
        short_entry_ch = low.rolling(self.entry_period).min().shift(1)
        short_exit_ch  = high.rolling(self.exit_period).max().shift(1)

        # pandas Series → numpy 배열 변환 (루프 안에서 빠르게 접근하기 위해)
        close_arr       = close.to_numpy(dtype=np.float64)
        long_entry_arr  = long_entry_ch.to_numpy(dtype=np.float64)
        long_exit_arr   = long_exit_ch.to_numpy(dtype=np.float64)
        short_entry_arr = short_entry_ch.to_numpy(dtype=np.float64)
        short_exit_arr  = short_exit_ch.to_numpy(dtype=np.float64)

        # ── 2. 국면 판단 배열 준비 ────────────────────────────────
        # is_trending : LONG 또는 SHORT 액션인 봉 → True
        # is_counter  : COUNTER 액션인 봉 (ADX횡보 + R²횡보) → True
        # is_cash_out : CASH 액션인 봉 → True (포지션 강제 청산 후 현금 대기)
        # is_bearish  : SHORT 액션인 봉 (ADX하락 + R²하락) → True
        is_trending = self.regime_filter.is_trending(df)
        is_counter  = self.regime_filter.is_counter_ranging(df)
        is_cash_out = self.regime_filter.is_cash_out(df)
        regime      = self.regime_filter.get_regime(df).to_numpy(dtype=np.int8)
        is_bearish  = regime == -1   # SHORT 액션 봉 (ADX하락 + R²하락)

        # ADX 값 배열 (shift(1) 적용 완료 — adx_filter.compute() 내부 처리)
        # SHORT_ENTRY_ADX_CAP 비교 및 미래 역추세 진입 로직에서 사용
        adx_arr = self.regime_filter.adx_filter.compute(df).to_numpy(dtype=np.float64)

        # 리포트 출력용: 각 봉의 ADX국면 / R²국면 / 결합 액션을 인스턴스에 저장
        # 엔진이 거래 체결 시 해당 봉의 국면 정보를 참조할 수 있도록 함
        self._adx_regime_arr      = self.regime_filter.adx_filter.get_regime(df).to_numpy(dtype=np.int8)
        self._r2_regime_arr       = self.regime_filter.r2_filter.get_regime(df).to_numpy(dtype=np.int8)
        self._combined_action_arr = self.regime_filter.get_combined_action(df).to_numpy(dtype=np.int8)

        # ── 3. 시그널 생성 (상태 머신) ────────────────────────────
        #
        # [Option B 핵심 개념]
        # base_position: "추세 로직이 판단한 기반 방향" (-1 / 0 / 1)
        #   - is_trending 블록에서만 갱신됨
        #   - COUNTER 로직은 이 값을 절대 변경하지 않음
        #   - "우리가 추세적으로 어느 방향을 바라보고 있는가"를 나타냄
        #
        # signals[i]: 실제로 포트폴리오에 전달되는 시그널
        #   - 일반 구간: base_position과 동일
        #   - COUNTER 구간: base_position × -1 (반대매매)
        #   - COUNTER 종료 시: 국면 적합성 검사 후 base_position 또는 0

        signals       = np.zeros(n, dtype=np.int8)
        base_position = 0   # 추세 로직 기반 포지션: 0=중립, 1=롱 기반, -1=숏 기반

        # 반대매매(COUNTER) 상태 추적 변수
        counter_trade_active = False  # 현재 반대매매 진행 중 여부
        counter_entry_price  = 0.0    # 반대매매 진입가 (손절 기준점)
        counter_stopped_out  = False  # 이번 COUNTER 구간에서 손절/채널이탈 발생 여부
        prev_is_counter      = False  # 직전 봉이 COUNTER 구간이었는지 (구간 전환 감지용)

        for i in range(n):

            # ── 워밍업 구간 처리 ──────────────────────────────────
            # 채널 계산에 필요한 봉이 아직 충분히 쌓이지 않은 초반 구간
            # NaN(숫자 아님)이 하나라도 있으면 이 봉은 건너뜀
            if (np.isnan(long_entry_arr[i]) or np.isnan(long_exit_arr[i]) or
                    np.isnan(short_entry_arr[i]) or np.isnan(short_exit_arr[i])):
                signals[i] = 0
                continue

            # ── [Option B 수정] COUNTER 종료 감지: 국면 적합성 검사 ──
            #
            # COUNTER가 끝나는 첫 봉에서 base_position과 새 국면의 방향이
            # 충돌하는지 확인한다.
            #
            # 적합성 판단 기준:
            #   LONG 국면 (is_trending=True, is_bearish=False):
            #     base_position >= 0 → 적합 (롱/중립 기반 → LONG 국면에 자연스러움)
            #     base_position  < 0 → 충돌 (숏 기반인데 LONG 국면 → 초기화)
            #   SHORT 국면 (is_trending=True, is_bearish=True):
            #     base_position <= 0 → 적합 (숏/중립 기반 → SHORT 국면에 자연스러움)
            #     base_position  > 0 → 충돌 (롱 기반인데 SHORT 국면 → 초기화)
            #   비추세 국면 (HOLD/CASH/COUNTER):
            #     방향 판단 불가 → 항상 초기화
            #
            # Option A와의 차이:
            #   Option A = 항상 0으로 초기화 (단순, 확실)
            #   Option B = 국면과 방향이 일치하면 유지 (더 정교, 재진입 지연 없음)
            if prev_is_counter and not is_counter[i]:
                regime_compatible = (
                    # LONG 국면이고 base_position이 롱 또는 중립
                    (is_trending[i] and not is_bearish[i] and base_position >= 0) or
                    # SHORT 국면이고 base_position이 숏 또는 중립
                    (is_trending[i] and     is_bearish[i] and base_position <= 0)
                )
                if not regime_compatible:
                    base_position = 0   # 국면 충돌 또는 비추세 → 초기화

            if is_trending[i]:
                # ── 추세장 (LONG 또는 SHORT 액션) ────────────────
                if base_position == 0:
                    # 현재 포지션 없음 → 진입 조건 확인
                    if close_arr[i] > long_entry_arr[i]:
                        # 종가가 entry_period 최고가를 돌파 → 롱 진입
                        base_position = 1
                    elif self.use_short and is_bearish[i] and close_arr[i] < short_entry_arr[i]:
                        # 종가가 entry_period 최저가를 하향 돌파
                        # + ADX하락 + R²하락 (is_bearish) 동시 충족 → 숏 진입 후보

                        # ADX 과열 여부 판단
                        # ADX NaN(워밍업) 구간은 과열 아님으로 처리 (숏 허용)
                        adx_overheated = (
                            not np.isnan(adx_arr[i])
                            and adx_arr[i] >= SHORT_ENTRY_ADX_CAP
                        )

                        if adx_overheated:
                            # ADX 과열 → 신규 숏 진입 차단
                            # 추세 소진 구간에서의 추격 숏은 고점 매도 위험이 있음
                            #
                            # ── [미래 확장 포인트] ─────────────────────────────
                            # ADX 과열 하락장에서 역추세 롱 진입을 추가하려면
                            # 이 블록 안에 아래와 같은 로직을 넣으면 됩니다:
                            #
                            #   USE_CONTRARIAN_LONG = True  (설정 섹션에 추가)
                            #   if USE_CONTRARIAN_LONG and base_position == 0:
                            #       base_position = 1  # 역추세 롱 진입
                            #
                            # ──────────────────────────────────────────────────
                            pass   # 현재는 아무것도 하지 않음 (현금 또는 기존 포지션 유지)
                        else:
                            # ADX 정상 범위 → 숏 진입
                            base_position = -1

                elif base_position == 1:
                    # 현재 롱 기반 → 청산 조건 확인
                    if close_arr[i] < long_exit_arr[i]:
                        # 종가가 exit_period 최저가 아래로 이탈 → 롱 청산
                        base_position = 0

                elif base_position == -1:
                    # 현재 숏 기반 → 청산 조건 확인
                    if close_arr[i] > short_exit_arr[i]:
                        # 종가가 exit_period 최고가 위로 돌파 → 숏 청산
                        base_position = 0

            else:
                # ── 비추세 구간 (HOLD / CASH / COUNTER 액션) ─────
                if is_cash_out[i]:
                    # CASH: 포지션 강제 청산 → 현금 보유
                    base_position = 0

                elif is_counter[i]:
                    # ── COUNTER: ADX 횡보 + R² 횡보 ──────────────────────────
                    # base_position은 이 블록에서 절대 변경하지 않음
                    # (COUNTER는 기반 방향을 바꾸는 게 아니라 일시적으로 반대 포지션을 취하는 것)

                    # ① 새 COUNTER 구간 시작 감지 → 반대매매 상태 초기화
                    #    (직전 봉이 COUNTER가 아니었다면 = 지금이 구간의 첫 번째 봉)
                    if not prev_is_counter:
                        counter_trade_active = False
                        counter_entry_price  = 0.0
                        counter_stopped_out  = False

                    # ② 채널 이탈 청산 (반대매매보다 우선)
                    #    원래 포지션이 채널을 크게 이탈했을 때 강제 청산
                    if base_position == 1 and close_arr[i] < long_exit_arr[i] * self.counter_long_exit_coeff:
                        # 롱 기반 포지션이 청산 채널 × 계수 아래로 이탈 → 롱 강제 청산
                        base_position        = 0
                        counter_trade_active = False
                        counter_stopped_out  = True   # 이 구간은 재진입 없음
                    elif base_position == -1 and close_arr[i] > short_exit_arr[i] * self.counter_short_exit_coeff:
                        # 숏 기반 포지션이 청산 채널 × 계수 위로 이탈 → 숏 강제 청산
                        base_position        = 0
                        counter_trade_active = False
                        counter_stopped_out  = True   # 이 구간은 재진입 없음

                    # ③ 반대매매 로직 (USE_COUNTER_TRADING 스위치로 on/off)
                    elif self.use_counter_trading:

                        if counter_stopped_out:
                            # 이번 COUNTER 구간에서 이미 손절 또는 채널이탈 청산됨
                            # → 구간이 끝날 때까지 재진입하지 않고 HOLD처럼 유지
                            pass

                        elif counter_trade_active:
                            # 반대매매 진행 중 → 손절 조건 확인
                            if self.use_counter_stop_loss and base_position != 0:
                                # is_short_ct: 반대매매가 숏인지 여부
                                #   base_position=1(롱 기반)의 반대매매 = 숏 → True
                                #   base_position=-1(숏 기반)의 반대매매 = 롱 → False
                                is_short_ct = (base_position == 1)
                                sl_hit = (
                                    is_short_ct and
                                    close_arr[i] > counter_entry_price * (1 + self.counter_stop_loss_pct)
                                ) or (
                                    not is_short_ct and
                                    close_arr[i] < counter_entry_price * (1 - self.counter_stop_loss_pct)
                                )
                                if sl_hit:
                                    # 손절 발동: 반대매매 종료, 원래 포지션으로 복귀
                                    # 이 구간에서 재진입하지 않도록 stopped_out 표시
                                    counter_trade_active = False
                                    counter_stopped_out  = True
                                    # fall-through → signals[i] = base_position (기반 포지션으로 복귀)
                                else:
                                    # 손절 미충족 → 반대매매 신호 유지
                                    signals[i] = base_position * -1
                                    prev_is_counter = True
                                    continue
                            else:
                                # USE_COUNTER_STOP_LOSS = False → 손절 없이 반대매매 유지
                                signals[i] = base_position * -1
                                prev_is_counter = True
                                continue

                        else:
                            # 반대매매 신규 진입 (base_position이 있을 때만)
                            if base_position != 0:
                                counter_trade_active = True
                                counter_entry_price  = close_arr[i]
                                signals[i]           = base_position * -1
                                prev_is_counter      = True
                                continue
                            # base_position == 0이면 반대매매할 기반 포지션 없음 → fall-through (신호 0)

                    # USE_COUNTER_TRADING = False → HOLD처럼 동작
                    # (아무것도 하지 않음 = 아래 signals[i] = base_position 이 실행됨)

                # HOLD: 그 외 모든 비추세 구간 → 현재 포지션 그대로 유지
                # (아무것도 하지 않음 = 아래 signals[i] = base_position 이 실행됨)

            signals[i]      = base_position
            prev_is_counter = is_counter[i]  # 다음 봉의 COUNTER 구간 전환 감지를 위해 갱신

        return signals
