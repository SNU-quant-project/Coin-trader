"""
전략 베이스 클래스

look-ahead bias 방지 규칙:
- generate_signals()에서 rolling(), shift() 등 과거 데이터만 사용
- df['close'].mean() 처럼 전체 평균 사용 금지
- 시그널 i는 바 i 종가까지의 데이터로 계산 → 바 i+1 시가에 체결
"""

from abc import ABC, abstractmethod
import numpy as np
import pandas as pd


class BaseStrategy(ABC):

    @abstractmethod
    def generate_signals(self, df: pd.DataFrame) -> np.ndarray:
        """
        전체 데이터에 대한 시그널을 벡터화하여 계산합니다.

        반환값: np.ndarray (int8, shape=(N,))
            1  = 롱 진입
           -1  = 숏 진입
            0  = 포지션 없음 (초기 워밍업 구간 등)

        시그널 i는 바 i의 종가 이후 발생한 것으로 처리됩니다.
        실제 체결은 바 i+1의 시가에 이루어집니다.
        """
        pass
