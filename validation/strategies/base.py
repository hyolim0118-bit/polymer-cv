"""
SplitStrategy
-------------
fold 분할 방식을 추상화한 인터페이스.
지금은 GroupKFoldStrategy 하나만 구현하지만, 나중에 StratifiedGroupKFold
등으로 교체하더라도 ValidationManager나 trainer 호출부는 전혀 바뀌지 않는다.
(STEP7의 BackboneFactory / BaseBackbone과 동일한 패턴)
"""

from abc import ABC, abstractmethod
from typing import Iterator, Tuple

import pandas as pd
import numpy as np


class SplitStrategy(ABC):
    """모든 fold 분할 전략의 부모 클래스."""

    @abstractmethod
    def split(
        self, df: pd.DataFrame, group_col: str
    ) -> Iterator[Tuple[int, np.ndarray, np.ndarray]]:
        """
        Parameters
        ----------
        df : pd.DataFrame
            전체 학습 데이터 (STEP3 load_split_dataframes 결과)
        group_col : str
            그룹 키 컬럼명 (SMILES 또는 canonical SMILES)

        Yields
        ------
        (fold_id, train_indices, val_indices)
        """
        raise NotImplementedError
