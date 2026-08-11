"""
GroupKFoldStrategy
-------------------
polymer identity(canonical_smiles) 기준으로 그룹을 나눠 같은 polymer가
train/val에 동시에 들어가지 않도록 fold를 생성한다. group_col은 반드시
canonical_smiles여야 한다 — raw SMILES는 방향환 표기 순서 등으로 같은
분자가 다른 문자열이 될 수 있어 leakage를 놓칠 수 있다.
"""

from typing import Iterator, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from .base import SplitStrategy


class GroupKFoldStrategy(SplitStrategy):
    def __init__(self, n_splits: int = 5, random_state: int = 42):
        # 주의: sklearn의 GroupKFold는 shuffle/random_state를 지원하지 않음
        # (그룹 단위 분할이라 셔플 옵션이 없음). random_state는 로깅 목적으로만 보관.
        self.n_splits = n_splits
        self.random_state = random_state
        self._splitter = GroupKFold(n_splits=n_splits)

    def split(
        self, df: pd.DataFrame, group_col: str
    ) -> Iterator[Tuple[int, np.ndarray, np.ndarray]]:
        groups = df[group_col].values
        X_dummy = np.zeros(len(df))  # GroupKFold는 X를 실제로 쓰지 않음

        for fold_id, (train_idx, val_idx) in enumerate(
            self._splitter.split(X_dummy, groups=groups)
        ):
            yield fold_id, train_idx, val_idx
