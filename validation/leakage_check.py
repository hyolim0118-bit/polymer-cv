"""
leakage_check.py
-----------------
fold 생성 직후, train/val 사이에 같은 polymer(SMILES)가 겹치는지
자동으로 검사한다. 겹치면 실험을 바로 중단시키는 명시적 에러를 낸다.
"""

from typing import Iterable

import pandas as pd


class LeakageError(Exception):
    """train/val 그룹(SMILES)이 겹칠 때 발생시키는 예외."""


def check_no_leakage(
    df: pd.DataFrame,
    train_idx: Iterable[int],
    val_idx: Iterable[int],
    group_col: str,
    fold_id: int,
) -> None:
    """
    Parameters
    ----------
    df : pd.DataFrame
        전체 학습 데이터
    train_idx, val_idx : array-like
        해당 fold의 train/val 인덱스
    group_col : str
        그룹 키 컬럼 (SMILES 등)
    fold_id : int
        에러 메시지에 표시할 fold 번호

    Raises
    ------
    LeakageError
        train/val 그룹 집합이 겹치는 경우
    """
    train_groups = set(df.iloc[list(train_idx)][group_col])
    val_groups = set(df.iloc[list(val_idx)][group_col])

    overlap = train_groups & val_groups
    if overlap:
        sample = list(overlap)[:5]
        raise LeakageError(
            f"[Fold {fold_id}] data leakage 감지: train/val에 동시에 존재하는 "
            f"polymer group이 {len(overlap)}개 있습니다. (예시: {sample}) "
            f"GroupKFold 분할 또는 데이터 병합 과정을 확인하세요."
        )
