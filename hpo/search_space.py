"""
search_space.py
----------------
- config의 search_space 정의를 읽어 Optuna trial에서 하이퍼파라미터를 샘플링
- STEP6.5의 SplitStrategy(GroupKFoldStrategy)를 그대로 재사용해서
  "search_fold"에 해당하는 fold 하나의 (train_idx, val_idx)만 뽑아낸다.
  -> 새로운 split 로직을 만들지 않고, 기존 GroupKFold 분할과 완전히 동일한
     fold 경계를 사용한다 (요구사항: "No new validation split should be generated").
"""

from typing import Any, Dict, Tuple

import numpy as np
import optuna
import pandas as pd

from validation.strategies.base import SplitStrategy


def sample_hyperparams(
    trial: optuna.Trial, search_space: Dict[str, Any]
) -> Dict[str, Any]:
    """config.yaml의 search_space 정의를 읽어 하이퍼파라미터를 샘플링."""
    params: Dict[str, Any] = {}

    for name, spec in search_space.items():
        spec_type = spec["type"]

        if spec_type == "loguniform":
            params[name] = trial.suggest_float(
                name, float(spec["low"]), float(spec["high"]), log=True
            )
        elif spec_type == "uniform":
            params[name] = trial.suggest_float(name, float(spec["low"]), float(spec["high"]))
        elif spec_type == "categorical":
            params[name] = trial.suggest_categorical(name, spec["choices"])
        elif spec_type == "int":
            params[name] = trial.suggest_int(name, int(spec["low"]), int(spec["high"]))
        else:
            raise ValueError(f"지원하지 않는 search space 타입: {spec_type}")

    return params


def get_fixed_fold_indices(
    df: pd.DataFrame,
    group_col: str,
    split_strategy: SplitStrategy,
    search_fold: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    STEP6.5와 동일한 SplitStrategy(GroupKFoldStrategy)를 그대로 사용해서
    fold 번호가 search_fold인 (train_idx, val_idx)만 골라 반환한다.

    새 split 로직을 만들지 않고, split_strategy.split()이 생성하는
    fold 경계를 그대로 재사용한다.
    """
    for fold_id, train_idx, val_idx in split_strategy.split(df, group_col):
        if fold_id == search_fold:
            return train_idx, val_idx

    raise ValueError(
        f"search_fold={search_fold}이 split_strategy가 생성하는 fold 범위에 없습니다."
    )
