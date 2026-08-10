"""
confirm.py
----------
Confirmation Stage. Round 2에서 나온 best_trial.yaml을 읽어,
STEP6.5 ValidationManager.run()을 그대로 호출해서 5-fold 전체를 재실행한다.
(ValidationManager/GroupKFoldStrategy는 전혀 수정하지 않음 — STEP6.5와 100% 동일 코드 재사용)
"""

from pathlib import Path
from typing import Any, Callable, Dict, Union

import yaml

from validation import ValidationManager, GroupKFoldStrategy


def save_best_trial_yaml(study, output_path: Union[str, Path]) -> Dict[str, Any]:
    """Round 2 study.best_trial을 best_trial.yaml로 저장."""
    best_params = study.best_trial.params
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        yaml.safe_dump(
            {"params": best_params, "val_MAE": study.best_trial.value}, f
        )
    return best_params


def load_best_trial_yaml(path: Union[str, Path]) -> Dict[str, Any]:
    with open(path) as f:
        data = yaml.safe_load(f)
    return data["params"]


def run_confirmation(
    best_trial_yaml_path: Union[str, Path],
    df,
    properties: list,
    group_col: str,
    fixed_params: Dict[str, Any],
    n_splits: int,
    output_dir: Union[str, Path],
    train_fn: Callable,
    predict_fn: Callable,
    compute_fold_stats_fn=None,
):
    """
    best_trial.yaml의 하이퍼파라미터 + fixed_params를 합쳐서,
    STEP6.5 ValidationManager로 5-fold 전체를 재실행하고 cv_summary를 반환.

    train_fn은 STEP6.5의 시그니처(trial 인자 없음)를 그대로 사용한다 —
    Confirmation Stage에서는 pruning이 필요 없으므로 Search Stage의
    train_fn과 동일한 함수를 그대로 재사용해도 되고, trial=None으로
    감싸는 얇은 wrapper 하나만 있으면 된다.
    """
    best_params = load_best_trial_yaml(best_trial_yaml_path)
    config = {**fixed_params, **best_params}

    def train_fn_with_config(fold_id, train_idx, val_idx, fold_stats, fold_dir):
        checkpoint_path, _ = train_fn(
            fold_id, train_idx, val_idx, fold_stats, fold_dir, config, None
        )
        return checkpoint_path

    manager = ValidationManager(
        df=df,
        properties=properties,
        group_col=group_col,
        split_strategy=GroupKFoldStrategy(n_splits=n_splits),
        output_dir=output_dir,
        compute_fold_stats_fn=compute_fold_stats_fn,
    )

    cv_summary = manager.run(train_fn=train_fn_with_config, predict_fn=predict_fn)
    return cv_summary, config
