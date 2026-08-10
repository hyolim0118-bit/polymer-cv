"""
search.py
---------
Search Stage 실행기. Round 1 / Round 2 모두 이 함수 하나로 처리한다
(round마다 search_space, n_trials, storage 경로만 다르게 넘기면 됨).

기존 trainer/dataset/GroupKFoldStrategy/BackboneFactory는 전혀 건드리지 않고,
매 trial마다 config만 바꿔서 기존 train_fn/predict_fn(STEP6.5와 동일한
콜백 인터페이스)을 호출한다.
"""

import shutil
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple, Union

import numpy as np
import optuna
import pandas as pd

from validation.strategies.base import SplitStrategy
from validation.leakage_check import check_no_leakage
from validation.metrics import compute_property_metrics, average_across_properties

from .search_space import sample_hyperparams, get_fixed_fold_indices
from .tracking import append_trial_record

# train_fn은 STEP6.5와 동일한 시그니처에 trial(선택)만 추가로 받는다.
# trial이 주어지면, 기존 trainer 내부에서 epoch마다
#     trial.report(val_loss, epoch)
#     if trial.should_prune(): raise optuna.TrialPruned()
# 를 호출하도록 얇게 연결해주면 MedianPruner가 동작한다.
# (trainer 로직 자체는 바뀌지 않고, 이 두 줄만 추가하면 됨)
TrainFn = Callable[
    [int, np.ndarray, np.ndarray, dict, Path, Dict[str, Any], Optional[optuna.Trial]],
    Tuple[Union[str, Path], int],  # (checkpoint_path, best_epoch)
]
PredictFn = Callable[[Union[str, Path], np.ndarray, dict], Tuple[np.ndarray, np.ndarray]]


def run_search_round(
    round_name: str,
    df: pd.DataFrame,
    properties: list,
    group_col: str,
    split_strategy: SplitStrategy,
    search_fold: int,
    search_space: Dict[str, Any],
    fixed_params: Dict[str, Any],
    n_trials: int,
    output_dir: Union[str, Path],
    train_fn: TrainFn,
    predict_fn: PredictFn,
    compute_fold_stats_fn: Optional[Callable] = None,
    storage: Optional[str] = None,
    sampler: Optional[optuna.samplers.BaseSampler] = None,
    pruner: Optional[optuna.pruners.BasePruner] = None,
    save_all_checkpoints: bool = False,
) -> optuna.Study:
    """
    Parameters
    ----------
    round_name : str
        "round1" 또는 "round2" (로그/디렉토리 구분용)
    search_space : dict
        이번 라운드에서 사용할 search space (round2는 좁힌 범위를 그대로 여기 전달)
    n_trials : int
        이번 라운드의 trial 수 (round1: 15~20, round2: 10~20 권장)
    storage : str, optional
        예: "sqlite:///experiments/optuna/round1_study.db" — resume 지원
    save_all_checkpoints : bool, default False
        False면 매 trial의 checkpoint 중 "현재까지 best trial"만 남기고 나머지는 삭제.
        (요구사항: Search Stage 기본 동작은 best trial checkpoint만 보존)

    Returns
    -------
    optuna.Study — round 종료 후 study.best_trial로 최적 하이퍼파라미터 확인 가능
    """
    output_dir = Path(output_dir)
    trials_csv_path = output_dir / "trials.csv"

    # STEP6.5와 완전히 동일한 fold 경계 재사용 (새 split 생성 없음)
    train_idx, val_idx = get_fixed_fold_indices(df, group_col, split_strategy, search_fold)
    check_no_leakage(df, train_idx, val_idx, group_col, search_fold)
    fold_stats = compute_fold_stats_fn(df, train_idx) if compute_fold_stats_fn else {}

    best_trial_number = {"value": None, "mae": float("inf")}  # closure로 추적

    def objective(trial: optuna.Trial) -> float:
        hparams = sample_hyperparams(trial, search_space)
        config_overrides = {**fixed_params, **hparams}

        trial_dir = output_dir / f"trial_{trial.number:03d}"
        trial_dir.mkdir(parents=True, exist_ok=True)

        start_time = time.time()
        checkpoint_path, best_epoch = train_fn(
            search_fold, train_idx, val_idx, fold_stats, trial_dir, config_overrides, trial
        )
        training_time = time.time() - start_time

        y_true, y_pred = predict_fn(checkpoint_path, val_idx, fold_stats)
        property_metrics = compute_property_metrics(y_true, y_pred, properties)
        fold_summary = average_across_properties(property_metrics)

        record = {
            "round": round_name,
            "trial_id": trial.number,
            "fold": search_fold,
            **hparams,
            "val_MAE": fold_summary["MAE"],
            "val_RMSE": fold_summary["RMSE"],
            "val_R2": fold_summary["R2"],
            "training_time_sec": training_time,
            "best_epoch": best_epoch,
        }
        append_trial_record(record, trials_csv_path)

        # checkpoint 정리: 기본은 best trial 것만 보존
        if not save_all_checkpoints:
            if fold_summary["MAE"] < best_trial_number["mae"]:
                # 이전 best의 checkpoint 삭제, 이번 trial을 새 best로
                if best_trial_number["value"] is not None:
                    prev_dir = output_dir / f"trial_{best_trial_number['value']:03d}"
                    ckpt = prev_dir / "checkpoint.pth"
                    if ckpt.exists():
                        ckpt.unlink()
                best_trial_number["value"] = trial.number
                best_trial_number["mae"] = fold_summary["MAE"]
            else:
                # 이번 trial은 best가 아니므로 checkpoint 삭제 (메타데이터는 남김)
                ckpt = trial_dir / "checkpoint.pth"
                if ckpt.exists():
                    ckpt.unlink()

        return fold_summary["MAE"]

    study = optuna.create_study(
        study_name=round_name,
        direction="minimize",
        storage=storage,
        sampler=sampler or optuna.samplers.TPESampler(),
        pruner=pruner or optuna.pruners.MedianPruner(),
        load_if_exists=True,  # storage가 이미 있으면 이어서 진행 (Resume 지원)
    )
    study.optimize(objective, n_trials=n_trials)

    return study
