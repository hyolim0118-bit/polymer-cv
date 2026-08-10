"""
visualization.py
-----------------
Optuna 기본 시각화 함수를 감싸는 optional 유틸리티.
config의 hpo.visualization.enabled가 true일 때만 호출하도록 사용측에서 분기.
"""

from pathlib import Path
from typing import Union

import optuna
import optuna.visualization.matplotlib as optuna_viz


def save_optimization_history(study: optuna.Study, save_path: Union[str, Path]) -> None:
    ax = optuna_viz.plot_optimization_history(study)
    fig = ax.get_figure()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")


def save_param_importance(study: optuna.Study, save_path: Union[str, Path]) -> None:
    # trial 수가 너무 적으면 importance 계산이 불안정할 수 있어 최소 trial 수 체크
    if len(study.trials) < 5:
        return
    ax = optuna_viz.plot_param_importances(study)
    fig = ax.get_figure()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
