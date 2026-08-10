"""
metrics.py
----------
기존 validate.py에서 쓰던 property별 masked MAE/RMSE/R² 계산 로직을
재사용 가능한 함수로 분리한 것. validate.py는 이 함수를 import해서 쓰도록
바꾸면 되고, 계산 로직 자체는 바꾸지 않는다.

라벨에 결측치(NaN)가 있는 multi-task 회귀이므로, 각 property마다
결측이 아닌 샘플만 골라 계산한다 (기존 Masked MSE와 동일한 마스킹 원칙).
"""

from typing import Dict, List

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def compute_property_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, properties: List[str]
) -> Dict[str, Dict[str, float]]:
    """
    Parameters
    ----------
    y_true : np.ndarray, shape (N, num_properties)
        결측은 NaN으로 표시된 정답값 (역정규화된 실제 단위)
    y_pred : np.ndarray, shape (N, num_properties)
        모델 예측값 (역정규화된 실제 단위)
    properties : list of str
        property 이름 (컬럼 순서와 일치)

    Returns
    -------
    dict: {property_name: {"MAE": ..., "RMSE": ..., "R2": ..., "n_samples": ...}}
    """
    results: Dict[str, Dict[str, float]] = {}

    for i, prop in enumerate(properties):
        mask = ~np.isnan(y_true[:, i])
        n_samples = int(mask.sum())

        if n_samples == 0:
            results[prop] = {"MAE": np.nan, "RMSE": np.nan, "R2": np.nan, "n_samples": 0}
            continue

        yt = y_true[mask, i]
        yp = y_pred[mask, i]

        mae = mean_absolute_error(yt, yp)
        rmse = float(np.sqrt(mean_squared_error(yt, yp)))
        r2 = r2_score(yt, yp) if n_samples > 1 else np.nan

        results[prop] = {"MAE": mae, "RMSE": rmse, "R2": r2, "n_samples": n_samples}

    return results


def average_across_properties(
    property_metrics: Dict[str, Dict[str, float]]
) -> Dict[str, float]:
    """모든 property의 MAE/RMSE/R²를 단순 평균 (fold 단위 요약용)."""
    maes = [v["MAE"] for v in property_metrics.values() if not np.isnan(v["MAE"])]
    rmses = [v["RMSE"] for v in property_metrics.values() if not np.isnan(v["RMSE"])]
    r2s = [v["R2"] for v in property_metrics.values() if not np.isnan(v["R2"])]

    return {
        "MAE": float(np.mean(maes)) if maes else np.nan,
        "RMSE": float(np.mean(rmses)) if rmses else np.nan,
        "R2": float(np.mean(r2s)) if r2s else np.nan,
    }
