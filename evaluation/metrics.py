"""Validation Metric 계산: 물성별 MAE, RMSE, R² (mask==0 제외)."""

from __future__ import annotations

import logging
from typing import Dict, List

import numpy as np
from sklearn.metrics import r2_score

logger = logging.getLogger(__name__)


def compute_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
    masks: np.ndarray,
    label_names: List[str],
) -> Dict[str, Dict[str, float]]:
    """물성별 + 평균 MAE/RMSE/R²를 계산한다.

    mask == 0인 위치는 모든 지표 계산에서 제외한다.

    Args:
        predictions: 예측값 배열, shape (N, NUM_TARGETS).
        targets: 정답값 배열, shape (N, NUM_TARGETS).
        masks: 결측 마스크 배열, shape (N, NUM_TARGETS). 1=존재, 0=결측.
        label_names: 물성 이름 리스트 (predictions의 컬럼 순서와 일치해야 함).

    Returns:
        ``{property_name: {"mae":..., "rmse":..., "r2":...}}`` 형태의
        딕셔너리. 마지막에 전체 평균을 담은 ``"average"`` 키가 추가된다.
        유효 샘플이 2개 미만이면 r2는 float("nan")으로 표시된다.
    """
    results: Dict[str, Dict[str, float]] = {}

    for i, name in enumerate(label_names):
        valid = masks[:, i] > 0
        n_valid = int(valid.sum())

        if n_valid == 0:
            logger.warning("'%s' 물성에 유효 라벨이 없어 metric을 건너뜁니다.", name)
            results[name] = {"mae": float("nan"), "rmse": float("nan"), "r2": float("nan")}
            continue

        y_true = targets[valid, i]
        y_pred = predictions[valid, i]

        mae = float(np.mean(np.abs(y_true - y_pred)))
        rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

        if n_valid >= 2 and np.var(y_true) > 0:
            r2 = float(r2_score(y_true, y_pred))
        else:
            r2 = float("nan")

        results[name] = {"mae": mae, "rmse": rmse, "r2": r2}

    # 물성별 평균 (nan은 제외하고 평균)
    for metric_key in ("mae", "rmse", "r2"):
        values = [results[name][metric_key] for name in label_names]
        valid_values = [v for v in values if not np.isnan(v)]
        avg = float(np.mean(valid_values)) if valid_values else float("nan")
        results.setdefault("average", {})[metric_key] = avg

    return results
