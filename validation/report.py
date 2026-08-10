"""
report.py
---------
각 fold의 결과(dict)를 모아서 cv_summary.csv를 생성한다.
Fold별 행 + 마지막에 Mean/Std 행을 추가한다.
"""

from pathlib import Path
from typing import Dict, List, Union

import numpy as np
import pandas as pd


def build_cv_summary(fold_results: List[Dict[str, Union[int, float]]]) -> pd.DataFrame:
    """
    Parameters
    ----------
    fold_results : list of dict
        예: [{"fold": 0, "MAE": 0.34, "RMSE": 0.54, "R2": 0.70}, ...]

    Returns
    -------
    pd.DataFrame
        fold별 행 + 마지막에 "mean", "std" 행이 추가된 요약 테이블
    """
    df = pd.DataFrame(fold_results)
    metric_cols = [c for c in df.columns if c != "fold"]

    mean_row = {"fold": "mean", **{c: df[c].mean() for c in metric_cols}}
    std_row = {"fold": "std", **{c: df[c].std() for c in metric_cols}}

    summary = pd.concat([df, pd.DataFrame([mean_row, std_row])], ignore_index=True)
    return summary


def save_cv_summary(
    fold_results: List[Dict[str, Union[int, float]]], output_path: Union[str, Path]
) -> pd.DataFrame:
    """cv_summary.csv로 저장하고, 생성된 DataFrame도 함께 반환."""
    summary = build_cv_summary(fold_results)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_path, index=False)
    return summary
