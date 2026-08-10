"""
visualization.py
-----------------
논문/발표 자료용 시각화 유틸리티 (선택적으로 사용).
- Fold Size Distribution
- Property Distribution
- Missing Label Distribution
"""

from pathlib import Path
from typing import List, Union

import matplotlib.pyplot as plt
import pandas as pd


def plot_fold_size_distribution(
    fold_metadata: List[dict], save_path: Union[str, Path] = None
):
    """fold별 train/val 샘플 수를 막대그래프로."""
    fig, ax = plt.subplots(figsize=(6, 4))
    fold_ids = [m["fold_id"] for m in fold_metadata]
    train_sizes = [m["train_samples"] for m in fold_metadata]
    val_sizes = [m["val_samples"] for m in fold_metadata]

    width = 0.35
    x = range(len(fold_ids))
    ax.bar([i - width / 2 for i in x], train_sizes, width, label="Train")
    ax.bar([i + width / 2 for i in x], val_sizes, width, label="Validation")
    ax.set_xticks(list(x))
    ax.set_xticklabels([f"Fold {i}" for i in fold_ids])
    ax.set_ylabel("Sample Count")
    ax.set_title("Fold Size Distribution")
    ax.legend()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_property_distribution(
    df: pd.DataFrame, properties: List[str], save_path: Union[str, Path] = None
):
    """전체 데이터의 property별 분포 (히스토그램)."""
    n = len(properties)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 3.5))
    if n == 1:
        axes = [axes]

    for ax, prop in zip(axes, properties):
        df[prop].dropna().hist(ax=ax, bins=30)
        ax.set_title(prop)
        ax.set_xlabel("value")

    fig.suptitle("Property Distribution")
    fig.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_missing_label_distribution(
    df: pd.DataFrame, properties: List[str], save_path: Union[str, Path] = None
):
    """property별 결측 라벨 비율."""
    missing_ratio = [df[prop].isna().mean() for prop in properties]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(properties, missing_ratio)
    ax.set_ylabel("Missing Ratio")
    ax.set_title("Missing Label Distribution")
    ax.set_ylim(0, 1)

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
