"""
run_cv.py — 사용 예시
---------------------
기존 train.py / validate.py / compute_fold_stats를 그대로 두고,
얇은 wrapper 두 개(run_training, run_prediction)만 작성해서
ValidationManager에 넘겨주는 방식.

* 아래 import 경로(training.train, training.validate, data.dataset)는
  실제 repo 구조에 맞게 한 줄만 수정하면 됩니다.
"""

import pandas as pd

from validation import ValidationManager, GroupKFoldStrategy

# --- 기존 STEP3/STEP5/STEP6 코드 재사용 (경로는 실제 repo에 맞게 수정) ---
from data.dataset import load_split_dataframes, compute_fold_stats  # STEP3
from training.train import train_one_fold                            # STEP5 (기존 train.py 래핑)
from training.validate import predict_one_fold                       # 기존 validate.py 래핑


PROPERTIES = ["Tg", "Density", "FFV", "Tc", "Rg"]
# raw SMILES가 아니라 canonical_smiles를 그룹 키로 쓴다 — 방향환 표기 순서 등이
# 달라 문자열은 다르지만 같은 분자인 경우를 raw SMILES 기준으로는 못 걸러내기
# 때문 (training/training real/cross_validation.py와 동일한 정책).
GROUP_COL = "canonical_smiles"


def run_training(fold_id, train_idx, val_idx, fold_stats, fold_dir):
    """
    기존 train.py의 학습 함수를 그대로 호출.
    trainer 코드 자체는 수정하지 않고, 여기서 fold별 인자만 넘겨준다.
    """
    checkpoint_path = train_one_fold(
        train_idx=train_idx,
        val_idx=val_idx,
        label_mean=fold_stats.get("label_mean"),
        label_std=fold_stats.get("label_std"),
        save_dir=fold_dir,
    )
    return checkpoint_path


def run_prediction(checkpoint_path, val_idx, fold_stats):
    """기존 validate.py의 역정규화 로직을 그대로 호출."""
    y_true, y_pred = predict_one_fold(
        checkpoint_path=checkpoint_path,
        val_idx=val_idx,
        label_mean=fold_stats.get("label_mean"),
        label_std=fold_stats.get("label_std"),
    )
    return y_true, y_pred


if __name__ == "__main__":
    df = load_split_dataframes()  # STEP3 기존 함수 그대로

    manager = ValidationManager(
        df=df,
        properties=PROPERTIES,
        group_col=GROUP_COL,
        split_strategy=GroupKFoldStrategy(n_splits=5, random_state=42),
        output_dir="outputs/step6_5_cv",
        compute_fold_stats_fn=compute_fold_stats,  # STEP3 기존 함수 그대로
    )

    cv_summary = manager.run(train_fn=run_training, predict_fn=run_prediction)
    print(cv_summary)
