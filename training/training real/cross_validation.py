"""STEP6.5 — K-Fold Cross Validation.

기존 STEP6 모듈(train.py/trainer.py/validate.py/loss.py/metrics.py/
scheduler.py)의 내부 로직은 전혀 수정하지 않는다. Trainer를 Fold마다
새로 생성해서 fit()을 반복 호출하는 방식으로만 K-Fold를 구현한다.

Data Leakage 방지: 각 Fold의 정규화 통계(label_mean/label_std)는
반드시 그 Fold의 "train 쪽" 데이터만으로 계산해서, 같은 Fold의
train/valid 양쪽에 동일하게 적용한다 (STEP3.1과 동일한 원칙).
"""

from __future__ import annotations

import csv
import logging
import shutil
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader

from datasets.dataset import LABEL_ORDER, PolymerDataset, compute_fold_stats
from datasets.transforms import get_train_transform, get_valid_transform
from models.factory import build_model
from training.trainer import Trainer

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def _build_fold_loaders(
    train_fold_df: pd.DataFrame, valid_fold_df: pd.DataFrame, config: dict
) -> tuple[DataLoader, DataLoader]:
    """한 Fold의 train/valid dataframe으로 DataLoader 두 개를 만든다.

    정규화 통계는 이 Fold의 train 쪽(train_fold_df)만으로 계산해서
    train/valid DataLoader 양쪽에 동일하게 적용한다 (Data Leakage 방지).

    Args:
        train_fold_df: 이번 Fold에서 train으로 쓸 행들.
        valid_fold_df: 이번 Fold에서 valid로 쓸 행들.
        config: 전체 config 딕셔너리.

    Returns:
        (train_loader, valid_loader) 튜플.
    """
    image_cfg = config["image"]
    aug_cfg = config["augmentation"]
    data_cfg = config["data"]
    loader_cfg = config["dataloader"]

    label_mean, label_std = compute_fold_stats(train_fold_df, LABEL_ORDER)

    train_transform = get_train_transform(
        image_cfg["size"], image_cfg["normalize_mean"], image_cfg["normalize_std"], aug_cfg
    )
    valid_transform = get_valid_transform(
        image_cfg["size"], image_cfg["normalize_mean"], image_cfg["normalize_std"]
    )

    train_dataset = PolymerDataset(
        train_fold_df, data_cfg["image_root"], train_transform, label_mean, label_std
    )
    valid_dataset = PolymerDataset(
        valid_fold_df, data_cfg["image_root"], valid_transform, label_mean, label_std
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=loader_cfg["batch_size"],
        shuffle=True,
        num_workers=loader_cfg["num_workers"],
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=loader_cfg["batch_size"],
        shuffle=False,
        num_workers=loader_cfg["num_workers"],
    )
    return train_loader, valid_loader


def run_kfold(config: dict) -> List[Dict[str, float]]:
    """config["k_fold"] 설정에 따라 K-Fold Cross Validation을 실행한다.

    Fold마다 새로운 Model을 만들고, 그 Model로 새 Trainer를 생성해
    (Optimizer/Scheduler는 Trainer.__init__이 매번 새로 초기화한다)
    trainer.fit()을 그대로 호출한다. Trainer 내부 학습 로직은 건드리지 않는다.

    Args:
        config: 전체 config 딕셔너리 (k_fold, data, model, training 등 포함).

    Returns:
        Fold별 결과 리스트. 각 원소는
        {"fold", "train_loss", "val_loss", "mae", "rmse", "r2"} 딕셔너리.
    """
    data_cfg = config["data"]
    kfold_cfg = config["k_fold"]
    base_checkpoint_dir = Path(config["training"]["checkpoint_dir"])
    base_log_dir = Path(config["training"]["log_dir"])
    base_checkpoint_dir.mkdir(parents=True, exist_ok=True)

    full_metadata = pd.read_csv(data_cfg["metadata_path"])
    # K-Fold는 valid를 미리 떼어두지 않고 "train" split 전체를 K개로 회전시켜 사용한다.
    pool_df = full_metadata[full_metadata["split"] == "train"].reset_index(drop=True)

    kfold = KFold(
        n_splits=kfold_cfg["n_splits"],
        shuffle=kfold_cfg.get("shuffle", True),
        random_state=config["training"]["seed"],
    )

    fold_results: List[Dict[str, float]] = []

    for fold_idx, (train_idx, valid_idx) in enumerate(kfold.split(pool_df), start=1):
        logger.info("========== Fold %d/%d 시작 ==========", fold_idx, kfold_cfg["n_splits"])

        train_fold_df = pool_df.iloc[train_idx].reset_index(drop=True)
        valid_fold_df = pool_df.iloc[valid_idx].reset_index(drop=True)
        train_loader, valid_loader = _build_fold_loaders(train_fold_df, valid_fold_df, config)

        # Fold마다 새로운 Model 생성
        model = build_model(
            model_name=config["model"]["name"],
            pretrained=config["model"]["pretrained"],
            dropout=config["model"]["dropout"],
        )

        # 이 Fold 전용 checkpoint/log 경로로 config를 얕은 복사해서 사용
        # (Trainer 내부는 그대로, 저장 경로만 Fold별로 분리)
        fold_config = dict(config)
        fold_config["training"] = dict(config["training"])
        fold_checkpoint_dir = base_checkpoint_dir / f"_fold{fold_idx}_tmp"
        fold_config["training"]["checkpoint_dir"] = str(fold_checkpoint_dir)
        fold_config["training"]["log_dir"] = str(base_log_dir / f"fold{fold_idx}")

        # Optimizer/Scheduler는 Trainer.__init__이 이 config로 매번 새로 만듦
        trainer = Trainer(
            model=model, train_loader=train_loader, valid_loader=valid_loader, config=fold_config
        )
        trainer.fit()

        # Best epoch(최저 val_loss) 시점의 지표를 이 Fold의 대표 성능으로 사용
        best_idx = int(np.argmin(trainer.history["val_loss"]))
        result = {
            "fold": fold_idx,
            "train_loss": trainer.history["train_loss"][best_idx],
            "val_loss": trainer.history["val_loss"][best_idx],
            "mae": trainer.history["val_mae"][best_idx],
            "rmse": trainer.history["val_rmse"][best_idx],
            "r2": trainer.history["val_r2"][best_idx],
        }
        fold_results.append(result)

        logger.info(
            "Fold %d\n  Train Loss : %.6f\n  Valid Loss : %.6f\n  MAE : %.6f\n  RMSE : %.6f\n  R2 : %.6f",
            fold_idx,
            result["train_loss"],
            result["val_loss"],
            result["mae"],
            result["rmse"],
            result["r2"],
        )

        # 이 Fold의 best checkpoint를 checkpoints/fold{N}_best.pth 로 저장
        fold_best_src = fold_checkpoint_dir / "best_model.pth"
        fold_best_dst = base_checkpoint_dir / f"fold{fold_idx}_best.pth"
        if fold_best_src.exists():
            shutil.copy(fold_best_src, fold_best_dst)
            logger.info("Fold %d Best Model 저장 완료: %s", fold_idx, fold_best_dst)
        shutil.rmtree(fold_checkpoint_dir, ignore_errors=True)

        logger.info("========== Fold %d/%d 종료 ==========", fold_idx, kfold_cfg["n_splits"])

    _save_and_summarize(fold_results)
    return fold_results


def _save_and_summarize(fold_results: List[Dict[str, float]]) -> None:
    """Fold별 결과를 results/kfold_results.csv로 저장하고, 평균/표준편차를 로그로 출력한다.

    Args:
        fold_results: run_kfold()가 만든 fold별 결과 리스트.
    """
    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)
    csv_path = results_dir / "kfold_results.csv"

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Fold", "TrainLoss", "ValLoss", "MAE", "RMSE", "R2"])
        for r in fold_results:
            writer.writerow([r["fold"], r["train_loss"], r["val_loss"], r["mae"], r["rmse"], r["r2"]])

    logger.info("Fold별 결과 CSV 저장 완료: %s", csv_path)

    mae_values = [r["mae"] for r in fold_results]
    rmse_values = [r["rmse"] for r in fold_results]
    r2_values = [r["r2"] for r in fold_results]

    logger.info(
        "\n========== K-Fold Result ==========\n"
        "MAE\n  Mean : %.6f\n  Std  : %.6f\n"
        "RMSE\n  Mean : %.6f\n  Std  : %.6f\n"
        "R2\n  Mean : %.6f\n  Std  : %.6f",
        float(np.mean(mae_values)),
        float(np.std(mae_values)),
        float(np.mean(rmse_values)),
        float(np.std(rmse_values)),
        float(np.mean(r2_values)),
        float(np.std(r2_values)),
    )
