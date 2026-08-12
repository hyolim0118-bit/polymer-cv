"""ensemble_common.py

STEP11 앙상블 관련 스크립트(oof_evaluation.py, test_ensemble.py)가 공통으로
쓰는 함수 모음. 이 파일 자체는 실행하지 않는다.
"""

import copy
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from datasets.dataset import LABEL_ORDER, PolymerDataset, compute_fold_stats, load_config
from datasets.transforms import get_test_transform, get_valid_transform
from models.factory import build_model

spec = importlib.util.spec_from_file_location(
    "cross_validation", "training/training real/cross_validation.py"
)
cross_validation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cross_validation)

N_SPLITS = 5
CHECKPOINT_DIR = Path("results/step8_hpo/confirmation_checkpoints")
FOLD_CHECKPOINT_TEMPLATE = "fold{fold_id}_best.pth"  # fold_id: 1~5

# STEP7(backbone)/STEP9.4(loss) 결과로 확정된 값 — 이 스크립트들은 항상 이 조합으로 동작
FIXED_OVERRIDES = {
    "model": {"name": "resnet18"},
    "loss": {
        "element_loss": {"name": "huber", "params": {"delta": 1.0}},
        "weighting": {"name": "uniform"},
        "reduction": "mean",
    },
}


def build_final_config():
    """STEP8 Confirmation 때와 동일한 config (backbone=resnet18, loss=huber,
    HPO best 하이퍼파라미터 포함)를 재구성."""
    base_config = load_config(Path("training/config.yaml"))
    config = copy.deepcopy(base_config)
    for key, value in FIXED_OVERRIDES.items():
        if isinstance(config.get(key), dict) and isinstance(value, dict):
            config[key].update(value)
        else:
            config[key] = value

    with open("results/step8_hpo/round2/best_params.json") as f:
        best_params = json.load(f)
    config["training"]["learning_rate"] = best_params["learning_rate"]
    config["training"]["weight_decay"] = best_params["weight_decay"]
    config["model"]["dropout"] = best_params["dropout"]
    config["dataloader"]["batch_size"] = best_params["batch_size"]
    return config


def get_fold_splits(pool_df):
    """cross_validation.py의 run_kfold()와 완전히 동일한 fold 분할을 재현.

    fold_id는 1부터 시작 (run_kfold의 enumerate(..., start=1)과 동일하게 맞춤
    -> checkpoint 파일명(fold{N}_best.pth)과 fold_id가 정확히 대응됨).
    """
    kfold = cross_validation.GroupKFold(n_splits=N_SPLITS)
    return list(
        enumerate(kfold.split(pool_df, groups=pool_df["canonical_smiles"]), start=1)
    )


def load_fold_model(fold_id, config, device):
    """fold_id(1~5)에 해당하는 학습된 모델을 불러온다."""
    ckpt_path = CHECKPOINT_DIR / FOLD_CHECKPOINT_TEMPLATE.format(fold_id=fold_id)
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"체크포인트를 찾을 수 없습니다: {ckpt_path}\n"
            f"CHECKPOINT_DIR/FOLD_CHECKPOINT_TEMPLATE가 실제 저장 경로/파일명과 "
            f"맞는지 확인하세요 (ls {CHECKPOINT_DIR} 로 실제 파일명 확인)."
        )
    model = build_model(
        model_name=config["model"]["name"],
        pretrained=False,
        dropout=config["model"]["dropout"],
    )
    checkpoint = torch.load(ckpt_path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def predict_normalized(model, loader, device):
    """모델의 정규화된(z-score) 예측값을 (N, 5) numpy 배열로 반환."""
    all_preds = []
    for images, _, _ in loader:
        images = images.to(device)
        preds = model(images)
        all_preds.append(preds.cpu().numpy())
    return np.concatenate(all_preds, axis=0)


def denormalize(preds_norm, label_mean, label_std, label_columns=LABEL_ORDER):
    """(N, 5) 정규화 예측값을 실제 물성 단위로 역정규화."""
    mean = np.array([label_mean[c] for c in label_columns])
    std = np.array([label_std[c] for c in label_columns])
    return preds_norm * std + mean


def masked_metrics(y_true, y_pred, mask, label_columns=LABEL_ORDER):
    """property별 masked MAE/RMSE/R2 + 전체 평균."""
    results = {}
    for i, col in enumerate(label_columns):
        m = mask[:, i].astype(bool)
        if m.sum() == 0:
            continue
        yt, yp = y_true[m, i], y_pred[m, i]
        mae = np.mean(np.abs(yt - yp))
        rmse = np.sqrt(np.mean((yt - yp) ** 2))
        ss_res = np.sum((yt - yp) ** 2)
        ss_tot = np.sum((yt - yt.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        results[col] = {"MAE": mae, "RMSE": rmse, "R2": r2, "n": int(m.sum())}

    overall = {
        "MAE": np.mean([v["MAE"] for v in results.values()]),
        "RMSE": np.mean([v["RMSE"] for v in results.values()]),
        "R2": np.mean([v["R2"] for v in results.values()]),
    }
    return results, overall


def build_valid_dataset(fold_train_df, fold_valid_df, config):
    """fold의 train 쪽으로 label_mean/std를 계산해, valid 쪽 PolymerDataset을 만든다.
    (label_mean, label_std)도 같이 반환 (역정규화에 필요)."""
    image_cfg = config["image"]
    data_cfg = config["data"]
    label_mean, label_std = compute_fold_stats(fold_train_df, LABEL_ORDER)
    transform = get_valid_transform(
        image_cfg["size"], image_cfg["normalize_mean"], image_cfg["normalize_std"]
    )
    dataset = PolymerDataset(fold_valid_df, data_cfg["image_root"], transform, label_mean, label_std)
    return dataset, label_mean, label_std


def build_test_dataset(fold_train_df, test_df, config):
    """fold의 train 쪽으로 label_mean/std를 계산해, test PolymerDataset을 만든다."""
    image_cfg = config["image"]
    data_cfg = config["data"]
    label_mean, label_std = compute_fold_stats(fold_train_df, LABEL_ORDER)
    transform = get_test_transform(
        image_cfg["size"], image_cfg["normalize_mean"], image_cfg["normalize_std"]
    )
    dataset = PolymerDataset(test_df, data_cfg["image_root"], transform, label_mean, label_std)
    return dataset, label_mean, label_std


def load_pool_and_metadata(config):
    full_metadata = pd.read_csv(config["data"]["metadata_path"])
    pool_df = full_metadata[full_metadata["split"] == "train"].reset_index(drop=True)
    return pool_df, full_metadata
