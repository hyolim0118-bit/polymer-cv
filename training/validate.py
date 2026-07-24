"""학습된 모델로 예측을 생성하고 CSV로 저장하는 모듈 (STEP6).

train.py가 학습 종료 후 best checkpoint로 test set 예측을 만들 때 이 모듈의
함수들을 그대로 가져다 쓰고, 이 파일을 단독으로 실행하면 임의의 checkpoint에
대해 다시 예측만 뽑아볼 수도 있다.

STEP3.1 반영: 정규화 역변환(denormalize)에 STEP2의 stats.json 대신,
create_dataloader()와 완전히 동일한 방식(load_split_dataframes)으로 계산한
label_mean/label_std를 사용한다. 이렇게 해야 학습 때 실제로 쓰인 정규화
기준과 추론 시 역정규화 기준이 항상 정확히 일치한다 (Data Leakage 방지 조치와
정합성을 맞추기 위함).

실행 방법 (프로젝트 루트에서, 단독 실행 시):
    python -m training.validate --checkpoint training/checkpoints/best_model.pth
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from datasets.dataset import PolymerDataset, load_config, load_split_dataframes
from datasets.transforms import get_test_transform
from models.factory import build_model
from training.trainer import get_device

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Kaggle 제출 포맷과 동일한 컬럼 순서 (Dataset.LABEL_ORDER와 다름에 주의)
SUBMISSION_COLUMN_ORDER: List[str] = ["id", "Tg", "FFV", "Tc", "Density", "Rg"]


def denormalize(
    values: np.ndarray,
    label_names: List[str],
    label_mean: Dict[str, float],
    label_std: Dict[str, float],
) -> np.ndarray:
    """정규화된 예측값(z-score)을 원래 물성 스케일로 되돌린다.

    Args:
        values: 정규화된 예측값, shape (N, len(label_names)).
        label_names: 컬럼 순서 (Dataset.LABEL_ORDER와 동일해야 함).
        label_mean: 학습 시 사용한 것과 동일한 Train Fold 기준 물성별 평균.
        label_std: 학습 시 사용한 것과 동일한 Train Fold 기준 물성별 표준편차.

    Returns:
        원래 스케일로 변환된 값 배열 (x * std + mean).
    """
    result = values.copy().astype(float)
    for i, name in enumerate(label_names):
        result[:, i] = result[:, i] * label_std[name] + label_mean[name]
    return result


@torch.no_grad()
def generate_predictions(
    model: nn.Module,
    data_loader: DataLoader,
    ids: List[int],
    device: torch.device,
    label_names: List[str],
    label_mean: Dict[str, float],
    label_std: Dict[str, float],
) -> pd.DataFrame:
    """DataLoader 전체에 대해 예측을 생성하고 원래 스케일로 되돌린 DataFrame을 만든다.

    Args:
        model: 추론에 사용할 모델 (이미 checkpoint 가중치가 로드된 상태여야 함).
        data_loader: 예측 대상 DataLoader (shuffle=False여야 ids와 순서가 맞음).
        ids: data_loader와 동일한 순서의 샘플 id 리스트.
        device: 추론에 사용할 device.
        label_names: 예측 컬럼 순서 (Dataset.LABEL_ORDER).
        label_mean: 정규화를 되돌리기 위한, 학습에 쓰인 것과 동일한 Train Fold 평균.
        label_std: 정규화를 되돌리기 위한, 학습에 쓰인 것과 동일한 Train Fold 표준편차.

    Returns:
        Kaggle 제출 포맷(id, Tg, FFV, Tc, Density, Rg)과 동일한 컬럼 순서의 DataFrame.
    """
    model.eval()
    all_preds: List[np.ndarray] = []

    for images, _, _ in data_loader:
        images = images.to(device, non_blocking=True)
        preds = model(images).detach().cpu().numpy()
        all_preds.append(preds)

    predictions = np.concatenate(all_preds, axis=0)
    if len(predictions) != len(ids):
        raise ValueError(
            f"예측 개수({len(predictions)})와 id 개수({len(ids)})가 일치하지 않습니다. "
            "DataLoader의 shuffle=False 여부를 확인하세요."
        )

    predictions = denormalize(predictions, label_names, label_mean, label_std)

    df = pd.DataFrame(predictions, columns=label_names)
    df.insert(0, "id", ids)
    return df[SUBMISSION_COLUMN_ORDER]


def build_test_dataset_and_loader(
    config: dict,
) -> Tuple[PolymerDataset, DataLoader, Dict[str, float], Dict[str, float]]:
    """config로부터 test PolymerDataset/DataLoader와 정규화 통계를 함께 만든다.

    create_dataloader()가 내부에서 하는 것과 동일하게 load_split_dataframes로
    train/valid/test를 나누고 label_mean/label_std를 Train Fold에서만 계산한
    뒤, test 쪽 dataframe으로 Dataset을 만든다. ids를 예측 결과와 순서대로
    맞추려면 Dataset 인스턴스(=metadata)에 직접 접근해야 하므로 이 헬퍼에서
    Dataset을 먼저 만들고 그 위에 DataLoader를 씌운다.
    (PolymerDataset/transforms 로직 자체는 재사용만 하고 수정하지 않음)

    Args:
        config: 전체 config 딕셔너리.

    Returns:
        (test_dataset, test_loader, label_mean, label_std) 튜플.
    """
    img_cfg = config["image"]
    data_cfg = config["data"]
    split_cfg = config["split"]
    loader_cfg = config["dataloader"]

    _, _, test_df, label_mean, label_std = load_split_dataframes(
        metadata_path=Path(data_cfg["metadata_path"]),
        val_ratio=split_cfg["val_ratio"],
        seed=split_cfg["seed"],
    )

    test_transform = get_test_transform(
        img_cfg["size"], img_cfg["normalize_mean"], img_cfg["normalize_std"]
    )
    test_dataset = PolymerDataset(
        dataframe=test_df,
        image_root=Path(data_cfg["image_root"]),
        transform=test_transform,
        label_mean=label_mean,
        label_std=label_std,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=loader_cfg["batch_size"],
        shuffle=False,  # id 순서를 보장하기 위해 반드시 False
        num_workers=loader_cfg["num_workers"],
    )
    return test_dataset, test_loader, label_mean, label_std


def main() -> None:
    """checkpoint를 로드해 test set 예측 CSV를 저장한다 (단독 실행용)."""
    parser = argparse.ArgumentParser(description="저장된 checkpoint로 예측 CSV 생성")
    parser.add_argument("--config", type=Path, default=Path("training/config.yaml"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("reports/predictions.csv"))
    args = parser.parse_args()

    config = load_config(args.config)
    device = get_device()

    checkpoint = torch.load(args.checkpoint, map_location=device)
    model = build_model(
        model_name=config["model"]["name"],
        pretrained=False,  # checkpoint 가중치로 덮어쓸 것이므로 사전학습 가중치는 불필요
        dropout=config["model"]["dropout"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    test_dataset, test_loader, label_mean, label_std = build_test_dataset_and_loader(config)

    predictions_df = generate_predictions(
        model=model,
        data_loader=test_loader,
        ids=test_dataset.metadata["id"].tolist(),
        device=device,
        label_names=test_dataset.LABEL_ORDER,
        label_mean=label_mean,
        label_std=label_std,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    predictions_df.to_csv(args.output, index=False)
    logger.info("예측 CSV 저장 완료: %s (%d행)", args.output, len(predictions_df))


if __name__ == "__main__":
    main()
