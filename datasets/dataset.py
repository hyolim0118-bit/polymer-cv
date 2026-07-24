"""processed_metadata.csv 기반 PyTorch Dataset 및 DataLoader.

STEP2에서 생성한 ``data/processed/processed_metadata.csv`` 와
``images/render/**`` 이미지를 그대로 사용하며, 새로운 CSV는 생성하지 않는다.

STEP2의 ``processed_metadata.csv``에는 train/test split만 존재하므로,
validation은 이 모듈에서 train 데이터를 config에 정의된 비율로
(고정 시드 기반, 재현 가능하게) 나눠서 만든다. CSV 파일 자체는 수정하지 않는다.

STEP3.1: Cross Validation에서의 Data Leakage 방지를 위해, 라벨 정규화를
더 이상 CSV의 ``*_norm`` 컬럼(=STEP2에서 전체 train 기준으로 미리 계산되어
고정된 값)에 의존하지 않는다. 대신 Dataset은 원본 라벨(Tg/Density/FFV/Tc/Rg)만
읽고, 외부에서 전달받은 label_mean/label_std로 그때그때 정규화를 수행한다.
mean/std 계산 자체는 Dataset이 하지 않으며, 반드시 Train Fold 데이터만으로
계산되어 Dataset에 주입되어야 한다 (이 계산은 이 모듈의 compute_fold_stats,
그리고 fold 구성은 load_split_dataframes가 담당).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split
from torch import Tensor
import torch
from torch.utils.data import DataLoader, Dataset

from datasets.transforms import get_test_transform, get_train_transform, get_valid_transform

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

VALID_SPLITS = ("train", "valid", "test")

# Dataset이 반환하는 label/mask 벡터의 컬럼 순서 (STEP3 스펙에서 지정한 순서)
LABEL_ORDER: List[str] = ["Tg", "Density", "FFV", "Tc", "Rg"]


def load_config(config_path: Path) -> Dict:
    """YAML config 파일을 읽어 dict로 반환한다.

    Args:
        config_path: config.yaml 경로.

    Returns:
        설정값 딕셔너리.
    """
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _split_train_valid(
    metadata: pd.DataFrame, val_ratio: float, seed: int
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """원본 'train' 행을 train/valid로 재현 가능하게 분할한다.

    Args:
        metadata: split == "train"인 행만 포함된 데이터프레임.
        val_ratio: validation으로 뗄 비율.
        seed: 재현성을 위한 랜덤 시드.

    Returns:
        (train_df, valid_df) 튜플. 인덱스는 원본 순서를 보존한다.
    """
    # random_state(seed)를 고정해두면 코드를 몇 번 다시 실행해도
    # 항상 똑같은 행들이 train/valid로 나뉨 (재현성 확보)
    train_df, valid_df = train_test_split(
        metadata, test_size=val_ratio, random_state=seed, shuffle=True
    )
    return train_df.reset_index(drop=True), valid_df.reset_index(drop=True)


def compute_fold_stats(
    df: pd.DataFrame, label_columns: List[str] = LABEL_ORDER
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """결측(NaN)을 제외한 값으로 물성별 mean/std를 계산한다.

    Data Leakage 방지를 위해, 이 함수에는 반드시 "Train Fold"에 해당하는
    행만 전달해야 한다 (Validation/Test 행이 섞이면 Leakage가 발생한다).
    Dataset 내부에서는 이 함수를 호출하지 않는다 — mean/std 계산은 항상
    Dataset 외부(이 모듈의 load_split_dataframes 등)에서 이루어진다.

    Args:
        df: 통계를 계산할 데이터프레임 (Train Fold만 포함해야 함).
        label_columns: 통계를 계산할 물성 컬럼 목록.

    Returns:
        (label_mean, label_std) 튜플. 각각 {물성명: 값} 형태의 dict.
    """
    label_mean: Dict[str, float] = {}
    label_std: Dict[str, float] = {}

    for col in label_columns:
        values = df[col].dropna()
        mean = float(values.mean()) if len(values) > 0 else 0.0
        std = float(values.std()) if len(values) > 1 else 1.0
        # 표준편차가 0/NaN이면 (x-mean)/std에서 0으로 나누는 사고가 나므로 1로 방어
        if std == 0.0 or pd.isna(std):
            std = 1.0
        label_mean[col] = mean
        label_std[col] = std

    return label_mean, label_std


def load_split_dataframes(
    metadata_path: Path, val_ratio: float, seed: int
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, float], Dict[str, float]]:
    """processed_metadata.csv를 읽어 train/valid/test로 나누고, Train Fold 기준
    정규화 통계(label_mean/label_std)를 함께 계산한다.

    Args:
        metadata_path: processed_metadata.csv 경로.
        val_ratio: train 중 validation으로 뗄 비율.
        seed: train/valid 분할용 랜덤 시드.

    Returns:
        (train_df, valid_df, test_df, label_mean, label_std) 튜플.
        label_mean/label_std는 train_df만으로 계산되어, valid/test에도
        동일하게 (Leakage 없이) 적용된다.

    Raises:
        FileNotFoundError: metadata_path가 존재하지 않는 경우.
    """
    metadata_path = Path(metadata_path)
    if not metadata_path.exists():
        raise FileNotFoundError(f"metadata 파일을 찾을 수 없습니다: {metadata_path}")

    full_metadata = pd.read_csv(metadata_path)
    raw_train = full_metadata[full_metadata["split"] == "train"].reset_index(drop=True)
    raw_test = full_metadata[full_metadata["split"] == "test"].reset_index(drop=True)

    train_df, valid_df = _split_train_valid(raw_train, val_ratio, seed)

    # Data Leakage 방지 핵심: mean/std는 오직 train_df(Train Fold)만으로 계산
    label_mean, label_std = compute_fold_stats(train_df, LABEL_ORDER)

    return train_df, valid_df, raw_test, label_mean, label_std


class PolymerDataset(Dataset):
    """SMILES 렌더링 이미지 + 5개 물성 라벨 + 결측 마스크를 반환하는 Dataset.

    STEP3.1: 이 Dataset은 자신이 Train Fold인지 Validation Fold인지 알지
    못하며, mean/std 계산도 하지 않는다. 이미 만들어진 dataframe(어느 fold의
    행인지는 호출부가 결정)과, 외부에서 전달받은 label_mean/label_std만
    가지고 정규화를 수행한다.
    """

    LABEL_ORDER: List[str] = LABEL_ORDER

    def __init__(
        self,
        dataframe: pd.DataFrame,
        image_root: Path,
        transform,
        label_mean: Dict[str, float],
        label_std: Dict[str, float],
    ) -> None:
        """Dataset을 초기화한다.

        Args:
            dataframe: 이 Dataset 인스턴스가 사용할 행만 담긴 데이터프레임
                (train/valid/test 중 어느 fold인지는 호출부에서 이미 결정됨).
            image_root: image_path 컬럼의 기준이 되는 루트 디렉토리.
            transform: albumentations Compose transform.
            label_mean: 물성별 평균 (Train Fold에서만 계산된 값이어야 함).
            label_std: 물성별 표준편차 (Train Fold에서만 계산된 값이어야 함).
        """
        self.metadata = dataframe.reset_index(drop=True)
        self.image_root = Path(image_root)
        self.transform = transform
        self.label_mean = label_mean
        self.label_std = label_std

        logger.info("Dataset 생성 완료. 샘플 수: %d", len(self.metadata))

    def __len__(self) -> int:
        """Dataset의 전체 샘플 수를 반환한다."""
        return len(self.metadata)

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor, Tensor]:
        """인덱스에 해당하는 (image, label, mask)를 반환한다.

        Args:
            index: 샘플 인덱스.

        Returns:
            image: torch.FloatTensor, shape (3, 224, 224)
            label: torch.FloatTensor, shape (5,) — label_mean/label_std로
                정규화된 물성값 (z-score), 결측은 0으로 채움
            mask: torch.FloatTensor, shape (5,) — 1이면 라벨 존재, 0이면 결측

        Raises:
            FileNotFoundError: 이미지 파일이 존재하지 않는 경우.
            ValueError: 이미지가 손상되어 읽을 수 없거나, label/mask 길이가
                맞지 않는 경우.
        """
        # index번째 샘플의 메타데이터 한 행 (id, image_path, 원본 라벨 컬럼들)
        row = self.metadata.iloc[index]

        # STEP2가 저장해둔 image_path(상대경로)와 image_root를 합쳐 실제 파일 경로 구성
        image_path = self.image_root / row["image_path"]
        if not image_path.exists():
            raise FileNotFoundError(f"이미지 파일이 존재하지 않습니다: {image_path}")

        # cv2.imread는 기본적으로 BGR 순서로 읽으므로 RGB로 다시 변환
        image_bgr = cv2.imread(str(image_path))
        if image_bgr is None:
            raise ValueError(f"이미지가 손상되어 읽을 수 없습니다: {image_path}")
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        # transforms.py에서 만든 Resize+Normalize+ToTensor 파이프라인 적용 (변경 없음)
        transformed = self.transform(image=image_rgb)
        image = transformed["image"]

        # 원본 라벨(Tg/Density/FFV/Tc/Rg)을 읽고, 외부에서 받은 mean/std로 그때그때 정규화
        label_values: List[float] = []
        mask_values: List[float] = []
        for col in self.LABEL_ORDER:
            raw_value = row[col]
            has_label = not pd.isna(raw_value)
            mask_values.append(1.0 if has_label else 0.0)  # Missing Label Mask 로직은 그대로 유지

            if has_label:
                mean = self.label_mean[col]
                std = self.label_std[col]
                label_values.append((float(raw_value) - mean) / std)
            else:
                # masked loss에서 mask=0인 값은 어차피 무시되므로 0으로 채워도 학습에 영향 없음
                label_values.append(0.0)

        if len(label_values) != 5 or len(mask_values) != 5:
            raise ValueError(
                f"label/mask 길이가 5가 아닙니다: label={len(label_values)}, "
                f"mask={len(mask_values)}"
            )

        label = torch.tensor(label_values, dtype=torch.float32)
        mask = torch.tensor(mask_values, dtype=torch.float32)

        # DataLoader가 배치로 묶어주면 최종적으로 (B,3,224,224), (B,5), (B,5) 가 됨
        return image, label, mask


def create_dataloader(
    split: str, config_path: Path = Path("training/config.yaml")
) -> DataLoader:
    """config.yaml 설정을 바탕으로 DataLoader를 생성한다.

    내부적으로 train/valid/test 세 fold를 모두 나누고, label_mean/label_std를
    반드시 train fold에서만 계산해 세 Dataset에 동일하게 적용한다 (Data
    Leakage 방지). 외부에 노출되는 함수 시그니처는 이전과 동일하다.

    Args:
        split: "train", "valid", "test" 중 하나.
        config_path: config.yaml 경로.

    Returns:
        해당 split의 torch DataLoader.

    Raises:
        ValueError: split 값이 유효하지 않은 경우.
    """
    if split not in VALID_SPLITS:
        raise ValueError(f"split은 {VALID_SPLITS} 중 하나여야 합니다. 입력값: {split}")

    config = load_config(config_path)  # training/config.yaml 통째로 dict로 로드

    image_cfg = config["image"]
    aug_cfg = config["augmentation"]
    data_cfg = config["data"]
    split_cfg = config["split"]
    loader_cfg = config["dataloader"]

    train_df, valid_df, test_df, label_mean, label_std = load_split_dataframes(
        metadata_path=Path(data_cfg["metadata_path"]),
        val_ratio=split_cfg["val_ratio"],
        seed=split_cfg["seed"],
    )
    dataframe_by_split = {"train": train_df, "valid": valid_df, "test": test_df}

    # train만 augmentation 적용, valid/test는 augmentation 없는 transform 사용
    if split == "train":
        transform = get_train_transform(
            image_cfg["size"], image_cfg["normalize_mean"], image_cfg["normalize_std"], aug_cfg
        )
    elif split == "valid":
        transform = get_valid_transform(
            image_cfg["size"], image_cfg["normalize_mean"], image_cfg["normalize_std"]
        )
    else:
        transform = get_test_transform(
            image_cfg["size"], image_cfg["normalize_mean"], image_cfg["normalize_std"]
        )

    dataset = PolymerDataset(
        dataframe=dataframe_by_split[split],
        image_root=Path(data_cfg["image_root"]),
        transform=transform,
        label_mean=label_mean,
        label_std=label_std,
    )

    # PyTorch DataLoader가 Dataset을 배치 단위로 묶고, train일 때만 매 epoch 순서를 섞음(shuffle)
    return DataLoader(
        dataset,
        batch_size=loader_cfg["batch_size"],
        shuffle=loader_cfg["shuffle"][split],
        num_workers=loader_cfg["num_workers"],  # 이미지 로딩을 병렬로 처리할 프로세스 수
    )


if __name__ == "__main__":
    train_loader = create_dataloader(split="train")
    valid_loader = create_dataloader(split="valid")
    test_loader = create_dataloader(split="test")

    images, labels, masks = next(iter(train_loader))
    logger.info("Image Shape: %s", tuple(images.shape))
    logger.info("Label Shape: %s", tuple(labels.shape))
    logger.info("Mask Shape: %s", tuple(masks.shape))
