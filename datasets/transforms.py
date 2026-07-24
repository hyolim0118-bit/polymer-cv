"""Train / Validation / Test 이미지 Transform 정의.

분자 2D 구조 이미지는 좌우가 뒤집혀도, 살짝 회전해도 화학적 의미가
크게 훼손되지 않지만, 과도한 왜곡(Cutout, Elastic 등)은 결합/각도
정보를 무너뜨릴 수 있어 사용하지 않는다.
"""

from __future__ import annotations

from typing import Dict, List

import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_train_transform(
    image_size: int,
    normalize_mean: List[float],
    normalize_std: List[float],
    aug_cfg: Dict[str, float],
) -> A.Compose:
    """학습용 Transform을 생성한다 (약한 augmentation만 적용).

    Args:
        image_size: Resize할 정사각형 크기.
        normalize_mean: 채널별 정규화 평균.
        normalize_std: 채널별 정규화 표준편차.
        aug_cfg: augmentation 확률/강도 설정값 딕셔너리.

    Returns:
        albumentations Compose 객체.
    """
    return A.Compose(
        [
            A.Resize(image_size, image_size),
            A.HorizontalFlip(p=aug_cfg["horizontal_flip_p"]),
            A.Rotate(limit=aug_cfg["rotate_limit"], p=aug_cfg["rotate_p"], border_mode=0),
            A.RandomBrightnessContrast(p=aug_cfg["brightness_contrast_p"]),
            A.Normalize(mean=normalize_mean, std=normalize_std),
            ToTensorV2(),
        ]
    )


def get_valid_transform(
    image_size: int,
    normalize_mean: List[float],
    normalize_std: List[float],
) -> A.Compose:
    """Validation용 Transform을 생성한다 (augmentation 없음).

    Args:
        image_size: Resize할 정사각형 크기.
        normalize_mean: 채널별 정규화 평균.
        normalize_std: 채널별 정규화 표준편차.

    Returns:
        albumentations Compose 객체.
    """
    return A.Compose(
        [
            A.Resize(image_size, image_size),
            A.Normalize(mean=normalize_mean, std=normalize_std),
            ToTensorV2(),
        ]
    )


def get_test_transform(
    image_size: int,
    normalize_mean: List[float],
    normalize_std: List[float],
) -> A.Compose:
    """Test용 Transform을 생성한다 (Validation과 동일).

    Args:
        image_size: Resize할 정사각형 크기.
        normalize_mean: 채널별 정규화 평균.
        normalize_std: 채널별 정규화 표준편차.

    Returns:
        albumentations Compose 객체.
    """
    return get_valid_transform(image_size, normalize_mean, normalize_std)
