"""Masked Multi-task Regression Loss.

Dataset이 반환하는 (prediction, target, mask)를 그대로 받아,
mask == 0인 (결측) 위치는 loss 계산에서 제외한 뒤 배치 평균을 반환한다.
"""

from __future__ import annotations

import logging
from typing import Dict, Type

import torch
import torch.nn.functional as F
from torch import Tensor, nn

logger = logging.getLogger(__name__)


def _masked_reduce(elementwise_loss: Tensor, mask: Tensor) -> Tensor:
    """원소별 loss에 mask를 곱해 평균을 낸다 (mask==0 위치는 제외).

    Args:
        elementwise_loss: mask와 동일한 shape (B, NUM_TARGETS)의 원소별 loss.
        mask: 결측 여부 마스크, shape (B, NUM_TARGETS). 1=존재, 0=결측.

    Returns:
        스칼라 loss 텐서. 배치 내 유효 라벨이 하나도 없으면 0을 반환한다
        (requires_grad는 유지되어 backward 시 에러가 나지 않는다).
    """
    valid_count = mask.sum()
    if valid_count == 0:
        logger.warning("배치 내 모든 Label이 Missing이라 loss를 0으로 처리합니다.")
        return (elementwise_loss * mask).sum()  # 항상 0이지만 grad graph 유지
    return (elementwise_loss * mask).sum() / valid_count


class MaskedMSELoss(nn.Module):
    """mask==0 위치를 제외한 Masked MSE Loss (배치 평균)."""

    def forward(self, prediction: Tensor, target: Tensor, mask: Tensor) -> Tensor:
        """Masked MSE를 계산한다.

        Args:
            prediction: 모델 예측값, shape (B, NUM_TARGETS).
            target: 정답값, shape (B, NUM_TARGETS).
            mask: 결측 마스크, shape (B, NUM_TARGETS).

        Returns:
            스칼라 loss 텐서.
        """
        elementwise = (prediction - target) ** 2
        return _masked_reduce(elementwise, mask)


class MaskedMAELoss(nn.Module):
    """mask==0 위치를 제외한 Masked MAE(L1) Loss (배치 평균)."""

    def forward(self, prediction: Tensor, target: Tensor, mask: Tensor) -> Tensor:
        """Masked MAE를 계산한다.

        Args:
            prediction: 모델 예측값, shape (B, NUM_TARGETS).
            target: 정답값, shape (B, NUM_TARGETS).
            mask: 결측 마스크, shape (B, NUM_TARGETS).

        Returns:
            스칼라 loss 텐서.
        """
        elementwise = torch.abs(prediction - target)
        return _masked_reduce(elementwise, mask)


class MaskedHuberLoss(nn.Module):
    """mask==0 위치를 제외한 Masked Huber(Smooth L1) Loss (배치 평균)."""

    def __init__(self, delta: float = 1.0) -> None:
        """MaskedHuberLoss를 초기화한다.

        Args:
            delta: Huber loss의 전환점(beta) 값.
        """
        super().__init__()
        self.delta = delta

    def forward(self, prediction: Tensor, target: Tensor, mask: Tensor) -> Tensor:
        """Masked Huber Loss를 계산한다.

        Args:
            prediction: 모델 예측값, shape (B, NUM_TARGETS).
            target: 정답값, shape (B, NUM_TARGETS).
            mask: 결측 마스크, shape (B, NUM_TARGETS).

        Returns:
            스칼라 loss 텐서.
        """
        elementwise = F.smooth_l1_loss(prediction, target, beta=self.delta, reduction="none")
        return _masked_reduce(elementwise, mask)


_LOSS_REGISTRY: Dict[str, Type[nn.Module]] = {
    "mse": MaskedMSELoss,
    "mae": MaskedMAELoss,
    "huber": MaskedHuberLoss,
}


def build_loss(name: str, huber_delta: float = 1.0) -> nn.Module:
    """config 기반으로 Loss 인스턴스를 생성한다.

    Args:
        name: "mse", "mae", "huber" 중 하나.
        huber_delta: name이 "huber"일 때만 사용되는 delta 값.

    Returns:
        nn.Module 형태의 loss criterion.

    Raises:
        ValueError: 지원하지 않는 loss 이름인 경우.
    """
    if name not in _LOSS_REGISTRY:
        supported = list(_LOSS_REGISTRY.keys())
        raise ValueError(f"지원하지 않는 loss입니다: '{name}'. 지원 목록: {supported}")

    if name == "huber":
        return MaskedHuberLoss(delta=huber_delta)
    return _LOSS_REGISTRY[name]()
