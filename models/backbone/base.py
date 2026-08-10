"""
BaseBackbone
------------
모든 backbone이 공통으로 구현해야 하는 추상 인터페이스.
MultiTaskModel(head)은 이 인터페이스만 알고 있으면 되고,
구체적인 backbone(ResNet18 / EfficientNet-B0 / 향후 ConvNeXt, Swin 등)의
내부 구조는 전혀 알 필요가 없다.
"""

from abc import ABC, abstractmethod
from typing import Optional

import torch
import torch.nn as nn


class BaseBackbone(nn.Module, ABC):
    """모든 backbone의 부모 클래스."""

    def __init__(self):
        super().__init__()

    @property
    @abstractmethod
    def feature_dim(self) -> int:
        """이 backbone이 출력하는 feature vector의 차원."""
        raise NotImplementedError

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        입력 이미지를 받아 (batch, feature_dim) 형태의 feature vector를 반환.
        """
        raise NotImplementedError

    def freeze(self) -> None:
        """backbone 파라미터를 전부 고정 (fine-tuning 방지)."""
        for p in self.parameters():
            p.requires_grad = False

    def unfreeze(self) -> None:
        """backbone 파라미터를 전부 학습 가능 상태로 되돌림."""
        for p in self.parameters():
            p.requires_grad = True

    def get_cam_target_layer(self) -> Optional[nn.Module]:
        """
        STEP12 Grad-CAM 등 시각화가 대상으로 삼을 레이어를 반환.

        - CNN 계열(ResNet, EfficientNet, ConvNeXt 등): 마지막 conv block 반환
        - Transformer 계열(Swin 등, 향후 확장 시): 마지막 attention block을
          반환하거나, conv 기반 CAM이 적용되지 않음을 나타내기 위해 None 반환

        기본 구현은 None. 각 backbone 서브클래스가 필요 시 override.
        """
        return None

    def num_parameters(self, trainable_only: bool = False) -> int:
        """실험 기록용: 파라미터 개수."""
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())
