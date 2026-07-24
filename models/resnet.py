"""torchvision ResNet18 기반 Backbone (Feature Extractor)."""

from __future__ import annotations

import logging

import torch
from torch import Tensor, nn
from torchvision.models import ResNet18_Weights, resnet18

logger = logging.getLogger(__name__)


class ResNet18Backbone(nn.Module):
    """마지막 Fully Connected Layer를 제거한 ResNet18 Feature Extractor."""

    def __init__(self, pretrained: bool = True) -> None:
        """ResNet18 Backbone을 초기화한다.

        Args:
            pretrained: True면 torchvision의 ImageNet 사전학습 가중치를
                사용하고, False면 무작위 초기화된 가중치를 사용한다.
        """
        super().__init__()

        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = resnet18(weights=weights)

        # 마지막 1000-class classifier(fc)를 제거하고, avgpool까지의
        # feature extractor만 사용한다. 출력 shape: (B, C, 1, 1).
        self.feature_extractor = nn.Sequential(*list(backbone.children())[:-1])

        # 이후 Head에 연결할 때 필요한 feature 차원을 자동으로 추출해둔다.
        self.feature_dim: int = backbone.fc.in_features

        logger.info(
            "ResNet18Backbone 생성 완료 (pretrained=%s, feature_dim=%d)",
            pretrained,
            self.feature_dim,
        )

    def forward(self, x: Tensor) -> Tensor:
        """이미지 배치를 받아 flatten된 feature vector를 반환한다.

        Args:
            x: 입력 이미지 텐서, shape (B, 3, H, W).

        Returns:
            feature 텐서, shape (B, feature_dim).
        """
        features = self.feature_extractor(x)
        return torch.flatten(features, 1)
