"""
ResNetBackbone
--------------
기존 STEP4에서 사용하던 ResNet18을 BaseBackbone 인터페이스로 감싼 버전.
동작 자체는 기존 구현과 동일하며, 인터페이스만 통일한다.
"""

from typing import Optional

import torch
import torch.nn as nn
import torchvision.models as tv_models

from .base import BaseBackbone


class ResNetBackbone(BaseBackbone):
    """
    torchvision의 ResNet18/34/50을 감싸는 backbone.
    기본값은 resnet18 (기존 STEP4~6과 동일).
    """

    _FEATURE_DIMS = {
        "resnet18": 512,
        "resnet34": 512,
        "resnet50": 2048,
    }

    def __init__(self, variant: str = "resnet18", pretrained: bool = True):
        super().__init__()
        if variant not in self._FEATURE_DIMS:
            raise ValueError(
                f"지원하지 않는 ResNet variant: {variant}. "
                f"사용 가능: {list(self._FEATURE_DIMS.keys())}"
            )
        self.variant = variant
        self._feature_dim = self._FEATURE_DIMS[variant]

        weights = "DEFAULT" if pretrained else None
        base_model = getattr(tv_models, variant)(weights=weights)

        # 마지막 fc layer(classifier)는 제거하고 conv feature extractor만 사용
        # avgpool까지 포함해서 (batch, feature_dim, 1, 1) 형태로 출력
        self.stem_and_blocks = nn.Sequential(*list(base_model.children())[:-2])
        self.avgpool = base_model.avgpool

        # Grad-CAM 등에서 참조할 마지막 conv block (layer4)
        self._last_conv_block = base_model.layer4

    @property
    def feature_dim(self) -> int:
        return self._feature_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat_map = self.stem_and_blocks(x)          # (B, C, H, W)
        pooled = self.avgpool(feat_map)              # (B, C, 1, 1)
        return torch.flatten(pooled, 1)               # (B, C)

    def get_cam_target_layer(self) -> Optional[nn.Module]:
        # ResNet은 CNN 계열이므로 마지막 conv block(layer4)을 그대로 반환
        return self._last_conv_block
