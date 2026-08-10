"""
EfficientNetBackbone
---------------------
torchvision의 EfficientNet-B0을 BaseBackbone 인터페이스로 감싼 버전.
STEP7에서 ResNet18과 동일 조건으로 비교할 두 번째 backbone.
"""

from typing import Optional

import torch
import torch.nn as nn
import torchvision.models as tv_models

from .base import BaseBackbone


class EfficientNetBackbone(BaseBackbone):
    """
    torchvision의 EfficientNet-B0/B1 등을 감싸는 backbone.
    기본값은 efficientnet_b0.
    """

    _FEATURE_DIMS = {
        "efficientnet_b0": 1280,
        "efficientnet_b1": 1280,
    }

    _MODEL_FN = {
        "efficientnet_b0": tv_models.efficientnet_b0,
        "efficientnet_b1": tv_models.efficientnet_b1,
    }

    def __init__(self, variant: str = "efficientnet_b0", pretrained: bool = True):
        super().__init__()
        if variant not in self._FEATURE_DIMS:
            raise ValueError(
                f"지원하지 않는 EfficientNet variant: {variant}. "
                f"사용 가능: {list(self._FEATURE_DIMS.keys())}"
            )
        self.variant = variant
        self._feature_dim = self._FEATURE_DIMS[variant]

        weights = "DEFAULT" if pretrained else None
        base_model = self._MODEL_FN[variant](weights=weights)

        # base_model.features: conv stem + MBConv blocks (classifier 제외)
        self.features = base_model.features
        self.avgpool = base_model.avgpool

        # Grad-CAM 대상: features의 마지막 block (마지막 conv block)
        self._last_conv_block = self.features[-1]

    @property
    def feature_dim(self) -> int:
        return self._feature_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat_map = self.features(x)                  # (B, C, H, W)
        pooled = self.avgpool(feat_map)                # (B, C, 1, 1)
        return torch.flatten(pooled, 1)                 # (B, C)

    def get_cam_target_layer(self) -> Optional[nn.Module]:
        return self._last_conv_block
