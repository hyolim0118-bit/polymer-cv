"""Backbone과 Head를 조합해 최종 모델을 생성하는 Factory 모듈.

새로운 backbone(EfficientNet, ViT 등)을 추가할 때는 해당 backbone 모듈을
만들고 ``_BACKBONE_REGISTRY``에 생성 함수 한 줄만 등록하면 ``build_model``이
그대로 동작한다.
"""

from __future__ import annotations

import logging
from typing import Callable, Dict, Tuple

import torch
from torch import Tensor, nn

from models.multitask_head import MultiTaskHead
from models.resnet import ResNet18Backbone

logger = logging.getLogger(__name__)


def _build_resnet18(pretrained: bool) -> Tuple[nn.Module, int]:
    """resnet18 backbone 인스턴스와 feature 차원을 반환한다.

    Args:
        pretrained: ImageNet 사전학습 가중치 사용 여부.

    Returns:
        (backbone, feature_dim) 튜플.
    """
    backbone = ResNet18Backbone(pretrained=pretrained)
    return backbone, backbone.feature_dim


# model_name -> (pretrained) -> (backbone, feature_dim) 생성 함수 레지스트리.
# 향후 "efficientnet_b0" 같은 backbone을 추가하려면 이 딕셔너리에
# 생성 함수만 추가하면 되고, build_model 로직은 변경할 필요가 없다.
_BACKBONE_REGISTRY: Dict[str, Callable[[bool], Tuple[nn.Module, int]]] = {
    "resnet18": _build_resnet18,
}


class PolymerRegressor(nn.Module):
    """Backbone(Feature Extractor) + MultiTaskHead를 연결한 전체 회귀 모델."""

    def __init__(self, backbone: nn.Module, head: nn.Module) -> None:
        """PolymerRegressor를 초기화한다.

        Args:
            backbone: 이미지를 feature vector로 변환하는 모듈.
            head: feature vector를 5개 물성 예측값으로 변환하는 모듈.
        """
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, x: Tensor) -> Tensor:
        """이미지 배치를 받아 5개 물성에 대한 예측값을 반환한다.

        Missing Label Mask는 이 모델에서 다루지 않으며, Trainer의 Loss
        계산 단계에서 별도로 적용한다.

        Args:
            x: 입력 이미지 텐서, shape (B, 3, H, W).

        Returns:
            예측 텐서, shape (B, NUM_TARGETS).
        """
        features = self.backbone(x)
        return self.head(features)


def build_model(model_name: str, pretrained: bool, dropout: float) -> nn.Module:
    """설정값을 바탕으로 Backbone + Head 모델을 생성한다.

    Args:
        model_name: 사용할 backbone 이름. 현재는 "resnet18"만 지원하며,
            지원 목록은 ``_BACKBONE_REGISTRY``의 키를 통해 확인할 수 있다.
        pretrained: True면 ImageNet 사전학습 가중치를 사용한다.
        dropout: MultiTaskHead의 Dropout 확률.

    Returns:
        PolymerRegressor 인스턴스 (nn.Module).

    Raises:
        ValueError: 지원하지 않는 model_name이 입력된 경우.
    """
    if model_name not in _BACKBONE_REGISTRY:
        supported = list(_BACKBONE_REGISTRY.keys())
        raise ValueError(
            f"지원하지 않는 model_name입니다: '{model_name}'. 지원 목록: {supported}"
        )

    backbone, feature_dim = _BACKBONE_REGISTRY[model_name](pretrained)
    head = MultiTaskHead(in_features=feature_dim, dropout=dropout)
    model = PolymerRegressor(backbone, head)

    logger.info(
        "모델 생성 완료: model_name=%s, pretrained=%s, dropout=%.2f, feature_dim=%d",
        model_name,
        pretrained,
        dropout,
        feature_dim,
    )
    return model


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    model = build_model(model_name="resnet18", pretrained=True, dropout=0.3)
    model.eval()

    dummy_input = torch.randn(4, 3, 224, 224)
    with torch.no_grad():
        output = model(dummy_input)

    logger.info("Output Shape: %s", tuple(output.shape))
