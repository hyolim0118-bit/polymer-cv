"""
BackboneFactory
---------------
config.yaml의 model.backbone 값 하나로 적절한 backbone 인스턴스를 생성한다.
새로운 backbone을 추가할 때는 파일 하나 + 아래 _REGISTRY 등록 한 줄이면 된다.
(예: ConvNeXt-Tiny를 추가하려면 convnext.py 작성 후 _REGISTRY에 한 줄 추가)
"""

from typing import Any, Dict

from .base import BaseBackbone
from .resnet import ResNetBackbone
from .efficientnet import EfficientNetBackbone


class BackboneFactory:
    """backbone 이름 -> 생성 함수 레지스트리."""

    _REGISTRY = {
        "resnet18": lambda pretrained: ResNetBackbone("resnet18", pretrained),
        "resnet34": lambda pretrained: ResNetBackbone("resnet34", pretrained),
        "resnet50": lambda pretrained: ResNetBackbone("resnet50", pretrained),
        "efficientnet_b0": lambda pretrained: EfficientNetBackbone(
            "efficientnet_b0", pretrained
        ),
        "efficientnet_b1": lambda pretrained: EfficientNetBackbone(
            "efficientnet_b1", pretrained
        ),
        # 향후 확장 예시 (구현 후 주석 해제):
        # "convnext_tiny": lambda pretrained: ConvNeXtBackbone("convnext_tiny", pretrained),
        # "swin_t": lambda pretrained: SwinBackbone("swin_t", pretrained),
    }

    @classmethod
    def create(cls, config: Dict[str, Any]) -> BaseBackbone:
        """
        Parameters
        ----------
        config : dict
            config.yaml의 `model` 섹션. 예:
            {"backbone": "resnet18", "pretrained": True}

        Returns
        -------
        BaseBackbone 인스턴스
        """
        backbone_name = config.get("backbone", "resnet18")
        pretrained = config.get("pretrained", True)

        if backbone_name not in cls._REGISTRY:
            raise ValueError(
                f"등록되지 않은 backbone: '{backbone_name}'. "
                f"사용 가능한 backbone: {list(cls._REGISTRY.keys())}"
            )

        backbone = cls._REGISTRY[backbone_name](pretrained)

        freeze = config.get("freeze_backbone", False)
        if freeze:
            backbone.freeze()

        return backbone

    @classmethod
    def available_backbones(cls):
        return list(cls._REGISTRY.keys())

    @classmethod
    def register(cls, name: str, constructor_fn) -> None:
        """
        새로운 backbone을 런타임에 등록하고 싶을 때 사용.
        constructor_fn(pretrained: bool) -> BaseBackbone 형태의 callable.
        """
        cls._REGISTRY[name] = constructor_fn
