"""LossFactory — STEP4 BackboneFactory와 동일한 Registry Pattern.

config를 읽어 ElementLoss + WeightingStrategy를 각각 생성한 뒤,
MaskedWeightedLoss로 조립해서 BaseLoss 인스턴스를 반환한다.
"""

from typing import Any, Dict, List, Type

from .base import BaseLoss
from .masked_weighted_loss import MaskedWeightedLoss
from .elements.base import ElementLoss
from .elements.mse import MaskedMSEElement
from .elements.mae import MaskedMAEElement
from .elements.huber import MaskedHuberElement
from .weighting.base import WeightingStrategy
from .weighting.uniform import UniformWeighting
from .weighting.static import StaticPropertyWeighting


class LossFactory:
    """config -> BaseLoss 인스턴스를 생성하는 factory.

    두 개의 독립적인 레지스트리(``_ELEMENT_REGISTRY``, ``_WEIGHTING_REGISTRY``)를
    두어, 오차 계산 방식과 가중치 전략을 각각 별도로 확장할 수 있게 한다.
    새 loss/weighting을 추가할 때는 클래스 하나 작성 + ``register_*`` 호출
    (또는 레지스트리 직접 등록) 한 줄이면 충분하며, 이 클래스나
    ``MaskedWeightedLoss``는 수정하지 않는다.

    Example:
        >>> config = {
        ...     "loss": {
        ...         "element_loss": {"name": "huber", "params": {"delta": 1.0}},
        ...         "weighting": {
        ...             "name": "static",
        ...             "property_weight": {"Tg": 1.0, "FFV": 2.0},
        ...         },
        ...         "reduction": "mean",
        ...     }
        ... }
        >>> criterion = LossFactory.build(config, property_names=["Tg", "FFV"])
        >>> loss = criterion(prediction, target, mask)
    """

    _ELEMENT_REGISTRY: Dict[str, Type[ElementLoss]] = {
        "mse": MaskedMSEElement,
        "mae": MaskedMAEElement,
        "huber": MaskedHuberElement,
    }

    _WEIGHTING_REGISTRY: Dict[str, Type[WeightingStrategy]] = {
        "uniform": UniformWeighting,
        "static": StaticPropertyWeighting,
        # "uncertainty": UncertaintyWeighting,  # STEP9.3에서 구현 완료 후 등록
    }

    @classmethod
    def build(cls, config: Dict[str, Any], property_names: List[str]) -> BaseLoss:
        """config로부터 ``MaskedWeightedLoss`` 인스턴스를 생성한다.

        Args:
            config: ``config["loss"]["element_loss"]["name"]``,
                ``config["loss"]["weighting"]["name"]`` 등을 포함하는 전체 config.
            property_names: property 이름 리스트 (컬럼 순서와 일치).

        Returns:
            ``BaseLoss``를 상속한 ``MaskedWeightedLoss`` 인스턴스.

        Raises:
            ValueError: config 구조가 올바르지 않거나, 등록되지 않은
                element_loss/weighting 이름이 지정된 경우.
        """
        loss_cfg = cls._validate_config(config)

        element_loss = cls._build_element_loss(loss_cfg["element_loss"])
        weighting = cls._build_weighting(loss_cfg["weighting"])
        reduction = loss_cfg.get("reduction", "mean")

        return MaskedWeightedLoss(
            element_loss=element_loss,
            weighting=weighting,
            property_names=property_names,
            reduction=reduction,
        )

    @classmethod
    def _validate_config(cls, config: Dict[str, Any]) -> Dict[str, Any]:
        """config 최상위 구조를 검증하고 ``loss`` 섹션을 반환한다.

        Args:
            config: 전체 config 딕셔너리.

        Returns:
            ``config["loss"]`` 섹션.

        Raises:
            ValueError: ``loss``, ``loss.element_loss.name``,
                ``loss.weighting.name`` 중 하나라도 없는 경우.
        """
        if "loss" not in config:
            raise ValueError("config에 'loss' 섹션이 없습니다.")
        loss_cfg = config["loss"]

        if "element_loss" not in loss_cfg or "name" not in loss_cfg["element_loss"]:
            raise ValueError("config.loss.element_loss.name이 필요합니다.")
        if "weighting" not in loss_cfg or "name" not in loss_cfg["weighting"]:
            raise ValueError("config.loss.weighting.name이 필요합니다.")

        return loss_cfg

    @classmethod
    def _build_element_loss(cls, element_cfg: Dict[str, Any]) -> ElementLoss:
        """config.loss.element_loss 섹션으로부터 ElementLoss를 생성한다.

        Args:
            element_cfg: ``{"name": ..., "params": {...}}`` 형태의 딕셔너리.

        Returns:
            ``ElementLoss`` 인스턴스.

        Raises:
            ValueError: 등록되지 않은 이름이거나, 잘못된 파라미터가 전달된 경우.
        """
        name = element_cfg["name"]
        if name not in cls._ELEMENT_REGISTRY:
            raise ValueError(
                f"등록되지 않은 element_loss: '{name}'. "
                f"사용 가능: {list(cls._ELEMENT_REGISTRY.keys())}"
            )
        params = element_cfg.get("params", {})
        try:
            return cls._ELEMENT_REGISTRY[name](**params)
        except TypeError as e:
            raise ValueError(
                f"element_loss '{name}'에 잘못된 params가 전달되었습니다: {params}. "
                f"원본 오류: {e}"
            ) from e

    @classmethod
    def _build_weighting(cls, weighting_cfg: Dict[str, Any]) -> WeightingStrategy:
        """config.loss.weighting 섹션으로부터 WeightingStrategy를 생성한다.

        Args:
            weighting_cfg: ``{"name": ..., "property_weight": {...}}`` 형태의 딕셔너리.
                ``name``이 ``"static"``일 때만 ``property_weight``가 필요하다.

        Returns:
            ``WeightingStrategy`` 인스턴스.

        Raises:
            ValueError: 등록되지 않은 이름이거나, ``static``인데
                ``property_weight``가 없는 경우.
        """
        name = weighting_cfg["name"]
        if name not in cls._WEIGHTING_REGISTRY:
            raise ValueError(
                f"등록되지 않은 weighting: '{name}'. "
                f"사용 가능: {list(cls._WEIGHTING_REGISTRY.keys())}"
            )

        if name == "static":
            property_weight = weighting_cfg.get("property_weight")
            if not property_weight:
                raise ValueError(
                    "weighting.name이 'static'이면 property_weight가 필요합니다."
                )
            return StaticPropertyWeighting(property_weight)

        return cls._WEIGHTING_REGISTRY[name]()

    @classmethod
    def register_element_loss(cls, name: str, loss_cls: Type[ElementLoss]) -> None:
        """새로운 element_loss를 런타임에 등록한다.

        Args:
            name: config에서 사용할 이름.
            loss_cls: ``ElementLoss``를 상속한 클래스 (인스턴스가 아님).
        """
        cls._ELEMENT_REGISTRY[name] = loss_cls

    @classmethod
    def register_weighting(cls, name: str, weighting_cls: Type[WeightingStrategy]) -> None:
        """새로운 weighting 전략을 런타임에 등록한다.

        Args:
            name: config에서 사용할 이름.
            weighting_cls: ``WeightingStrategy``를 상속한 클래스 (인스턴스가 아님).
        """
        cls._WEIGHTING_REGISTRY[name] = weighting_cls
