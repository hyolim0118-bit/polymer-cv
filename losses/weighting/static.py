"""StaticPropertyWeighting — config에 고정된 property별 가중치."""

from typing import Dict, List, Optional

import torch
from torch import Tensor

from .base import WeightingStrategy


class StaticPropertyWeighting(WeightingStrategy):
    """config에서 지정한 고정 property별 가중치를 그대로 사용하는 전략.

    Args:
        property_weight: ``{property_name: weight}`` 형태의 딕셔너리.
            예: ``{"Tg": 1.0, "FFV": 2.0}``.

    Raises:
        ValueError: ``property_weight``가 비어 있거나, 가중치 값이 음수인 경우.

    Example:
        >>> weighting = StaticPropertyWeighting({"Tg": 1.0, "FFV": 2.0})
        >>> weighting.get_weights(["Tg", "FFV"])
        tensor([1., 2.])
    """

    def __init__(self, property_weight: Dict[str, float]) -> None:
        if not property_weight:
            raise ValueError("property_weight가 비어 있습니다.")
        for name, weight in property_weight.items():
            if weight < 0:
                raise ValueError(
                    f"property_weight['{name}']는 음수일 수 없습니다: {weight}"
                )
        self.property_weight = property_weight

    def get_weights(
        self, property_names: List[str], epoch: Optional[int] = None
    ) -> Tensor:
        """property_names 순서에 맞춰 config의 가중치를 텐서로 변환한다.

        Args:
            property_names: property 이름 리스트.
            epoch: 사용하지 않음 (인터페이스 호환을 위해 존재).

        Returns:
            property_names 순서의 가중치 텐서, shape ``(num_properties,)``.

        Raises:
            ValueError: ``property_names`` 중 ``property_weight``에 없는 항목이 있는 경우.
        """
        missing = [p for p in property_names if p not in self.property_weight]
        if missing:
            raise ValueError(
                f"static weighting config에 다음 property의 가중치가 없습니다: {missing}"
            )
        return torch.tensor(
            [self.property_weight[p] for p in property_names], dtype=torch.float32
        )
