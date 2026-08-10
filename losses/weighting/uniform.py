"""UniformWeighting — 모든 property에 동일한 가중치를 부여."""

from typing import List, Optional

import torch
from torch import Tensor

from .base import WeightingStrategy


class UniformWeighting(WeightingStrategy):
    """모든 property에 1.0의 동일한 가중치를 부여하는 기본 전략.

    Example:
        >>> weighting = UniformWeighting()
        >>> weighting.get_weights(["Tg", "FFV"])
        tensor([1., 1.])
    """

    def get_weights(
        self, property_names: List[str], epoch: Optional[int] = None
    ) -> Tensor:
        """모든 property에 1.0을 부여한다.

        Args:
            property_names: property 이름 리스트.
            epoch: 사용하지 않음 (인터페이스 호환을 위해 존재).

        Returns:
            전부 1.0으로 채워진 텐서, shape ``(num_properties,)``.

        Raises:
            ValueError: ``property_names``가 빈 리스트인 경우.
        """
        if not property_names:
            raise ValueError("property_names가 비어 있습니다.")
        return torch.ones(len(property_names), dtype=torch.float32)
