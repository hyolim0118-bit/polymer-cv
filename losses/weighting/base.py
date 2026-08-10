"""WeightingStrategy — 오차 계산 방식을 전혀 모르는 property별 가중치 인터페이스."""

from abc import ABC, abstractmethod
from typing import List, Optional

from torch import Tensor


class WeightingStrategy(ABC):
    """property별 가중치 벡터를 생성하는 추상 인터페이스.

    오차가 MSE로 계산됐는지 Huber로 계산됐는지는 전혀 알지 못하며,
    오직 "property별로 얼마나 가중치를 줄 것인가"만 책임진다.

    Example:
        >>> weighting = UniformWeighting()
        >>> weights = weighting.get_weights(["Tg", "FFV", "Tc", "Density", "Rg"])
        >>> weights.shape
        torch.Size([5])
    """

    @abstractmethod
    def get_weights(
        self, property_names: List[str], epoch: Optional[int] = None
    ) -> Tensor:
        """property_names 순서에 맞는 가중치 텐서를 반환한다.

        Args:
            property_names: property 이름 리스트. 반환 텐서의 순서는 이 리스트와 일치.
            epoch: 현재 학습 epoch. ``DynamicWeighting`` 등 학습 진행에 따라
                가중치가 바뀌는 향후 확장을 위해 예약된 인자이며,
                ``UniformWeighting``/``StaticPropertyWeighting``은 사용하지 않는다.

        Returns:
            property별 가중치, shape ``(num_properties,)``.
        """
        raise NotImplementedError
