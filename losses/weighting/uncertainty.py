"""UncertaintyWeighting — 인터페이스만 정의, 구현은 STEP9.3 이후.

STEP9.2 범위에서는 계산 로직을 구현하지 않는다. Kendall et al. (2018)
방식의 학습 가능한 log-variance 기반 task weighting을 염두에 두고
인터페이스만 먼저 고정해둔다.
"""

from typing import Any, List, Optional

from torch import Tensor

from .base import WeightingStrategy


class UncertaintyWeighting(WeightingStrategy):
    """(STEP9.3+ 구현 예정) 학습 가능한 uncertainty 기반 property 가중치.

    property별로 학습 가능한 log-variance 파라미터를 두고, 불확실성이 큰
    property의 가중치를 자동으로 낮추는 전략을 구현할 예정이다.

    이번 STEP9.2에서는 인터페이스만 남기고 실제 계산 로직은 구현하지 않는다.
    ``LossFactory``의 registry에도 아직 등록하지 않는다.

    Raises:
        NotImplementedError: 모든 메서드 호출 시.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError(
            "UncertaintyWeighting은 STEP9.3 이후 단계에서 구현될 예정입니다."
        )

    def get_weights(
        self, property_names: List[str], epoch: Optional[int] = None
    ) -> Tensor:
        raise NotImplementedError(
            "UncertaintyWeighting은 STEP9.3 이후 단계에서 구현될 예정입니다."
        )
