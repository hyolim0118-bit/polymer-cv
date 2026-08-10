"""MaskedWeightedLoss — STEP9.1 Architecture의 유일한 BaseLoss 구현체.

ElementLoss(오차 계산)와 WeightingStrategy(property별 가중치)를 합성해서
mask 적용 -> weighting 적용 -> reduction까지 수행한다.
"""

from typing import List

from torch import Tensor

from .base import BaseLoss
from .elements.base import ElementLoss
from .weighting.base import WeightingStrategy

_VALID_REDUCTIONS = ("mean", "sum")


class MaskedWeightedLoss(BaseLoss):
    """ElementLoss + WeightingStrategy를 조합한 masked multi-task loss.

    데이터 흐름::

        raw_loss = element_loss.compute(prediction, target)      # (B, P)
        weighted_mask = mask * weighting.get_weights(...)         # (B, P)
        weighted_loss = raw_loss * weighted_mask
        result = weighted_loss.sum() / weighted_mask.sum()        # reduction="mean"

    Args:
        element_loss: 오차 계산 방식 (예: ``MaskedMSEElement``).
        weighting: property별 가중치 전략 (예: ``UniformWeighting``).
        property_names: property 이름 리스트. prediction/target/mask의
            컬럼 순서와 일치해야 하며, ``weighting.get_weights``에 그대로 전달된다.
        reduction: ``"mean"`` 또는 ``"sum"``. 기본값 ``"mean"``.

    Raises:
        ValueError: ``reduction``이 허용된 값이 아니거나 ``property_names``가 빈 경우.

    Example:
        >>> element_loss = MaskedMSEElement()
        >>> weighting = UniformWeighting()
        >>> criterion = MaskedWeightedLoss(
        ...     element_loss, weighting,
        ...     property_names=["Tg", "FFV"], reduction="mean",
        ... )
        >>> loss = criterion(prediction, target, mask)
        >>> loss.backward()
    """

    def __init__(
        self,
        element_loss: ElementLoss,
        weighting: WeightingStrategy,
        property_names: List[str],
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        if reduction not in _VALID_REDUCTIONS:
            raise ValueError(
                f"reduction은 {_VALID_REDUCTIONS} 중 하나여야 합니다. "
                f"받은 값: '{reduction}'"
            )
        if not property_names:
            raise ValueError("property_names가 비어 있습니다.")

        self.element_loss = element_loss
        self.weighting = weighting
        self.property_names = property_names
        self.reduction = reduction

    def forward(self, prediction: Tensor, target: Tensor, mask: Tensor) -> Tensor:
        """masked, weighted multi-task loss를 계산한다.

        Args:
            prediction: shape ``(batch_size, num_properties)``.
            target: shape ``(batch_size, num_properties)``.
            mask: shape ``(batch_size, num_properties)``. 1이면 라벨 존재, 0이면 결측.

        Returns:
            스칼라 loss 텐서.

        Raises:
            ValueError: prediction/target/mask의 shape이 서로 다르거나,
                property 차원이 ``property_names`` 길이와 다른 경우,
                또는 배치 내에 유효한(mask=1) 라벨이 하나도 없는 경우
                (``reduction="mean"``일 때만 해당).
        """
        if prediction.shape != target.shape or prediction.shape != mask.shape:
            raise ValueError(
                "prediction/target/mask의 shape이 모두 동일해야 합니다. "
                f"받은 shape: prediction={tuple(prediction.shape)}, "
                f"target={tuple(target.shape)}, mask={tuple(mask.shape)}"
            )
        if prediction.dim() != 2 or prediction.shape[1] != len(self.property_names):
            raise ValueError(
                f"prediction의 property 차원({tuple(prediction.shape)})이 "
                f"property_names 길이({len(self.property_names)})와 일치하지 않습니다."
            )

        raw_loss = self.element_loss.compute(prediction, target)  # (B, P)

        weights = self.weighting.get_weights(self.property_names).to(
            device=prediction.device, dtype=prediction.dtype
        )  # (P,)
        weighted_mask = mask * weights.unsqueeze(0)  # (B, P), 결측 위치는 0

        weighted_loss = raw_loss * weighted_mask

        if self.reduction == "sum":
            return weighted_loss.sum()

        denom = weighted_mask.sum()
        if denom.item() == 0:
            raise ValueError(
                "이 배치에 유효한(mask=1) 라벨이 하나도 없습니다. "
                "reduction='mean'으로는 loss를 계산할 수 없습니다 — "
                "배치 구성 또는 property_weight 설정을 확인하세요."
            )
        return weighted_loss.sum() / denom
