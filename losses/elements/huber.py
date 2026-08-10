"""MaskedHuberElement — Huber Loss의 원소 단위 계산."""

import torch
from torch import Tensor

from .base import ElementLoss


class MaskedHuberElement(ElementLoss):
    """Huber Loss: ``delta`` 이하 오차는 quadratic, 이상은 linear로 처리.

    outlier에 덜 민감하면서도 원점 근처에서는 MSE처럼 매끄러운 gradient를
    제공하는 오차 함수.

    Args:
        delta: quadratic/linear 구간을 나누는 threshold. 기본값 1.0.

    Raises:
        ValueError: ``delta``가 0 이하인 경우.

    Example:
        >>> loss_fn = MaskedHuberElement(delta=1.0)
        >>> raw = loss_fn.compute(prediction, target)  # (B, P)
    """

    def __init__(self, delta: float = 1.0) -> None:
        if delta <= 0:
            raise ValueError(f"delta는 양수여야 합니다. 받은 값: {delta}")
        self.delta = delta

    def compute(self, prediction: Tensor, target: Tensor) -> Tensor:
        """Huber loss를 원소별로 계산한다.

        Args:
            prediction: shape ``(batch_size, num_properties)``.
            target: shape ``(batch_size, num_properties)``.

        Returns:
            원소별 Huber loss, shape ``(batch_size, num_properties)``.
        """
        self._validate_shapes(prediction, target)
        diff = prediction - target
        abs_diff = diff.abs()
        quadratic = torch.clamp(abs_diff, max=self.delta)
        linear = abs_diff - quadratic
        return 0.5 * quadratic**2 + self.delta * linear
