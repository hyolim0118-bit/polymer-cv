"""MaskedMAEElement — Mean Absolute Error의 원소 단위 계산."""

from torch import Tensor

from .base import ElementLoss


class MaskedMAEElement(ElementLoss):
    """Mean Absolute Error: ``|prediction - target|``.

    Example:
        >>> loss_fn = MaskedMAEElement()
        >>> raw = loss_fn.compute(prediction, target)  # (B, P)
    """

    def compute(self, prediction: Tensor, target: Tensor) -> Tensor:
        """|prediction - target|을 원소별로 계산한다.

        Args:
            prediction: shape ``(batch_size, num_properties)``.
            target: shape ``(batch_size, num_properties)``.

        Returns:
            원소별 absolute error, shape ``(batch_size, num_properties)``.
        """
        self._validate_shapes(prediction, target)
        return (prediction - target).abs()
