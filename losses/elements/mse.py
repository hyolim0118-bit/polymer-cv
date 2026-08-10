"""MaskedMSEElement — Mean Squared Error의 원소 단위 계산."""

from torch import Tensor

from .base import ElementLoss


class MaskedMSEElement(ElementLoss):
    """Mean Squared Error: ``(prediction - target) ** 2``.

    "Masked"라는 이름은 이 클래스가 mask를 적용한다는 뜻이 아니라,
    이 프로젝트의 masked multi-task 파이프라인에서 쓰이는 오차 함수라는
    맥락을 나타낸다. 실제 mask 적용은 ``MaskedWeightedLoss``가 수행한다.

    Example:
        >>> loss_fn = MaskedMSEElement()
        >>> raw = loss_fn.compute(prediction, target)  # (B, P)
    """

    def compute(self, prediction: Tensor, target: Tensor) -> Tensor:
        """(prediction - target) ** 2를 원소별로 계산한다.

        Args:
            prediction: shape ``(batch_size, num_properties)``.
            target: shape ``(batch_size, num_properties)``.

        Returns:
            원소별 squared error, shape ``(batch_size, num_properties)``.
        """
        self._validate_shapes(prediction, target)
        return (prediction - target) ** 2
