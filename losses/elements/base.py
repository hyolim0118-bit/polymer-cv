"""ElementLoss — mask/weighting을 전혀 모르는 원소 단위 오차 계산 인터페이스."""

from abc import ABC, abstractmethod

from torch import Tensor


class ElementLoss(ABC):
    """오차 함수(MSE/MAE/Huber 등)의 추상 인터페이스.

    구현체는 ``prediction``/``target``만 보고, mask나 property별 가중치가
    적용되기 전의 원소별(row=sample, col=property) 오차 텐서를 반환해야 한다.
    mask 적용과 가중치 적용, reduction은 모두 ``MaskedWeightedLoss``의 책임이다.

    Example:
        >>> element_loss = MaskedMSEElement()
        >>> raw = element_loss.compute(prediction, target)
        >>> raw.shape
        torch.Size([32, 5])
    """

    @abstractmethod
    def compute(self, prediction: Tensor, target: Tensor) -> Tensor:
        """원소별 오차를 계산한다.

        Args:
            prediction: shape ``(batch_size, num_properties)``.
            target: shape ``(batch_size, num_properties)``.

        Returns:
            mask/weight 적용 전의 원소별 오차, shape ``(batch_size, num_properties)``.

        Raises:
            ValueError: ``prediction``과 ``target``의 shape이 다르거나 2차원이 아닌 경우.
        """
        raise NotImplementedError

    @staticmethod
    def _validate_shapes(prediction: Tensor, target: Tensor) -> None:
        """prediction/target의 shape을 검증한다.

        Args:
            prediction: 검증할 예측값 텐서.
            target: 검증할 정답값 텐서.

        Raises:
            ValueError: shape이 서로 다르거나, 2차원 텐서가 아닌 경우.
        """
        if prediction.shape != target.shape:
            raise ValueError(
                "prediction과 target의 shape이 다릅니다: "
                f"{tuple(prediction.shape)} vs {tuple(target.shape)}"
            )
        if prediction.dim() != 2:
            raise ValueError(
                "prediction/target은 (batch_size, num_properties) 형태의 "
                f"2차원 텐서여야 합니다. 받은 shape: {tuple(prediction.shape)}"
            )
