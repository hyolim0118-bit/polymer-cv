"""BaseLoss — Trainer가 의존하는 유일한 Loss 인터페이스."""

from abc import ABC, abstractmethod

import torch
from torch import Tensor


class BaseLoss(torch.nn.Module, ABC):
    """모든 loss 구현체의 최상위 인터페이스.

    Trainer는 이 인터페이스의 ``forward`` 시그니처만 알고 있으면 되며,
    내부적으로 어떤 오차 함수/가중치 전략이 쓰이는지는 전혀 알 필요가 없다.

    Example:
        >>> criterion = LossFactory.build(config, property_names=["Tg", "FFV"])
        >>> loss = criterion(prediction, target, mask)
        >>> loss.backward()
    """

    @abstractmethod
    def forward(self, prediction: Tensor, target: Tensor, mask: Tensor) -> Tensor:
        """Loss를 계산한다.

        Args:
            prediction: 모델 예측값, shape ``(batch_size, num_properties)``.
            target: 정답값, shape ``(batch_size, num_properties)``.
                결측 위치의 값은 임의의 값이어도 무방하며, ``mask``로 제외된다.
            mask: 라벨 존재 여부, shape ``(batch_size, num_properties)``.
                1이면 해당 (샘플, property) 라벨이 존재, 0이면 결측.

        Returns:
            스칼라 loss 텐서.
        """
        raise NotImplementedError
