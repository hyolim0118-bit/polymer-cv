"""공유 feature를 받아 5개 물성을 동시에 예측하는 Multi-task Regression Head."""

from __future__ import annotations

import logging

from torch import Tensor, nn

logger = logging.getLogger(__name__)

# 예측 대상 물성 개수: Tg, Density, FFV, Tc, Rg
# (순서는 datasets.dataset.PolymerDataset.LABEL_ORDER와 반드시 일치해야 한다)
NUM_TARGETS: int = 5


class MultiTaskHead(nn.Module):
    """Dropout -> BatchNorm1d -> Linear 순으로 구성된 회귀 Head."""

    def __init__(self, in_features: int, dropout: float = 0.3) -> None:
        """MultiTaskHead를 초기화한다.

        Args:
            in_features: Backbone에서 전달되는 feature 차원.
            dropout: Dropout 확률.
        """
        super().__init__()

        self.dropout = nn.Dropout(p=dropout)
        self.batch_norm = nn.BatchNorm1d(in_features)
        self.fc = nn.Linear(in_features, NUM_TARGETS)

        logger.info(
            "MultiTaskHead 생성 완료 (in_features=%d, dropout=%.2f, num_targets=%d)",
            in_features,
            dropout,
            NUM_TARGETS,
        )

    def forward(self, x: Tensor) -> Tensor:
        """공유 feature 벡터로부터 5개 물성 예측값을 계산한다.

        mask 처리는 이 Head가 아닌 Trainer의 Loss 단계에서 이루어진다.

        Args:
            x: Backbone에서 나온 feature 텐서, shape (B, in_features).

        Returns:
            예측 텐서, shape (B, NUM_TARGETS). 컬럼 순서는
            ["Tg", "Density", "FFV", "Tc", "Rg"]를 따른다.
        """
        x = self.dropout(x)
        x = self.batch_norm(x)
        return self.fc(x)
