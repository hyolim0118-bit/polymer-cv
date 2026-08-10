"""
model.py
--------
STEP7 마이그레이션의 핵심 변경 파일.
기존 코드에서 바뀌는 부분은 "backbone을 어떻게 만드는가" 뿐이며,
head / trainer / dataset은 전혀 수정하지 않는다.
"""

from typing import Any, Dict

import torch
import torch.nn as nn

from .backbone import BackboneFactory
from .head.multitask_head import MultiTaskHead  # 기존 STEP4 Head, 변경 없음


class MultiTaskModel(nn.Module):
    """
    backbone(교체 가능) + multi-task regression head(기존 그대로)를 조립하는 모델.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Parameters
        ----------
        config : dict
            config.yaml 전체 중 필요한 부분:
            {
                "model": {"backbone": "resnet18", "pretrained": true, "freeze_backbone": false},
                "head": {"properties": ["Tg", "Density", "FFV", "Tc", "Rg"]}
            }
        """
        super().__init__()

        model_cfg = config.get("model", {})
        head_cfg = config.get("head", {})

        # ---- 변경된 부분: factory를 통해 backbone 생성 ----
        self.backbone = BackboneFactory.create(model_cfg)

        # ---- 변경 없는 부분: 기존 head를 backbone.feature_dim으로 초기화 ----
        properties = head_cfg.get("properties", ["Tg", "Density", "FFV", "Tc", "Rg"])
        self.head = MultiTaskHead(
            in_features=self.backbone.feature_dim,
            properties=properties,
        )

        # 실험 재현성: backbone 메타데이터 기록 (Design Goal #6)
        self.backbone_meta = {
            "backbone_name": getattr(self.backbone, "variant", model_cfg.get("backbone")),
            "num_parameters": self.backbone.num_parameters(),
            "trainable_parameters": self.backbone.num_parameters(trainable_only=True),
            "feature_dim": self.backbone.feature_dim,
        }

    def forward(self, x: torch.Tensor):
        features = self.backbone(x)          # (B, feature_dim)
        predictions = self.head(features)    # 기존 head의 출력 형태 그대로 (예: dict[str, Tensor])
        return predictions

    def get_cam_target_layer(self):
        """STEP12 Grad-CAM에서 사용할 대상 레이어를 그대로 전달."""
        return self.backbone.get_cam_target_layer()

    def get_experiment_meta(self) -> Dict[str, Any]:
        """실험 로깅용 메타데이터 (backbone 이름, 파라미터 수, feature_dim)."""
        return dict(self.backbone_meta)
