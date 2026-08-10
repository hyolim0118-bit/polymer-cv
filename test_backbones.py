"""
test_backbones.py
-----------------
STEP7 구현이 끝난 뒤 가장 먼저 돌려볼 스모크 테스트.
- 두 backbone(resnet18, efficientnet_b0)이 동일한 인터페이스로 동작하는지
- forward() 출력 shape이 기대한 feature_dim과 일치하는지
- get_cam_target_layer()가 정상적으로 반환되는지
를 확인한다. 실제 학습 없이 몇 초 안에 끝난다.

실행: python test_backbones.py
"""

import torch

from models.backbone import BackboneFactory


def check_backbone(name: str, pretrained: bool = False):
    print(f"\n=== {name} ===")
    backbone = BackboneFactory.create({"backbone": name, "pretrained": pretrained})
    backbone.eval()

    dummy_input = torch.randn(2, 3, 224, 224)  # batch=2 스모크 테스트
    with torch.no_grad():
        out = backbone(dummy_input)

    assert out.shape == (2, backbone.feature_dim), (
        f"{name}: 출력 shape {out.shape} != 기대값 (2, {backbone.feature_dim})"
    )

    cam_layer = backbone.get_cam_target_layer()
    print(f"feature_dim       : {backbone.feature_dim}")
    print(f"output shape       : {tuple(out.shape)}")
    print(f"num_parameters     : {backbone.num_parameters():,}")
    print(f"cam_target_layer   : {type(cam_layer).__name__ if cam_layer is not None else None}")
    print(f"OK: {name} 정상 동작")


if __name__ == "__main__":
    print("사용 가능한 backbone:", BackboneFactory.available_backbones())

    for backbone_name in ["resnet18", "efficientnet_b0"]:
        check_backbone(backbone_name, pretrained=False)

    print("\n모든 backbone이 동일 인터페이스로 정상 동작합니다.")
