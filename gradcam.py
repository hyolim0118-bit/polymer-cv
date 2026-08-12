"""gradcam.py

STEP12: Grad-CAM으로 모델이 어느 이미지 영역을 보고 property를 예측하는지
시각화한다. 대표로 fold1 모델(ensemble의 첫 번째 fold)을 사용한다 — Grad-CAM은
"해석"이 목적이라, 5개 fold 전부를 쓸 필요는 없다.

Target layer: ResNet18Backbone.feature_extractor의 layer4
(children() 순서: conv1, bn1, relu, maxpool, layer1, layer2, layer3, layer4,
avgpool -> fc 제거된 상태라 인덱스 7이 layer4)

실행: python gradcam.py
(ensemble/ 폴더와 같은 레벨, 프로젝트 루트에서 PYTHONPATH=. python gradcam.py)
"""

import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent / "ensemble"))
from ensemble_common import build_final_config, load_fold_model, load_pool_and_metadata, get_fold_splits  # noqa: E402
from datasets.dataset import LABEL_ORDER, PolymerDataset, compute_fold_stats  # noqa: E402
from datasets.transforms import get_valid_transform  # noqa: E402

FOLD_FOR_VISUALIZATION = 1  # 대표로 사용할 fold (1~5 중 하나)
TARGET_LAYER_INDEX = 7  # feature_extractor 안에서 layer4의 인덱스
N_SAMPLES_PER_PROPERTY = 3  # property별로 시각화할 샘플 수
OUTPUT_DIR = Path("results/step12_gradcam")


class GradCAM:
    """표준 Grad-CAM 구현. target_layer의 activation/gradient를 hook으로 잡는다."""

    def __init__(self, model, target_layer):
        self.model = model
        self.activations = None
        self.gradients = None
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, image_tensor, property_index):
        """image_tensor: (1, 3, H, W). property_index: 0~4 (LABEL_ORDER 순서).

        Returns: (H, W) 0~1 정규화된 히트맵 (원본 이미지 크기로 resize됨).
        """
        self.model.zero_grad()
        output = self.model(image_tensor)  # (1, 5)
        score = output[0, property_index]
        score.backward()

        # gradient를 공간 축(H, W)에 대해 global average pooling -> 채널별 가중치
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)  # (1, 1, h, w)
        cam = F.relu(cam)

        cam = cam.squeeze().cpu().numpy()
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()

        cam = cv2.resize(cam, (image_tensor.shape[3], image_tensor.shape[2]))
        return cam


def unnormalize_image(image_tensor, mean, std):
    """정규화된 이미지 텐서를 0~255 RGB uint8 이미지로 되돌린다 (시각화용)."""
    img = image_tensor.squeeze().cpu().numpy().transpose(1, 2, 0)  # (H, W, 3)
    img = img * np.array(std) + np.array(mean)
    img = np.clip(img * 255, 0, 255).astype(np.uint8)
    return img


def overlay_heatmap(image_rgb, cam, alpha=0.4):
    """원본 이미지 위에 Grad-CAM 히트맵을 겹쳐서 반환."""
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = (heatmap * alpha + image_rgb * (1 - alpha)).astype(np.uint8)
    return overlay


def main():
    config = build_final_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"사용 device: {device}, 시각화용 fold: {FOLD_FOR_VISUALIZATION}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    pool_df, _ = load_pool_and_metadata(config)
    splits = get_fold_splits(pool_df)
    train_idx, valid_idx = dict(splits)[FOLD_FOR_VISUALIZATION]
    train_df = pool_df.iloc[train_idx].reset_index(drop=True)
    valid_df = pool_df.iloc[valid_idx].reset_index(drop=True)
    label_mean, label_std = compute_fold_stats(train_df, LABEL_ORDER)

    image_cfg = config["image"]
    data_cfg = config["data"]
    transform = get_valid_transform(
        image_cfg["size"], image_cfg["normalize_mean"], image_cfg["normalize_std"]
    )
    dataset = PolymerDataset(valid_df, data_cfg["image_root"], transform, label_mean, label_std)

    model = load_fold_model(FOLD_FOR_VISUALIZATION, config, device)
    target_layer = model.backbone.feature_extractor[TARGET_LAYER_INDEX]
    gradcam = GradCAM(model, target_layer)

    for prop_idx, prop_name in enumerate(LABEL_ORDER):
        # 이 property에 라벨이 있는 샘플만 골라서 시각화 (mask=1)
        candidate_indices = [
            i for i in range(len(dataset)) if dataset[i][2][prop_idx].item() == 1.0
        ][:N_SAMPLES_PER_PROPERTY]

        if not candidate_indices:
            print(f"  {prop_name}: 라벨 있는 샘플 없음, 스킵")
            continue

        fig, axes = plt.subplots(1, len(candidate_indices), figsize=(5 * len(candidate_indices), 5))
        if len(candidate_indices) == 1:
            axes = [axes]

        for ax, idx in zip(axes, candidate_indices):
            image_tensor, label, mask = dataset[idx]
            image_tensor = image_tensor.unsqueeze(0).to(device)

            cam = gradcam.generate(image_tensor, prop_idx)
            image_rgb = unnormalize_image(
                image_tensor, image_cfg["normalize_mean"], image_cfg["normalize_std"]
            )
            overlay = overlay_heatmap(image_rgb, cam)

            ax.imshow(overlay)
            ax.set_title(f"{prop_name} (sample {idx})", fontsize=10)
            ax.axis("off")

        fig.suptitle(f"Grad-CAM: {prop_name}", fontsize=13)
        fig.tight_layout()
        save_path = OUTPUT_DIR / f"gradcam_{prop_name}.png"
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
        print(f"  {prop_name}: 저장 완료 -> {save_path}")

    print(f"\n완료. {OUTPUT_DIR}/ 에 property별 Grad-CAM 이미지 저장됨.")


if __name__ == "__main__":
    main()
