"""guided_gradcam.py

STEP12 확장: Guided Grad-CAM.
기존 gradcam.py의 GradCAM(coarse, class-discriminative)에
Guided Backpropagation(fine-grained, pixel-level)을 곱해서
더 세밀한 픽셀 단위 근거를 시각화한다.

원리:
- Grad-CAM: "어느 영역"이 중요한지 (coarse, layer4 해상도)
- Guided Backprop: "어느 픽셀"이 중요한지 (fine, 입력 해상도) —
  단, 모든 ReLU의 backward에서 음수 gradient를 0으로 죽여서
  "양의 영향을 준 부분만" 강조하는 방식
- Guided Grad-CAM = 위 둘을 element-wise 곱 (Grad-CAM으로 영역을 좁히고,
  그 안에서 Guided Backprop으로 세밀한 픽셀 패턴을 봄)

실행: python guided_gradcam.py
(gradcam.py와 같은 위치, 프로젝트 루트에서 실행)
"""

import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent / "ensemble"))
from ensemble_common import build_final_config, load_fold_model, load_pool_and_metadata, get_fold_splits  # noqa: E402
from datasets.dataset import LABEL_ORDER, PolymerDataset, compute_fold_stats  # noqa: E402
from datasets.transforms import get_valid_transform  # noqa: E402

# gradcam.py의 GradCAM, unnormalize_image, overlay_heatmap 재사용
from gradcam import GradCAM, unnormalize_image, overlay_heatmap  # noqa: E402

FOLD_FOR_VISUALIZATION = 1
TARGET_LAYER_INDEX = 7
N_SAMPLES_PER_PROPERTY = 3
OUTPUT_DIR = Path("results/step12_guided_gradcam")


class GuidedBackprop:
    """모든 ReLU의 backward hook을 걸어서, 음수 gradient를 0으로 죽인다.
    (표준 Guided Backpropagation, Springenberg et al. 2015)
    """

    def __init__(self, model):
        self.model = model
        self._hooks = []
        self._register_hooks()

    def _register_hooks(self):
        def relu_backward_hook(module, grad_input, grad_output):
            # forward에서 나온 output이 0 이하인 부분과, backward gradient가
            # 음수인 부분을 둘 다 0으로 죽인다 (guided = deconvnet + backprop 결합)
            return (torch.clamp(grad_input[0], min=0.0),)

        for module in self.model.modules():
            if isinstance(module, nn.ReLU):
                # inplace=True인 ReLU에 backward hook을 걸면 autograd가
                # "view + inplace 수정" 충돌로 에러를 낸다 (RuntimeError:
                # Output 0 of BackwardHookFunctionBackward is a view...).
                # inplace를 꺼서 별도 텐서를 만들게 하면 해결됨.
                module.inplace = False
                h = module.register_full_backward_hook(relu_backward_hook)
                self._hooks.append(h)

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()

    def generate(self, image_tensor, property_index):
        """image_tensor: (1, 3, H, W), requires_grad=True 필요.

        Returns: (H, W) 0~1 정규화된 saliency map.
        """
        image_tensor = image_tensor.clone().detach().requires_grad_(True)
        self.model.zero_grad()
        output = self.model(image_tensor)
        score = output[0, property_index]
        score.backward()

        # 입력 이미지에 대한 gradient (채널 3개 -> 절대값 최대로 1채널 축약)
        grad = image_tensor.grad.detach().squeeze().cpu().numpy()  # (3, H, W)
        grad = np.transpose(grad, (1, 2, 0))  # (H, W, 3)
        saliency = np.max(np.abs(grad), axis=2)  # (H, W)

        saliency = saliency - saliency.min()
        if saliency.max() > 0:
            saliency = saliency / saliency.max()
        return saliency


def make_guided_gradcam(gradcam_map, guided_map):
    """Grad-CAM(coarse)과 Guided Backprop(fine)을 element-wise 곱.
    둘 다 (H, W), 0~1 정규화된 상태여야 함 (크기 같아야 함).
    """
    combined = gradcam_map * guided_map
    combined = combined - combined.min()
    if combined.max() > 0:
        combined = combined / combined.max()
    return combined


def saliency_to_rgb(saliency):
    """0~1 saliency map을 grayscale RGB 이미지로 변환 (오버레이 없이 그 자체로 시각화)."""
    gray = (saliency * 255).astype(np.uint8)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)


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
    guided_bp = GuidedBackprop(model)

    for prop_idx, prop_name in enumerate(LABEL_ORDER):
        candidate_indices = [
            i for i in range(len(dataset)) if dataset[i][2][prop_idx].item() == 1.0
        ][:N_SAMPLES_PER_PROPERTY]

        if not candidate_indices:
            print(f"  {prop_name}: 라벨 있는 샘플 없음, 스킵")
            continue

        # 3행: 원본 Grad-CAM overlay / Guided Backprop / Guided Grad-CAM
        fig, axes = plt.subplots(3, len(candidate_indices), figsize=(5 * len(candidate_indices), 14))
        if len(candidate_indices) == 1:
            axes = axes.reshape(3, 1)

        for col, idx in enumerate(candidate_indices):
            image_tensor, label, mask = dataset[idx]
            image_tensor = image_tensor.unsqueeze(0).to(device)

            # 1) 기존 Grad-CAM (coarse)
            cam = gradcam.generate(image_tensor, prop_idx)
            image_rgb = unnormalize_image(
                image_tensor, image_cfg["normalize_mean"], image_cfg["normalize_std"]
            )
            gradcam_overlay = overlay_heatmap(image_rgb, cam)

            # 2) Guided Backprop (fine, pixel-level)
            guided_saliency = guided_bp.generate(image_tensor, prop_idx)
            guided_rgb = saliency_to_rgb(guided_saliency)

            # 3) Guided Grad-CAM = 둘의 곱
            combined = make_guided_gradcam(cam, guided_saliency)
            combined_rgb = saliency_to_rgb(combined)

            axes[0, col].imshow(gradcam_overlay)
            axes[0, col].set_title(f"Grad-CAM (sample {idx})", fontsize=10)
            axes[0, col].axis("off")

            axes[1, col].imshow(guided_rgb)
            axes[1, col].set_title(f"Guided Backprop (sample {idx})", fontsize=10)
            axes[1, col].axis("off")

            axes[2, col].imshow(combined_rgb)
            axes[2, col].set_title(f"Guided Grad-CAM (sample {idx})", fontsize=10)
            axes[2, col].axis("off")

        fig.suptitle(f"Guided Grad-CAM: {prop_name}", fontsize=14)
        fig.tight_layout()
        save_path = OUTPUT_DIR / f"guided_gradcam_{prop_name}.png"
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
        print(f"  {prop_name}: 저장 완료 -> {save_path}")

    guided_bp.remove_hooks()
    print(f"\n완료. {OUTPUT_DIR}/ 에 property별 Guided Grad-CAM 이미지 저장됨.")


if __name__ == "__main__":
    main()