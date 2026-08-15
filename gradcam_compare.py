"""
gradcam_compare.py

외부 검증에서 Tg 예측 성공(PET) vs 실패(PDMS) 사례를 Grad-CAM으로 나란히 비교.
app.py의 GradCAM(TTA+배경마스킹) 로직 그대로 재사용. 재학습 없음, fold1 모델 사용.

실행: python gradcam_compare.py
(프로젝트 루트에서, app.py와 같은 레벨)

출력: gradcam_success_vs_failure.png (같은 폴더에 저장)
"""

import sys
import tempfile
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent / "ensemble"))
from ensemble_common import build_final_config, load_fold_model, load_pool_and_metadata, get_fold_splits  # noqa: E402
from datasets.dataset import LABEL_ORDER, compute_fold_stats  # noqa: E402
from datasets.transforms import get_valid_transform  # noqa: E402

from preprocessing.smiles_to_mol import parse_smiles  # noqa: E402
from preprocessing.mol_to_image import render_mol_to_png, IMAGE_SIZE  # noqa: E402

GRADCAM_FOLD = 1
TARGET_LAYER_INDEX = 7
TTA_ANGLES = (0, 90, 180, 270)

# 비교할 두 케이스 (외부 검증 결과 기준: PET=성공, PDMS=실패, Tg 절대오차 기준)
CASES = [
    {
        "label": "PET (성공, AbsErr_Tg=18.3)",
        "smiles": "*OCCOC(=O)c1ccc(C(=O)*)cc1",
    },
    {
        "label": "PDMS (실패, AbsErr_Tg=211.9)",
        "smiles": "*O[Si](C)(C)*",
    },
]

TG_INDEX = LABEL_ORDER.index("Tg")


def _rotate(img, angle, border_value):
    h, w = img.shape[:2]
    center = (w / 2, h / 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(img, M, (w, h), borderValue=border_value)


class GradCAM:
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
        self.model.zero_grad()
        output = self.model(image_tensor)
        score = output[0, property_index]
        score.backward()
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = cam.squeeze().cpu().numpy()
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()
        cam = cv2.resize(cam, (image_tensor.shape[3], image_tensor.shape[2]))
        return cam


def generate_gradcam_tta(image_np, prop_idx, gradcam, transform, device, angles=TTA_ANGLES):
    cams = []
    for angle in angles:
        rotated_img = _rotate(image_np, angle, border_value=(255, 255, 255))
        image_tensor = transform(image=rotated_img)["image"].unsqueeze(0).to(device)
        cam_rotated = gradcam.generate(image_tensor, prop_idx)
        cam_back = _rotate(cam_rotated, -angle, border_value=0)
        cams.append(cam_back)
    cam_avg = np.mean(cams, axis=0)
    cam_avg = cam_avg - cam_avg.min()
    if cam_avg.max() > 0:
        cam_avg = cam_avg / cam_avg.max()
    return cam_avg


def build_background_mask(image_rgb, white_thresh=245):
    gray = image_rgb.mean(axis=2)
    mask = (gray < white_thresh).astype(np.float32)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=1)
    return mask


def apply_background_mask(cam, mask):
    masked = cam * mask
    if masked.max() > 0:
        masked = masked / masked.max()
    return masked


def unnormalize_image(image_tensor, mean, std):
    img = image_tensor.squeeze().cpu().numpy().transpose(1, 2, 0)
    img = img * np.array(std) + np.array(mean)
    return np.clip(img * 255, 0, 255).astype(np.uint8)


def overlay_heatmap(image_rgb, cam, alpha=0.4):
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    return (heatmap * alpha + image_rgb * (1 - alpha)).astype(np.uint8)


def render_smiles_to_image(smiles):
    mol = parse_smiles(smiles)
    if mol is None:
        raise ValueError(f"SMILES 파싱 실패: {smiles}")
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir) / "render.png"
        ok = render_mol_to_png(mol, tmp_path, size=IMAGE_SIZE)
        if not ok:
            raise ValueError(f"렌더링 실패: {smiles}")
        img = Image.open(tmp_path).convert("RGB")
        img.load()
    return img


def main():
    print("모델 로딩 중 (fold1)...")
    config = build_final_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pool_df, _ = load_pool_and_metadata(config)
    splits = dict(get_fold_splits(pool_df))
    train_idx, _ = splits[GRADCAM_FOLD]
    train_df = pool_df.iloc[train_idx].reset_index(drop=True)
    label_mean, label_std = compute_fold_stats(train_df, LABEL_ORDER)

    model = load_fold_model(GRADCAM_FOLD, config, device)
    image_cfg = config["image"]
    transform = get_valid_transform(
        image_cfg["size"], image_cfg["normalize_mean"], image_cfg["normalize_std"]
    )

    target_layer = model.backbone.feature_extractor[TARGET_LAYER_INDEX]
    gradcam = GradCAM(model, target_layer)

    mean_arr = np.array([label_mean[n] for n in LABEL_ORDER])
    std_arr = np.array([label_std[n] for n in LABEL_ORDER])

    fig, axes = plt.subplots(1, len(CASES), figsize=(6 * len(CASES), 6))
    if len(CASES) == 1:
        axes = [axes]

    for ax, case in zip(axes, CASES):
        print(f"처리 중: {case['label']}")
        pil_img = render_smiles_to_image(case["smiles"])
        image_np = np.array(pil_img)
        image_tensor = transform(image=image_np)["image"].unsqueeze(0).to(device)

        with torch.no_grad():
            out = model(image_tensor).cpu().numpy()[0]
        pred_tg = out[TG_INDEX] * std_arr[TG_INDEX] + mean_arr[TG_INDEX]

        cam = generate_gradcam_tta(image_np, TG_INDEX, gradcam, transform, device)
        bg_mask = build_background_mask(image_np)
        cam = apply_background_mask(cam, bg_mask)
        image_rgb = unnormalize_image(
            image_tensor, image_cfg["normalize_mean"], image_cfg["normalize_std"]
        )
        overlay = overlay_heatmap(image_rgb, cam)

        ax.imshow(overlay)
        ax.set_title(f"{case['label']}\nAI 예측 Tg={pred_tg:.1f}", fontsize=11)
        ax.axis("off")

    fig.suptitle("Grad-CAM: Tg 예측 성공 vs 실패 사례 비교", fontsize=13)
    fig.tight_layout()
    out_path = Path("gradcam_success_vs_failure.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"저장 완료: {out_path}")


if __name__ == "__main__":
    main()
