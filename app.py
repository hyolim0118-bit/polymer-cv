"""
app.py

발표 라이브 시연용 Gradio 데모 앱.
SMILES 입력 -> 이미지 렌더링 -> 5-fold 앙상블 예측 -> Grad-CAM(fold1 대표) 출력.

실행: python app.py
(프로젝트 루트에서, ensemble/ 와 같은 레벨)

=== 실행 전 체크리스트 ===
1. [TODO] render_smiles_to_image() 함수 안에 실제 프로젝트의 렌더링 로직 채워넣기
   (preprocessing/ 또는 dataset 클래스에서 찾은 함수 그대로 가져오거나 호출)
2. 랩실 SMILES(PA66, PPE 등)로 미리 한 번 실행해보고 파싱/렌더링 에러 없는지 확인
3. EXAMPLE_SMILES 리스트에 검증된 SMILES 채워넣기 (라이브 중 오타 방지용 드롭다운)
"""

import sys
import tempfile
from pathlib import Path

import cv2
import gradio as gr
import matplotlib.pyplot as plt
import numpy as np
import requests
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent / "ensemble"))
from ensemble_common import build_final_config, load_fold_model, load_pool_and_metadata, get_fold_splits  # noqa: E402
from datasets.dataset import LABEL_ORDER, compute_fold_stats  # noqa: E402
from datasets.transforms import get_valid_transform  # noqa: E402

# 실제 프로젝트 렌더링 함수 그대로 사용 (preprocessing/ 폴더)
from preprocessing.smiles_to_mol import parse_smiles  # noqa: E402
from preprocessing.mol_to_image import render_mol_to_png, IMAGE_SIZE  # noqa: E402

N_FOLDS = 5
GRADCAM_FOLD = 1  # Grad-CAM 시각화는 fold1 대표 (STEP12와 동일)
TARGET_LAYER_INDEX = 7  # feature_extractor 안 layer4 인덱스

# OOF 평가 기준 property별 MAE (oof_per_property.csv 결과, 신뢰구간 표시용)
OOF_MAE = {
    "Tg": 53.886189425005746,
    "Density": 0.03266138727603087,
    "FFV": 0.0064683021001621915,
    "Tc": 0.03055107072443361,
    "Rg": 1.716363617641058,
}


# 자주 쓰이는 고분자 이름 -> 반복단위 SMILES (PubChem에는 대부분 없는 정보라 자체 보유)
# 이름 매칭은 대소문자/공백 무시. 필요하면 계속 추가.
COMMON_POLYMER_SMILES = {
    "pa66": "*NCCCCCCNC(=O)CCCCC(=O)*",
    "nylon66": "*NCCCCCCNC(=O)CCCCC(=O)*",
    "nylon 6,6": "*NCCCCCCNC(=O)CCCCC(=O)*",
    "ppe": "*Oc1cc(C)c(*)cc1C",
    "polyphenyleneether": "*Oc1cc(C)c(*)cc1C",
    "polyethylene": "*CC*",
    "pe": "*CC*",
    "polypropylene": "*CC(C)*",
    "pp": "*CC(C)*",
    "polystyrene": "*CC(c1ccccc1)*",
    "ps": "*CC(c1ccccc1)*",
    "pvc": "*CC(Cl)*",
    "polyvinylchloride": "*CC(Cl)*",
    "pet": "*OCCOC(=O)c1ccc(C(=O)*)cc1",
    "polyethyleneterephthalate": "*OCCOC(=O)c1ccc(C(=O)*)cc1",
    "pmma": "*CC(C)(C(=O)OC)*",
    "polymethylmethacrylate": "*CC(C)(C(=O)OC)*",
}


def _normalize_name(name: str) -> str:
    return name.strip().lower().replace(" ", "").replace("-", "").replace("_", "")


# ---------------------------------------------------------------------------
# 0. 이름 -> SMILES 변환 (자체 고분자 사전 우선, 없으면 PubChem 조회)
# ---------------------------------------------------------------------------
def name_to_smiles(name: str, timeout: float = 5.0):
    """1) 자체 고분자 이름 사전에서 먼저 찾고, 2) 없으면 PubChem PUG REST로 조회.
    PubChem은 개별 소분자 위주 DB라 '폴리에틸렌' 같은 고분자 이름 자체는 대부분 없음.
    """
    normalized = _normalize_name(name)
    if normalized in COMMON_POLYMER_SMILES:
        return COMMON_POLYMER_SMILES[normalized]

    try:
        url = (
            "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/"
            f"{name}/property/CanonicalSMILES/TXT"
        )
        resp = requests.get(url, timeout=timeout)
        if resp.status_code == 200:
            smiles = resp.text.strip().splitlines()[0].strip()
            if smiles:
                return smiles
    except Exception:
        pass
    return None


def resolve_input_to_smiles(text: str):
    """입력이 SMILES면 그대로, 아니면 이름으로 간주해 자체 사전 -> PubChem 순으로 조회.
    반환: (최종 smiles 또는 None, 안내 메시지 또는 None)
    """
    text = text.strip()

    # 1) 우선 SMILES로 바로 파싱 시도
    if parse_smiles(text) is not None:
        return text, None

    # 2) SMILES 파싱 실패 -> 화합물 이름으로 간주하고 조회
    resolved = name_to_smiles(text)
    if resolved is not None:
        return resolved, f"'{text}' -> SMILES 자동 변환: {resolved}"

    return None, (
        f"'{text}'를 SMILES로도, 화합물 이름으로도 인식하지 못했어요. "
        "SMILES를 직접 입력하거나 철자를 확인해주세요."
    )


# ---------------------------------------------------------------------------
# 1. SMILES -> 이미지 렌더링 (학습 때 쓴 preprocessing/ 함수 그대로 사용)
# ---------------------------------------------------------------------------
def render_smiles_to_image(smiles: str):
    """preprocessing.smiles_to_mol.parse_smiles + preprocessing.mol_to_image.render_mol_to_png
    를 그대로 호출해서, 학습 이미지와 동일한 스타일(224px, bondLineWidth=2, padding=0.12)로 렌더링.
    """
    mol = parse_smiles(smiles)
    if mol is None:
        return None, "SMILES 파싱 실패. 문자열을 확인해주세요."

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir) / "render.png"
        ok = render_mol_to_png(mol, tmp_path, size=IMAGE_SIZE)
        if not ok:
            return None, "이미지 렌더링 실패."
        img = Image.open(tmp_path).convert("RGB")
        img.load()  # 임시 파일 삭제 전에 완전히 메모리로 로드
    return img, None


# ---------------------------------------------------------------------------
# 2. 모델 준비 (앱 시작 시 한 번만 로드 -> 전역 캐시)
# ---------------------------------------------------------------------------
print("모델 로딩 중 (5-fold)...")
_config = build_final_config()
_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_pool_df, _ = load_pool_and_metadata(_config)
_splits = dict(get_fold_splits(_pool_df))

_models = {}
_label_stats = {}
for fold in range(1, N_FOLDS + 1):
    train_idx, _ = _splits[fold]
    train_df = _pool_df.iloc[train_idx].reset_index(drop=True)
    label_mean, label_std = compute_fold_stats(train_df, LABEL_ORDER)
    _label_stats[fold] = (label_mean, label_std)
    _models[fold] = load_fold_model(fold, _config, _device)
    print(f"  fold{fold} 로드 완료")

_image_cfg = _config["image"]
_transform = get_valid_transform(
    _image_cfg["size"], _image_cfg["normalize_mean"], _image_cfg["normalize_std"]
)
print(f"모델 로딩 완료. device={_device}")


# ---------------------------------------------------------------------------
# 3. Grad-CAM (STEP12와 동일 로직, fold1 모델 사용)
# ---------------------------------------------------------------------------
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


_gradcam_model = _models[GRADCAM_FOLD]
_target_layer = _gradcam_model.backbone.feature_extractor[TARGET_LAYER_INDEX]
_gradcam = GradCAM(_gradcam_model, _target_layer)


def unnormalize_image(image_tensor, mean, std):
    img = image_tensor.squeeze().cpu().numpy().transpose(1, 2, 0)
    img = img * np.array(std) + np.array(mean)
    return np.clip(img * 255, 0, 255).astype(np.uint8)


def overlay_heatmap(image_rgb, cam, alpha=0.4):
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    return (heatmap * alpha + image_rgb * (1 - alpha)).astype(np.uint8)


# ---------------------------------------------------------------------------
# Grad-CAM TTA (Test-Time Rotation Augmentation) + 배경 마스킹
# gradcam_chem_analysis.py 실험에서 4방향 회전 평균이 배경(캔버스 위치) 아티팩트를
# 크게 줄이고 화학적으로 의미 있는 attention을 강화하는 것으로 확인되어 앱에도 적용.
# 재학습 없이 추론 시점에서만 동작 (모델 가중치 변경 없음).
# ---------------------------------------------------------------------------
TTA_ANGLES = (0, 90, 180, 270)


def _rotate(img, angle, border_value):
    h, w = img.shape[:2]
    center = (w / 2, h / 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(img, M, (w, h), borderValue=border_value)


def generate_gradcam_tta(image_np, prop_idx, gradcam, transform, device, angles=TTA_ANGLES):
    """이미지를 여러 각도로 회전 -> 각각 Grad-CAM 계산 -> 원래 각도로 되돌려 평균."""
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
    """분자 영역(=흰 배경이 아닌 픽셀)만 1, 배경은 0인 마스크."""
    gray = image_rgb.mean(axis=2)
    mask = (gray < white_thresh).astype(np.float32)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=1)
    return mask


def apply_background_mask(cam, mask):
    """배경 영역 CAM을 0으로 강제하고 재정규화."""
    masked = cam * mask
    if masked.max() > 0:
        masked = masked / masked.max()
    return masked


# ---------------------------------------------------------------------------
# 4. 예측 + Grad-CAM 파이프라인 (Gradio 콜백)
# ---------------------------------------------------------------------------
def predict(smiles: str):
    if not smiles or not smiles.strip():
        return None, "SMILES 또는 화합물 이름을 입력해주세요.", None

    resolved_smiles, note = resolve_input_to_smiles(smiles)
    if resolved_smiles is None:
        return None, note, None

    # 도메인 체크: 이 모델은 고분자 반복단위(* 연결점 포함) 전용으로 학습됨.
    # 일반 소분자(예: H2O, 벤젠 등)는 학습 범위 밖이라 예측값이 무의미하므로 차단.
    if "*" not in resolved_smiles:
        msg = (
            "이 모델은 고분자 반복단위(SMILES에 '*' 연결점 포함)만 예측 가능해요.\n"
            f"입력하신 값은 일반 분자({resolved_smiles})로 인식되어 예측을 진행하지 않았습니다.\n"
            "고분자 이름이나 반복단위가 포함된 SMILES를 입력해주세요."
        )
        if note:
            msg = note + "\n\n" + msg
        return None, msg, None

    pil_img, err = render_smiles_to_image(resolved_smiles)
    if err:
        return None, err, None

    image_np = np.array(pil_img)
    image_tensor = _transform(image=image_np)["image"].unsqueeze(0).to(_device)

    # 5-fold 앙상블 예측 (fold별 정규화 통계로 역변환 후 평균)
    fold_preds = []
    for fold in range(1, N_FOLDS + 1):
        model = _models[fold]
        label_mean, label_std = _label_stats[fold]

        # label_mean/label_std는 {property_name: value} 형태의 dict이므로
        # LABEL_ORDER 순서에 맞춰 array로 변환해야 모델 출력(array)과 연산 가능
        mean_arr = np.array([label_mean[name] for name in LABEL_ORDER])
        std_arr = np.array([label_std[name] for name in LABEL_ORDER])

        with torch.no_grad():
            out = model(image_tensor).cpu().numpy()[0]
        real_values = out * std_arr + mean_arr
        fold_preds.append(real_values)
    ensemble_pred = np.mean(fold_preds, axis=0)

    result_text = "\n".join(
        f"{name}: {value:.4f}  (±{OOF_MAE[name]:.4f}, OOF MAE 기준)"
        for name, value in zip(LABEL_ORDER, ensemble_pred)
    )
    if note:
        result_text = note + "\n\n" + result_text

    # Grad-CAM (fold1 모델, TTA 4방향 회전평균 + 배경마스킹, property별 5장)
    fig, axes = plt.subplots(1, len(LABEL_ORDER), figsize=(4 * len(LABEL_ORDER), 4))
    label_mean1, label_std1 = _label_stats[GRADCAM_FOLD]
    for i, prop_name in enumerate(LABEL_ORDER):
        cam = generate_gradcam_tta(image_np, i, _gradcam, _transform, _device)
        bg_mask = build_background_mask(image_np)
        cam = apply_background_mask(cam, bg_mask)
        image_rgb = unnormalize_image(
            image_tensor, _image_cfg["normalize_mean"], _image_cfg["normalize_std"]
        )
        overlay = overlay_heatmap(image_rgb, cam)
        axes[i].imshow(overlay)
        axes[i].set_title(prop_name, fontsize=11)
        axes[i].axis("off")
    fig.tight_layout()

    return pil_img, result_text, fig


# ---------------------------------------------------------------------------
# 5. Gradio UI
# ---------------------------------------------------------------------------
with gr.Blocks(title="Polymer Property Predictor") as demo:
    gr.Markdown("# 고분자 물성 예측 데모 (Tg / Density / FFV / Tc / Rg)")
    gr.Markdown("SMILES 또는 화합물 이름을 자유롭게 입력하세요. 이름이면 자동으로 SMILES로 변환됩니다.")

    with gr.Row():
        with gr.Column(scale=1):
            smiles_input = gr.Textbox(
                label="SMILES 또는 화합물 이름 입력",
                placeholder="예: *NCCCCCCNC(=O)* 또는 polyethylene 등",
            )
            predict_btn = gr.Button("예측 실행", variant="primary")

        with gr.Column(scale=1):
            mol_image = gr.Image(label="렌더링된 분자 구조")
            pred_text = gr.Textbox(label="예측값", lines=6)

    gradcam_plot = gr.Plot(label="Grad-CAM (property별)")

    predict_btn.click(predict, inputs=smiles_input, outputs=[mol_image, pred_text, gradcam_plot])


if __name__ == "__main__":
    demo.launch(share=True, show_api=False)