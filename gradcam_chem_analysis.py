"""
gradcam_chem_analysis.py

Property별 Grad-CAM attention이 화학적으로 의미 있는 부분(작용기/헤테로원자)에
실제로 집중되는지 대표 샘플군(N개)에 대해 정량화하는 분석 스크립트.

논문 Discussion용 구조:
  1. 가설 설정 (property별 화학적 결정 요인)
  2. 대표 분자군 Grad-CAM 정량 분석 (본 스크립트)
  3. 단일 샘플 예시 + 한계 명시

실행 위치: 프로젝트 루트 (app.py와 같은 레벨, ensemble/ 와 같은 레벨)
실행: python gradcam_chem_analysis.py

출력:
  outputs/gradcam_chem_quant_<tag>.csv     - property별 정량 결과 표
  outputs/gradcam_chem_summary_<tag>.md    - 논문에 바로 삽입 가능한 markdown 표
  outputs/gradcam_chem_panel_<prop>_<tag>.png - property별 대표 샘플 시각화 패널
  (<tag>는 layer/배경마스킹/TTA 설정을 담은 식별자, 재현성을 위해 파일명에 포함)
"""

import argparse
import csv
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D

sys.path.insert(0, str(Path(__file__).parent / "ensemble"))
from ensemble_common import build_final_config, load_fold_model  # noqa: E402
from datasets.transforms import get_valid_transform  # noqa: E402
from preprocessing.smiles_to_mol import parse_smiles  # noqa: E402
from preprocessing.mol_to_image import render_mol_to_png, IMAGE_SIZE  # noqa: E402

# ---------------------------------------------------------------------------
# 설정
# ---------------------------------------------------------------------------
METADATA_CSV = "data/processed_merged/processed_metadata.csv"  # 실제 경로 확인됨
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

GRADCAM_FOLD = 1
TARGET_LAYER_INDEX = 7  # --layer 인자로 덮어쓸 수 있음 (기본 layer4)
BACKGROUND_MASK = True  # --no-bgmask로 끌 수 있음 (분자 바깥 흰 배경 영역 CAM 강제 0)
N_SAMPLES_PER_PROPERTY = 30
N_VIS_PER_PROPERTY = 6  # 그림에 넣을 대표 샘플 수
SMILES_MAX_LEN = 90
RADIUS_PX = 14  # 관심 원자 주변 마스크 반경 (픽셀)
RANDOM_SEED = 42

LABEL_ORDER = ["Tg", "Density", "FFV", "Tc", "Rg"]

# property별 화학적 가설 -> SMARTS 패턴 (여러 개면 매칭되는 모든 원자 사용)
# Tc, Rg는 국소 작용기가 아니라 전역 구조(대칭성/전체 크기) 가설이라 별도 처리
PROPERTY_SMARTS = {
    "Tg": ["[NX3][CX3](=[OX1])", "[#7]", "[#16]"],       # 아마이드, N, S (극성/H-bond)
    "Density": ["[#16]", "[#7]", "[#8]"],                 # 무거운 헤테로원자
    "FFV": ["[C]#[N]", "[CH3]"],                           # 나이트릴/말단 벌키기
}
# Tc: 방향족 backbone(대칭 축 근사), Rg: 전역 분산도로 별도 평가


# ---------------------------------------------------------------------------
# GradCAM (app.py와 동일)
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


def unnormalize_image(image_tensor, mean, std):
    img = image_tensor.squeeze().cpu().numpy().transpose(1, 2, 0)
    img = img * np.array(std) + np.array(mean)
    return np.clip(img * 255, 0, 255).astype(np.uint8)


def build_background_mask(image_rgb, white_thresh=245):
    """분자 영역(=흰 배경이 아닌 픽셀)만 1, 배경은 0인 마스크.
    RDKit 렌더링은 배경이 거의 순백색이라 밝기 기준으로 간단히 분리 가능.
    """
    gray = image_rgb.mean(axis=2)
    mask = (gray < white_thresh).astype(np.float32)
    # 팽창(dilate)으로 결합선 주변 여유 조금 확보 (분자 바로 바깥 1~2px까지 포함)
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
# Grad-CAM TTA (Test-Time Rotation Augmentation)
# 재학습 없이, 이미지를 여러 각도로 돌려서 각각 Grad-CAM을 뽑고 원래 각도로
# 되돌려 평균낸다. 특정 모서리/여백에 attention이 고착되는 편향을 완화하려는 목적.
# 모델 가중치는 전혀 건드리지 않으므로 재학습 리스크 없음.
# ---------------------------------------------------------------------------
def _rotate(img, angle, border_value):
    h, w = img.shape[:2]
    center = (w / 2, h / 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(img, M, (w, h), borderValue=border_value)


def generate_gradcam_tta(image_np, prop_idx, gradcam, transform, device, angles=(0, 90, 180, 270)):
    """여러 각도로 회전한 이미지에 대해 Grad-CAM을 계산 후 원래 각도로 되돌려 평균.

    Args:
        image_np: 원본 렌더링 이미지 (H, W, 3), 배경 흰색.
        prop_idx: LABEL_ORDER 내 property 인덱스.
        gradcam: GradCAM 인스턴스.
        transform: valid transform (augmentation 없는 것).
        device: torch device.
        angles: 테스트할 회전 각도 목록.

    Returns:
        평균/정규화된 cam (224x224, 0~1).
    """
    cams = []
    for angle in angles:
        rotated_img = _rotate(image_np, angle, border_value=(255, 255, 255))
        image_tensor = transform(image=rotated_img)["image"].unsqueeze(0).to(device)
        cam_rotated = gradcam.generate(image_tensor, prop_idx)
        # 원래 각도로 되돌림 (반대 방향 회전, CAM 빈 공간은 0으로 채움)
        cam_back = _rotate(cam_rotated, -angle, border_value=0)
        cams.append(cam_back)

    cam_avg = np.mean(cams, axis=0)
    cam_avg = cam_avg - cam_avg.min()
    if cam_avg.max() > 0:
        cam_avg = cam_avg / cam_avg.max()
    return cam_avg


def tight_crop_for_display(overlay, margin=12, white_thresh=245):
    """시각화(그림 저장)용 후처리 전용 함수.
    분자 바깥 흰 배경을 bounding box 기준으로 잘라내 그림을 보기 좋게 만든다.
    주의: 이 함수는 저장할 그림에만 적용하며, 모델 입력/예측/정량 지표(enrichment_ratio 등)
    계산에는 절대 사용하지 않는다 (학습 때의 padding=0.12와 다른 스케일이 되면
    train-test mismatch가 생기기 때문).
    """
    gray = overlay.mean(axis=2)
    non_bg = gray < white_thresh
    if not non_bg.any():
        return overlay  # 전부 배경이면 원본 그대로

    ys, xs = np.where(non_bg)
    y0, y1 = max(ys.min() - margin, 0), min(ys.max() + margin, overlay.shape[0])
    x0, x1 = max(xs.min() - margin, 0), min(xs.max() + margin, overlay.shape[1])
    return overlay[y0:y1, x0:x1]


def overlay_heatmap(image_rgb, cam, alpha=0.4):
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    return (heatmap * alpha + image_rgb * (1 - alpha)).astype(np.uint8)


# ---------------------------------------------------------------------------
# 원자 픽셀 좌표 얻기 (학습 렌더링과 동일 설정으로 다시 drawer 실행)
# ---------------------------------------------------------------------------
def get_atom_pixel_coords(mol, size=IMAGE_SIZE, bond_line_width=2, padding=0.12):
    """RDKit drawer로 render_mol_to_png와 동일한 절차(mol 복사 + Compute2DCoords)를
    거쳐 그렸을 때 각 원자의 픽셀 좌표를 반환. 실제 학습 이미지와 좌표계를 맞추기 위해
    preprocessing/mol_to_image.py의 render_mol_to_png와 동일한 옵션을 사용한다.
    """
    from rdkit.Chem import AllChem

    mol_for_draw = Chem.Mol(mol)
    AllChem.Compute2DCoords(mol_for_draw)

    drawer = rdMolDraw2D.MolDraw2DCairo(size, size)
    opts = drawer.drawOptions()
    opts.bondLineWidth = bond_line_width
    opts.padding = padding
    opts.clearBackground = True
    opts.addStereoAnnotation = False
    rdMolDraw2D.PrepareAndDrawMolecule(drawer, mol_for_draw)
    drawer.FinishDrawing()

    coords = {}
    for atom in mol_for_draw.GetAtoms():
        idx = atom.GetIdx()
        pt = drawer.GetDrawCoords(idx)
        coords[idx] = (pt.x, pt.y)
    return coords


def match_atoms(mol, smarts_list):
    """SMARTS 패턴들에 매칭되는 모든 원자 인덱스 집합 반환."""
    matched = set()
    for smarts in smarts_list:
        patt = Chem.MolFromSmarts(smarts)
        if patt is None:
            continue
        for match in mol.GetSubstructMatches(patt):
            matched.update(match)
    return matched


def build_atom_mask(coords, atom_indices, size=IMAGE_SIZE, radius=RADIUS_PX):
    """관심 원자들 주변에 원형 마스크(0/1) 생성."""
    mask = np.zeros((size, size), dtype=np.float32)
    yy, xx = np.mgrid[0:size, 0:size]
    for idx in atom_indices:
        if idx not in coords:
            continue
        cx, cy = coords[idx]
        dist2 = (xx - cx) ** 2 + (yy - cy) ** 2
        mask[dist2 <= radius**2] = 1.0
    return mask


def compute_enrichment(cam, mask):
    """
    관심 영역에 쏠린 attention 비율(area 대비)을 baseline과 비교.
    enrichment = (mask 안 cam 에너지 비율) / (mask 면적 비율)
    1보다 크면 '우연보다 더 집중됐다'는 뜻.
    """
    total_energy = cam.sum()
    if total_energy <= 0:
        return None
    area_frac = mask.sum() / mask.size
    if area_frac <= 0:
        return None
    energy_frac = (cam * mask).sum() / total_energy
    return energy_frac / area_frac


def compute_dispersion(cam):
    """
    Rg처럼 '국소가 아니라 전역으로 퍼져야 자연스러운' property를 위한 지표.
    attention이 이미지 전체에 얼마나 고르게 퍼져 있는지 (엔트로피 기반, 0~1 정규화).
    값이 클수록 넓게 분산됨.
    """
    p = cam.flatten()
    p = p / (p.sum() + 1e-8)
    entropy = -(p * np.log(p + 1e-12)).sum()
    max_entropy = np.log(p.size)
    return entropy / max_entropy


# ---------------------------------------------------------------------------
# 샘플 선정
# ---------------------------------------------------------------------------
def select_samples(metadata_csv, property_name, n, seed=RANDOM_SEED):
    rows = []
    with open(metadata_csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["split"] != "train":
                continue
            if row.get(f"{property_name}_mask") != "1":
                continue
            if len(row["canonical_smiles"]) > SMILES_MAX_LEN:
                continue
            rows.append(row)
    random.Random(seed).shuffle(rows)
    return rows[:n]


# ---------------------------------------------------------------------------
# 메인
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--layer", type=int, default=TARGET_LAYER_INDEX,
        help="Grad-CAM target layer index (feature_extractor 내 인덱스). "
             "낮을수록(예: layer3) 해상도는 높아지지만 의미 추상도는 낮아짐.",
    )
    parser.add_argument(
        "--no-bgmask", action="store_true",
        help="배경(분자 바깥 흰 영역) 마스킹 끄기",
    )
    parser.add_argument(
        "--list-layers", action="store_true",
        help="feature_extractor 레이어 구조만 출력하고 종료 (레이어 인덱스 확인용)",
    )
    parser.add_argument(
        "--tta", action="store_true",
        help="Grad-CAM TTA(회전 평균) 사용. 재학습 없이 모서리 편향 완화 시도.",
    )
    parser.add_argument(
        "--tta-angles", type=str, default="0,90,180,270",
        help="TTA에 쓸 회전 각도 목록 (쉼표 구분). 예: '0,45,90,135,180,225,270,315'",
    )
    args = parser.parse_args()

    use_bgmask = BACKGROUND_MASK and not args.no_bgmask
    target_layer_index = args.layer
    use_tta = args.tta
    tta_angles = tuple(int(a) for a in args.tta_angles.split(","))

    print("모델 로딩...")
    config = build_final_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_fold_model(GRADCAM_FOLD, config, device)

    if args.list_layers:
        print("\n=== backbone.feature_extractor 레이어 구조 ===")
        for i, layer in enumerate(model.backbone.feature_extractor):
            print(f"  [{i}] {layer.__class__.__name__}")
        print("\n--layer <인덱스> 로 원하는 레이어 선택해서 재실행하세요.")
        return

    print(f"target layer index = {target_layer_index}, 배경마스킹 = {use_bgmask}")
    target_layer = model.backbone.feature_extractor[target_layer_index]
    gradcam = GradCAM(model, target_layer)

    image_cfg = config["image"]
    transform = get_valid_transform(
        image_cfg["size"], image_cfg["normalize_mean"], image_cfg["normalize_std"]
    )

    summary_rows = []

    for prop_idx, prop_name in enumerate(LABEL_ORDER):
        print(f"\n=== {prop_name} 분석 시작 ===")
        samples = select_samples(METADATA_CSV, prop_name, N_SAMPLES_PER_PROPERTY)
        print(f"  샘플 {len(samples)}개 선정")

        scores = []
        vis_saved = 0
        vis_panels = []

        for row in samples:
            smiles = row["canonical_smiles"]
            mol = parse_smiles(smiles)
            if mol is None:
                continue

            # 렌더링 (학습과 동일 스타일)
            import tempfile
            from PIL import Image

            with tempfile.TemporaryDirectory() as tmp_dir:
                tmp_path = Path(tmp_dir) / "render.png"
                ok = render_mol_to_png(mol, tmp_path, size=IMAGE_SIZE)
                if not ok:
                    continue
                pil_img = Image.open(tmp_path).convert("RGB")
                pil_img.load()

            image_np = np.array(pil_img)

            if use_tta:
                cam = generate_gradcam_tta(
                    image_np, prop_idx, gradcam, transform, device, angles=tta_angles
                )
                # 시각화/오버레이용 기준 텐서는 0도(원본) 기준으로 통일
                image_tensor = transform(image=image_np)["image"].unsqueeze(0).to(device)
            else:
                image_tensor = transform(image=image_np)["image"].unsqueeze(0).to(device)
                cam = gradcam.generate(image_tensor, prop_idx)

            if use_bgmask:
                # 배경(분자 바깥 흰 영역) CAM 강제 0 처리
                bg_mask = build_background_mask(image_np)
                cam = apply_background_mask(cam, bg_mask)

            if prop_name in PROPERTY_SMARTS:
                atom_idx = match_atoms(mol, PROPERTY_SMARTS[prop_name])
                if not atom_idx:
                    continue
                coords = get_atom_pixel_coords(mol)
                mask = build_atom_mask(coords, atom_idx)
                score = compute_enrichment(cam, mask)
                metric_name = "enrichment_ratio"
            elif prop_name == "Tc":
                # 방향족 backbone 원자를 '규칙성/대칭' 근사 영역으로 사용
                aromatic_idx = {a.GetIdx() for a in mol.GetAtoms() if a.GetIsAromatic()}
                if not aromatic_idx:
                    continue
                coords = get_atom_pixel_coords(mol)
                mask = build_atom_mask(coords, aromatic_idx)
                score = compute_enrichment(cam, mask)
                metric_name = "enrichment_ratio(aromatic_backbone)"
            else:  # Rg -> 전역 분산도
                score = compute_dispersion(cam)
                metric_name = "dispersion_index"

            if score is None:
                continue
            scores.append(score)

            # 대표 시각화 저장 (property당 N_VIS_PER_PROPERTY개)
            if vis_saved < N_VIS_PER_PROPERTY:
                image_rgb = unnormalize_image(
                    image_tensor, image_cfg["normalize_mean"], image_cfg["normalize_std"]
                )
                overlay = overlay_heatmap(image_rgb, cam)
                overlay_display = tight_crop_for_display(overlay)  # 그림 표시용 크롭만, 예측/지표엔 미사용
                vis_panels.append((overlay_display, row["id"], score))
                vis_saved += 1

        if scores:
            mean_score = float(np.mean(scores))
            std_score = float(np.std(scores))
        else:
            mean_score, std_score = float("nan"), float("nan")

        print(f"  {metric_name}: {mean_score:.3f} ± {std_score:.3f} (n={len(scores)})")
        summary_rows.append(
            {
                "property": prop_name,
                "metric": metric_name,
                "mean": mean_score,
                "std": std_score,
                "n_samples": len(scores),
            }
        )

        # 패널 그림 저장 (2행 x 3열 그리드)
        if vis_panels:
            n_vis = len(vis_panels)
            n_cols = 3
            n_rows = (n_vis + n_cols - 1) // n_cols
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
            axes = np.array(axes).reshape(-1)  # 1D로 통일 (n_rows=1이어도 동작)
            for ax, (overlay, sample_id, score) in zip(axes, vis_panels):
                ax.imshow(overlay)
                ax.set_title(f"id={sample_id}\nscore={score:.3f}", fontsize=9)
                ax.axis("off")
            # 남는 칸 숨기기
            for ax in axes[n_vis:]:
                ax.axis("off")
            fig.suptitle(f"{prop_name} — Grad-CAM representative samples", fontsize=13)
            fig.tight_layout()
            tag = f"layer{target_layer_index}" + ("_bgmask" if use_bgmask else "_nobgmask") + ("_tta" if use_tta else "")
            out_path = OUTPUT_DIR / f"gradcam_chem_panel_{prop_name}_{tag}.png"
            fig.savefig(out_path, dpi=150)
            plt.close(fig)
            print(f"  패널 저장: {out_path}")

    tag = f"layer{target_layer_index}" + ("_bgmask" if use_bgmask else "_nobgmask") + ("_tta" if use_tta else "")

    # 결과 표 저장 (csv + markdown)
    csv_path = OUTPUT_DIR / f"gradcam_chem_quant_{tag}.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["property", "metric", "mean", "std", "n_samples"])
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"\n표 저장: {csv_path}")

    md_path = OUTPUT_DIR / f"gradcam_chem_summary_{tag}.md"
    with open(md_path, "w") as f:
        f.write("| Property | Metric | Mean | Std | N |\n")
        f.write("|---|---|---|---|---|\n")
        for r in summary_rows:
            f.write(
                f"| {r['property']} | {r['metric']} | {r['mean']:.3f} | "
                f"{r['std']:.3f} | {r['n_samples']} |\n"
            )
    print(f"markdown 표 저장: {md_path}")
    print("\n완료. enrichment_ratio > 1 이면 가설대로 해당 화학기에 attention이 우연보다 더 집중된 것.")


if __name__ == "__main__":
    main()