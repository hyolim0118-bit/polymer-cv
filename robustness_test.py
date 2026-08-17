"""robustness_test.py

STEP13 (Robustness Test 일부): Rotation(90/180/270) + Flip(좌우/상하) 변형을
가한 이미지에 대해 5-fold 앙상블 예측값이 원본 대비 얼마나 흔들리는지 측정한다.

TTA(Test-Time Augmentation)와의 차이:
- TTA는 여러 회전 버전의 예측을 "평균 내서" 최종 예측을 안정화하는 용도
- 여기서는 그 반대로, 각 변형이 원본 예측과 "얼마나 다른지"를 직접 측정해서
  모델이 방향/반전에 얼마나 강건한지를 정량적으로 보고하는 용도

실행: python robustness_test.py
(프로젝트 루트에서, ensemble/ 와 같은 레벨)
출력: results/step13_robustness/robustness_summary.csv,
      results/step13_robustness/robustness_boxplot.png
"""

import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent / "ensemble"))
from ensemble_common import build_final_config, load_fold_model, load_pool_and_metadata, get_fold_splits  # noqa: E402
from datasets.dataset import LABEL_ORDER, compute_fold_stats  # noqa: E402
from datasets.transforms import get_valid_transform  # noqa: E402

N_FOLDS = 5
N_SAMPLES = 30  # 검증에 쓸 샘플 수 (전체 valid셋에서 랜덤 추출)
SEED = 42
OUTPUT_DIR = Path("results/step13_robustness")

# 적용할 변형들: (이름, cv2 변형 함수)
AUGMENTATIONS = {
    "rotate_90": lambda img: cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE),
    "rotate_180": lambda img: cv2.rotate(img, cv2.ROTATE_180),
    "rotate_270": lambda img: cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE),
    "flip_horizontal": lambda img: cv2.flip(img, 1),
    "flip_vertical": lambda img: cv2.flip(img, 0),
}


def ensemble_predict(image_tensor, models, label_stats):
    """5-fold 앙상블 예측 (역정규화까지 완료된 실제 값 반환)."""
    fold_preds = []
    for fold in range(1, N_FOLDS + 1):
        model = models[fold]
        label_mean, label_std = label_stats[fold]
        with torch.no_grad():
            out = model(image_tensor).cpu().numpy()[0]
        mean_arr = np.array([label_mean[c] for c in LABEL_ORDER])
        std_arr = np.array([label_std[c] for c in LABEL_ORDER])
        real_values = out * std_arr + mean_arr
        fold_preds.append(real_values)
    return np.mean(fold_preds, axis=0)


def main():
    config = build_final_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"사용 device: {device}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    pool_df, _ = load_pool_and_metadata(config)
    splits = dict(get_fold_splits(pool_df))

    # fold1의 valid 인덱스를 대표로 사용해 샘플 추출 (Grad-CAM 때와 동일한 관례)
    train_idx, valid_idx = splits[1]
    train_df = pool_df.iloc[train_idx].reset_index(drop=True)
    valid_df = pool_df.iloc[valid_idx].reset_index(drop=True)

    rng = np.random.default_rng(SEED)
    sample_indices = rng.choice(len(valid_df), size=min(N_SAMPLES, len(valid_df)), replace=False)

    # 5-fold 모델 + fold별 정규화 통계 로드 (app.py와 동일한 패턴)
    print("모델 로딩 중 (5-fold)...")
    models = {}
    label_stats = {}
    for fold in range(1, N_FOLDS + 1):
        f_train_idx, _ = splits[fold]
        f_train_df = pool_df.iloc[f_train_idx].reset_index(drop=True)
        label_mean, label_std = compute_fold_stats(f_train_df, LABEL_ORDER)
        label_stats[fold] = (label_mean, label_std)
        models[fold] = load_fold_model(fold, config, device)
    print("모델 로딩 완료.")

    image_cfg = config["image"]
    data_cfg = config["data"]
    image_root = Path(data_cfg["image_root"])
    transform = get_valid_transform(
        image_cfg["size"], image_cfg["normalize_mean"], image_cfg["normalize_std"]
    )

    records = []  # (sample_id, augmentation, property, original_pred, aug_pred, abs_diff)

    for i, sample_idx in enumerate(sample_indices):
        row = valid_df.iloc[sample_idx]
        image_path = image_root / row["image_path"]
        image_bgr = cv2.imread(str(image_path))
        if image_bgr is None:
            print(f"  [스킵] 이미지 로드 실패: {image_path}")
            continue
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        # 원본 예측
        orig_tensor = transform(image=image_rgb)["image"].unsqueeze(0).to(device)
        orig_pred = ensemble_predict(orig_tensor, models, label_stats)

        # 변형별 예측
        for aug_name, aug_fn in AUGMENTATIONS.items():
            aug_image = aug_fn(image_rgb)
            aug_tensor = transform(image=aug_image)["image"].unsqueeze(0).to(device)
            aug_pred = ensemble_predict(aug_tensor, models, label_stats)

            for prop_idx, prop_name in enumerate(LABEL_ORDER):
                # 결측 라벨인 property는 애초에 학습 신호가 약해 비교 의미가 떨어지므로
                # mask 상관없이 "모델이 뱉는 값" 자체의 변화를 그대로 기록한다
                records.append({
                    "sample_id": row.get("id", sample_idx),
                    "augmentation": aug_name,
                    "property": prop_name,
                    "original_pred": orig_pred[prop_idx],
                    "aug_pred": aug_pred[prop_idx],
                    "abs_diff": abs(aug_pred[prop_idx] - orig_pred[prop_idx]),
                })

        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{len(sample_indices)} 샘플 처리 완료")

    result_df = pd.DataFrame(records)
    csv_path = OUTPUT_DIR / "robustness_raw.csv"
    result_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"\nRaw 결과 저장: {csv_path}")

    # property x augmentation 별 평균 절대오차(MAE) 요약 테이블
    summary = result_df.groupby(["property", "augmentation"])["abs_diff"].agg(
        ["mean", "std", "max"]
    ).reset_index()
    summary_path = OUTPUT_DIR / "robustness_summary.csv"
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    print(f"요약 테이블 저장: {summary_path}")
    print("\n" + summary.to_string(index=False))

    # property별 boxplot (x축: augmentation, y축: abs_diff)
    fig, axes = plt.subplots(1, len(LABEL_ORDER), figsize=(5 * len(LABEL_ORDER), 5), sharey=False)
    for ax, prop_name in zip(axes, LABEL_ORDER):
        prop_df = result_df[result_df["property"] == prop_name]
        data_by_aug = [
            prop_df[prop_df["augmentation"] == aug]["abs_diff"].values
            for aug in AUGMENTATIONS.keys()
        ]
        ax.boxplot(data_by_aug, labels=list(AUGMENTATIONS.keys()))
        ax.set_title(prop_name)
        ax.set_ylabel("|pred - original_pred|")
        ax.tick_params(axis="x", rotation=45)
    fig.suptitle("Robustness Test: 변형별 예측값 절대 편차 분포", fontsize=14)
    fig.tight_layout()
    plot_path = OUTPUT_DIR / "robustness_boxplot.png"
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"박스플롯 저장: {plot_path}")

    print(f"\n완료. {OUTPUT_DIR}/ 확인하세요.")


if __name__ == "__main__":
    main()
