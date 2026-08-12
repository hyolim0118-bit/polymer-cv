"""test_ensemble.py

STEP11 (2/2): 진짜 앙상블 — Kaggle test set(라벨 없음)에 대해 5개 fold 모델의
예측을 평균 낸다. 정답이 없어 성능 검증은 못 하지만, sanity check로 5개
모델이 실제로 서로 다르게 예측하는지 등을 확인한다.

실행: python test_ensemble.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from datasets.dataset import LABEL_ORDER
from ensemble_common import (
    N_SPLITS,
    build_final_config,
    build_test_dataset,
    denormalize,
    get_fold_splits,
    load_fold_model,
    load_pool_and_metadata,
    predict_normalized,
)


def run_test_ensemble(config, pool_df, full_metadata, device, output_path):
    print("Test set 앙상블 (5개 모델 평균) + sanity check 시작")

    test_df = full_metadata[full_metadata["split"] == "test"].reset_index(drop=True)
    fold_predictions = []  # 각 fold 모델의 (N, 5) 실제 단위 예측

    for fold_id, (train_idx, _) in get_fold_splits(pool_df):
        train_df = pool_df.iloc[train_idx].reset_index(drop=True)

        test_dataset, label_mean, label_std = build_test_dataset(train_df, test_df, config)
        test_loader = DataLoader(
            test_dataset, batch_size=config["dataloader"]["batch_size"], shuffle=False
        )

        model = load_fold_model(fold_id, config, device)
        preds_norm = predict_normalized(model, test_loader, device)
        preds_real = denormalize(preds_norm, label_mean, label_std)
        fold_predictions.append(preds_real)
        print(f"  fold {fold_id} 예측 완료: {preds_real.shape}")

    fold_predictions = np.stack(fold_predictions, axis=0)  # (5, N, 5)

    # --- sanity check 1: 모델들이 서로 다른 예측을 하는지 ---
    print("\n[Sanity Check] fold 간 예측 diversity")
    all_identical = True
    for i in range(N_SPLITS):
        for j in range(i + 1, N_SPLITS):
            if not np.allclose(fold_predictions[i], fold_predictions[j], atol=1e-6):
                all_identical = False
    if all_identical:
        print("  경고: 5개 fold 모델의 예측이 전부 동일합니다 -> 체크포인트 로딩 버그 의심!")
    else:
        std_across_folds = fold_predictions.std(axis=0).mean()
        print(f"  OK: fold 간 예측이 서로 다릅니다 (평균 표준편차: {std_across_folds:.4f})")

    # --- sanity check 2: NaN 없는지 ---
    if np.isnan(fold_predictions).any():
        print("  경고: 예측값에 NaN이 있습니다!")
    else:
        print("  OK: NaN 없음")

    ensemble_pred = fold_predictions.mean(axis=0)  # (N, 5)

    submission = pd.DataFrame(ensemble_pred, columns=LABEL_ORDER)
    submission.insert(0, "id", test_df["id"].values)
    submission.to_csv(output_path, index=False)
    print(f"\n앙상블 예측 저장: {output_path} ({len(submission)}행)")

    return submission


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"사용 device: {device}")

    config = build_final_config()
    pool_df, full_metadata = load_pool_and_metadata(config)

    output_dir = Path("results/step11_ensemble")
    output_dir.mkdir(parents=True, exist_ok=True)

    run_test_ensemble(config, pool_df, full_metadata, device, output_dir / "ensemble_submission.csv")


if __name__ == "__main__":
    main()
