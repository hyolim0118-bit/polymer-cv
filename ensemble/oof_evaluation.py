"""oof_evaluation.py

STEP11 (1/2): OOF(Out-of-Fold) 평가.
각 샘플을 "그 샘플이 validation이었던 fold의 모델"로만 예측해서, leakage 없이
전체 train pool에 대한 공정한 성능을 계산한다. 이게 논문에 보고할 핵심 숫자.

실행: python oof_evaluation.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from ensemble_common import (
    build_final_config,
    build_valid_dataset,
    denormalize,
    get_fold_splits,
    load_fold_model,
    load_pool_and_metadata,
    masked_metrics,
    predict_normalized,
)


def run_oof_evaluation(config, pool_df, device):
    print("OOF(Out-of-Fold) 평가 시작 — leakage 없는 공정한 성능 계산")

    all_true, all_pred, all_mask = [], [], []

    for fold_id, (train_idx, valid_idx) in get_fold_splits(pool_df):
        train_df = pool_df.iloc[train_idx].reset_index(drop=True)
        valid_df = pool_df.iloc[valid_idx].reset_index(drop=True)

        valid_dataset, label_mean, label_std = build_valid_dataset(train_df, valid_df, config)
        valid_loader = DataLoader(
            valid_dataset, batch_size=config["dataloader"]["batch_size"], shuffle=False
        )

        model = load_fold_model(fold_id, config, device)
        preds_norm = predict_normalized(model, valid_loader, device)
        preds_real = denormalize(preds_norm, label_mean, label_std)

        true_norm = np.stack([valid_dataset[i][1].numpy() for i in range(len(valid_dataset))])
        mask = np.stack([valid_dataset[i][2].numpy() for i in range(len(valid_dataset))])
        true_real = denormalize(true_norm, label_mean, label_std)

        all_true.append(true_real)
        all_pred.append(preds_real)
        all_mask.append(mask)

        print(f"  fold {fold_id}: {len(valid_df)}개 샘플 예측 완료")

    y_true = np.concatenate(all_true, axis=0)
    y_pred = np.concatenate(all_pred, axis=0)
    mask = np.concatenate(all_mask, axis=0)

    per_property, overall = masked_metrics(y_true, y_pred, mask)

    print(f"\nOOF 전체 결과 (샘플 수: {len(y_true)}):")
    print(f"  MAE  : {overall['MAE']:.4f}")
    print(f"  RMSE : {overall['RMSE']:.4f}")
    print(f"  R2   : {overall['R2']:.4f}")
    print("\nproperty별:")
    for col, v in per_property.items():
        print(f"  {col:10s} MAE={v['MAE']:.4f} RMSE={v['RMSE']:.4f} R2={v['R2']:.4f} (n={v['n']})")

    return per_property, overall


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"사용 device: {device}")

    config = build_final_config()
    pool_df, _ = load_pool_and_metadata(config)

    output_dir = Path("results/step11_ensemble")
    output_dir.mkdir(parents=True, exist_ok=True)

    per_property, overall = run_oof_evaluation(config, pool_df, device)

    pd.DataFrame(per_property).T.to_csv(output_dir / "oof_per_property.csv")
    pd.DataFrame([overall]).to_csv(output_dir / "oof_overall.csv", index=False)
    print(f"\n결과 저장: {output_dir}/oof_per_property.csv, oof_overall.csv")


if __name__ == "__main__":
    main()
