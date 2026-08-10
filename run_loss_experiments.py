"""run_loss_experiments.py

STEP9.4: Exp1(Masked MSE, baseline) ~ Exp4(Weighted MSE)를 실제
cross_validation.py의 run_kfold()를 그대로 재사용해서 4번 실행하고,
결과를 하나의 비교표(loss_comparison.csv)로 모은다.
"""

import copy
import importlib.util
import shutil
import time
from pathlib import Path

import pandas as pd

from datasets.dataset import LABEL_ORDER, load_config

spec = importlib.util.spec_from_file_location(
    "cross_validation", "training/training real/cross_validation.py"
)
cross_validation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cross_validation)


def compute_inverse_frequency_weights(pool_df, label_columns=LABEL_ORDER):
    label_counts = {col: int(pool_df[col].notna().sum()) for col in label_columns}
    raw_weights = {col: 1.0 / label_counts[col] for col in label_columns}
    mean_weight = sum(raw_weights.values()) / len(raw_weights)
    weights = {col: w / mean_weight for col, w in raw_weights.items()}
    return weights, label_counts


def build_experiment_configs(pool_df):
    inv_weights, label_counts = compute_inverse_frequency_weights(pool_df)
    print("Property별 라벨 개수:", label_counts)
    print("역수 기반 weight (평균 1 정규화):", inv_weights)

    return {
        "exp1_masked_mse": {
            "element_loss": {"name": "mse", "params": {}},
            "weighting": {"name": "uniform"},
            "reduction": "mean",
        },
        "exp2_masked_mae": {
            "element_loss": {"name": "mae", "params": {}},
            "weighting": {"name": "uniform"},
            "reduction": "mean",
        },
        "exp3_masked_huber": {
            "element_loss": {"name": "huber", "params": {"delta": 1.0}},
            "weighting": {"name": "uniform"},
            "reduction": "mean",
        },
        "exp4_weighted_mse": {
            "element_loss": {"name": "mse", "params": {}},
            "weighting": {"name": "static", "property_weight": inv_weights},
            "reduction": "mean",
        },
    }


def main():
    config_path = Path("training/config.yaml")
    output_dir = Path("results/step9_4")
    output_dir.mkdir(parents=True, exist_ok=True)

    base_config = load_config(config_path)

    full_metadata = pd.read_csv(base_config["data"]["metadata_path"])
    pool_df = full_metadata[full_metadata["split"] == "train"].reset_index(drop=True)

    experiments = build_experiment_configs(pool_df)

    comparison_rows = []
    for exp_name, loss_cfg in experiments.items():
        print(f"\n=== {exp_name} 시작 ===")

        config = copy.deepcopy(base_config)
        config["loss"] = loss_cfg
        config["training"]["checkpoint_dir"] = str(
            Path(base_config["training"]["checkpoint_dir"]) / exp_name
        )
        config["training"]["log_dir"] = str(Path(base_config["training"]["log_dir"]) / exp_name)

        start_time = time.time()
        fold_results = cross_validation.run_kfold(config)
        elapsed = time.time() - start_time
        avg_fold_time = elapsed / len(fold_results)

        default_csv = Path("results/kfold_results.csv")
        if default_csv.exists():
            shutil.copy(default_csv, output_dir / f"kfold_results_{exp_name}.csv")

        mae = pd.Series([r["mae"] for r in fold_results])
        rmse = pd.Series([r["rmse"] for r in fold_results])
        r2 = pd.Series([r["r2"] for r in fold_results])
        val_loss = pd.Series([r["val_loss"] for r in fold_results])

        comparison_rows.append({
            "experiment": exp_name,
            "loss": loss_cfg["element_loss"]["name"],
            "weighting": loss_cfg["weighting"]["name"],
            "val_loss_mean": val_loss.mean(),
            "val_loss_std": val_loss.std(),
            "MAE_mean": mae.mean(),
            "MAE_std": mae.std(),
            "RMSE_mean": rmse.mean(),
            "RMSE_std": rmse.std(),
            "R2_mean": r2.mean(),
            "R2_std": r2.std(),
            "avg_fold_training_time_sec": avg_fold_time,
        })

    comparison_df = pd.DataFrame(comparison_rows)
    comparison_path = output_dir / "loss_comparison.csv"
    comparison_df.to_csv(comparison_path, index=False)
    print(f"\n전체 비교 결과 저장: {comparison_path}")
    print(comparison_df)


if __name__ == "__main__":
    main()
