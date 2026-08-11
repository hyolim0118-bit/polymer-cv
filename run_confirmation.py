"""run_confirmation.py

STEP8 Confirmation Stage. round2/best_params.json을 읽어서, 그 하이퍼파라미터로
cross_validation.py의 run_kfold()를 그대로 호출해 5-fold 전체를 재실행한다.
(오늘 backbone/loss 비교와 완전히 동일한 패턴 — 새 인프라 없음)
"""

import copy
import importlib.util
import json
import shutil
from pathlib import Path

from datasets.dataset import load_config

spec = importlib.util.spec_from_file_location(
    "cross_validation", "training/training real/cross_validation.py"
)
cross_validation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cross_validation)

FIXED_OVERRIDES = {
    "model": {"name": "resnet18"},
    "loss": {
        "element_loss": {"name": "huber", "params": {"delta": 1.0}},
        "weighting": {"name": "uniform"},
        "reduction": "mean",
    },
}


def apply_overrides(config, overrides):
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(config.get(key), dict):
            apply_overrides(config[key], value)
        else:
            config[key] = value
    return config


def main():
    base_config = load_config(Path("training/config.yaml"))
    apply_overrides(base_config, FIXED_OVERRIDES)

    best_params_path = Path("results/step8_hpo/round2/best_params.json")
    with open(best_params_path) as f:
        best_params = json.load(f)

    print("Confirmation에 사용할 최종 하이퍼파라미터:", best_params)

    config = copy.deepcopy(base_config)
    config["training"]["learning_rate"] = best_params["learning_rate"]
    config["training"]["weight_decay"] = best_params["weight_decay"]
    config["model"]["dropout"] = best_params["dropout"]
    config["dataloader"]["batch_size"] = best_params["batch_size"]
    config["scheduler"]["t_max"] = config["training"]["epochs"]

    config["training"]["checkpoint_dir"] = "results/step8_hpo/confirmation_checkpoints"
    config["training"]["log_dir"] = "results/step8_hpo/confirmation_logs"

    fold_results = cross_validation.run_kfold(config)

    shutil.copy("results/kfold_results.csv", "results/step8_hpo/kfold_results_confirmation.csv")

    print("\n=== STEP8 Confirmation 완료 (5-fold) ===")
    for r in fold_results:
        print(f"  Fold {r['fold']}: MAE={r['mae']:.4f} RMSE={r['rmse']:.4f} R2={r['r2']:.4f}")


if __name__ == "__main__":
    main()
