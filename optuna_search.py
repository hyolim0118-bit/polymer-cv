"""optuna_search.py

STEP8 HPO Search Stage. cross_validation.py의 내부 함수
(_build_fold_loaders, build_model, Trainer)를 그대로 재사용해서,
fold 하나(search_fold)만으로 빠르게 하이퍼파라미터를 탐색한다.

backbone(resnet18)과 loss(huber)는 고정 — STEP7/STEP9.4 결과 기준.

사용법:
    python optuna_search.py round1
    (round1 결과 보고 아래 ROUND2_SPACE를 직접 수정한 뒤)
    python optuna_search.py round2
"""

import copy
import importlib.util
import json
import shutil
import sys
from pathlib import Path

import optuna
import pandas as pd

from datasets.dataset import load_config

# 폴더명에 공백이 있어서 importlib로 불러옴 (STEP9.4와 동일한 방식)
spec = importlib.util.spec_from_file_location(
    "cross_validation", "training/training real/cross_validation.py"
)
cross_validation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cross_validation)


SEARCH_FOLD = 0  # Round1/Round2 모두 이 fold 하나만 사용

# 고정값 — STEP7(backbone=resnet18), STEP9.4(loss=huber) 결과 기준
FIXED_OVERRIDES = {
    "model": {"name": "resnet18"},
    "loss": {
        "element_loss": {"name": "huber", "params": {"delta": 1.0}},
        "weighting": {"name": "uniform"},
        "reduction": "mean",
    },
}

# Round1: 넓은 범위
ROUND1_SPACE = {
    "learning_rate": {"low": 1e-5, "high": 1e-3, "log": True},
    "weight_decay": {"low": 1e-6, "high": 1e-2, "log": True},
    "dropout": {"choices": [0.2, 0.3, 0.4]},
    "batch_size": {"choices": [16, 32, 64]},
}
ROUND1_N_TRIALS = 18

# Round2: round1 결과 보고 이 부분을 직접 좁혀서 수정 후 실행
ROUND2_SPACE = {
    "learning_rate": {"low": 1e-4, "high": 1e-3, "log": True},
    "weight_decay": {"low": 1e-6, "high": 1e-4, "log": True},
    "dropout": {"choices": [0.2]},
    "batch_size": {"choices": [64]},
}
ROUND2_N_TRIALS = 15


def apply_overrides(config, overrides):
    """config에 FIXED_OVERRIDES를 재귀적으로 병합 (nested dict)."""
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(config.get(key), dict):
            apply_overrides(config[key], value)
        else:
            config[key] = value
    return config


def get_fold0_split(pool_df, n_splits, search_fold):
    """cross_validation.py와 동일한 GroupKFold(canonical_smiles)로 fold 하나만 추출."""
    kfold = cross_validation.GroupKFold(n_splits=n_splits)
    splits = list(kfold.split(pool_df, groups=pool_df["canonical_smiles"]))
    train_idx, valid_idx = splits[search_fold]
    train_df = pool_df.iloc[train_idx].reset_index(drop=True)
    valid_df = pool_df.iloc[valid_idx].reset_index(drop=True)
    return train_df, valid_df


def run_round(round_name, search_space, n_trials, base_config, pool_df, output_dir):
    output_dir = Path(output_dir) / round_name
    output_dir.mkdir(parents=True, exist_ok=True)

    train_df, valid_df = get_fold0_split(
        pool_df, base_config["k_fold"]["n_splits"], SEARCH_FOLD
    )

    trials_log = []

    def objective(trial):
        config = copy.deepcopy(base_config)
        apply_overrides(config, FIXED_OVERRIDES)

        lr_space = search_space["learning_rate"]
        wd_space = search_space["weight_decay"]

        config["training"]["learning_rate"] = trial.suggest_float(
            "learning_rate", lr_space["low"], lr_space["high"], log=lr_space["log"]
        )
        config["training"]["weight_decay"] = trial.suggest_float(
            "weight_decay", wd_space["low"], wd_space["high"], log=wd_space["log"]
        )
        config["model"]["dropout"] = trial.suggest_categorical(
            "dropout", search_space["dropout"]["choices"]
        )
        config["dataloader"]["batch_size"] = trial.suggest_categorical(
            "batch_size", search_space["batch_size"]["choices"]
        )
        # scheduler t_max는 보통 epochs와 맞춰야 하므로 동일하게 유지
        config["scheduler"]["t_max"] = config["training"]["epochs"]

        trial_dir = output_dir / f"trial_{trial.number:03d}"
        config["training"]["checkpoint_dir"] = str(trial_dir)
        config["training"]["log_dir"] = str(trial_dir / "logs")

        train_loader, valid_loader = cross_validation._build_fold_loaders(
            train_df, valid_df, config
        )
        model = cross_validation.build_model(
            model_name=config["model"]["name"],
            pretrained=config["model"]["pretrained"],
            dropout=config["model"]["dropout"],
        )
        trainer = cross_validation.Trainer(
            model=model, train_loader=train_loader, valid_loader=valid_loader, config=config
        )
        trainer.fit()

        best_idx = trainer.history["val_loss"].index(min(trainer.history["val_loss"]))
        val_loss = trainer.history["val_loss"][best_idx]

        trials_log.append({
            "trial": trial.number,
            "learning_rate": config["training"]["learning_rate"],
            "weight_decay": config["training"]["weight_decay"],
            "dropout": config["model"]["dropout"],
            "batch_size": config["dataloader"]["batch_size"],
            "val_loss": val_loss,
            "val_mae": trainer.history["val_mae"][best_idx],
        })

        # 탐색 단계는 checkpoint 필요 없음 -> 디스크 절약을 위해 삭제
        shutil.rmtree(trial_dir, ignore_errors=True)

        return val_loss

    study = optuna.create_study(
        study_name=round_name,
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=base_config["training"]["seed"]),
    )
    study.optimize(objective, n_trials=n_trials)

    pd.DataFrame(trials_log).to_csv(output_dir / "trials.csv", index=False)
    with open(output_dir / "best_params.json", "w") as f:
        json.dump(study.best_trial.params, f, indent=2)

    print(f"\n=== {round_name} 완료 ===")
    print(f"Best val_loss: {study.best_trial.value:.4f}")
    print(f"Best params: {study.best_trial.params}")
    print(f"결과 저장: {output_dir}")

    return study


def main():
    if len(sys.argv) != 2 or sys.argv[1] not in ("round1", "round2"):
        print("사용법: python optuna_search.py [round1|round2]")
        sys.exit(1)

    round_name = sys.argv[1]

    base_config = load_config(Path("training/config.yaml"))
    full_metadata = pd.read_csv(base_config["data"]["metadata_path"])
    pool_df = full_metadata[full_metadata["split"] == "train"].reset_index(drop=True)

    output_dir = "results/step8_hpo"

    if round_name == "round1":
        run_round("round1", ROUND1_SPACE, ROUND1_N_TRIALS, base_config, pool_df, output_dir)
    else:
        run_round("round2", ROUND2_SPACE, ROUND2_N_TRIALS, base_config, pool_df, output_dir)


if __name__ == "__main__":
    main()
