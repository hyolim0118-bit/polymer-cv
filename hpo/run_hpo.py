"""Optuna HPO runner for the project's training pipeline.

Run a quick CPU verification from the repository root:

    python -m hpo.run_hpo --smoke-test

The run writes one resolved config per trial, ``trials.csv``, and
``best_trial.yaml``.  Those artifacts make it possible to verify that sampled
parameters were actually used to construct the DataLoader, model, and Trainer.
"""

from __future__ import annotations

import argparse
import copy
import csv
from pathlib import Path
from typing import Any, Dict

import optuna
import yaml
from torch.utils.data import DataLoader, Subset

from datasets.dataset import PolymerDataset, load_config, load_split_dataframes
from datasets.transforms import get_train_transform, get_valid_transform
from hpo.search_space import sample_hyperparams
from models.factory import build_model
from training.trainer import Trainer


PARAMETER_PATHS = {
    "learning_rate": ("training", "learning_rate"),
    "weight_decay": ("training", "weight_decay"),
    "dropout": ("model", "dropout"),
    "batch_size": ("dataloader", "batch_size"),
    "epochs": ("training", "epochs"),
    "amp": ("training", "amp"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Optuna HPO against the training pipeline.")
    parser.add_argument("--training-config", type=Path, default=Path("training/config.yaml"))
    parser.add_argument("--hpo-config", type=Path, default=Path("hpo/config_example.yaml"))
    parser.add_argument("--round", choices=("round1", "round2"), default="round1")
    parser.add_argument("--trials", type=int, default=None, help="Override the configured trial count.")
    parser.add_argument("--smoke-test", action="store_true", help="CPU-friendly: 2 trials, 1 epoch, 16/8 samples.")
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-valid-samples", type=int, default=None)
    return parser.parse_args()


def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def apply_parameters(base_config: Dict[str, Any], parameters: Dict[str, Any]) -> Dict[str, Any]:
    """Return a trial config with every HPO parameter mapped to its real config key."""
    config = copy.deepcopy(base_config)
    for name, value in parameters.items():
        if name not in PARAMETER_PATHS:
            raise ValueError(f"Unsupported HPO parameter: {name}")
        section, key = PARAMETER_PATHS[name]
        config[section][key] = value

    # Keep the scheduler aligned when HPO changes the number of epochs.
    config["scheduler"]["t_max"] = config["training"]["epochs"]
    return config


def make_loaders(config: Dict[str, Any], max_train: int | None, max_valid: int | None) -> tuple[DataLoader, DataLoader]:
    data_cfg, image_cfg, aug_cfg = config["data"], config["image"], config["augmentation"]
    split_cfg, loader_cfg = config["split"], config["dataloader"]
    train_df, valid_df, _, label_mean, label_std = load_split_dataframes(
        Path(data_cfg["metadata_path"]), split_cfg["val_ratio"], split_cfg["seed"]
    )
    train_dataset = PolymerDataset(
        train_df, Path(data_cfg["image_root"]),
        get_train_transform(image_cfg["size"], image_cfg["normalize_mean"], image_cfg["normalize_std"], aug_cfg),
        label_mean, label_std,
    )
    valid_dataset = PolymerDataset(
        valid_df, Path(data_cfg["image_root"]),
        get_valid_transform(image_cfg["size"], image_cfg["normalize_mean"], image_cfg["normalize_std"]),
        label_mean, label_std,
    )
    if max_train is not None:
        train_dataset = Subset(train_dataset, range(min(max_train, len(train_dataset))))
    if max_valid is not None:
        valid_dataset = Subset(valid_dataset, range(min(max_valid, len(valid_dataset))))

    batch_size = loader_cfg["batch_size"]
    return (
        DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=loader_cfg["num_workers"]),
        DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=loader_cfg["num_workers"]),
    )


def write_trial_row(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(row))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def main() -> None:
    args = parse_args()
    base_config = load_config(args.training_config)
    hpo_config = load_yaml(args.hpo_config)
    round_config = hpo_config["hpo"][args.round]
    fixed = hpo_config["hpo"].get("fixed", {})
    n_trials = args.trials if args.trials is not None else round_config["n_trials"]
    max_train, max_valid = args.max_train_samples, args.max_valid_samples

    if args.smoke_test:
        n_trials = args.trials if args.trials is not None else 2
        fixed = {**fixed, "epochs": 1, "amp": False}
        max_train = max_train if max_train is not None else 8
        max_valid = max_valid if max_valid is not None else 4

    output_dir = Path(round_config["output_dir"])
    if args.smoke_test:
        output_dir = output_dir / "smoke_test"
    output_dir.mkdir(parents=True, exist_ok=True)
    trials_csv = output_dir / "trials.csv"
    # A smoke study is in-memory and always restarts from trial 0.  Start its
    # small verification report fresh instead of mixing it with an older run.
    if args.smoke_test and trials_csv.exists():
        trials_csv.unlink()

    storage = None if args.smoke_test else round_config.get("storage")
    if storage and storage.startswith("sqlite:///"):
        Path(storage[len("sqlite:///"):]).parent.mkdir(parents=True, exist_ok=True)
    study = optuna.create_study(
        study_name=f"{args.round}{'_smoke' if args.smoke_test else ''}",
        direction="minimize", storage=storage, load_if_exists=bool(storage),
        sampler=optuna.samplers.TPESampler(seed=42),
    )

    def objective(trial: optuna.Trial) -> float:
        sampled = sample_hyperparams(trial, round_config["search_space"])
        trial_parameters = {**fixed, **sampled}
        config = apply_parameters(base_config, trial_parameters)
        if args.smoke_test:
            # Keep the verification fast on CPU without changing normal HPO behavior.
            config["image"]["size"] = 64
            config["dataloader"]["num_workers"] = 0
        trial_dir = output_dir / f"trial_{trial.number:03d}"
        config["training"]["checkpoint_dir"] = str(trial_dir / "checkpoints")
        config["training"]["log_dir"] = str(trial_dir / "logs")
        trial_dir.mkdir(parents=True, exist_ok=True)
        with (trial_dir / "resolved_config.yaml").open("w", encoding="utf-8") as file:
            yaml.safe_dump(config, file, sort_keys=False)

        train_loader, valid_loader = make_loaders(config, max_train, max_valid)
        model = build_model(config["model"]["name"], pretrained=False if args.smoke_test else config["model"]["pretrained"], dropout=config["model"]["dropout"])
        trainer = Trainer(model, train_loader, valid_loader, config)
        trainer.fit()
        val_mae = min(trainer.history["val_mae"])
        if val_mae != val_mae:  # NaN cannot be optimized safely.
            raise RuntimeError("Validation MAE is NaN; inspect the trial data and labels.")

        row = {"trial_id": trial.number, **sampled, "val_MAE": val_mae, "best_val_loss": trainer.best_score}
        write_trial_row(trials_csv, row)
        print(f"trial={trial.number} params={sampled} val_MAE={val_mae:.6f}")
        return val_mae

    study.optimize(objective, n_trials=n_trials)
    best_path = output_dir / "best_trial.yaml"
    with best_path.open("w", encoding="utf-8") as file:
        yaml.safe_dump({"params": {**fixed, **study.best_trial.params}, "val_MAE": study.best_value}, file, sort_keys=False)
    print(f"Best trial: {study.best_trial.number}, val_MAE={study.best_value:.6f}")
    print(f"Artifacts: {trials_csv} and {best_path}")


if __name__ == "__main__":
    main()
