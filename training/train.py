"""STEP6 전체 학습 진입점.

STEP3(Dataset/DataLoader), STEP4(Model), STEP5(Trainer)를 config 하나로
연결해서 학습을 실행하고, 학습이 끝나면 다음을 자동으로 생성한다.
    - Learning Curve (train/valid loss 그래프) PNG
    - test set에 대한 예측 CSV (best checkpoint 기준, 원래 물성 스케일로 역정규화)

실행 방법 (프로젝트 루트에서):
    python -m training.train                                  # 전체 학습 (GPU 권장)
    python -m training.train --dry-run                        # CPU로 몇 분 안에 끝나는 축소 실행
    python -m training.train --dry-run --epochs 3 --max-train-samples 128
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")  # GUI 없는 환경(서버)에서도 안전하게 그림 저장
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset

from datasets.dataset import create_dataloader, load_config
from models.factory import build_model
from training.trainer import Trainer
from training.validate import build_test_dataset_and_loader, generate_predictions

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = Path("training/config.yaml")

# --dry-run일 때 기본으로 적용되는 축소값 (CLI 인자로 덮어쓸 수 있음)
DRY_RUN_EPOCHS = 2
DRY_RUN_MAX_TRAIN_SAMPLES = 64
DRY_RUN_MAX_VALID_SAMPLES = 32


def plot_learning_curve(history: dict, save_path: Path) -> None:
    """Trainer.history를 이용해 train/valid loss 곡선을 그려 PNG로 저장한다.

    Args:
        history: Trainer.history (epoch/train_loss/val_loss/val_mae/val_rmse/val_r2).
        save_path: 저장할 PNG 경로.
    """
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].plot(history["epoch"], history["train_loss"], label="train_loss")
    axes[0].plot(history["epoch"], history["val_loss"], label="val_loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Learning Curve (Loss)")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(history["epoch"], history["val_mae"], label="val_MAE")
    axes[1].plot(history["epoch"], history["val_rmse"], label="val_RMSE")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Metric")
    axes[1].set_title("Validation MAE / RMSE")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)

    logger.info("Learning Curve 저장 완료: %s", save_path)


def parse_args() -> argparse.Namespace:
    """CLI 인자를 파싱한다."""
    parser = argparse.ArgumentParser(description="STEP6 전체 학습 (또는 --dry-run으로 축소 실행)")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="CPU에서도 몇 분 안에 끝나도록 epoch 수와 데이터 양을 축소해서 실행 (코드 동작 확인용)",
    )
    parser.add_argument("--epochs", type=int, default=None, help="config의 training.epochs를 덮어씀")
    parser.add_argument(
        "--max-train-samples", type=int, default=None, help="train 샘플 수를 이 값으로 제한"
    )
    parser.add_argument(
        "--max-valid-samples", type=int, default=None, help="valid 샘플 수를 이 값으로 제한"
    )
    return parser.parse_args()


def main() -> None:
    """config.yaml을 읽어 학습 -> Learning Curve -> 예측 CSV까지 전부 실행한다."""
    args = parse_args()
    config = load_config(args.config)

    # --dry-run이면 기본 축소값을 먼저 적용하고, 그 위에 명시적 CLI 값이 있으면 덮어씀
    epochs = args.epochs
    max_train_samples = args.max_train_samples
    max_valid_samples = args.max_valid_samples

    if args.dry_run:
        epochs = epochs if epochs is not None else DRY_RUN_EPOCHS
        max_train_samples = max_train_samples if max_train_samples is not None else DRY_RUN_MAX_TRAIN_SAMPLES
        max_valid_samples = max_valid_samples if max_valid_samples is not None else DRY_RUN_MAX_VALID_SAMPLES

        config["training"]["checkpoint_dir"] = "training/dry_run_checkpoints"
        config["training"]["log_dir"] = "runs/dry_run"
        config["output"]["learning_curve_path"] = "reports/dry_run_learning_curve.png"
        config["output"]["predictions_path"] = "reports/dry_run_predictions.csv"
        logger.info(
            "--dry-run 모드: epochs=%s, max_train_samples=%s, max_valid_samples=%s",
            epochs, max_train_samples, max_valid_samples,
        )

    if epochs is not None:
        config["training"]["epochs"] = epochs
        config["scheduler"]["t_max"] = epochs

    train_loader = create_dataloader(split="train", config_path=args.config)
    valid_loader = create_dataloader(split="valid", config_path=args.config)

    if max_train_samples is not None:
        train_loader = DataLoader(
            Subset(train_loader.dataset, list(range(min(max_train_samples, len(train_loader.dataset))))),
            batch_size=train_loader.batch_size,
            shuffle=True,
            num_workers=train_loader.num_workers,
        )
    if max_valid_samples is not None:
        valid_loader = DataLoader(
            Subset(valid_loader.dataset, list(range(min(max_valid_samples, len(valid_loader.dataset))))),
            batch_size=valid_loader.batch_size,
            shuffle=False,
            num_workers=valid_loader.num_workers,
        )

    model = build_model(
        model_name=config["model"]["name"],
        pretrained=config["model"]["pretrained"] and not args.dry_run,  # dry-run은 인터넷 없이도 되게
        dropout=config["model"]["dropout"],
    )

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        config=config,
    )

    # 중단된 학습을 이어가려면 아래 줄의 주석을 해제하세요.
    # trainer.load_checkpoint(Path(config["training"]["checkpoint_dir"]) / "last_model.pth")

    trainer.fit()
    logger.info("학습 완료. best_score=%.6f", trainer.best_score)

    # 1) Learning Curve 저장
    plot_learning_curve(trainer.history, Path(config["output"]["learning_curve_path"]))

    # 2) best checkpoint로 test set 예측 -> CSV 저장
    best_ckpt_path = Path(config["training"]["checkpoint_dir"]) / "best_model.pth"
    trainer.load_checkpoint(best_ckpt_path)

    test_dataset, test_loader, label_mean, label_std = build_test_dataset_and_loader(config)

    predictions_df = generate_predictions(
        model=trainer.model,
        data_loader=test_loader,
        ids=test_dataset.metadata["id"].tolist(),
        device=trainer.device,
        label_names=test_dataset.LABEL_ORDER,
        label_mean=label_mean,
        label_std=label_std,
    )

    predictions_path = Path(config["output"]["predictions_path"])
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    predictions_df.to_csv(predictions_path, index=False)
    logger.info("예측 CSV 저장 완료: %s (%d행)", predictions_path, len(predictions_df))

    if args.dry_run:
        print("\n--dry-run 완료: 전체 파이프라인(STEP3~6)이 정상 동작하는 것을 확인했습니다.")


if __name__ == "__main__":
    main()
