"""PyTorch 학습 엔진(Trainer).

STEP1~STEP4에서 만든 Dataset/Model/Config를 그대로 받아 학습만 담당한다.
Dataset이나 Model을 다시 만들지 않으며, DataLoader/Model 내부 구조에도
관여하지 않는다 (연결 역할만 수행).

사용 예시:
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        config=config,
    )
    trainer.fit()
"""

from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from evaluation.metrics import compute_metrics
from losses.factory import LossFactory
from training.scheduler import build_scheduler

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    """random / numpy / torch / torch.cuda의 시드를 모두 고정한다.

    Args:
        seed: 고정할 랜덤 시드값.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    logger.info("Random seed 고정: %d", seed)


def get_device() -> torch.device:
    """CUDA -> MPS -> CPU 순으로 사용 가능한 device를 자동 선택한다.

    Returns:
        선택된 torch.device.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    logger.info("사용 device: %s", device)
    return device


class EarlyStopping:
    """Validation Loss 기준 Early Stopping 판정기."""

    def __init__(self, patience: int, min_delta: float) -> None:
        """EarlyStopping을 초기화한다.

        Args:
            patience: 개선 없이 허용할 최대 epoch 수.
            min_delta: "개선"으로 인정할 최소 loss 감소폭.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.best_score: Optional[float] = None
        self.counter = 0
        self.should_stop = False

    def step(self, val_loss: float) -> bool:
        """이번 epoch의 validation loss로 상태를 갱신한다.

        Args:
            val_loss: 이번 epoch의 validation loss.

        Returns:
            이번 epoch이 지금까지 중 최고 성능(best)이면 True.
        """
        if self.best_score is None or val_loss < self.best_score - self.min_delta:
            self.best_score = val_loss
            self.counter = 0
            return True

        self.counter += 1
        if self.counter >= self.patience:
            self.should_stop = True
            logger.info("Early Stopping 조건 충족 (patience=%d)", self.patience)
        return False


class Trainer:
    """Train/Validation Loop, Checkpoint, Early Stopping을 담당하는 학습 엔진."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        valid_loader: DataLoader,
        config: dict,
    ) -> None:
        """Trainer를 초기화한다.

        Args:
            model: STEP4의 build_model()로 생성된 모델.
            train_loader: STEP3의 create_dataloader(split="train") 결과.
            valid_loader: STEP3의 create_dataloader(split="valid") 결과.
            config: STEP1~4에서 사용한 것과 동일한 config 딕셔너리.
        """
        self.config = config
        train_cfg = config["training"]

        set_seed(train_cfg.get("seed", 42))

        self.device = get_device()
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.valid_loader = valid_loader

        self.label_names: List[str] = config["labels"]["order"]

        self.optimizer = AdamW(
            self.model.parameters(),
            lr=train_cfg["learning_rate"],
            weight_decay=train_cfg["weight_decay"],
        )
        self.scheduler = build_scheduler(self.optimizer, config)
        self.criterion = LossFactory.build(config, property_names=self.label_names)

        self.amp_enabled = bool(train_cfg.get("amp", True)) and self.device.type == "cuda"
        if train_cfg.get("amp", True) and not self.amp_enabled:
            logger.info("AMP 요청됨이나 CUDA가 아니므로 fp32로 동작합니다 (device=%s).", self.device)
        # torch.cuda.amp API는 최신 torch에서 deprecated이지만 스펙 요구사항에 맞춰 사용.
        # enabled=False이면 CPU/MPS에서도 안전하게 no-op으로 동작한다.
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.amp_enabled)

        self.grad_clip_norm: float = train_cfg["grad_clip_norm"]
        self.early_stopping = EarlyStopping(
            patience=train_cfg["early_stopping"]["patience"],
            min_delta=train_cfg["early_stopping"]["min_delta"],
        )

        self.checkpoint_dir = Path(train_cfg["checkpoint_dir"])
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        log_dir = Path(train_cfg["log_dir"])
        log_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir=str(log_dir))

        self.start_epoch = 0
        self.best_score = float("inf")

        # STEP6 Learning Curve 저장을 위한 epoch별 기록
        self.history: Dict[str, List[float]] = {
            "epoch": [],
            "train_loss": [],
            "val_loss": [],
            "val_mae": [],
            "val_rmse": [],
            "val_r2": [],
        }

    def _forward_batch(
        self, images: torch.Tensor, labels: torch.Tensor, masks: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """한 배치에 대해 forward + loss 계산만 수행한다 (backward는 하지 않음).

        Args:
            images: 입력 이미지 배치.
            labels: 정답 라벨 배치.
            masks: 결측 마스크 배치.

        Returns:
            (prediction, loss) 튜플.
        """
        with torch.cuda.amp.autocast(enabled=self.amp_enabled):
            prediction = self.model(images)
            loss = self.criterion(prediction, labels, masks)
        return prediction, loss

    def train_one_epoch(self, epoch: int) -> float:
        """한 epoch만큼 학습하고 평균 train loss를 반환한다.

        Args:
            epoch: 현재 epoch 번호 (로깅용).

        Returns:
            이번 epoch의 평균 train loss.
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        progress = tqdm(self.train_loader, desc=f"[Epoch {epoch}] Train")
        for images, labels, masks in progress:
            # BatchNorm1d는 배치 크기 1에서 학습 모드로 동작할 수 없으므로 건너뛴다.
            if images.size(0) <= 1:
                logger.warning("배치 크기가 1 이하라 이번 배치를 건너뜁니다 (epoch=%d).", epoch)
                continue

            if masks.sum().item() == 0:
                logger.warning("배치 내 모든 Label이 Missing이라 이번 배치를 건너뜁니다.")
                continue

            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            masks = masks.to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)
            _, loss = self._forward_batch(images, labels, masks)

            if torch.isnan(loss) or torch.isinf(loss):
                logger.error("NaN/Inf loss가 발생해 이번 배치를 건너뜁니다 (epoch=%d).", epoch)
                continue

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item()
            num_batches += 1
            progress.set_postfix(loss=loss.item())

        return total_loss / num_batches if num_batches > 0 else float("nan")

    @torch.no_grad()
    def validate(self, epoch: int) -> Tuple[float, Dict[str, Dict[str, float]]]:
        """Validation Loop을 수행하고 (평균 loss, metric 딕셔너리)를 반환한다.

        Args:
            epoch: 현재 epoch 번호 (로깅용).

        Returns:
            (val_loss, metrics) 튜플. metrics는 evaluation.metrics.compute_metrics 결과.
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        all_predictions: List[np.ndarray] = []
        all_targets: List[np.ndarray] = []
        all_masks: List[np.ndarray] = []

        progress = tqdm(self.valid_loader, desc=f"[Epoch {epoch}] Valid")
        for images, labels, masks in progress:
            if masks.sum().item() == 0:
                logger.warning("Validation 배치 내 모든 Label이 Missing이라 건너뜁니다.")
                continue

            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            masks = masks.to(self.device, non_blocking=True)

            prediction, loss = self._forward_batch(images, labels, masks)

            if not (torch.isnan(loss) or torch.isinf(loss)):
                total_loss += loss.item()
                num_batches += 1

            all_predictions.append(prediction.detach().cpu().numpy())
            all_targets.append(labels.detach().cpu().numpy())
            all_masks.append(masks.detach().cpu().numpy())

        val_loss = total_loss / num_batches if num_batches > 0 else float("nan")

        if all_predictions:
            metrics = compute_metrics(
                predictions=np.concatenate(all_predictions, axis=0),
                targets=np.concatenate(all_targets, axis=0),
                masks=np.concatenate(all_masks, axis=0),
                label_names=self.label_names,
            )
        else:
            logger.warning("Validation에 유효한 배치가 없어 metric을 계산할 수 없습니다.")
            metrics = {}

        return val_loss, metrics

    def save_checkpoint(self, epoch: int, is_best: bool) -> None:
        """현재 학습 상태를 체크포인트로 저장한다 (last, 필요시 best).

        Args:
            epoch: 현재 epoch 번호.
            is_best: True이면 best_model.pth도 함께 저장한다.
        """
        state = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "scaler_state_dict": self.scaler.state_dict(),
            "best_score": self.best_score,
            "config": self.config,
        }

        last_path = self.checkpoint_dir / "last_model.pth"
        torch.save(state, last_path)

        if is_best:
            best_path = self.checkpoint_dir / "best_model.pth"
            torch.save(state, best_path)
            logger.info("Best model 갱신 및 저장: %s (epoch=%d)", best_path, epoch)

    def load_checkpoint(self, checkpoint_path: Path) -> None:
        """체크포인트를 불러와 학습을 이어갈 수 있도록 상태를 복원한다 (Resume Training).

        Args:
            checkpoint_path: 불러올 .pth 파일 경로.

        Raises:
            FileNotFoundError: 파일이 존재하지 않는 경우.
            RuntimeError: 체크포인트가 손상되어 로드할 수 없는 경우.
        """
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"체크포인트 파일을 찾을 수 없습니다: {checkpoint_path}")

        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
        except Exception as exc:
            raise RuntimeError(f"체크포인트가 손상되어 로드할 수 없습니다: {checkpoint_path}") from exc

        required_keys = {
            "epoch",
            "model_state_dict",
            "optimizer_state_dict",
            "scheduler_state_dict",
            "scaler_state_dict",
            "best_score",
        }
        missing = required_keys - checkpoint.keys()
        if missing:
            raise RuntimeError(f"체크포인트에 필수 키가 없습니다: {missing}")

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.scaler.load_state_dict(checkpoint["scaler_state_dict"])
        self.best_score = checkpoint["best_score"]
        self.start_epoch = checkpoint["epoch"] + 1

        logger.info(
            "체크포인트 로드 완료: %s (재시작 epoch=%d, best_score=%.6f)",
            checkpoint_path, self.start_epoch, self.best_score,
        )

    def fit(self) -> None:
        """전체 학습 루프를 실행한다 (Train -> Validate -> Log -> Checkpoint -> Early Stop)."""
        epochs = self.config["training"]["epochs"]

        for epoch in range(self.start_epoch, epochs):
            train_loss = self.train_one_epoch(epoch)
            val_loss, metrics = self.validate(epoch)

            current_lr = self.optimizer.param_groups[0]["lr"]
            self.scheduler.step()

            self.writer.add_scalar("Loss/train", train_loss, epoch)
            self.writer.add_scalar("Loss/valid", val_loss, epoch)
            self.writer.add_scalar("LearningRate", current_lr, epoch)

            avg_mae = metrics.get("average", {}).get("mae", float("nan")) if metrics else float("nan")
            avg_rmse = metrics.get("average", {}).get("rmse", float("nan")) if metrics else float("nan")
            avg_r2 = metrics.get("average", {}).get("r2", float("nan")) if metrics else float("nan")

            if metrics:
                self.writer.add_scalar("MAE/average", avg_mae, epoch)
                self.writer.add_scalar("RMSE/average", avg_rmse, epoch)
                self.writer.add_scalar("R2/average", avg_r2, epoch)

            self.history["epoch"].append(epoch)
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["val_mae"].append(avg_mae)
            self.history["val_rmse"].append(avg_rmse)
            self.history["val_r2"].append(avg_r2)

            # STEP6 요구사항: Loss / MAE / RMSE / R² 콘솔 출력
            logger.info(
                "[Epoch %d] train_loss=%.6f val_loss=%.6f MAE=%.6f RMSE=%.6f R2=%.6f lr=%.8f",
                epoch, train_loss, val_loss, avg_mae, avg_rmse, avg_r2, current_lr,
            )

            is_best = self.early_stopping.step(val_loss)
            if is_best:
                self.best_score = val_loss
            self.save_checkpoint(epoch, is_best)

            if self.early_stopping.should_stop:
                logger.info("Early Stopping으로 학습을 조기 종료합니다 (epoch=%d).", epoch)
                break

        self.writer.close()
