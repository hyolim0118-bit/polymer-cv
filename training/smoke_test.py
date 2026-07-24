"""STEP5 Trainer 스모크 테스트.

전체 7973장을 다 돌리지 않고, train/valid에서 일부만 뽑아
2 epoch만 돌려서 Trainer가 에러 없이 동작하는지(생성 -> forward ->
loss 계산 -> backward -> checkpoint 저장)만 빠르게 확인한다.

GPU 없이 CPU로 몇 분 안에 끝난다.

실행 방법 (프로젝트 루트에서):
    python -m training.smoke_test
"""

from __future__ import annotations

from pathlib import Path

from torch.utils.data import DataLoader, Subset

from datasets.dataset import PolymerDataset, load_config, load_split_dataframes
from datasets.transforms import get_train_transform, get_valid_transform
from models.factory import build_model
from training.trainer import Trainer

CONFIG_PATH = Path("training/config.yaml")
NUM_TRAIN_SAMPLES = 64
NUM_VALID_SAMPLES = 32


def main() -> None:
    """소량 데이터로 Trainer 전체 파이프라인을 빠르게 검증한다."""
    config = load_config(CONFIG_PATH)

    # 스모크 테스트용으로 epoch 수와 저장 경로만 임시로 축소
    config["training"]["epochs"] = 2
    config["training"]["checkpoint_dir"] = "training/smoke_checkpoints"
    config["training"]["log_dir"] = "runs/smoke_test"
    config["scheduler"]["t_max"] = 2

    img_cfg = config["image"]
    aug_cfg = config["augmentation"]
    data_cfg = config["data"]
    split_cfg = config["split"]

    train_tf = get_train_transform(
        img_cfg["size"], img_cfg["normalize_mean"], img_cfg["normalize_std"], aug_cfg
    )
    valid_tf = get_valid_transform(
        img_cfg["size"], img_cfg["normalize_mean"], img_cfg["normalize_std"]
    )

    # train/valid/test 분할과, Train Fold만으로 계산된 label_mean/std를 함께 얻음
    train_df, valid_df, _, label_mean, label_std = load_split_dataframes(
        metadata_path=data_cfg["metadata_path"],
        val_ratio=split_cfg["val_ratio"],
        seed=split_cfg["seed"],
    )

    train_ds = PolymerDataset(train_df, data_cfg["image_root"], train_tf, label_mean, label_std)
    valid_ds = PolymerDataset(valid_df, data_cfg["image_root"], valid_tf, label_mean, label_std)

    # 전체 중 일부(64/32장)만 잘라서 사용 -> CPU로도 몇 분 안에 끝남
    train_small = Subset(train_ds, list(range(min(NUM_TRAIN_SAMPLES, len(train_ds)))))
    valid_small = Subset(valid_ds, list(range(min(NUM_VALID_SAMPLES, len(valid_ds)))))

    train_loader = DataLoader(train_small, batch_size=8, shuffle=True)
    valid_loader = DataLoader(valid_small, batch_size=8, shuffle=False)

    # pretrained=False로 해야 인터넷 없이도 바로 테스트 가능
    model = build_model(
        model_name=config["model"]["name"], pretrained=False, dropout=config["model"]["dropout"]
    )

    trainer = Trainer(model=model, train_loader=train_loader, valid_loader=valid_loader, config=config)
    trainer.fit()

    print("\n스모크 테스트 완료: Trainer 생성 -> Loss 계산 -> Checkpoint 저장까지 정상 동작 확인.")


if __name__ == "__main__":
    main()
