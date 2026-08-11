"""overfit_test.py

작은 배치(32개)로 모델이 100 epoch 안에 loss를 충분히 낮출 수 있는지
확인하는 sanity check. Data/Model/Loss/Trainer 파이프라인이 최소한
"암기(overfit)"는 할 수 있는 상태인지를 빠르게 검증하는 용도다.

같은 32개 샘플을 train/valid에 그대로 재사용한다 (일부러 held-out을
두지 않음 — 이 스크립트의 목적은 일반화가 아니라 "이 모델/파이프라인이
이 32개를 외울 능력이 있는가"를 보는 것).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from torch.utils.data import DataLoader

from datasets.dataset import LABEL_ORDER, PolymerDataset, compute_fold_stats, load_config
from datasets.transforms import get_valid_transform
from models.factory import build_model
from training.trainer import Trainer

CONFIG_PATH = Path("training/config.yaml")
NUM_SAMPLES = 32
EPOCHS = 100


def main() -> None:
    config = load_config(CONFIG_PATH)

    data_cfg = config["data"]
    image_cfg = config["image"]

    full_metadata = pd.read_csv(data_cfg["metadata_path"])
    pool_df = full_metadata[full_metadata["split"] == "train"].reset_index(drop=True)

    image_root = Path(data_cfg["image_root"])
    exists_mask = pool_df["image_path"].apply(lambda p: (image_root / p).exists())
    pool_df = pool_df[exists_mask].reset_index(drop=True)

    sample_df = pool_df.sample(n=NUM_SAMPLES, random_state=0).reset_index(drop=True)

    # 오버핏 테스트이므로 augmentation 없이(고정된 입력으로) 암기 능력만 본다.
    transform = get_valid_transform(
        image_cfg["size"], image_cfg["normalize_mean"], image_cfg["normalize_std"]
    )

    label_mean, label_std = compute_fold_stats(sample_df, LABEL_ORDER)

    dataset = PolymerDataset(sample_df, image_root, transform, label_mean, label_std)

    # 배치 하나(=32개 전부)를 매 epoch 그대로 재사용. train/valid를 같은
    # 데이터로 구성해 "이 파이프라인이 이 32개를 외울 수 있는가"만 본다.
    train_loader = DataLoader(dataset, batch_size=NUM_SAMPLES, shuffle=False, num_workers=0)
    valid_loader = DataLoader(dataset, batch_size=NUM_SAMPLES, shuffle=False, num_workers=0)

    config["training"]["epochs"] = EPOCHS
    config["training"]["checkpoint_dir"] = "training/overfit_checkpoints"
    config["training"]["log_dir"] = "runs/overfit_test"
    # Early Stopping이 도중에 끊지 못하도록 사실상 비활성화 (100 epoch 전부 관찰)
    config["training"]["early_stopping"]["patience"] = EPOCHS + 1
    config["training"]["early_stopping"]["min_delta"] = 0.0
    config["scheduler"]["t_max"] = EPOCHS
    config["model"]["pretrained"] = False  # 인터넷 없이도 바로 돌아가도록

    model = build_model(
        model_name=config["model"]["name"],
        pretrained=config["model"]["pretrained"],
        dropout=config["model"]["dropout"],
    )

    trainer = Trainer(model=model, train_loader=train_loader, valid_loader=valid_loader, config=config)
    trainer.fit()

    train_loss = trainer.history["train_loss"]
    val_loss = trainer.history["val_loss"]
    val_r2 = trainer.history["val_r2"]

    # train_loss는 train() 모드(dropout 활성)라 매 epoch 다른 서브네트워크를
    # 통과해 값이 들쭉날쭉하다. eval() 모드(dropout 비활성)라 결과가
    # 결정적인 val_loss/val_r2를 판정 기준으로 쓴다 (train/valid가 같은
    # 32개 데이터이므로 val_loss가 곧 "이 데이터를 얼마나 외웠는가"를 뜻함).
    initial_val_loss = val_loss[0]
    final_val_loss = val_loss[-1]
    ratio = final_val_loss / initial_val_loss if initial_val_loss else float("nan")

    print("\n========== Overfit Test 결과 ==========")
    print(f"샘플 수: {NUM_SAMPLES}, Epoch 수: {len(train_loss)}")
    print(f"Epoch 0   train_loss={train_loss[0]:.6f}  val_loss={val_loss[0]:.6f}  val_r2={val_r2[0]:.6f}")
    print(f"Epoch {len(train_loss)-1:<3} train_loss={train_loss[-1]:.6f}  val_loss={val_loss[-1]:.6f}  val_r2={val_r2[-1]:.6f}")
    print(f"val_loss 감소율: {final_val_loss:.6f} / {initial_val_loss:.6f} = {ratio:.4%}")

    if ratio < 0.05 and val_r2[-1] > 0.9:
        print("판정: PASS - 32개 샘플을 100 epoch 안에 충분히 암기함 (val_loss가 초기값의 5% 미만, R2>0.9).")
    else:
        print("판정: FAIL - loss가 충분히 낮아지지 않음. 파이프라인(Data/Model/Loss/Optimizer) 점검 필요.")


if __name__ == "__main__":
    main()
