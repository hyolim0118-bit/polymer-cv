"""CSV(SMILES + 물성 라벨) -> 이미지 데이터셋 + processed_metadata.csv 생성.

STEP2 요구사항을 구현하는 최상위 실행 모듈이다.
    1. data/raw/{train,test}.csv 를 읽는다.
    1-1. (선택) data/raw/external/*.csv 의 물성 라벨을 train에 병합한다.
    2. SMILES를 파싱해 224x224 PNG로 저장한다 (images/render/{split}/{id}.png).
    3. 물성 라벨을 train 통계 기준으로 정규화하고, 결측 여부를 마스크로 남긴다.
    4. train/test 동일한 스키마로 data/processed/processed_metadata.csv,
       data/processed/stats.json 을 저장한다.

이 데이터셋(NeurIPS Open Polymer Prediction 2025)의 물성 컬럼은
Tg, FFV, Tc, Density, Rg 5개이며, 각 행마다 일부 컬럼만 라벨이 채워져
있는 경우가 대부분이다 (Missing Label 유지 요구사항).
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
from rdkit import Chem
from tqdm import tqdm

from preprocessing.mol_to_image import render_mol_to_png
from preprocessing.smiles_to_mol import parse_smiles

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Metadata CSV column order follows the source/submission schema. Training reconstructs
# label vectors by datasets.dataset.LABEL_ORDER (column name), so this does not affect it.
LABEL_COLUMNS: List[str] = ["Tg", "FFV", "Tc", "Density", "Rg"]

# 외부 보강 데이터: (파일명, {외부 CSV 컬럼명: LABEL_COLUMNS 중 대응 컬럼명})
# dataset2.csv(라벨 없는 SMILES)는 지도학습에 쓸 수 없어 제외한다.
EXTERNAL_SOURCES: List[Tuple[str, Dict[str, str]]] = [
    ("dataset1.csv", {"TC_mean": "Tc"}),
    ("dataset3.csv", {"Tg": "Tg"}),
    ("dataset4.csv", {"FFV": "FFV"}),
]


def merge_external_data(train_df: pd.DataFrame, external_dir: Path) -> pd.DataFrame:
    """외부 물성 데이터를 train_df에 병합한다 (train.csv 값이 항상 우선).

    정책:
        1. SMILES가 train_df에 이미 있으면, 그 물성이 결측일 때만 채운다
           (train.csv에 값이 있으면 그대로 두고 덮어쓰지 않는다).
        2. SMILES가 train_df에 없으면 새 행으로 추가한다 (해당 물성만 라벨, 나머지는 결측).
        3. 같은 병합 과정에서 여러 외부 파일에 걸쳐 등장하는 신규 SMILES는
           같은 새 행에 물성들을 이어서 채운다 (행이 중복 생성되지 않도록).

    Args:
        train_df: 원본 train 데이터프레임.
        external_dir: dataset1.csv, dataset3.csv, dataset4.csv가 있는 디렉토리.

    Returns:
        외부 데이터가 병합된 새 데이터프레임 (train_df는 변경하지 않음).
    """
    merged = train_df.copy()
    smiles_to_row_idx = {s: i for i, s in enumerate(merged["SMILES"])}
    # 이번 병합 중 새로 추가되는 분자들을 SMILES 기준으로 임시 보관
    # (여러 외부 파일에 같은 SMILES가 나오면 한 행에 계속 이어서 채우기 위함)
    pending_new_rows: Dict[str, Dict] = {}
    # 기존 id와 절대 겹치지 않도록 충분히 큰 오프셋에서 새 id 채번 시작
    next_id = int(merged["id"].max()) + 1_000_000_000

    filled_count = 0
    added_count = 0

    for filename, col_map in EXTERNAL_SOURCES:
        path = external_dir / filename
        if not path.exists():
            logger.warning("외부 데이터 파일을 찾을 수 없어 건너뜁니다: %s", path)
            continue

        ext_df = pd.read_csv(path)
        for _, ext_row in ext_df.iterrows():
            smiles = ext_row["SMILES"]

            for ext_col, target_col in col_map.items():
                value = ext_row[ext_col]
                if pd.isna(value):
                    continue

                if smiles in smiles_to_row_idx:
                    # train.csv에 이미 있는 분자 -> 결측인 물성만 채움 (train.csv 값 우선)
                    idx = smiles_to_row_idx[smiles]
                    if pd.isna(merged.at[idx, target_col]):
                        merged.at[idx, target_col] = value
                        filled_count += 1
                elif smiles in pending_new_rows:
                    # 이번 병합에서 이미 새로 추가하기로 한 분자 -> 다른 물성도 이어서 채움
                    if pd.isna(pending_new_rows[smiles][target_col]):
                        pending_new_rows[smiles][target_col] = value
                        filled_count += 1
                else:
                    # 완전히 새로운 분자 -> 새 행 준비 (나머지 물성은 전부 결측)
                    new_row = {col: pd.NA for col in merged.columns}
                    new_row["id"] = next_id
                    next_id += 1
                    new_row["SMILES"] = smiles
                    new_row[target_col] = value
                    pending_new_rows[smiles] = new_row
                    added_count += 1

    if pending_new_rows:
        merged = pd.concat([merged, pd.DataFrame(pending_new_rows.values())], ignore_index=True)

    logger.info(
        "외부 데이터 병합 완료: 기존 분자 결측 %d개 채움, 신규 분자 %d개 추가 (train 총 %d -> %d행)",
        filled_count, added_count, len(train_df), len(merged),
    )
    return merged


def compute_label_stats(train_df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """train 데이터의 결측을 제외한 값으로 물성별 평균/표준편차를 계산한다.

    Args:
        train_df: 원본 train 데이터프레임 (물성 컬럼 포함).

    Returns:
        {물성명: {"mean": ..., "std": ...}} 형태의 딕셔너리.
    """
    stats: Dict[str, Dict[str, float]] = {}
    for col in LABEL_COLUMNS:
        values = train_df[col].dropna()  # 결측(NaN)인 행은 통계 계산에서 제외
        mean = float(values.mean()) if len(values) > 0 else 0.0
        std = float(values.std()) if len(values) > 1 else 1.0
        # 표준편차가 0이면 나중에 (x-mean)/std에서 0으로 나누는 사고가 나므로 1로 방어
        if std == 0.0 or pd.isna(std):
            std = 1.0
        stats[col] = {"mean": mean, "std": std}
    return stats


def process_split(
    df: pd.DataFrame,
    split: str,
    image_dir: Path,
    stats: Dict[str, Dict[str, float]],
) -> pd.DataFrame:
    """한 split(train 또는 test)에 대해 이미지 생성 + 메타데이터 행을 구성한다.

    Args:
        df: 해당 split의 원본 데이터프레임.
        split: "train" 또는 "test".
        image_dir: 이미지를 저장할 최상위 디렉토리 (images/render).
        stats: compute_label_stats로 계산한 물성별 평균/표준편차.

    Returns:
        image_path, split, 원본 라벨, 정규화 라벨, mask 컬럼을 포함한 데이터프레임.
    """
    records = []  # 각 행(샘플)에 대한 결과를 dict로 쌓아뒀다가 마지막에 DataFrame으로 변환
    split_image_dir = image_dir / split  # 예: images/render/train

    # tqdm으로 감싸면 for문 진행 상황이 진행바(%)로 표시됨
    for row in tqdm(df.itertuples(index=False), total=len(df), desc=f"[{split}] 처리 중"):
        row_dict = row._asdict()
        sample_id = row_dict["id"]
        smiles = row_dict["SMILES"]

        # 1단계: SMILES -> Mol 객체 (smiles_to_mol.py)
        mol = parse_smiles(smiles)
        if mol is None:
            logger.warning("[%s] id=%s: SMILES 무효로 건너뜀 -> %s", split, sample_id, smiles)
            continue  # 이 샘플은 통째로 스킵하고 다음 행으로

        # 2단계: Mol 객체 -> PNG 이미지 저장 (mol_to_image.py)
        image_path = split_image_dir / f"{sample_id}.png"
        success = render_mol_to_png(mol, image_path)
        if not success:
            logger.warning("[%s] id=%s: 이미지 렌더링 실패로 건너뜀", split, sample_id)
            continue

        # 3단계: metadata 한 행 구성 시작 (id, 이미지 경로, split만 우선 채움)
        record = {
            "id": sample_id,
            "image_path": str(image_path.relative_to(image_dir.parent)),
            "split": split,
            "SMILES": smiles,
            # 표기(방향환 순서 등)만 다른 동일 분자를 GroupKFold에서 하나의
            # group으로 묶기 위한 정규화된 SMILES.
            "canonical_smiles": Chem.MolToSmiles(mol),
        }

        # 4단계: 물성 5개 각각에 대해 (원본값, 정규화값, 결측마스크) 3종 컬럼 생성
        for col in LABEL_COLUMNS:
            raw_value = row_dict.get(col, None)
            has_label = raw_value is not None and not pd.isna(raw_value)
            record[col] = raw_value if has_label else None       # 원본값 (결측이면 None 유지)
            record[f"{col}_mask"] = 1 if has_label else 0         # 1=라벨 있음, 0=결측
            if has_label:
                # z-score 정규화: (원본값 - train평균) / train표준편차
                mean, std = stats[col]["mean"], stats[col]["std"]
                record[f"{col}_norm"] = (raw_value - mean) / std
            else:
                # masked loss에서 mask=0인 값은 어차피 무시되므로 0으로 채워도 학습에 영향 없음.
                record[f"{col}_norm"] = 0.0

        records.append(record)

    return pd.DataFrame.from_records(records)


def run(
    raw_dir: Path,
    image_dir: Path,
    processed_dir: Path,
    external_dir: Path = None,
    use_external: bool = True,
) -> None:
    """전체 전처리 파이프라인을 실행한다.

    Args:
        raw_dir: train.csv / test.csv가 있는 디렉토리 (data/raw).
        image_dir: 렌더링된 이미지를 저장할 디렉토리 (images/render).
        processed_dir: metadata/stats 출력 디렉토리 (data/processed).
        external_dir: 외부 보강 데이터(dataset1/3/4.csv)가 있는 디렉토리.
            None이면 raw_dir/"external"을 사용한다.
        use_external: True이면 외부 데이터를 train에 병합한다.
    """
    train_path = raw_dir / "train.csv"
    test_path = raw_dir / "test.csv"

    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(f"train.csv 또는 test.csv를 {raw_dir}에서 찾을 수 없습니다.")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    # test.csv에는 물성 컬럼이 없으므로 동일 스키마를 위해 결측으로 추가한다.
    for col in LABEL_COLUMNS:
        if col not in test_df.columns:
            test_df[col] = pd.NA

    if use_external:
        external_dir = external_dir if external_dir is not None else raw_dir / "external"
        train_df = merge_external_data(train_df, external_dir)

    # 정규화 통계는 반드시 (외부 데이터 병합 후의) train 데이터만으로 계산
    # -> test 정보가 섞이면 data leakage
    stats = compute_label_stats(train_df)

    processed_dir.mkdir(parents=True, exist_ok=True)
    with open(processed_dir / "stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    logger.info("정규화 통계 저장 완료: %s", processed_dir / "stats.json")

    # train, test 각각에 대해 이미지 생성 + metadata 행 구성 (핵심 로직은 process_split)
    train_meta = process_split(train_df, "train", image_dir, stats)
    test_meta = process_split(test_df, "test", image_dir, stats)

    # 두 결과를 하나의 CSV로 합쳐서 저장 (split 컬럼으로 나중에 구분 가능)
    metadata = pd.concat([train_meta, test_meta], ignore_index=True)
    metadata_path = processed_dir / "processed_metadata.csv"
    metadata.to_csv(metadata_path, index=False)

    logger.info(
        "전처리 완료: train %d/%d행, test %d/%d행 -> %s",
        len(train_meta), len(train_df),
        len(test_meta), len(test_df),
        metadata_path,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Polymer SMILES -> 이미지 데이터셋 전처리")
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--image-dir", type=Path, default=Path("images/render"))
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--external-dir", type=Path, default=None)
    parser.add_argument(
        "--no-external", action="store_true", help="외부 보강 데이터(dataset1/3/4.csv)를 병합하지 않는다."
    )
    args = parser.parse_args()

    run(
        args.raw_dir,
        args.image_dir,
        args.processed_dir,
        external_dir=args.external_dir,
        use_external=not args.no_external,
    )


if __name__ == "__main__":
    main()
