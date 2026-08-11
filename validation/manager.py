"""
manager.py
----------
STEP6.5의 핵심 파일. fold 생성부터 결과 집계까지 전 과정을 조율한다.

ValidationManager는 trainer/dataset의 실제 구현을 알지 못한다.
대신 아래 두 콜백을 주입받아 기존 train.py / validate.py를 그대로 호출한다.

    train_fn(fold_id, train_idx, val_idx, fold_stats, output_dir) -> checkpoint_path
    predict_fn(checkpoint_path, val_idx, fold_stats) -> (y_true, y_pred)  # 역정규화된 값

이렇게 하면 STEP7(backbone 교체), STEP9(loss 교체) 등 이후 실험들이
train_fn만 바꿔서 넘기면 되고, ValidationManager/leakage check/report 코드는
전혀 손댈 필요가 없다.
"""

from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from .strategies.base import SplitStrategy
from .leakage_check import check_no_leakage
from .metrics import compute_property_metrics, average_across_properties
from .report import save_cv_summary

TrainFn = Callable[[int, np.ndarray, np.ndarray, dict, Path], Union[str, Path]]
PredictFn = Callable[[Union[str, Path], np.ndarray, dict], Tuple[np.ndarray, np.ndarray]]


class ValidationManager:
    def __init__(
        self,
        df: pd.DataFrame,
        properties: List[str],
        group_col: str,
        split_strategy: SplitStrategy,
        output_dir: Union[str, Path],
        compute_fold_stats_fn: Optional[Callable[[pd.DataFrame, np.ndarray], dict]] = None,
    ):
        """
        Parameters
        ----------
        df : pd.DataFrame
            STEP3 load_split_dataframes 결과 (전체 학습 데이터)
        properties : list of str
            예측 대상 property 목록 (Tg, FFV, Tc, Density, Rg 등)
        group_col : str
            그룹 키 컬럼. 반드시 canonical_smiles를 사용한다 (raw SMILES는
            방향환 표기 순서 등으로 같은 분자가 다른 문자열이 될 수 있어
            leakage를 놓칠 수 있음).
        split_strategy : SplitStrategy
            현재는 GroupKFoldStrategy를 주입
        output_dir : str or Path
            fold별 결과(체크포인트/메트릭/config)와 cv_summary.csv를 저장할 경로
        compute_fold_stats_fn : callable, optional
            기존 STEP3의 compute_fold_stats 함수를 그대로 주입.
            train fold 데이터로 label_mean/label_std를 계산하는 함수라면 무엇이든 가능.
            (데이터 leakage 방지를 위해 train fold로만 계산해야 함 - STEP3와 동일 원칙)
        """
        self.df = df.reset_index(drop=True)
        self.properties = properties
        self.group_col = group_col
        self.split_strategy = split_strategy
        self.output_dir = Path(output_dir)
        self.compute_fold_stats_fn = compute_fold_stats_fn

        self.fold_metadata: List[dict] = []

    def run(self, train_fn: TrainFn, predict_fn: PredictFn) -> pd.DataFrame:
        """
        전체 K-Fold 검증을 실행하고 cv_summary DataFrame을 반환.

        train_fn / predict_fn은 기존 train.py / validate.py를 감싸는
        얇은 wrapper이므로, trainer/dataset 코드 자체는 수정하지 않는다.
        """
        fold_results: List[Dict[str, float]] = []

        for fold_id, train_idx, val_idx in self.split_strategy.split(self.df, self.group_col):
            fold_dir = self.output_dir / f"fold_{fold_id}"
            fold_dir.mkdir(parents=True, exist_ok=True)

            # 1) Leakage Prevention: train/val group 중복 자동 검사
            check_no_leakage(self.df, train_idx, val_idx, self.group_col, fold_id)

            # 2) fold-stat 계산 (train fold로만, STEP3 로직 재사용)
            fold_stats = (
                self.compute_fold_stats_fn(self.df, train_idx)
                if self.compute_fold_stats_fn is not None
                else {}
            )

            # 3) Fold Metadata 기록
            metadata = self._build_fold_metadata(fold_id, train_idx, val_idx)
            self.fold_metadata.append(metadata)
            self._save_fold_metadata(metadata, fold_dir)

            # 4) Fold Training (기존 train.py 호출, trainer 무변경)
            checkpoint_path = train_fn(fold_id, train_idx, val_idx, fold_stats, fold_dir)

            # 5) Fold Evaluation (기존 validate.py 호출, 무변경)
            y_true, y_pred = predict_fn(checkpoint_path, val_idx, fold_stats)
            property_metrics = compute_property_metrics(y_true, y_pred, self.properties)
            fold_summary = average_across_properties(property_metrics)
            fold_summary["fold"] = fold_id

            # per-property metric도 fold_dir에 저장 (checkpoint/metrics/config를
            # fold별로 분리 저장 -> 이전 fold를 덮어쓰지 않음)
            pd.DataFrame(property_metrics).T.to_csv(fold_dir / "property_metrics.csv")

            fold_results.append(fold_summary)

        # 6) Result Aggregation: cv_summary.csv (mean/std 포함)
        cv_summary = save_cv_summary(fold_results, self.output_dir / "cv_summary.csv")
        return cv_summary

    def _build_fold_metadata(
        self, fold_id: int, train_idx: np.ndarray, val_idx: np.ndarray
    ) -> dict:
        train_df = self.df.iloc[train_idx]
        val_df = self.df.iloc[val_idx]

        missing_stats = {
            f"missing_{prop}_ratio": float(val_df[prop].isna().mean())
            for prop in self.properties
            if prop in val_df.columns
        }

        return {
            "fold_id": fold_id,
            "train_samples": len(train_idx),
            "val_samples": len(val_idx),
            "unique_polymers_train": train_df[self.group_col].nunique(),
            "unique_polymers_val": val_df[self.group_col].nunique(),
            **missing_stats,
        }

    @staticmethod
    def _save_fold_metadata(metadata: dict, fold_dir: Path) -> None:
        pd.Series(metadata).to_csv(fold_dir / "fold_metadata.csv", header=False)
