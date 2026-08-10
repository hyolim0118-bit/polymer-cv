"""
tracking.py
-----------
Search Stage에서 trial마다 남기는 기록(trials.csv)을 관리.
"""

from pathlib import Path
from typing import Any, Dict, Union

import pandas as pd


def append_trial_record(record: Dict[str, Any], trials_csv_path: Union[str, Path]) -> None:
    """
    trial 하나의 기록을 trials.csv에 추가. 파일이 없으면 새로 만든다.
    (round1/round2가 각자 다른 trials.csv 경로를 쓰므로 라운드 간에도 안 섞임)
    """
    trials_csv_path = Path(trials_csv_path)
    trials_csv_path.parent.mkdir(parents=True, exist_ok=True)

    row = pd.DataFrame([record])
    if trials_csv_path.exists():
        row.to_csv(trials_csv_path, mode="a", header=False, index=False)
    else:
        row.to_csv(trials_csv_path, mode="w", header=True, index=False)
