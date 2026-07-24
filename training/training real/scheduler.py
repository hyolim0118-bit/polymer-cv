"""Optimizer Scheduler Factory.

새 scheduler를 추가하려면 ``_SCHEDULER_REGISTRY``에 생성 함수만
등록하면 되고, ``build_scheduler`` 로직은 바꿀 필요가 없다.
"""

from __future__ import annotations

import logging
from typing import Callable, Dict

from torch.optim import Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR, LRScheduler

logger = logging.getLogger(__name__)


def _build_cosine(optimizer: Optimizer, config: dict) -> LRScheduler:
    """CosineAnnealingLR scheduler를 생성한다.

    Args:
        optimizer: 대상 optimizer.
        config: 전체 config 딕셔너리 (config["scheduler"]를 사용).

    Returns:
        CosineAnnealingLR 인스턴스.
    """
    sched_cfg = config["scheduler"]
    return CosineAnnealingLR(
        optimizer,
        T_max=sched_cfg["t_max"],
        eta_min=sched_cfg["eta_min"],
    )


_SCHEDULER_REGISTRY: Dict[str, Callable[[Optimizer, dict], LRScheduler]] = {
    "cosine": _build_cosine,
}


def build_scheduler(optimizer: Optimizer, config: dict) -> LRScheduler:
    """config["scheduler"]["name"]에 따라 scheduler를 생성한다.

    Args:
        optimizer: 대상 optimizer.
        config: 전체 config 딕셔너리.

    Returns:
        torch LRScheduler 인스턴스.

    Raises:
        ValueError: 지원하지 않는 scheduler 이름인 경우.
    """
    name = config["scheduler"]["name"]
    if name not in _SCHEDULER_REGISTRY:
        supported = list(_SCHEDULER_REGISTRY.keys())
        raise ValueError(f"지원하지 않는 scheduler입니다: '{name}'. 지원 목록: {supported}")

    logger.info("Scheduler 생성: %s", name)
    return _SCHEDULER_REGISTRY[name](optimizer, config)
