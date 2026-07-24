"""SMILES 문자열을 RDKit Mol 객체로 파싱하고 유효성을 검증하는 모듈."""

from __future__ import annotations

import logging
from typing import Optional

from rdkit import Chem, RDLogger

# RDKit 자체 경고/에러 로그(예: 파싱 실패 시 stderr 출력)를 끄고
# 아래 logger를 통해 일관된 방식으로만 기록한다.
RDLogger.DisableLog("rdApp.*")

logger = logging.getLogger(__name__)


def parse_smiles(smiles: str) -> Optional[Chem.Mol]:
    """SMILES 문자열을 RDKit Mol 객체로 변환한다.

    폴리머 데이터셋의 SMILES는 결합 지점을 나타내는 ``*`` (dummy atom)를
    포함할 수 있으며, RDKit은 이를 정상적인 원자로 처리한다.

    Args:
        smiles: 파싱할 SMILES 문자열.

    Returns:
        파싱에 성공하면 RDKit Mol 객체, 실패하면 None.
    """
    if not isinstance(smiles, str) or not smiles.strip():
        logger.warning("빈 SMILES 문자열이 입력되었습니다.")
        return None

    try:
        mol = Chem.MolFromSmiles(smiles)
    except Exception as exc:  # RDKit 내부 예외에 대한 방어적 처리
        logger.warning("SMILES 파싱 중 예외 발생: %s | %s", smiles, exc)
        return None

    if mol is None:
        logger.warning("SMILES 파싱 실패(유효하지 않은 구조): %s", smiles)
        return None

    return mol


def is_valid_smiles(smiles: str) -> bool:
    """SMILES 문자열의 유효성만 빠르게 확인한다.

    Args:
        smiles: 검증할 SMILES 문자열.

    Returns:
        유효하면 True, 아니면 False.
    """
    return parse_smiles(smiles) is not None
