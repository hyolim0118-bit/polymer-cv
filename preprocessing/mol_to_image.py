"""RDKit Mol 객체를 고정 규격의 2D 구조 PNG 이미지로 렌더링하는 모듈.

모든 이미지가 동일한 크기/스타일로 생성되어야 CNN 입력으로 일관되게
사용할 수 있으므로, 렌더링 옵션을 이 모듈 한 곳에서만 정의한다.
"""

from __future__ import annotations

import logging
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Draw import rdMolDraw2D

logger = logging.getLogger(__name__)

IMAGE_SIZE = 224


def render_mol_to_png(mol: Chem.Mol, output_path: Path, size: int = IMAGE_SIZE) -> bool:
    """RDKit Mol 객체를 2D 구조 PNG 이미지로 저장한다.

    Args:
        mol: 렌더링할 RDKit Mol 객체.
        output_path: 저장할 PNG 파일 경로.
        size: 정사각형 이미지의 한 변 픽셀 크기 (기본 224).

    Returns:
        저장에 성공하면 True, 실패하면 False.
    """
    try:
        mol_for_draw = Chem.Mol(mol)
        AllChem.Compute2DCoords(mol_for_draw)

        drawer = rdMolDraw2D.MolDraw2DCairo(size, size)
        options = drawer.drawOptions()
        options.bondLineWidth = 2
        options.padding = 0.12
        options.clearBackground = True
        options.addStereoAnnotation = False

        rdMolDraw2D.PrepareAndDrawMolecule(drawer, mol_for_draw)
        drawer.FinishDrawing()

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(drawer.GetDrawingText())
        return True
    except Exception as exc:  # 좌표 계산/드로잉 단계에서 발생 가능한 예외 방어
        logger.warning("이미지 렌더링 실패 (%s): %s", output_path.name, exc)
        return False
