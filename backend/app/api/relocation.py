"""Repeatable prep/abutment relocation endpoints for CAD-1 test workflows.

The original designer endpoints intentionally blocked relocation once a Margin,
axis, outer crown, or sibling design existed.  That policy is too restrictive
for engineering regression: the same scan pair is routinely reused while
iterating algorithms.  These routes deliberately allow re-running localization.

They are registered *before* ``designer_router`` so the public URLs stay
unchanged.  Downstream artifacts are still protected by the existing dependency
digests; once support/Margin inputs change, stale outer/final artifacts are no
longer considered current.
"""
from pathlib import Path
from uuid import uuid4

import numpy as np
from fastapi import APIRouter, HTTPException

from .designer import DraftPreparation, Locate, _draft_implant_preparation
from .routes import asset, design_row, now, VALID_FDI, save_extracted_support
from ..config import CASE_DIR
from ..db import db
from ..geometry.draft_margin import draft_margin
from ..geometry.io import load_mesh, save_mesh, mesh_info
from ..geometry.tooth_location import locate_target_tooth, extract_support
from ..models import RoiRequest
from ..revisions import current_margin, record_auto_margin_candidate

router = APIRouter(prefix="/api")


@router.post("/designs/{design_id}/draft-preparation")
def draft_preparation_repeatable(design_id: str, p: DraftPreparation):
    """Re-run preparation localization even if the case/design was used before.

    No dedup/history gate is applied here.  This is intentional for CAD-1
    engineering tests where identical STL inputs and repeated Margin runs are
    expected.
    """
    d = design_row(design_id)
    prior = current_margin(design_id)
    source = "lower" if d["target_fdi"] >= 31 else "upper"
    scan = asset(d["case_id"], source)
    if not scan:
        raise HTTPException(400, "请先上传目标颌")

    if d["support_type"] == "abutment":
        return _draft_implant_preparation(d, p, scan, source)

    if p.center is None or p.axis is None:
        raise HTTPException(422, "请在基牙顶部选择中心与方向")
    center = np.asarray(p.center, float)
    direction = np.asarray(p.axis, float)
    if (
        center.shape != (3,)
        or direction.shape != (3,)
        or not np.isfinite([center, direction]).all()
        or np.linalg.norm(direction) < 1e-8
    ):
        raise HTTPException(422, "请在预备体顶部选择有效定位点")

    jaw = load_mesh(Path(scan["path"]))
    nearby = np.linalg.norm(jaw.triangles_center - center, axis=1) < 1.5
    if nearby.any():
        averaged = np.average(jaw.face_normals[nearby], axis=0, weights=jaw.area_faces[nearby])
        if np.linalg.norm(averaged) > 0.2 and np.dot(averaged, direction) > 0:
            direction = averaged / np.linalg.norm(averaged)

    try:
        points, axis = draft_margin(jaw, p.center, direction, p.radius_mm)
    except ValueError as exc:
        if prior is None:
            save_extracted_support(
                d["case_id"], RoiRequest(source=source, center=p.center, radius_mm=p.radius_mm)
            )
        raise HTTPException(422, str(exc)) from exc

    # Repeated runs are allowed: saving a fresh support asset changes the input
    # dependency digest, so any old outer/final result becomes stale naturally.
    save_extracted_support(
        d["case_id"], RoiRequest(source=source, center=p.center, radius_mm=p.radius_mm)
    )
    record_auto_margin_candidate(
        design_id,
        points,
        algorithm_version="welooai/draft-section/1.0",
        note="ENGINEERING DRAFT: repeatable localization; section 3mm below designer seed",
    )
    return {
        "points": points,
        "axis": axis,
        "status": "DRAFT_REVIEW_REQUIRED",
        "method": "seeded_section_3mm",
        "repeatable": True,
        "note": "粗边缘线与牙弓方向，允许同病例重复定位用于工程测试",
    }


@router.post("/cases/{case_id}/traditional-location")
def traditional_location_repeatable(case_id: str, p: Locate):
    """Traditional localization without the old 'case already has geometry' gate."""
    if p.target_fdi not in VALID_FDI:
        raise HTTPException(400, "无效 FDI")
    source = "lower" if p.target_fdi >= 31 else "upper"
    scan = asset(case_id, source)
    if not scan:
        raise HTTPException(400, "请先上传工作颌")

    try:
        jaw = load_mesh(Path(scan["path"]))
        if p.center is not None:
            seed = np.asarray(p.center, float)
            if seed.shape != (3,) or not np.isfinite(seed).all():
                raise ValueError("无效中心坐标")
        located = locate_target_tooth(jaw, p.target_fdi, user_seed=p.center)
        if p.center is None:
            return located.as_dict()

        segmented = extract_support(jaw, located, target_fdi=p.target_fdi)
        if segmented.mesh is None:
            return segmented.as_dict()
        path = CASE_DIR / case_id / f"support_geodesic_{uuid4().hex}.stl"
        save_mesh(segmented.mesh, path)
        with db() as connection:
            connection.execute(
                "INSERT INTO assets(id,case_id,kind,path,original_name,mesh_info_json,created_at) VALUES(?,?,?,?,?,?,?)",
                (
                    uuid4().hex,
                    case_id,
                    "support",
                    str(path),
                    "extracted_geodesic.stl",
                    __import__("json").dumps(mesh_info(segmented.mesh)),
                    now(),
                ),
            )
        return {
            **segmented.as_dict(),
            "location": located.as_dict(),
            "review_required": True,
            "repeatable": True,
        }
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
