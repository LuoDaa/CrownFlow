"""Fast-path inputs for CAD-1 crowns on an existing abutment.

The normal natural-prep workflow remains deliberately review-heavy.  Existing
abutment crowns are different: the finish line and insertion direction are
already strongly constrained by the abutment geometry, and the incoming upper /
lower scans are expected to be exported in the bite relationship.

This router supplies only the two pieces that were still forcing unnecessary UI
steps in that case:

* an engineering patient frame derived from the working arch around the confirmed
  abutment; and
* small mesial/distal contact-surface patches detected from the working-jaw scan.

These are deterministic geometry heuristics, not deep learning.  Every result is
versioned and remains visible in the normal AutoFit/QC reports.
"""
from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import Field

from .auto_fit import StrictInput, checked, frame_from, save_confirmation
from .design_inputs import scan_reference
from .routes import asset, now
from ..config import CASE_DIR
from ..geometry.crown_outer.patient_frame import PatientFrame, validate_patient_frame
from ..geometry.io import load_mesh, save_mesh
from ..revisions import DesignRevision, RevisionKind
from ..services.design_dependencies import file_digest, publish_artifacts


router = APIRouter(prefix="/api/designs")


class AutoNeighborInput(StrictInput):
    patch_radius_mm: float = Field(default=3.0, ge=1.5, le=5.0)


def _unit(vector, message):
    value = np.asarray(vector, dtype=float)
    norm = float(np.linalg.norm(value))
    if value.shape != (3,) or not np.isfinite(value).all() or norm < 1e-9:
        raise ValueError(message)
    return value / norm


def _automatic_abutment_frame(work_mesh, origin, insertion_axis):
    """Return a right-handed local frame with Y pointing to the arch exterior.

    For a single tooth position the vector from the robust working-jaw centroid
    to the abutment is a useful buccolingual radial cue.  It is considerably more
    stable than asking the operator to flip a PCA eigenvector from the abutment
    margin.  X is then the tangent implied by the confirmed insertion direction.
    """
    origin = np.asarray(origin, dtype=float)
    z = _unit(insertion_axis, "插入方向无效")
    vertices = np.asarray(work_mesh.vertices, dtype=float)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) < 20:
        raise ValueError("工作颌网格不足，无法自动建立基台坐标")

    # Median is intentionally used instead of mean: scanners often leave long
    # triangles / small detached fragments that would otherwise pull the center.
    arch_center = np.median(vertices, axis=0)
    radial = origin - arch_center
    radial -= z * float(radial @ z)

    if np.linalg.norm(radial) < 0.5:
        # Anterior / cropped scans can put the robust center close to the target.
        # Fall back to the farthest projected bbox direction, still without a
        # manual orientation step.
        projected = vertices - origin
        projected -= (projected @ z)[:, None] * z[None, :]
        lengths = np.linalg.norm(projected, axis=1)
        radial = -projected[int(np.argmax(lengths))]

    y = _unit(radial, "牙列方向退化，无法自动推导颊舌方向")
    x = _unit(np.cross(y, z), "牙列方向与插入轴平行，无法自动推导近远中方向")
    y = _unit(np.cross(z, x), "自动坐标正交化失败")
    if float(y @ radial) < 0:
        x *= -1.0
        y *= -1.0

    frame = PatientFrame(origin, z, x, y, revision_id="auto-abutment-frame-preview")
    validate_patient_frame(frame)
    return x, y, z, {
        "arch_center": arch_center.tolist(),
        "radial_length_mm": float(np.linalg.norm(radial)),
        "method": "WORKING_ARCH_RADIAL_PLUS_INSERTION_AXIS",
    }


@router.post("/{design_id}/auto-fit/abutment-frame")
def confirm_automatic_abutment_frame(design_id: str, p: StrictInput):
    d = checked(design_id, p.expected_input_digest, natural_only=False)
    if d["support_type"] != "abutment":
        raise HTTPException(409, "自动基台坐标只用于已有实体基台冠")

    margin = np.asarray(json.loads(d.get("margin_json") or "null"), dtype=float)
    axis = np.asarray(json.loads(d.get("axis_json") or "null"), dtype=float)
    if margin.ndim != 2 or margin.shape[0] < 3 or margin.shape[1] != 3:
        raise HTTPException(409, "请先自动确认基台 Margin")
    if not d.get("confirmed_margin_revision_id") or not d.get("current_axis_revision_id"):
        raise HTTPException(409, "请先自动确认基台 Margin 与插入方向")

    source_kind = "lower" if d["target_fdi"] >= 31 else "upper"
    source = asset(d["case_id"], source_kind)
    support = asset(d["case_id"], "support")
    if not source or not support:
        raise HTTPException(409, "缺少工作颌或基台支撑网格")

    work_mesh = load_mesh(Path(source["path"]))
    try:
        x, y, z, diagnostics = _automatic_abutment_frame(work_mesh, margin.mean(axis=0), axis)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc

    value = {
        "x": x.tolist(),
        "y": y.tolist(),
        "z": z.tolist(),
        "buccal": y.tolist(),
        "units": "mm",
        "axis_revision_id": d["current_axis_revision_id"],
        "prep_sha256": file_digest(support["path"]),
        "margin_revision_id": d["confirmed_margin_revision_id"],
        "method": "AUTO_ABUTMENT_ARCH_FRAME_CAD1",
        "authentication_status": "ENGINEERING_AUTO",
        "diagnostics": diagnostics,
    }
    return save_confirmation(
        d, "autofit_frame_json", value, RevisionKind.GEOMETRY_FRAME, p.expected_input_digest,
    )


def _connected_patch(mesh, allowed, seed, radius_mask):
    allowed = np.asarray(allowed, dtype=bool) & np.asarray(radius_mask, dtype=bool)
    if seed < 0 or seed >= len(allowed) or not allowed[seed]:
        return np.zeros(0, dtype=int)
    adjacency = [[] for _ in range(len(mesh.faces))]
    for a, b in np.asarray(mesh.face_adjacency, dtype=int):
        adjacency[int(a)].append(int(b))
        adjacency[int(b)].append(int(a))
    queue = [int(seed)]
    seen = {int(seed)}
    while queue:
        face = queue.pop()
        for other in adjacency[face]:
            if other not in seen and allowed[other]:
                seen.add(other)
                queue.append(other)
    return np.asarray(sorted(seen), dtype=int)


def _automatic_neighbor_patch(mesh, frame, margin, side, radius_mm):
    centers = np.asarray(mesh.triangles_center, dtype=float)
    normals = np.asarray(mesh.face_normals, dtype=float)
    areas = np.asarray(mesh.area_faces, dtype=float)
    matrix = np.asarray(frame.matrix(), dtype=float)
    local = (centers - np.asarray(frame.origin, dtype=float)) @ matrix
    margin_local = (np.asarray(margin, dtype=float) - np.asarray(frame.origin, dtype=float)) @ matrix

    half_x = max(float(np.ptp(margin_local[:, 0])) / 2.0, 0.8)
    half_y = max(float(np.ptp(margin_local[:, 1])) / 2.0, 0.8)
    sign = 1.0 if side == "mesial" else -1.0
    outward = -sign * np.asarray(frame.x, dtype=float)  # neighbor surface faces the crown gap
    side_x = sign * local[:, 0]
    facing = normals @ outward

    # Contact surfaces should sit just outside the abutment footprint, above the
    # finish line and face into the edentulous/abutment gap.  The intentionally
    # generous Y/Z envelope tolerates different cusp heights while rejecting most
    # gingiva and remote teeth before connected-patch growth.
    allowed = (
        (side_x > half_x + 0.25)
        & (side_x < half_x + 8.0)
        & (np.abs(local[:, 1]) < half_y + 4.5)
        & (local[:, 2] > 0.35)
        & (local[:, 2] < 9.0)
        & (facing > -0.15)
        & (areas > 1e-8)
    )
    candidates = np.flatnonzero(allowed)
    if not len(candidates):
        return None, {"status": "UNKNOWN", "reason": "NO_FACING_SURFACE_IN_SEARCH_ENVELOPE"}

    # Prefer the first inward-facing surface encountered from the crown gap.  A
    # small vertical/lateral penalty avoids picking an isolated cusp tip.
    score = (
        1.25 * (side_x[candidates] - (half_x + 0.25))
        + 0.10 * np.abs(local[candidates, 1])
        + 0.06 * np.abs(local[candidates, 2] - 3.0)
        - 0.45 * np.clip(facing[candidates], 0.0, 1.0)
    )
    seed = int(candidates[int(np.argmin(score))])
    seed_center = centers[seed]

    ids = np.zeros(0, dtype=int)
    for scale in (1.0, 1.35):
        distance = np.linalg.norm(centers - seed_center, axis=1)
        ids = _connected_patch(
            mesh,
            allowed & (facing > -0.35),
            seed,
            distance <= float(radius_mm) * scale,
        )
        if len(ids) >= 12:
            break
    if len(ids) < 3:
        return None, {"status": "UNKNOWN", "reason": "CONTACT_PATCH_TOO_SMALL", "seed_face": seed}

    region = mesh.submesh([ids], append=True, repair=False)
    region.remove_unreferenced_vertices()
    if len(region.faces) < 3 or float(region.area) < 0.20:
        return None, {"status": "UNKNOWN", "reason": "CONTACT_PATCH_AREA_TOO_SMALL", "seed_face": seed}

    # Open patches need an explicit outward direction for signed clearance.
    from ..geometry.crown_outer.auto_fit_adapter import FitSurface
    try:
        FitSurface(region, outward)
    except ValueError as exc:
        return None, {"status": "UNKNOWN", "reason": str(exc), "seed_face": seed}

    return (region, ids, outward), {
        "status": "CONFIRMED",
        "seed_face": seed,
        "face_count": int(len(ids)),
        "area_mm2": float(region.area),
        "closest_side_x_mm": float(np.min(side_x[ids])),
        "half_margin_width_mm": half_x,
        "patch_radius_mm": float(radius_mm),
        "method": "LOCAL_CONNECTED_FACING_SURFACE_PATCH",
    }


@router.post("/{design_id}/auto-fit/auto-neighbors")
def confirm_automatic_neighbors(design_id: str, p: AutoNeighborInput):
    d = checked(design_id, p.expected_input_digest, natural_only=False)
    if d["support_type"] != "abutment":
        raise HTTPException(409, "自动邻牙接触面只用于基台快速流程；天然牙继续人工复核")
    frame, fref = frame_from(d)
    source_kind = "lower" if d["target_fdi"] >= 31 else "upper"
    source = asset(d["case_id"], source_kind)
    if not source:
        raise HTTPException(409, "缺少工作颌扫描")

    mesh = load_mesh(Path(source["path"]))
    margin = np.asarray(json.loads(d["margin_json"]), dtype=float)
    source_ref = scan_reference(d["case_id"], source_kind)
    choices = json.loads(d.get("neighbors_json") or "{}")
    diagnostics = {}
    revision = DesignRevision(
        design_id=design_id,
        revision_no=0,
        kind=RevisionKind.NEIGHBOR_CONFIRMED,
        payload={},
        algorithm_version="welooai/abutment-auto-neighbor/1.0",
    )

    for side in ("mesial", "distal"):
        existing = choices.get(side, {})
        if existing.get("confirmation_status") in ("CONFIRMED", "ABSENT") and str(existing.get("scope", "")).startswith("HUMAN_"):
            diagnostics[side] = {"status": "PRESERVED_MANUAL_INPUT"}
            continue

        extracted, report = _automatic_neighbor_patch(mesh, frame, margin, side, p.patch_radius_mm)
        diagnostics[side] = report
        if extracted is None:
            # Never silently call a missed detection "ABSENT".  UNKNOWN keeps the
            # automatic-mode eligibility honest and forces a fallback only in the
            # rare case where the deterministic patch search cannot prove a side.
            choices[side] = {
                "status": "UNSET",
                "confirmation_status": "UNKNOWN",
                "instance_ref": uuid4().hex,
                "instance_name": f"{side} 自动邻牙未确认",
                "frame_revision_id": fref["revision_id"],
                "scope": "AUTO_CAD1_LOCAL_CONTACT_PATCH",
                "diagnostics": report,
                "revision_id": revision.id,
            }
            continue

        region, ids, outward = extracted
        path = CASE_DIR / d["case_id"] / f"neighbor_{side}_auto_{uuid4().hex}.stl"
        save_mesh(region, path)
        choices[side] = {
            "status": "SELECTED",
            "confirmation_status": "CONFIRMED",
            "instance_ref": uuid4().hex,
            "instance_name": f"{side} 邻牙自动接触面",
            "frame_revision_id": fref["revision_id"],
            "source": source_ref,
            "path": str(path),
            "region_sha256": file_digest(path),
            "face_ids": ids.tolist(),
            "outward_normal": np.asarray(outward, dtype=float).tolist(),
            "pollution_reviewed": False,
            "pollution_audit": {
                "classification": "ENGINEERING_GEOMETRIC_HEURISTIC",
                "human_reviewed": False,
            },
            "scope": "AUTO_CAD1_LOCAL_CONTACT_PATCH",
            "diagnostics": report,
            "revision_id": revision.id,
        }

    revision.payload = {
        "mode": "AUTO_ABUTMENT_LOCAL_CONTACT_PATCH",
        "frame_revision_id": fref["revision_id"],
        "neighbors": choices,
        "diagnostics": diagnostics,
    }
    publish_artifacts(
        d,
        p.expected_input_digest,
        {
            "neighbors_json": json.dumps(choices, ensure_ascii=False),
            "final_path": None,
            "cutter_path": None,
            "qc_json": None,
            "final_dependency_digest": None,
            "updated_at": now(),
        },
        revision=revision,
    )
    return {
        "neighbors": choices,
        "diagnostics": diagnostics,
        "frame_revision_id": fref["revision_id"],
        "release_status": "ENGINEERING_AUTO_REVIEWABLE",
    }
