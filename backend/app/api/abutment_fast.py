"""Deterministic fast-path references for CAD-1 crowns on an existing abutment.

Existing-abutment orders are intentionally different from natural-prep orders:
upper/lower scans arrive in their bite relationship and the user should not have
to click Margin / Axis / Neighbor reference surfaces for a routine case.

This router therefore derives two hidden engineering references:

* a right-handed patient frame from the working arch + insertion axis; and
* mesial/distal neighbor contact patches from the working-jaw scan.

The neighbor detector is deliberately anatomy/spatial driven rather than normal
only.  IOS face winding is not reliable enough to make a hard facing-normal gate.
The local-frame strategy mirrors the useful part of the legacy abutment engine:
look in a coronal band, split by the mesiodistal coordinate and use the closest
side percentile as the neighbor location.  A connected local patch is then saved
for the existing signed-contact engine.
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
    patch_radius_mm: float = Field(default=3.5, ge=1.5, le=6.0)


def _unit(vector, message):
    value = np.asarray(vector, dtype=float)
    norm = float(np.linalg.norm(value))
    if value.shape != (3,) or not np.isfinite(value).all() or norm < 1e-9:
        raise ValueError(message)
    return value / norm


def _automatic_abutment_frame(work_mesh, origin, insertion_axis):
    """Return a right-handed local frame with Y pointing to arch exterior."""
    origin = np.asarray(origin, dtype=float)
    z = _unit(insertion_axis, "插入方向无效")
    vertices = np.asarray(work_mesh.vertices, dtype=float)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) < 20:
        raise ValueError("工作颌网格不足，无法自动建立基台坐标")

    # Median is robust to scan fragments and long trimming triangles.
    arch_center = np.median(vertices, axis=0)
    radial = origin - arch_center
    radial -= z * float(radial @ z)
    if np.linalg.norm(radial) < 0.5:
        projected = vertices - origin
        projected -= (projected @ z)[:, None] * z[None, :]
        radial = -projected[int(np.argmax(np.linalg.norm(projected, axis=1)))]

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
        x, y, z, diagnostics = _automatic_abutment_frame(
            work_mesh, margin.mean(axis=0), axis,
        )
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc

    value = {
        "x": x.tolist(), "y": y.tolist(), "z": z.tolist(), "buccal": y.tolist(),
        "units": "mm",
        "axis_revision_id": d["current_axis_revision_id"],
        "prep_sha256": file_digest(support["path"]),
        "margin_revision_id": d["confirmed_margin_revision_id"],
        "method": "AUTO_ABUTMENT_ARCH_FRAME_CAD1",
        "authentication_status": "ENGINEERING_AUTO",
        "diagnostics": diagnostics,
    }
    return save_confirmation(
        d, "autofit_frame_json", value, RevisionKind.GEOMETRY_FRAME,
        p.expected_input_digest,
    )


def _connected_patch(mesh, allowed, seed, radius_mask):
    """Connected face flood constrained by a semantic mask and Euclidean radius."""
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


def _neighbor_search_masks(local, areas, margin_local, side):
    """Return strict/relaxed coronal masks and local geometric diagnostics.

    The important rule is the same as the reference abutment engine: first isolate
    a coronal band and only then split it by +/- mesiodistal coordinate.  This
    rejects most gingiva without requiring a pre-segmented tooth mesh.
    """
    half_x = max(float(np.ptp(margin_local[:, 0])) / 2.0, 0.8)
    half_y = max(float(np.ptp(margin_local[:, 1])) / 2.0, 0.8)
    z_hi = float(np.percentile(margin_local[:, 2], 90))
    sign = 1.0 if side == "mesial" else -1.0
    side_x = sign * local[:, 0]

    base = (
        (side_x > half_x + 0.25)
        & (side_x < half_x + 10.0)
        & (np.abs(local[:, 1]) < half_y + 5.5)
        & (areas > 1e-8)
    )
    strict = base & (local[:, 2] > z_hi + 0.50) & (local[:, 2] < z_hi + 6.5)
    relaxed = base & (local[:, 2] > z_hi + 0.15) & (local[:, 2] < z_hi + 8.0)
    return strict, relaxed, side_x, half_x, half_y, z_hi


def _automatic_neighbor_patch(mesh, frame, margin, side, radius_mm):
    """Find a mesial/distal neighbor patch without any user click.

    Face normals are used only as a soft score.  The hard evidence is spatial:
    a coronal-band surface on the correct side of the margin footprint.  The
    closest-side percentile identifies the adjacent tooth; connected growth then
    provides a compact open patch for the signed contact engine.
    """
    centers = np.asarray(mesh.triangles_center, dtype=float)
    normals = np.asarray(mesh.face_normals, dtype=float)
    areas = np.asarray(mesh.area_faces, dtype=float)
    matrix = np.asarray(frame.matrix(), dtype=float)
    local = (centers - np.asarray(frame.origin, dtype=float)) @ matrix
    margin_local = (
        np.asarray(margin, dtype=float) - np.asarray(frame.origin, dtype=float)
    ) @ matrix

    strict, relaxed, side_x, half_x, half_y, z_hi = _neighbor_search_masks(
        local, areas, margin_local, side,
    )
    sign = 1.0 if side == "mesial" else -1.0
    outward = -sign * np.asarray(frame.x, dtype=float)
    facing = normals @ outward

    candidates = np.flatnonzero(strict)
    search_mode = "CORONAL_STRICT"
    if len(candidates) < 3:
        candidates = np.flatnonzero(relaxed)
        search_mode = "CORONAL_RELAXED"
    if not len(candidates):
        # No coronal anatomy within 10 mm on this side is strong evidence of a
        # genuinely absent adjacent tooth in the fast single-crown workflow.
        return None, {
            "status": "ABSENT",
            "reason": "NO_CORONAL_NEIGHBOR_WITHIN_SEARCH_RANGE",
            "half_margin_width_mm": half_x,
            "search_reach_mm": 10.0,
            "method": "CORONAL_MD_PERCENTILE_AUTO_V2",
        }

    # Legacy/reference logic used a low/high percentile of coronal +/-u points.
    # Use the same robust idea on face centers: the low side percentile is the
    # first surface encountered outside the target-tooth footprint.
    nearest_x = float(np.percentile(side_x[candidates], 5))
    anatomical_band = relaxed & (side_x <= nearest_x + 3.0)
    band_candidates = np.flatnonzero(anatomical_band)
    if len(band_candidates) >= 3:
        candidates = band_candidates

    target_z = z_hi + 3.0
    score = (
        1.50 * np.maximum(side_x[candidates] - nearest_x, 0.0)
        + 0.11 * np.abs(local[candidates, 1])
        + 0.07 * np.abs(local[candidates, 2] - target_z)
        - 0.20 * np.clip(facing[candidates], 0.0, 1.0)
    )
    seed = int(candidates[int(np.argmin(score))])
    seed_center = centers[seed]
    distance = np.linalg.norm(centers - seed_center, axis=1)

    ids = np.zeros(0, dtype=int)
    for scale in (1.0, 1.35, 1.75):
        ids = _connected_patch(
            mesh, anatomical_band, seed,
            distance <= float(radius_mm) * scale,
        )
        if len(ids) >= 10:
            break

    # If scan triangulation is sparse, connected growth can legitimately yield
    # fewer faces.  Keep a small valid connected patch rather than forcing manual
    # selection; the contact engine later rejects ambiguous patch-edge queries.
    if len(ids) < 3:
        return None, {
            "status": "UNKNOWN",
            "reason": "CONNECTED_CORONAL_PATCH_TOO_SMALL",
            "seed_face": seed,
            "candidate_faces": int(len(candidates)),
            "method": "CORONAL_MD_PERCENTILE_AUTO_V2",
        }

    region = mesh.submesh([ids], append=True, repair=False)
    region.remove_unreferenced_vertices()
    if len(region.faces) < 3 or float(region.area) < 0.12:
        return None, {
            "status": "UNKNOWN",
            "reason": "CONTACT_PATCH_AREA_TOO_SMALL",
            "seed_face": seed,
            "area_mm2": float(region.area),
            "method": "CORONAL_MD_PERCENTILE_AUTO_V2",
        }

    # Open patches have no intrinsic inside/outside.  Supply an explicit direction
    # toward the crown gap; FitSurface/contact measurement owns the signed query.
    from ..geometry.crown_outer.auto_fit_adapter import FitSurface
    try:
        FitSurface(region, outward)
    except ValueError as exc:
        return None, {
            "status": "UNKNOWN", "reason": str(exc), "seed_face": seed,
            "method": "CORONAL_MD_PERCENTILE_AUTO_V2",
        }

    return (region, ids, outward), {
        "status": "CONFIRMED",
        "seed_face": seed,
        "face_count": int(len(ids)),
        "area_mm2": float(region.area),
        "nearest_side_percentile_mm": nearest_x,
        "closest_side_x_mm": float(np.min(side_x[ids])),
        "half_margin_width_mm": half_x,
        "half_margin_buccolingual_mm": half_y,
        "coronal_z_reference_mm": z_hi,
        "patch_radius_mm": float(radius_mm),
        "search_mode": search_mode,
        "normal_facing_median": float(np.median(facing[ids])),
        "method": "CORONAL_MD_PERCENTILE_AUTO_V2",
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
        algorithm_version="welooai/abutment-auto-neighbor/2.0",
    )

    for side in ("mesial", "distal"):
        existing = choices.get(side, {})
        if (
            existing.get("confirmation_status") in ("CONFIRMED", "ABSENT")
            and str(existing.get("scope", "")).startswith("HUMAN_")
        ):
            diagnostics[side] = {"status": "PRESERVED_MANUAL_INPUT"}
            continue

        extracted, report = _automatic_neighbor_patch(
            mesh, frame, margin, side, p.patch_radius_mm,
        )
        diagnostics[side] = report

        if extracted is None and report.get("status") == "ABSENT":
            choices[side] = {
                "status": "ABSENT",
                "confirmation_status": "ABSENT",
                "instance_ref": uuid4().hex,
                "instance_name": f"{side} 自动判定缺牙",
                "frame_revision_id": fref["revision_id"],
                "scope": "AUTO_CAD1_CORONAL_MD_NEIGHBOR_V2",
                "diagnostics": report,
                "revision_id": revision.id,
            }
            continue

        if extracted is None:
            choices[side] = {
                "status": "UNSET",
                "confirmation_status": "UNKNOWN",
                "instance_ref": uuid4().hex,
                "instance_name": f"{side} 自动邻牙未确认",
                "frame_revision_id": fref["revision_id"],
                "scope": "AUTO_CAD1_CORONAL_MD_NEIGHBOR_V2",
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
                "classification": "ENGINEERING_GEOMETRIC_HEURISTIC_V2",
                "human_reviewed": False,
            },
            "scope": "AUTO_CAD1_CORONAL_MD_NEIGHBOR_V2",
            "diagnostics": report,
            "revision_id": revision.id,
        }

    revision.payload = {
        "mode": "AUTO_ABUTMENT_CORONAL_MD_NEIGHBOR_V2",
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
