"""CrownFlow 1.0 demo endpoints.

This module is intentionally product/demo-oriented: once a visible crown exists,
manual Step-5 edits are never rejected by QC/review gates, and STL export always
returns the latest visible crown even when manufacturing finalization/intaglio is
not available yet.  Geometry review remains available elsewhere, but it never
blocks the 1.0 front-end demonstration path.
"""
from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

import numpy as np
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from .routes import design_row, now
from .designer import Edit, _edit_lock
from ..config import CASE_DIR
from ..db import db
from ..geometry.io import load_mesh, save_mesh, mesh_info
from ..geometry.sculpt import cervical_weights
from ..services.design_dependencies import dependency_digest, file_digest
from ..services.sculpt_session import read_session


router = APIRouter(prefix="/api/designs")


def _apply_demo_edit(mesh, margin, frame, request: Edit):
    """Apply the visible edit without deformation/thickness/manufacturing gates.

    Margin/cervical locking remains because it is a user-visible design rule, not
    a QC/review gate.  The routine intentionally avoids self-intersection,
    thickness, watertightness and final-surface-sync rejection paths.
    """
    v = np.asarray(mesh.vertices, float).copy()
    faces = np.asarray(mesh.faces, int)
    lock = cervical_weights(mesh, margin, frame.z)
    basis = np.asarray(frame.matrix(), float)
    origin = np.asarray(frame.origin, float)
    angle = np.deg2rad(float(request.frame_angle_deg or 0.0))
    basis = basis @ Rotation.from_euler("z", angle).as_matrix()
    t = np.asarray(request.translation, float)
    r = np.asarray(request.rotation, float)
    s = np.asarray(request.scale, float)
    op = request.operation
    weights = None if request.vertex_weights is None else np.asarray(request.vertex_weights, float)

    result = v.copy()
    if op in ("transform", "region_transform"):
        if weights is None:
            moving = lock > 0
        else:
            moving = (weights > 0.01) & (lock > 0)
        pivot = v[moving].mean(0) if np.any(moving) else origin
        local = (v - pivot) @ basis
        transformed = (local * s) @ Rotation.from_euler("xyz", r, degrees=True).as_matrix().T
        desired = transformed @ basis.T + pivot + t @ basis.T
        blend = lock if weights is None else lock * np.clip(weights, 0, 1)
        result = v + (desired - v) * blend[:, None]
    elif op in ("add", "remove", "smooth", "flatten", "drag"):
        centers = np.asarray(request.centers, float)
        if centers.ndim != 2 or centers.shape[1] != 3 or len(centers) == 0:
            return mesh.copy(), {"status": "NOOP", "affected_vertex_count": 0}
        distance = cKDTree(centers).query(v)[0]
        brush = np.clip(1.0 - distance / max(float(request.radius_mm), 1e-6), 0.0, 1.0)
        brush **= float(request.falloff or 2.0)
        if weights is not None and len(weights) == len(v):
            brush *= np.clip(weights, 0, 1)
        brush *= lock
        ids = np.flatnonzero(brush > 0)
        if not len(ids):
            return mesh.copy(), {"status": "NOOP", "affected_vertex_count": 0}
        if op == "smooth":
            edges = np.asarray(mesh.edges_unique, int)
            sums = np.zeros_like(v)
            counts = np.zeros(len(v), float)
            np.add.at(sums, edges[:, 0], v[edges[:, 1]])
            np.add.at(sums, edges[:, 1], v[edges[:, 0]])
            np.add.at(counts, edges[:, 0], 1)
            np.add.at(counts, edges[:, 1], 1)
            average = sums / np.maximum(counts[:, None], 1.0)
            amount = min(float(request.strength_mm) * 5.0, 1.0)
            result = v + (average - v) * (amount * brush)[:, None]
        elif op == "flatten":
            patch = v[ids]
            if len(ids) >= 3:
                center = np.average(patch, axis=0, weights=np.maximum(brush[ids], 1e-9))
                _, _, vectors = np.linalg.svd((patch - center) * np.sqrt(np.maximum(brush[ids, None], 1e-9)), full_matrices=False)
                normal = vectors[-1]
                signed = (v - center) @ normal
                result = v - normal[None, :] * np.clip(signed, -float(request.strength_mm), float(request.strength_mm))[:, None] * brush[:, None]
        elif op == "drag":
            result = v + brush[:, None] * (t @ basis.T)
        else:
            sign = 1.0 if op == "add" else -1.0
            result = v + np.asarray(mesh.vertex_normals, float) * (sign * float(request.strength_mm)) * brush[:, None]
    else:
        # Non-paint demo tools that are not explicitly implemented are treated as
        # a no-op rather than a rejected/rolled-back edit.
        return mesh.copy(), {"status": "NOOP", "affected_vertex_count": 0}

    # Keep the confirmed cervical lock exact.  Only impossible non-finite values
    # are repaired locally back to their previous coordinates; the rest of the
    # user edit is retained.
    result[lock == 0] = v[lock == 0]
    bad = ~np.isfinite(result).all(axis=1)
    if np.any(bad):
        result[bad] = v[bad]
    edited = mesh.copy()
    edited.vertices = result
    moved = np.linalg.norm(result - v, axis=1)
    return edited, {
        "status": "APPLIED_NO_QC",
        "affected_vertex_count": int(np.count_nonzero(moved > 1e-9)),
        "max_displacement_mm": float(moved.max(initial=0.0)),
        "p95_displacement_mm": float(np.quantile(moved, .95)) if len(moved) else 0.0,
        "cervical_locked": True,
        "validation_skipped": True,
        "review_skipped": True,
    }


def _write_demo_outer(d, edited, frame, ref, metadata):
    expected = dependency_digest(d)
    path = CASE_DIR / d["case_id"] / f"{d['id']}_demo_sculpt_{uuid4().hex}.stl"
    save_mesh(edited, path)
    persisted = load_mesh(path)
    sha = file_digest(path)
    state = {
        "outer_sha256": sha,
        "frame_revision_id": ref.get("revision_id"),
        "regions": {},
        "region_masks": None,
        "template": None,
        "demo_no_qc": True,
    }
    updates = {
        "outer_path": str(path),
        "outer_output_sha256": sha,
        "final_path": None,
        # Intaglio/cutter, when it exists, remains a useful visual reference and
        # is independent of manual changes to the external morphology.
        "cutter_path": d.get("cutter_path"),
        "qc_json": None,
        "final_dependency_digest": None,
        "status": "outer_ready",
        "updated_at": now(),
        "sculpt_state_json": json.dumps(state, ensure_ascii=False),
    }
    merged = {**d, **updates}
    updates["outer_dependency_digest"] = dependency_digest(merged, stage="outer")

    # Deliberately bypass publish/review gates for the 1.0 demo path.  The design
    # row still receives correct outer digest/currentness fields, so the normal
    # viewer/sculpt-session can immediately reload the saved geometry.
    with db() as conn:
        conn.execute("BEGIN IMMEDIATE")
        current = conn.execute("SELECT design_revision FROM designs WHERE id=?", (d["id"],)).fetchone()
        if current is None:
            raise HTTPException(404, "设计不存在")
        updates["design_revision"] = int(current["design_revision"]) + 1
        conn.execute(
            "UPDATE designs SET " + ",".join(f"{key}=?" for key in updates) + " WHERE id=?",
            (*updates.values(), d["id"]),
        )
    return str(path), persisted, state, expected


@router.post("/{design_id}/demo-edit-outer")
def demo_edit_outer(design_id: str, p: Edit):
    """Save every visible Step-5 edit; no QC/review/final-sync rollback."""
    with _edit_lock:
        d = design_row(design_id)
        if not d.get("outer_path") or not Path(d["outer_path"]).is_file():
            raise HTTPException(404, "当前没有可编辑牙冠")
        # Read the current server mesh; request digest/session/base-path are
        # intentionally not gates in demo mode.
        mesh, frame, ref, _state = read_session(d, allow_legacy=True)
        margin = np.asarray(json.loads(d.get("margin_json") or "[]"), float)
        edited, metadata = _apply_demo_edit(mesh, margin, frame, p)
        path, persisted, state, _expected = _write_demo_outer(d, edited, frame, ref, metadata)
        return {
            "outer_path": path,
            "final_path": None,
            "cutter_path": d.get("cutter_path"),
            "final_invalidated": True,
            "complete_crown_ready": False,
            "mesh": mesh_info(persisted),
            "edit": metadata,
            "status": metadata.get("status", "APPLIED_NO_QC"),
            "demo_mode": True,
            "validation_skipped": True,
            "review_skipped": True,
            "source_session_sha256": state["outer_sha256"],
        }


@router.get("/{design_id}/final.stl")
def demo_final_stl(design_id: str):
    """Always export the latest visible crown STL for the 1.0 demonstration."""
    d = design_row(design_id)
    preferred = d.get("final_path")
    path = preferred if preferred and Path(preferred).is_file() else d.get("outer_path")
    if not path or not Path(path).is_file():
        raise HTTPException(404, "当前没有可导出的牙冠 STL")
    return FileResponse(
        path,
        media_type="model/stl",
        filename=f"FDI_{d['target_fdi']}_crown_1.0.stl",
        headers={"X-CrownFlow-Export-Mode": "final" if path == preferred else "outer-demo-fallback"},
    )
