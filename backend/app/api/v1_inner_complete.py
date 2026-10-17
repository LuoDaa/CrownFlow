"""CrownFlow 1.0 patient-owned inner crown lifecycle.

For the 1.0 workflow the geometry contract is explicit:

* template/anatomy -> OUTER only;
* prepared abutment + insertion axis + cement gap -> INTAGLIO only;
* confirmed 3D Margin -> shared cervical seam between the two.

A generated inner therefore survives AUTO acceptance and every outer sculpt edit.
If an older/partial AUTO candidate did not publish its preview artifact, this layer
rebuilds the intaglio directly from the current prepared abutment instead of
showing a cutter bottom or silently dropping the inner crown.  The 1.0 STL export
also prefers a stitched hollow shell (outer + patient intaglio + shared Margin)
over an outer-only fallback.
"""
from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

import numpy as np
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from .routes import design_row
from .auto_fit import candidate
from .complete_generation import (
    CompleteAcceptInput,
    accept_complete_generated_crown as _accept_original,
    rebuild_cervical_idempotent as _rebuild_original,
)
from .demo_v1 import demo_edit_outer as _demo_edit_original
from .designer import Edit
from ..config import CASE_DIR
from ..db import db
from ..geometry.io import load_mesh, save_mesh
from ..geometry.margin_seam import stitch_cervical_surfaces
from ..geometry.nonplanar_crown import MarginCurtain, _single_open_surface_above_margin
from ..models import FinalizeRequest
from ..services.design_dependencies import file_digest


router = APIRouter(prefix="/api/designs")


def _existing_verified(path, sha256=None):
    if not path or not Path(path).is_file():
        return False
    return sha256 is None or file_digest(path) == sha256


def _candidate_inner(revision):
    expected = revision.payload.get("artifact_sha256") or {}
    for label in ("intaglio_surface", "intaglio", "intaglio_preview"):
        path = revision.artifacts.get(label)
        sha = expected.get(label)
        if _existing_verified(path, sha):
            return label, path, sha or file_digest(path)
    return None, None, None


def _read_qc(d):
    try:
        return json.loads(d.get("qc_json") or "{}")
    except (TypeError, json.JSONDecodeError):
        return {}


def _read_design_inner(d):
    qc = _read_qc(d)
    # A true finalized intaglio always wins if present.
    final_path = (qc.get("artifact_paths") or {}).get("intaglio_surface")
    if _existing_verified(final_path):
        return {
            "path": final_path,
            "sha256": file_digest(final_path),
            "role": "FINAL_INTAGLIO_SURFACE",
            "margin_owned": True,
            "generated_from_prepared_abutment": True,
        }, qc
    inner = qc.get("design_intaglio") or {}
    path = inner.get("path")
    sha = inner.get("sha256")
    if _existing_verified(path, sha):
        return inner, qc
    return None, qc


def _persist_inner_record(design_id: str, *, path: str, role: str,
                          source_candidate_id=None, source_artifact_label=None,
                          generation_report=None):
    d = design_row(design_id)
    qc = _read_qc(d)
    sha = file_digest(path)
    record = {
        "path": path,
        "sha256": sha,
        "role": role,
        "source_candidate_id": source_candidate_id,
        "source_artifact_label": source_artifact_label,
        "generated_from_prepared_abutment": True,
        "template_used_for_inner": False,
        "margin_owned": True,
        "shared_margin_semantics": "CONFIRMED_3D_MARGIN_IS_ONLY_CERVICAL_SEAM",
        "survives_outer_sculpt": True,
    }
    if generation_report is not None:
        record["generation_report"] = generation_report
    qc["design_intaglio"] = record
    qc.setdefault("artifact_paths", {})["design_intaglio_preview"] = path
    qc["inner_geometry_ready"] = True
    qc["hollow_crown_semantics"] = {
        "outer_source": "ANATOMY_TEMPLATE_DEFORMED_IN_PATIENT_CONTEXT",
        "inner_source": "PREPARED_ABUTMENT_BLOCKOUT_AND_CEMENT_GAP",
        "cervical_owner": "CONFIRMED_3D_MARGIN",
    }
    with db() as conn:
        conn.execute(
            "UPDATE designs SET qc_json=? WHERE id=?",
            (json.dumps(qc, ensure_ascii=False), design_id),
        )
    return record


def _persist_candidate_inner(design_id: str, proposal_id: str):
    r = candidate(design_id, proposal_id)
    label, path, _sha = _candidate_inner(r)
    if not path:
        return None
    return _persist_inner_record(
        design_id,
        path=path,
        role="DESIGN_INTAGLIO_PREVIEW",
        source_candidate_id=proposal_id,
        source_artifact_label=label,
        generation_report=(r.payload.get("report") or {}).get("INTAGLIO_SPACE_PREVIEW"),
    )


def _support_asset(d):
    with db() as conn:
        row = conn.execute(
            "SELECT * FROM assets WHERE case_id=? AND kind='support' ORDER BY created_at DESC LIMIT 1",
            (d["case_id"],),
        ).fetchone()
    return dict(row) if row else None


def _build_design_inner_now(design_id: str, *, source_candidate_id=None):
    """Generate a real patient intaglio even when the AUTO artifact was not published."""
    d = design_row(design_id)
    support_ref = _support_asset(d)
    if not support_ref or not Path(support_ref["path"]).is_file():
        return None
    try:
        margin = np.asarray(json.loads(d.get("margin_json") or "[]"), float)
        axis = np.asarray(json.loads(d.get("axis_json") or "[]"), float)
        params = json.loads(d.get("parameters_json") or "{}")
    except (TypeError, ValueError, json.JSONDecodeError):
        return None
    if margin.ndim != 2 or margin.shape[1:] != (3,) or len(margin) < 8 or axis.shape != (3,):
        return None

    support = load_mesh(Path(support_ref["path"]))
    support_contract_report = None
    try:
        from ..services.abutment_support import prepare_design_support
        support_contract, prepared = prepare_design_support(d)
        if prepared is not None:
            support = prepared.mesh
        support_contract_report = support_contract.as_report() if support_contract else None
    except Exception as exc:
        # 1.0 still attempts the patient-owned surface on the current support.
        # The failure is recorded; it is not replaced by template geometry.
        support_contract_report = {"status": "RAW_SUPPORT_FALLBACK", "reason": str(exc)}

    from ..services.intaglio_preview import build_intaglio_preview
    try:
        intaglio, _blockout, report = build_intaglio_preview(support, margin, axis, params)
    except Exception:
        return None
    if not len(intaglio.vertices) or not len(intaglio.faces):
        return None

    path = CASE_DIR / d["case_id"] / f"{design_id}_design_intaglio_{uuid4().hex}.stl"
    save_mesh(intaglio, path)
    generation_report = dict(report or {})
    generation_report["prepared_support_contract"] = support_contract_report
    generation_report["generated_on_demand"] = True
    return _persist_inner_record(
        design_id,
        path=str(path),
        role="DESIGN_INTAGLIO_PATIENT_GENERATED",
        source_candidate_id=source_candidate_id,
        source_artifact_label="on_demand_patient_intaglio",
        generation_report=generation_report,
    )


def _ensure_design_inner(design_id: str, proposal_id: str | None = None):
    d = design_row(design_id)
    inner, _qc = _read_design_inner(d)
    if inner:
        return inner
    if proposal_id:
        try:
            inner = _persist_candidate_inner(design_id, proposal_id)
        except HTTPException:
            inner = None
        if inner:
            return inner
    return _build_design_inner_now(design_id, source_candidate_id=proposal_id)


def _snap_open_boundary_to_margin(mesh, margin):
    from ..geometry.crown_outer.collar_body_outer import _boundary_loops
    from ..v1_margin_bl_completion import _nearest_points_on_closed_polyline
    loops = _boundary_loops(np.asarray(mesh.faces, np.int64))
    if len(loops) != 1:
        raise ValueError(f"需要唯一 Margin 开口，当前 {len(loops)} 个")
    ids = np.asarray(loops[0], np.int64)
    vertices = np.asarray(mesh.vertices, float).copy()
    projected, _distance = _nearest_points_on_closed_polyline(vertices[ids], margin)
    vertices[ids] = projected
    out = mesh.copy()
    out.vertices = vertices
    return out


def _build_hollow_shell(design_id: str, inner):
    """Stitch current visible outer to the patient intaglio at one shared Margin."""
    d = design_row(design_id)
    outer_path = d.get("outer_path")
    if not _existing_verified(outer_path):
        return None
    margin = np.asarray(json.loads(d.get("margin_json") or "[]"), float)
    axis = np.asarray(json.loads(d.get("axis_json") or "[]"), float)
    if margin.ndim != 2 or margin.shape[1:] != (3,) or len(margin) < 8 or axis.shape != (3,):
        return None

    outer_sha = file_digest(outer_path)
    inner_sha = inner.get("sha256") or file_digest(inner["path"])
    qc = _read_qc(d)
    cached = qc.get("demo_hollow_shell") or {}
    if (cached.get("outer_sha256") == outer_sha and cached.get("inner_sha256") == inner_sha
            and _existing_verified(cached.get("path"), cached.get("sha256"))):
        return cached

    try:
        outer = load_mesh(Path(outer_path))
        intaglio = load_mesh(Path(inner["path"]))
        curtain = MarginCurtain(margin, axis)
        outside, outer_extract = _single_open_surface_above_margin(
            outer, curtain, 0.0, "OUTER_DEMO_HOLLOW", invert=False
        )
        outside = _snap_open_boundary_to_margin(outside, margin)
        intaglio = _snap_open_boundary_to_margin(intaglio, margin)
        shell, seam = stitch_cervical_surfaces(
            outside, intaglio, margin, quality_mode="DESIGN"
        )
        if not len(shell.vertices) or not len(shell.faces) or not np.isfinite(np.asarray(shell.vertices)).all():
            return None
        path = CASE_DIR / d["case_id"] / f"{design_id}_hollow_crown_{uuid4().hex}.stl"
        save_mesh(shell, path)
        record = {
            "path": str(path),
            "sha256": file_digest(path),
            "outer_sha256": outer_sha,
            "inner_sha256": inner_sha,
            "watertight": bool(shell.is_watertight),
            "shared_margin": True,
            "outer_extraction": outer_extract,
            "seam": seam,
        }
        qc = _read_qc(design_row(design_id))
        qc["demo_hollow_shell"] = record
        with db() as conn:
            conn.execute(
                "UPDATE designs SET qc_json=? WHERE id=?",
                (json.dumps(qc, ensure_ascii=False), design_id),
            )
        return record
    except Exception:
        return None


@router.post('/{design_id}/auto-fit/{proposal_id}/accept')
def accept_with_persistent_inner(design_id: str, proposal_id: str, p: CompleteAcceptInput):
    result = _accept_original(design_id, proposal_id, p)
    inner = _ensure_design_inner(design_id, proposal_id)
    if inner:
        result["inner_geometry_ready"] = True
        result["inner_preview_url"] = f"/api/designs/{design_id}/inner-preview.stl"
        result["design_intaglio_role"] = inner["role"]
    return result


@router.get('/{design_id}/inner-preview.stl')
def current_persistent_inner(design_id: str):
    d = design_row(design_id)
    inner, _qc = _read_design_inner(d)
    if inner:
        return FileResponse(inner["path"], media_type="model/stl")

    # Backward compatibility for cases accepted before persistence was added.
    with db() as conn:
        rows = conn.execute(
            "SELECT payload_json FROM design_revisions WHERE design_id=? AND kind='outer' "
            "ORDER BY revision_no DESC LIMIT 24",
            (design_id,),
        ).fetchall()
    for row in rows:
        try:
            payload = json.loads(row["payload_json"] or "{}")
        except (TypeError, json.JSONDecodeError):
            continue
        proposal_id = payload.get("proposal_id")
        if not proposal_id:
            continue
        persisted = _ensure_design_inner(design_id, proposal_id)
        if persisted:
            return FileResponse(persisted["path"], media_type="model/stl")

    # If the AUTO candidate never published an intaglio artifact, generate it now
    # from the current prepared abutment instead of returning 404.
    built = _ensure_design_inner(design_id)
    if built:
        return FileResponse(built["path"], media_type="model/stl")
    raise HTTPException(404, "当前设计无法从基台生成内冠")


@router.post('/{design_id}/rebuild-cervical')
def rebuild_cervical_demo_safe(design_id: str, p: FinalizeRequest):
    d = design_row(design_id)
    if d.get("final_path") and Path(d["final_path"]).is_file():
        return _rebuild_original(design_id, p)
    inner = _ensure_design_inner(design_id)
    if inner:
        qc = _read_qc(design_row(design_id))
        return {
            "already_current": True,
            "complete_crown_ready": False,
            "inner_geometry_ready": True,
            "inner_preview_url": f"/api/designs/{design_id}/inner-preview.stl",
            "design_intaglio": inner,
            "qc": qc,
            "release_status": "ENGINEERING_ONLY",
            "demo_finalization_deferred": True,
        }
    return _rebuild_original(design_id, p)


@router.post('/{design_id}/demo-edit-outer')
def demo_edit_preserve_inner(design_id: str, p: Edit):
    before = design_row(design_id)
    inner, _qc = _read_design_inner(before)
    if inner is None:
        inner = _ensure_design_inner(design_id)
    result = _demo_edit_original(design_id, p)
    if inner:
        preserved = {
            "design_intaglio": inner,
            "artifact_paths": {"design_intaglio_preview": inner["path"]},
            "inner_geometry_ready": True,
            "demo_outer_edited_after_inner_generation": True,
            "hollow_crown_semantics": {
                "outer_source": "ANATOMY_TEMPLATE_DEFORMED_IN_PATIENT_CONTEXT",
                "inner_source": "PREPARED_ABUTMENT_BLOCKOUT_AND_CEMENT_GAP",
                "cervical_owner": "CONFIRMED_3D_MARGIN",
            },
        }
        with db() as conn:
            conn.execute(
                "UPDATE designs SET qc_json=? WHERE id=?",
                (json.dumps(preserved, ensure_ascii=False), design_id),
            )
        result["inner_geometry_ready"] = True
        result["inner_preview_url"] = f"/api/designs/{design_id}/inner-preview.stl"
    return result


@router.get('/{design_id}/final.stl')
def export_hollow_crown_v1(design_id: str):
    """Export a real hollow crown whenever a patient intaglio is available."""
    d = design_row(design_id)
    # A true finalized crown remains authoritative.
    if _existing_verified(d.get("final_path")):
        return FileResponse(
            d["final_path"], media_type="model/stl",
            filename=f"FDI_{d['target_fdi']}_crown_1.0.stl",
            headers={"X-CrownFlow-Export-Mode": "final"},
        )

    inner = _ensure_design_inner(design_id)
    if inner:
        hollow = _build_hollow_shell(design_id, inner)
        if hollow and _existing_verified(hollow.get("path"), hollow.get("sha256")):
            return FileResponse(
                hollow["path"], media_type="model/stl",
                filename=f"FDI_{d['target_fdi']}_hollow_crown_1.0.stl",
                headers={"X-CrownFlow-Export-Mode": "hollow-design-shell"},
            )

    # Last-resort 1.0 demo behavior remains visible/exportable, but it is clearly
    # identified as outer-only rather than pretending to be a hollow restoration.
    outer = d.get("outer_path")
    if not _existing_verified(outer):
        raise HTTPException(404, "当前没有可导出的牙冠 STL")
    return FileResponse(
        outer, media_type="model/stl",
        filename=f"FDI_{d['target_fdi']}_outer_only_1.0.stl",
        headers={"X-CrownFlow-Export-Mode": "outer-only-last-resort"},
    )
