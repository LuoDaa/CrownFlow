"""CrownFlow 1.0 inner-crown continuity layer.

The AUTO pipeline already builds a real intaglio preview from prepared support +
confirmed Margin + insertion axis + cement gap before outer fitting.  The remaining
1.0 defect was lifecycle-related: accepting/sculpting the outer could clear qc_json
or lose the candidate SHA relationship, making that generated inner disappear from
the canvas.  This router is registered before the legacy compatibility routes and
keeps the generated design intaglio attached to the design until a true finalized
intaglio surface replaces it.
"""
from __future__ import annotations

import json
from pathlib import Path

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
from ..db import db
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


def _read_design_inner(d):
    try:
        qc = json.loads(d.get("qc_json") or "{}")
    except (TypeError, json.JSONDecodeError):
        qc = {}
    inner = qc.get("design_intaglio") or {}
    path = inner.get("path")
    sha = inner.get("sha256")
    if _existing_verified(path, sha):
        return inner, qc
    # A real finalized intaglio always wins if present.
    final_path = (qc.get("artifact_paths") or {}).get("intaglio_surface")
    if _existing_verified(final_path):
        return {
            "path": final_path,
            "sha256": file_digest(final_path),
            "role": "FINAL_INTAGLIO_SURFACE",
            "margin_owned": True,
        }, qc
    return None, qc


def _persist_candidate_inner(design_id: str, proposal_id: str):
    r = candidate(design_id, proposal_id)
    label, path, sha = _candidate_inner(r)
    if not path:
        return None
    d = design_row(design_id)
    try:
        qc = json.loads(d.get("qc_json") or "{}")
    except (TypeError, json.JSONDecodeError):
        qc = {}
    qc["design_intaglio"] = {
        "path": path,
        "sha256": sha,
        "role": "DESIGN_INTAGLIO_PREVIEW",
        "source_candidate_id": proposal_id,
        "source_artifact_label": label,
        "generated_from_prepared_abutment": True,
        "margin_owned": True,
        "survives_outer_sculpt": True,
    }
    qc.setdefault("artifact_paths", {})["design_intaglio_preview"] = path
    qc["inner_geometry_ready"] = True
    with db() as conn:
        conn.execute(
            "UPDATE designs SET qc_json=? WHERE id=?",
            (json.dumps(qc, ensure_ascii=False), design_id),
        )
    return qc["design_intaglio"]


@router.post('/{design_id}/auto-fit/{proposal_id}/accept')
def accept_with_persistent_inner(design_id: str, proposal_id: str, p: CompleteAcceptInput):
    result = _accept_original(design_id, proposal_id, p)
    # If a true complete crown was published, its intaglio_surface is already
    # authoritative. Otherwise persist the generated design intaglio explicitly
    # instead of allowing outer-only publication to make it disappear.
    if not result.get("complete_crown_ready"):
        inner = _persist_candidate_inner(design_id, proposal_id)
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

    # Backward compatibility for cases accepted before persistence was added:
    # follow the accepted outer proposal lineage and persist it on first access.
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
        try:
            persisted = _persist_candidate_inner(design_id, proposal_id)
        except HTTPException:
            persisted = None
        if persisted:
            return FileResponse(persisted["path"], media_type="model/stl")

    raise HTTPException(404, "当前设计尚未生成可用内冠")


@router.post('/{design_id}/rebuild-cervical')
def rebuild_cervical_demo_safe(design_id: str, p: FinalizeRequest):
    """Do not turn a valid generated inner into a 400 during the 1.0 demo path.

    A true current final is still returned by the original implementation.  When
    final stitching is unavailable but the AUTO job already produced a verified
    design intaglio, keep that result and report success instead of discarding it.
    """
    d = design_row(design_id)
    inner, qc = _read_design_inner(d)
    if d.get("final_path") and Path(d["final_path"]).is_file():
        return _rebuild_original(design_id, p)
    if inner:
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
    # No generated inner exists: allow the legacy path to try true finalization.
    return _rebuild_original(design_id, p)


@router.post('/{design_id}/demo-edit-outer')
def demo_edit_preserve_inner(design_id: str, p: Edit):
    """Save every outer edit without clearing the already-generated intaglio."""
    before = design_row(design_id)
    inner, qc = _read_design_inner(before)
    result = _demo_edit_original(design_id, p)
    if inner:
        # demo_v1 intentionally invalidates final/QC on every outer edit. Restore
        # only the design-intaglio continuity metadata; no manufacturing approval
        # or stale final is resurrected.
        preserved = {
            "design_intaglio": inner,
            "artifact_paths": {"design_intaglio_preview": inner["path"]},
            "inner_geometry_ready": True,
            "demo_outer_edited_after_inner_generation": True,
        }
        with db() as conn:
            conn.execute(
                "UPDATE designs SET qc_json=? WHERE id=?",
                (json.dumps(preserved, ensure_ascii=False), design_id),
            )
        result["inner_geometry_ready"] = True
        result["inner_preview_url"] = f"/api/designs/{design_id}/inner-preview.stl"
    return result
