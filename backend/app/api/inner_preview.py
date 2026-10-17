"""Always expose the best inner geometry for the current CrownFlow 1.0 outer.

The design intaglio is generated from the prepared abutment/support, insertion
axis and confirmed cement-gap field. Once W3 finalization succeeds, the geometry
that should be shown to the user is the *actual intaglio surface of the crown*,
not the closed Boolean cutter. Before finalization succeeds, keep the generated
AUTO intaglio preview visible through the accepted proposal lineage, even after
the outer is accepted or sculpted and its file hash changes.
"""
from __future__ import annotations

import json
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from .routes import design_row
from ..db import db
from ..services.design_dependencies import file_digest


router = APIRouter(prefix="/api/designs")


def _existing(path):
    return bool(path and Path(path).is_file())


def _verified_candidate_inner(payload, artifacts):
    expected = payload.get('artifact_sha256') or {}
    for label in ('intaglio_surface', 'intaglio', 'intaglio_preview'):
        path = artifacts.get(label)
        digest = expected.get(label)
        if path and digest and Path(path).is_file() and file_digest(path) == digest:
            return path
    return None


def _candidate_inner_by_proposal(design_id: str, proposal_id: str | None):
    if not proposal_id:
        return None
    with db() as conn:
        row = conn.execute(
            "SELECT payload_json, artifacts_json FROM design_revisions "
            "WHERE design_id=? AND id=? AND kind='outer_autofit_candidate' LIMIT 1",
            (design_id, proposal_id),
        ).fetchone()
    if not row:
        return None
    try:
        payload = json.loads(row['payload_json'] or '{}')
        artifacts = json.loads(row['artifacts_json'] or '{}')
    except (TypeError, json.JSONDecodeError):
        return None
    return _verified_candidate_inner(payload, artifacts)


@router.get('/{design_id}/inner-preview.stl')
def current_inner_preview(design_id: str):
    d = design_row(design_id)
    try:
        qc = json.loads(d.get('qc_json') or '{}')
    except (TypeError, json.JSONDecodeError):
        qc = {}
    artifact_paths = qc.get('artifact_paths') or {}
    final_intaglio = artifact_paths.get('intaglio_surface')
    if _existing(final_intaglio):
        return FileResponse(final_intaglio, media_type='model/stl')

    # First preserve the generated inner through the accepted AUTO proposal.
    # This is intentionally independent of current outer SHA: sculpt edits change
    # the outer file, but they must not make the already-generated intaglio vanish.
    with db() as conn:
        accepted = conn.execute(
            "SELECT payload_json FROM design_revisions "
            "WHERE design_id=? AND kind='outer' ORDER BY revision_no DESC LIMIT 24",
            (design_id,),
        ).fetchall()
    for row in accepted:
        try:
            payload = json.loads(row['payload_json'] or '{}')
        except (TypeError, json.JSONDecodeError):
            continue
        path = _candidate_inner_by_proposal(design_id, payload.get('proposal_id'))
        if _existing(path):
            return FileResponse(path, media_type='model/stl')

    # Backward-compatible SHA ownership for older cases that have no proposal id
    # recorded on the accepted outer revision.
    outer = d.get('outer_path')
    if outer and Path(outer).is_file():
        current_outer_sha = file_digest(outer)
        with db() as conn:
            rows = conn.execute(
                "SELECT payload_json, artifacts_json FROM design_revisions "
                "WHERE design_id=? AND kind='outer_autofit_candidate' "
                "ORDER BY revision_no DESC LIMIT 24",
                (design_id,),
            ).fetchall()

        for row in rows:
            try:
                payload = json.loads(row['payload_json'] or '{}')
                artifacts = json.loads(row['artifacts_json'] or '{}')
            except (TypeError, json.JSONDecodeError):
                continue
            if payload.get('candidate_outer_sha256') != current_outer_sha:
                continue
            path = _verified_candidate_inner(payload, artifacts)
            if _existing(path):
                return FileResponse(path, media_type='model/stl')

    cutter = d.get('cutter_path')
    if _existing(cutter):
        return FileResponse(cutter, media_type='model/stl')

    if not outer or not Path(outer).is_file():
        raise HTTPException(404, '当前设计还没有外冠，无法匹配内冠')
    raise HTTPException(404, '当前外冠没有可匹配的内冠产物')


# The 1.0 demonstration endpoints share this early router so their edit/export
# behavior is available before the legacy engineering routes are registered.
from .demo_v1 import router as _demo_v1_router
router.routes.extend(_demo_v1_router.routes)
