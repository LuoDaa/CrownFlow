"""Always expose the best inner geometry for the current CrownFlow 1.0 outer.

The design intaglio is generated from the prepared abutment/support, insertion
axis and confirmed cement-gap field.  Once W3 finalization succeeds, the geometry
that should be shown to the user is the *actual intaglio surface of the crown*,
not the closed Boolean cutter (whose artificial bottom closure intentionally sits
below the Margin and can look misaligned in the viewer).

This read-only endpoint never invents an inner surface.  It returns, in order:
1. finalized ``intaglio_surface`` from the W3 artifact set — this shares the
   authoritative 3D Margin seam with the final crown;
2. the intaglio artifact from the AUTO candidate whose outer SHA exactly matches
   the current accepted outer;
3. the finalized closed cutter only as a compatibility fallback.
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


@router.get('/{design_id}/inner-preview.stl')
def current_inner_preview(design_id: str):
    d = design_row(design_id)

    # W3's exported intaglio surface is the real crown inner wall and shares the
    # exact Margin seam built by build_nonplanar_crown.  Prefer it over cutter.
    try:
        qc = json.loads(d.get('qc_json') or '{}')
    except (TypeError, json.JSONDecodeError):
        qc = {}
    artifact_paths = qc.get('artifact_paths') or {}
    final_intaglio = artifact_paths.get('intaglio_surface')
    if _existing(final_intaglio):
        return FileResponse(final_intaglio, media_type='model/stl')

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
            expected = payload.get('artifact_sha256') or {}
            for label in ('intaglio_surface', 'intaglio', 'intaglio_preview'):
                path = artifacts.get(label)
                digest = expected.get(label)
                if path and digest and Path(path).is_file() and file_digest(path) == digest:
                    return FileResponse(path, media_type='model/stl')

    # The cutter is still useful for legacy/finalized designs whose older QC did
    # not publish an intaglio_surface artifact, but it is intentionally last.
    cutter = d.get('cutter_path')
    if _existing(cutter):
        return FileResponse(cutter, media_type='model/stl')

    if not outer or not Path(outer).is_file():
        raise HTTPException(404, '当前设计还没有外冠，无法匹配内冠')
    raise HTTPException(404, '当前外冠没有可匹配的内冠产物')
