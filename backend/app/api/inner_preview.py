"""Always expose the best inner geometry for the current CrownFlow 1.0 outer.

A generated AUTO candidate already owns an ``intaglio`` artifact even when final
boolean/cervical construction is not complete.  The UI used to look only at
``design.cutter_path`` and therefore hid that geometry before acceptance and
again after an outer-only acceptance.

This read-only endpoint never invents an inner surface.  It returns, in order:
1. the current finalized cutter when present;
2. the intaglio artifact from the AUTO candidate whose outer SHA is exactly the
   same as the design's current outer SHA.
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


@router.get('/{design_id}/inner-preview.stl')
def current_inner_preview(design_id: str):
    d = design_row(design_id)

    cutter = d.get('cutter_path')
    if cutter and Path(cutter).is_file():
        return FileResponse(cutter, media_type='model/stl')

    outer = d.get('outer_path')
    if not outer or not Path(outer).is_file():
        raise HTTPException(404, '当前设计还没有外冠，无法匹配内冠')
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
        for label in ('intaglio', 'intaglio_surface', 'intaglio_preview'):
            path = artifacts.get(label)
            digest = expected.get(label)
            if path and digest and Path(path).is_file() and file_digest(path) == digest:
                return FileResponse(path, media_type='model/stl')

    raise HTTPException(404, '当前外冠没有可匹配的内冠产物')
