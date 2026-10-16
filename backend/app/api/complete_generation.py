"""Accept a one-click CAD-1 candidate as one complete crown.

The generation job already computes the patient-driven outer, the prep-driven
intaglio and the non-planar cervical stitch.  Acceptance must therefore publish
those exact artifacts together.  Publishing only ``outer_path`` and rebuilding
later made Step 5 operate on a solid shell and reintroduced a second finalize
workflow that the 1.0 product does not have.
"""
from __future__ import annotations

import json
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict
from typing import Literal

from .routes import design_row, now
from .auto_fit import candidate, checked
from ..models import FinalizeRequest
from ..revisions import DesignRevision, RevisionKind
from ..services.design_dependencies import (
    dependency_digest, file_digest, publish_artifacts, require_current_final,
)


router = APIRouter(prefix="/api/designs")


class CompleteAcceptInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    expected_input_digest: str
    confirmed: Literal[True]
    confirmed_template_anatomy: bool = False


def _verified_artifact(revision, label: str) -> str:
    path = revision.artifacts.get(label)
    expected = (revision.payload.get("artifact_sha256") or {}).get(label)
    if label == "outer" and expected is None:
        expected = revision.payload.get("candidate_outer_sha256")
    if not path or not expected or not Path(path).is_file() or file_digest(path) != expected:
        raise HTTPException(409, f"生成候选的 {label} 产物不存在或已变化，请重新生成")
    return path


@router.post('/{design_id}/auto-fit/{proposal_id}/accept')
def accept_complete_generated_crown(design_id: str, proposal_id: str, p: CompleteAcceptInput):
    """Publish AUTO candidates as outer + cutter + stitched final in one revision.

    This route intentionally shadows the legacy outer-only accept route because
    it is included before ``auto_fit.router``. Template-preview candidates keep
    their old outer-only semantics; one-click AUTO candidates must be complete.
    """
    d = checked(design_id, p.expected_input_digest, natural_only=False)
    r = candidate(design_id, proposal_id)
    if r.payload.get('input_digest') != p.expected_input_digest or r.payload.get('design_revision') != d['design_revision']:
        raise HTTPException(409, '候选已过期，不能采纳')

    is_auto = r.payload.get('candidate_kind') == 'AUTO_FIT_CANDIDATE'
    if not is_auto:
        # Preserve the existing template-preview acceptance contract. A template
        # draft is only an anatomical reference; it is not a complete crown.
        if not (r.payload.get('type') in ('template_placement', 'template_draft')
                and p.confirmed_template_anatomy
                and r.payload.get('status') == 'TEMPLATE_REVIEW_REQUIRED'):
            raise HTTPException(409, '该候选不是可采纳的一键牙冠；请重新生成')
        outer = _verified_artifact(r, 'outer')
        revision = DesignRevision(
            design_id=design_id, revision_no=0, kind=RevisionKind.OUTER,
            payload={
                'type': 'autofit_accepted', 'proposal_id': r.id,
                'consumed_revisions': r.payload.get('consumed_revisions', {}),
                'outer_dependency_digest': dependency_digest(d, stage='outer'),
                'output_sha256': file_digest(outer),
            },
            artifacts={'outer': outer}, algorithm_version='welooai/autofit/1.1-complete',
        )
        updates = {
            'outer_path': outer, 'final_path': None, 'cutter_path': None,
            'qc_json': None, 'final_dependency_digest': None,
            'status': 'outer_ready', 'updated_at': now(),
        }
        if r.payload.get('type') == 'template_placement':
            landmarks = json.loads(d.get('landmarks_json') or '{}')
            landmarks['_selected_template'] = {
                'source_path': r.payload['source_template_path'],
                'sha256': r.payload['template_sha256'],
                'metadata_path': r.payload['source_metadata_path'],
                'metadata_sha256': r.payload['metadata_sha256'],
                'library': r.payload['template']['library'],
                'proposal_id': r.id,
                'geometry_frame_revision_id': r.payload['geometry_frame_revision_id'],
            }
            updates['landmarks_json'] = json.dumps(landmarks, ensure_ascii=False)
            revision.payload['outer_dependency_digest'] = dependency_digest({**d, **updates}, stage='outer')
        publish_artifacts(d, p.expected_input_digest, updates, stage='outer', revision=revision)
        return {
            'outer_path': outer, 'revision_id': revision.id,
            'final_invalidated': True, 'complete_crown_ready': False,
            'release_status': 'NOT_PRODUCTION_APPROVED',
        }

    # AUTO generation is not accepted into Step 5 until the same job produced a
    # real intaglio and a watertight stitched final. Warnings may remain review
    # items, but missing geometry is not silently converted into outer-only work.
    if not r.payload.get('complete_crown_ready'):
        reason = (r.payload.get('w3_preflight') or {}).get('reason') or '最终闭合或内冠尚未构建完成'
        raise HTTPException(409, {'message': '完整牙冠尚未生成，不能进入微调', 'reason': reason})

    outer = _verified_artifact(r, 'outer')
    final = _verified_artifact(r, 'final')
    cutter = _verified_artifact(r, 'cutter')

    qc = dict((r.payload.get('w3_preflight') or {}).get('qc') or {})
    if not qc:
        raise HTTPException(409, '完整牙冠缺少生成阶段 QC，请重新生成')
    generated_paths = {}
    for label in ('outer_surface', 'intaglio_surface', 'manufacturing_qc', 'prepared_support', 'actual_support'):
        if label in r.artifacts:
            generated_paths[label] = _verified_artifact(r, label)
    qc['artifact_paths'] = {**(qc.get('artifact_paths') or {}), **generated_paths}
    qc['release_status'] = 'ENGINEERING_ONLY'
    qc['generation_candidate_id'] = r.id
    qc['complete_generation'] = {
        'status': 'COMPLETE_CROWN_READY',
        'outer_sha256': file_digest(outer),
        'cutter_sha256': file_digest(cutter),
        'final_sha256': file_digest(final),
        'intaglio_built_before_sculpt': True,
        'cervical_stitch_built_before_sculpt': True,
    }

    updates = {
        'outer_path': outer,
        'outer_output_sha256': file_digest(outer),
        'final_path': final,
        'cutter_path': cutter,
        'qc_json': json.dumps(qc, ensure_ascii=False),
        'status': 'final_ready',
        'updated_at': now(),
        'sculpt_state_json': None,
    }
    # ``publish_artifacts(stage='final')`` owns final dependency/hash publication;
    # outer/body currentness must also be established because Step 5 reads the
    # editable external body through the outer contract.
    updates['outer_dependency_digest'] = dependency_digest({**d, **updates}, stage='outer')
    landmarks = json.loads(d.get('landmarks_json') or '{}')
    landmarks['_body_dependency_digest'] = dependency_digest({**d, **updates}, stage='body')
    updates['landmarks_json'] = json.dumps(landmarks, ensure_ascii=False)

    revision = DesignRevision(
        design_id=design_id, revision_no=0, kind=RevisionKind.OUTER,
        payload={
            'type': 'autofit_accepted',
            'proposal_id': r.id,
            'complete_crown': True,
            'consumed_revisions': r.payload.get('consumed_revisions', {}),
            'outer_dependency_digest': updates['outer_dependency_digest'],
            'output_sha256': file_digest(outer),
            'final_sha256': file_digest(final),
            'cutter_sha256': file_digest(cutter),
        },
        artifacts={'outer': outer, 'final': final, 'cutter': cutter, **generated_paths},
        algorithm_version='welooai/autofit/1.1-complete',
    )
    publish_artifacts(d, p.expected_input_digest, updates, stage='final', revision=revision)
    return {
        'outer_path': outer,
        'final_path': final,
        'cutter_path': cutter,
        'revision_id': revision.id,
        'final_invalidated': False,
        'complete_crown_ready': True,
        'release_status': 'ENGINEERING_ONLY',
    }


@router.post('/{design_id}/rebuild-cervical')
def rebuild_cervical_idempotent(design_id: str, p: FinalizeRequest):
    """Compatibility endpoint for the old UI; never rebuild an already-current final.

    Generation now creates the intaglio and stitch before Step 5. Older clients
    still issue one ``rebuild-cervical`` immediately after candidate acceptance.
    Treat that call as a read of the already-current result instead of running the
    expensive geometry a second time. If sculpting invalidated the final, the same
    endpoint remains the internal sync operation used just before export.
    """
    d = design_row(design_id)
    try:
        require_current_final(d)
    except HTTPException:
        from .routes import _finalize
        return _finalize(design_id, p, preserve_body=True)
    return {
        'qc': json.loads(d.get('qc_json') or '{}'),
        'already_current': True,
        'final_path': d.get('final_path'),
        'cutter_path': d.get('cutter_path'),
        'release_status': 'ENGINEERING_ONLY',
    }
