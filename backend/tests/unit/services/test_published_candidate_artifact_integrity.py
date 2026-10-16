import json

import pytest
from fastapi import HTTPException

from app.services.design_dependencies import (
    _validate_published_candidate_artifacts,
    file_digest,
)


def _proposal(artifacts):
    return {
        'artifacts_json': json.dumps(artifacts),
        'payload_json': json.dumps({
            'artifact_sha256': {label: file_digest(path) for label, path in artifacts.items()},
        }),
    }


def test_changed_preflight_artifact_does_not_block_outer_acceptance(tmp_path):
    outer = tmp_path / 'outer.stl'
    preflight = tmp_path / 'preflight.json'
    outer.write_bytes(b'candidate outer')
    preflight.write_bytes(b'original audit')
    proposal = _proposal({'outer': str(outer), 'preflight': str(preflight)})
    preflight.write_bytes(b'updated audit')

    _validate_published_candidate_artifacts(proposal, {'outer': str(outer)})


def test_changed_outer_still_blocks_candidate_acceptance(tmp_path):
    outer = tmp_path / 'outer.stl'
    outer.write_bytes(b'candidate outer')
    proposal = _proposal({'outer': str(outer)})
    outer.write_bytes(b'changed outer')

    with pytest.raises(HTTPException, match='候选或预审产物已变化'):
        _validate_published_candidate_artifacts(proposal, {'outer': str(outer)})
