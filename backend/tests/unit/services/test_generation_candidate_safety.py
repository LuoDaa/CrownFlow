from app.services.generation_workflow import _apply_auto_candidate_safety


def test_abutment_auto_review_preserves_output_when_occlusal_stage_missing():
    d = {'support_type': 'abutment'}
    report = {
        'mode': 'PROXIMAL_ONLY',
        'status': 'PASS_ENGINEERING',
        'stages': [
            {'stage': 'PROXIMAL', 'status': 'PASS_ENGINEERING'},
        ],
        'warnings': [],
        'bite_revision_id': None,
        'candidate_kind': 'AUTO_FIT_CANDIDATE',
        'complete_crown_ready': True,
    }
    artifacts = {'outer': '/tmp/crown.stl'}

    result, kept = _apply_auto_candidate_safety(d, report, artifacts)

    assert kept is artifacts
    assert result['candidate_kind'] == 'AUTO_FIT_REVIEW'
    assert result['status'] == 'REVIEW_REQUIRED'
    assert result['complete_crown_ready'] is True
    assert result['automatic_accept_allowed'] is False
    assert result['generation_safety']['output_preserved'] is True
    codes = {item['code'] for item in result['generation_safety']['reasons']}
    assert 'ABUTMENT_FULL_SPACE_NOT_AVAILABLE' in codes
    assert 'OCCLUSAL_STAGE_MISSING' in codes


def test_large_shell_change_marks_review_but_does_not_remove_outer():
    d = {'support_type': 'prep'}
    report = {
        'mode': 'BOTH',
        'status': 'PASS_ENGINEERING',
        'stages': [
            {
                'stage': 'SHELL_REPAIR_DECISION',
                'status': 'SKIPPED_SMALL_LOCAL_CHANGE',
                'major_change': True,
                'moved_vertices': 1220,
                'moved_fraction': 0.203,
                'max_move_mm': 1.53,
            },
            {'stage': 'INNER_SHELL_FINAL_AUDIT', 'status': 'REVIEW_REQUIRED'},
        ],
        'warnings': [],
        'candidate_kind': 'AUTO_FIT_CANDIDATE',
        'complete_crown_ready': True,
    }
    artifacts = {'outer': '/tmp/crown.stl'}

    result, kept = _apply_auto_candidate_safety(d, report, artifacts)

    assert kept['outer'] == '/tmp/crown.stl'
    assert result['candidate_kind'] == 'AUTO_FIT_REVIEW'
    assert result['generation_safety']['output_preserved'] is True
    reason = result['generation_safety']['reasons'][0]
    assert reason['code'] == 'LARGE_SHELL_DEFORMATION_NOT_CERTIFIED'
    assert reason['moved_vertices'] == 1220
    assert reason['max_move_mm'] == 1.53


def test_certified_patient_space_keeps_candidate_kind():
    d = {'support_type': 'abutment'}
    report = {
        'mode': 'BOTH',
        'status': 'PASS_ENGINEERING',
        'stages': [
            {'stage': 'OCCLUSAL_HEIGHT_PREFIT', 'status': 'PASS_ENGINEERING'},
            {'stage': 'OCCLUSAL', 'status': 'PASS_ENGINEERING'},
            {
                'stage': 'SHELL_REPAIR_DECISION',
                'status': 'REPAIR_REQUIRED',
                'major_change': True,
                'moved_vertices': 100,
                'moved_fraction': 0.02,
                'max_move_mm': 0.20,
            },
            {'stage': 'INNER_SHELL_FINAL_AUDIT', 'status': 'PASS_ENGINEERING'},
        ],
        'warnings': [],
        'bite_revision_id': 'bite-rev-1',
        'candidate_kind': 'AUTO_FIT_CANDIDATE',
        'complete_crown_ready': True,
    }

    result, _ = _apply_auto_candidate_safety(d, report, {'outer': '/tmp/crown.stl'})

    assert result['candidate_kind'] == 'AUTO_FIT_CANDIDATE'
    assert result['generation_safety']['status'] == 'PASS_ENGINEERING'
    assert result['automatic_accept_allowed'] is True
