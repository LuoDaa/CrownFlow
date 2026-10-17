from pathlib import Path


def test_occlusal_height_prefit_is_not_global_drop():
    source = Path('backend/app/v1_occlusal_intaglio_completion_guard.py').read_text(encoding='utf-8')
    assert 'LOCAL_OCCLUSAL_SOLVER_OWNS_ANTAGONIST_HEIGHT_V1' in source
    assert 'BROAD_HEIGHT_PREFIT_DISABLED_TO_PRESERVE_POSTERIOR_ANATOMY' in source
    assert '_space._fit_antagonist_height = _local_solver_owns_antagonist_height' in source


def test_auto_generation_forces_w3_in_same_job():
    source = Path('backend/app/v1_occlusal_intaglio_completion_guard.py').read_text(encoding='utf-8')
    assert 'kwargs["run_w3_preflight"] = True' in source
    assert 'w3_generation": "SAME_JOB_AS_OUTER' in source
    assert '_workflow.generate_candidate = _complete_auto_generation' in source


def test_height_preservation_is_full_scale():
    source = Path('backend/app/v1_morphology_preservation_guard.py').read_text(encoding='utf-8')
    assert 'MORPH_HEIGHT_MIN_SCALE = 1.00' in source


def test_policy_loads_before_api_and_after_morphology_guard():
    source = Path('backend/app/main.py').read_text(encoding='utf-8')
    morph = source.index('v1_morphology_preservation_guard')
    owner = source.index('v1_occlusal_intaglio_completion_guard')
    api = source.index('from .api.crown import router as crown_router')
    assert morph < owner < api
