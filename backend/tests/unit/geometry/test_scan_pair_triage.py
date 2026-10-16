"""Private real-scan triage must find a detached prep without mistaking specks for teeth."""
import numpy as np
import trimesh

from app.geometry.scan_pair_triage import select_separate_support


def _ellipsoid(extents, subdivisions, offset):
    mesh = trimesh.creation.icosphere(subdivisions=subdivisions, radius=1.0)
    mesh.vertices *= np.asarray(extents, float) / 2.0
    mesh.apply_translation(np.asarray(offset, float))
    return mesh


def test_detached_tooth_scale_component_is_selected_over_speck():
    arch = _ellipsoid([50.0, 34.0, 20.0], 4, [0.0, 0.0, 0.0])
    prep = _ellipsoid([7.0, 9.0, 8.0], 3, [35.0, 0.0, 0.0])
    speck = _ellipsoid([0.2, 0.2, 0.2], 2, [45.0, 0.0, 0.0])
    scan = trimesh.util.concatenate([arch, prep, speck])

    selected, report = select_separate_support(scan)

    assert selected is not None
    assert report["status"] == "CANDIDATE_FOUND"
    assert report["selected_rank"] == 1
    np.testing.assert_allclose(np.sort(selected.extents), [7.0, 8.0, 9.0], atol=1e-6)


def test_single_arch_does_not_invent_support_component():
    arch = _ellipsoid([50.0, 34.0, 20.0], 4, [0.0, 0.0, 0.0])
    selected, report = select_separate_support(arch)

    assert selected is None
    assert report["status"] == "NOT_FOUND"
    assert report["reason"] == "NO_SECOND_MAJOR_COMPONENT"
