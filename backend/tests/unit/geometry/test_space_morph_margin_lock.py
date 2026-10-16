import numpy as np
import trimesh

from app.geometry.nonplanar_crown import MarginCurtain
from app.geometry.crown_outer.space_morph import (
    SpaceMorphConfig,
    enforce_inner_envelope_clearance,
)


def _circle_margin(radius=2.0, z=-2.0, n=32):
    a = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    return np.c_[radius * np.cos(a), radius * np.sin(a), np.full(n, z)]


def test_shell_clearance_never_moves_margin_owned_cervical_band():
    margin = _circle_margin()
    axis = np.array([0.0, 0.0, 1.0])
    frame = MarginCurtain(margin, axis).frame

    outer = trimesh.creation.cylinder(radius=2.0, height=4.0, sections=32)
    inner = trimesh.creation.cylinder(radius=1.85, height=4.0, sections=32)
    before = np.asarray(outer.vertices, float).copy()
    height = MarginCurtain(margin, axis).axial_above(before)

    out, report = enforce_inner_envelope_clearance(
        outer,
        inner,
        frame,
        margin,
        required_mm=0.50,
        config=SpaceMorphConfig(envelope_iterations=2),
    )

    locked = height <= 0.45
    assert np.array_equal(np.asarray(out.vertices)[locked], before[locked])
    assert report['cervical_immutable_mm'] == 0.45
    assert report['method'] == 'ACTUAL_INTAGLIO_SAFE_LATERAL_CLEARANCE_V4'

    # A shell repair must never turn a visible patch inside-out.
    dot = np.einsum('ij,ij->i', outer.face_normals, out.face_normals)
    assert np.all(dot > 0.0)
    assert np.all(out.area_faces >= 1e-10)
