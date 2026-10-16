import numpy as np
import trimesh

from app.geometry.crown_outer.auto_fit_adapter import FitSurface
from app.geometry.crown_outer.patient_envelope import (
    build_patient_envelope,
    morph_embryo_to_patient_envelope,
)
from app.geometry.crown_outer.patient_frame import confirmed_patient_frame


def _frame():
    return confirmed_patient_frame(
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    )


def _margin(rx=2.4, ry=2.0, n=128):
    t = np.linspace(0, 2*np.pi, n, endpoint=False)
    return np.c_[rx*np.cos(t), ry*np.sin(t), np.zeros(n)]


def _embryo(sx=1.0, sy=1.0, sz=1.0):
    profile = np.array([
        [0.0, 0.0], [1.8, 0.0], [2.0, 0.5], [2.2, 2.0],
        [2.3, 3.8], [1.7, 5.6], [0.0, 6.0],
    ])
    mesh = trimesh.creation.revolve(profile, sections=96)
    mesh.apply_scale([sx, sy, sz])
    return mesh


def _neighbor(x):
    m = trimesh.creation.box(extents=[1.0, 8.2, 7.0])
    m.apply_translation([x, 0.0, 3.2])
    return FitSurface(m)


def _uv_extents(mesh, env, frame):
    local = (np.asarray(mesh.vertices) - np.asarray(frame.origin)) @ np.asarray(frame.matrix())
    xy = local[:, :2] - np.asarray(env.center_xy)
    u = xy @ np.asarray(env.md_axis_xy)
    v = xy @ np.asarray(env.bl_axis_xy)
    body = local[:, 2] >= np.quantile(local[:, 2], .50)
    return np.quantile(u[body], [.02, .98]), np.quantile(v[body], [.02, .98])


def test_two_different_template_sizes_converge_to_same_patient_dimensions():
    frame = _frame(); margin = _margin()
    intaglio = trimesh.creation.cylinder(radius=2.1, height=4.2, sections=96)
    intaglio.apply_translation([0, 0, 2.1])
    surfaces = {"left": _neighbor(-5.2), "right": _neighbor(5.2)}
    env = build_patient_envelope(
        frame=frame, margin=margin, inner_envelope_mesh=intaglio,
        minimum_thickness_mm=.8, proximal_surfaces=surfaces,
        proximal_target_mm=-.02,
    )

    small, _ = morph_embryo_to_patient_envelope(_embryo(.65, .72, .8), frame, margin, env)
    large, _ = morph_embryo_to_patient_envelope(_embryo(1.45, 1.35, 1.25), frame, margin, env)
    su, sv = _uv_extents(small, env, frame)
    lu, lv = _uv_extents(large, env, frame)

    assert env.sources["confirmed_neighbor_count"] == 2
    assert env.sources["positive_neighbor"] and env.sources["negative_neighbor"]
    assert np.allclose(su, lu, atol=.20)
    assert np.allclose(sv, lv, atol=.20)


def test_margin_root_barely_moves_while_coronal_body_reaches_patient_width():
    frame = _frame(); margin = _margin()
    surfaces = {"left": _neighbor(-5.0), "right": _neighbor(5.0)}
    env = build_patient_envelope(
        frame=frame, margin=margin, proximal_surfaces=surfaces,
        proximal_target_mm=0.0,
    )
    embryo = _embryo(.72, .72, 1.0)
    before = np.asarray(embryo.vertices).copy()
    result, report = morph_embryo_to_patient_envelope(embryo, frame, margin, env)
    after = np.asarray(result.vertices)

    h = before[:, 2]
    cervical = h <= .20
    coronal = h >= 2.5
    assert np.max(np.linalg.norm(after[cervical] - before[cervical], axis=1)) < .08
    assert np.median(np.linalg.norm(after[coronal] - before[coronal], axis=1)) > .25
    assert report["template_size_authoritative"] is False


def test_proximal_bounds_are_owned_by_real_neighbors_and_allow_small_negative_target():
    frame = _frame(); margin = _margin()
    surfaces = {"anything_a": _neighbor(-5.0), "anything_b": _neighbor(5.4)}
    env0 = build_patient_envelope(
        frame=frame, margin=margin, proximal_surfaces=surfaces,
        proximal_target_mm=0.0,
    )
    envn = build_patient_envelope(
        frame=frame, margin=margin, proximal_surfaces=surfaces,
        proximal_target_mm=-.03,
    )

    assert envn.u_max > env0.u_max
    assert envn.u_min < env0.u_min
    assert abs((envn.u_max - env0.u_max) - .03) < .015
    assert abs((env0.u_min - envn.u_min) - .03) < .015


def test_absent_side_uses_margin_emergence_not_template_size():
    frame = _frame(); margin = _margin()
    env = build_patient_envelope(
        frame=frame, margin=margin,
        proximal_surfaces={"one_real_side": _neighbor(5.0)},
        proximal_target_mm=0.0,
    )
    huge = _embryo(2.2, 1.0, 1.0)
    result, _ = morph_embryo_to_patient_envelope(huge, frame, margin, env)
    u, _v = _uv_extents(result, env, frame)

    assert env.sources["confirmed_neighbor_count"] == 1
    assert max(abs(float(u[0])), abs(float(u[1]))) < 7.2
    assert env.md_width_mm < 12.0


def test_bl_width_comes_from_margin_shell_and_neighbor_width_not_template():
    frame = _frame(); margin = _margin(2.3, 1.6)
    intaglio = trimesh.creation.box(extents=[4.0, 3.8, 4.0])
    intaglio.apply_translation([0, 0, 2.0])
    surfaces = {"left": _neighbor(-5.1), "right": _neighbor(5.1)}
    env = build_patient_envelope(
        frame=frame, margin=margin, inner_envelope_mesh=intaglio,
        minimum_thickness_mm=.8, proximal_surfaces=surfaces,
    )

    assert env.bl_width_mm > 7.0
    assert env.sources["neighbor_bl_prior_count"] == 2


def test_antagonist_sets_coronal_ceiling_but_never_below_material_floor():
    frame = _frame(); margin = _margin()
    intaglio = trimesh.creation.cylinder(radius=2.0, height=4.2, sections=96)
    intaglio.apply_translation([0, 0, 2.1])
    antagonist = trimesh.creation.box(extents=[14, 14, 1.0])
    antagonist = antagonist.subdivide().subdivide()
    antagonist.apply_translation([0, 0, 8.5])
    env = build_patient_envelope(
        frame=frame, margin=margin, inner_envelope_mesh=intaglio,
        minimum_thickness_mm=.8, proximal_surfaces={"left": _neighbor(-5), "right": _neighbor(5)},
        antagonist_surface=FitSurface(antagonist), occlusal_target_mm=.05,
    )
    shell_top = env.sources["inner_shell_bounds"]["top_z"]

    assert env.sources["antagonist"]["used"] is True
    assert env.target_top_z >= shell_top - 1e-9
    assert 7.7 < env.target_top_z < 8.2
