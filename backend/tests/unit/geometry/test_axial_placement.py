"""轴向初放置约束测试（合成几何，不依赖病例资产）。

覆盖：
1. **可行**：冠太高但区间放得下 ⇒ 轴向缩放把冠顶压进区间，且**颈缘带位移精确为 0**；
2. **不可行**：lower > upper ⇒ 明确 ``INSUFFICIENT_OCCLUSAL_SPACE``，不返回被压坏的网格
   （不让下游咬合求解器去削 3–6mm）；
3. **Crown 1.0**：冠虽然已在可行区间内，但明显低于对颌安全上界时，有限增高到 upper，
   而不是把一个“咬不到”的短冠交给后续局部咬合求解器。
"""
import json

import numpy as np
import pytest
import trimesh

from app.geometry.crown_outer.auto_fit_adapter import FitSurface
from app.geometry.crown_outer.patient_frame import build_patient_frame
from app.services.axial_placement import (apply_axial_placement, coarse_height_fit, occlusal_envelope,
                                          plan_axial_placement)

AXIS = np.array([0.0, 0.0, 1.0])
SEGMENTS = 32


def _ring(radius, z):
    angle = np.linspace(0.0, 2.0 * np.pi, SEGMENTS, endpoint=False)
    return np.stack([radius * np.cos(angle), radius * np.sin(angle), np.full(SEGMENTS, z)], axis=1)


def _cone_crown(top_z=6.0, apex_z=7.0, bottom_radius=2.0, top_radius=1.2):
    """下环 = Margin(z=0) → 上环 → 顶点：一个简单合成冠。"""
    bottom, top = _ring(bottom_radius, 0.0), _ring(top_radius, top_z)
    vertices = np.vstack([bottom, top, np.array([[0.0, 0.0, apex_z]])])
    faces = []
    for index in range(SEGMENTS):
        nxt = (index + 1) % SEGMENTS
        faces += [[index, nxt, SEGMENTS + nxt], [index, SEGMENTS + nxt, SEGMENTS + index],
                  [SEGMENTS + index, SEGMENTS + nxt, 2 * SEGMENTS]]
    mesh = trimesh.Trimesh(vertices=vertices, faces=np.asarray(faces, np.int64), process=False)
    outward = np.asarray(mesh.triangles_center, float) * np.array([1.0, 1.0, 0.0])
    if float(np.mean(np.einsum("ij,ij->i", np.asarray(mesh.face_normals, float), outward))) < 0:
        mesh.faces = np.asarray(mesh.faces, np.int64)[:, [0, 2, 1]]
    return mesh


def _margin_curve():
    ring = _ring(2.0, 0.0)
    return np.vstack([ring, ring[:1]])


def _prep_file(tmp_path, height=5.0, radius=1.6):
    cylinder = trimesh.creation.cylinder(radius=radius, height=height, sections=SEGMENTS)
    cylinder.apply_translation([0.0, 0.0, height / 2.0])      # 跨度 h ∈ [0, height]
    path = tmp_path / "prep.stl"
    cylinder.export(path)
    return str(path)


def _antagonist_plate(z, *, half=4.0, divisions=20):
    grid = np.linspace(-half, half, divisions + 1)
    xx, yy = np.meshgrid(grid, grid)
    vertices = np.stack([xx.ravel(), yy.ravel(), np.full(xx.size, z)], axis=1)
    faces = []
    for row in range(divisions):
        for col in range(divisions):
            a = row * (divisions + 1) + col
            faces += [[a, a + divisions + 1, a + divisions + 2], [a, a + divisions + 2, a + 1]]
    plate = trimesh.Trimesh(vertices=vertices, faces=np.asarray(faces, np.int64), process=False)
    return FitSurface(plate, (-AXIS).tolist())


@pytest.fixture()
def frame():
    return build_patient_frame(_margin_curve(), AXIS)


def _design(prep_path):
    return {"margin_json": json.dumps(_margin_curve().tolist()),
            "parameters_json": json.dumps({"occlusal_target_mm": 0.05}),
            "cutter_path": prep_path, "target_fdi": 46}


def test_feasible_crown_is_axially_scaled_into_the_interval(tmp_path, frame):
    design, crown = _design(_prep_file(tmp_path)), _cone_crown()
    envelope = occlusal_envelope(design, _margin_curve(), AXIS, frame,
                                 surface=_antagonist_plate(6.5), crown_mesh=crown)

    assert envelope["status"] == "FEASIBLE", envelope
    assert envelope["prep_top_mm"] == pytest.approx(5.0, abs=0.02)
    assert envelope["lower_bound_mm"] == pytest.approx(5.5, abs=0.02)
    assert envelope["upper_bound_mm"] == pytest.approx(6.45, abs=0.02)
    assert envelope["available_space_mm"]["min"] == pytest.approx(1.5, abs=0.02)
    assert envelope["minimum_required_space_mm"] == pytest.approx(0.55)
    assert envelope["crown_top_mm"] == pytest.approx(7.0, abs=1e-6)

    plan = plan_axial_placement(envelope, crown_top_mm=envelope["crown_top_mm"])
    assert plan["mechanism"] == "AXIAL_SCALE", plan
    assert 0.75 <= plan["axial_scale"] < 1.0

    placed, report = apply_axial_placement(crown, _margin_curve(), AXIS, plan)
    assert report["applied"] and report["gate_ok"], report
    assert placed is not None
    assert report["crown_top_after_mm"] <= envelope["upper_bound_mm"] + 1e-6
    assert report["crown_top_after_mm"] < report["crown_top_before_mm"]
    assert report["cervical_band_displacement_max_mm"] == 0.0
    # 颈缘环（下环）逐点不动
    assert np.allclose(np.asarray(placed.vertices, float)[:_ring(2.0, 0.0).shape[0]],
                       np.asarray(crown.vertices, float)[:_ring(2.0, 0.0).shape[0]])


def test_short_but_feasible_crown_grows_toward_antagonist(tmp_path, frame):
    """冠顶只要还明显低于 upper，就不能因为“已经可行”而停在半空。"""
    design = _design(_prep_file(tmp_path, height=4.0))
    crown = _cone_crown(top_z=4.7, apex_z=5.2)
    envelope = occlusal_envelope(design, _margin_curve(), AXIS, frame,
                                 surface=_antagonist_plate(6.0), crown_mesh=crown)

    assert envelope["status"] == "FEASIBLE", envelope
    assert envelope["lower_bound_mm"] < envelope["crown_top_mm"] < envelope["upper_bound_mm"]
    plan = plan_axial_placement(envelope, crown_top_mm=envelope["crown_top_mm"])
    assert plan["mechanism"] == "AXIAL_SCALE", plan
    assert 1.0 < plan["axial_scale"] <= 1.35

    placed, report = apply_axial_placement(crown, _margin_curve(), AXIS, plan)
    assert placed is not None and report["applied"] and report["gate_ok"], report
    assert report["crown_top_after_mm"] == pytest.approx(envelope["upper_bound_mm"], abs=1e-6)
    assert report["crown_top_after_mm"] > report["crown_top_before_mm"]
    assert report["cervical_band_displacement_max_mm"] == 0.0
    np.testing.assert_allclose(placed.vertices[:SEGMENTS], crown.vertices[:SEGMENTS], atol=0)


def test_antagonist_deadband_does_not_churn_geometry():
    envelope = {"status": "FEASIBLE", "lower_bound_mm": 5.0, "upper_bound_mm": 6.0}
    plan = plan_axial_placement(envelope, crown_top_mm=5.95)
    assert plan["mechanism"] == "TRANSLATE" and plan["feasible"], plan
    assert plan["axial_scale"] == 1.0


def test_excessive_axial_growth_is_refused():
    envelope = {"status": "FEASIBLE", "lower_bound_mm": 4.5, "upper_bound_mm": 8.0}
    plan = plan_axial_placement(envelope, crown_top_mm=5.0)
    assert plan["mechanism"] == "REQUIRED_AXIAL_SCALE_OUT_OF_RANGE", plan
    assert not plan["feasible"] and plan["axial_scale"] > 1.35


def test_insufficient_occlusal_space_is_reported_not_carved(tmp_path, frame):
    design, crown = _design(_prep_file(tmp_path)), _cone_crown()
    envelope = occlusal_envelope(design, _margin_curve(), AXIS, frame,
                                 surface=_antagonist_plate(5.2), crown_mesh=crown)

    assert envelope["status"] == "INSUFFICIENT_OCCLUSAL_SPACE", envelope
    assert "INSUFFICIENT_OCCLUSAL_SPACE" in envelope["reasons"]
    assert envelope["lower_bound_mm"] > envelope["upper_bound_mm"]
    assert envelope["columns_below_required"] == envelope["measured_columns"]

    plan = plan_axial_placement(envelope, crown_top_mm=envelope["crown_top_mm"])
    assert plan["mechanism"] == "INSUFFICIENT_OCCLUSAL_SPACE" and not plan["feasible"]
    placed, report = apply_axial_placement(crown, _margin_curve(), AXIS, plan)
    assert placed is None and report["applied"] is False


def test_too_large_axial_scale_is_refused_instead_of_squashing(tmp_path, frame):
    """需要压到 75% 以下时不算"合理范围内的缩放" ⇒ 应换更短的模板，而不是硬压。"""
    design, crown = _design(_prep_file(tmp_path, height=5.0)), _cone_crown(top_z=12.0, apex_z=14.0)
    envelope = occlusal_envelope(design, _margin_curve(), AXIS, frame,
                                 surface=_antagonist_plate(8.0), crown_mesh=crown)
    assert envelope["status"] == "FEASIBLE", envelope
    plan = plan_axial_placement(envelope, crown_top_mm=envelope["crown_top_mm"])
    assert plan["mechanism"] == "REQUIRED_AXIAL_SCALE_OUT_OF_RANGE", plan
    assert plan["axial_scale"] < 0.75
    placed, report = apply_axial_placement(crown, _margin_curve(), AXIS, plan)
    assert placed is None and report["applied"] is False


def test_coarse_height_fit_keeps_margin_and_leaves_small_residual_for_fine(tmp_path, frame, monkeypatch):
    from app.services import contact_measurement
    design, crown = _design(_prep_file(tmp_path)), _cone_crown()
    surface = _antagonist_plate(6.5)
    monkeypatch.setattr(contact_measurement, 'confirmed_antagonist_surface',
                        lambda *a, **k: (surface, {'revision_id': 'bite'}, None))
    envelope = occlusal_envelope(design, _margin_curve(), AXIS, frame,
                                 surface=surface, crown_mesh=crown)
    fitted, report = coarse_height_fit(design, crown, _margin_curve(), AXIS, frame, envelope)
    assert fitted is not None and report['status'] == 'APPLIED', report
    assert report['after_min_clearance_mm'] >= -0.2
    assert report['margin_displacement_mm'] == 0
    assert report['md_scale'] == report['bl_scale'] == 1.0
    np.testing.assert_allclose(fitted.vertices[:SEGMENTS], crown.vertices[:SEGMENTS], atol=0)
