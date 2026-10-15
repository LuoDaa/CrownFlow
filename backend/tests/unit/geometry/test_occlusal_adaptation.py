"""咬合适配机制测试（合成几何，不依赖病例资产）。

覆盖：
1. **能减材**：冠顶高出对颌时，适配把咬合面压到目标间隙之下，且颈缘带 / 内表面位移精确为 0；
2. **不可行如实失败**：需要下压量超过咬合区自身高度时报 ``REQUIRED_DROP_EXCEEDS_OCCLUSAL_DOMAIN``，
   状态 ``OCCLUSAL_ADAPTATION_FAILED``，且**不返回被改坏的网格**。
"""
import json

import numpy as np
import pytest
import trimesh

from app.geometry.crown_outer.auto_fit_adapter import FitSurface
from app.geometry.crown_outer.patient_frame import build_patient_frame
from app.services import contact_measurement
from app.services.occlusal_adaptation import adapt_occlusal

SEGMENTS = 48
AXIS = np.array([0.0, 0.0, 1.0])


def _ring(radius, z):
    angle = np.linspace(0.0, 2.0 * np.pi, SEGMENTS, endpoint=False)
    return np.stack([radius * np.cos(angle), radius * np.sin(angle), np.full(SEGMENTS, z)], axis=1)


def _crown_mesh(*, bottom_z=0.0, top_z=6.0, apex_z=8.0, radius=2.0, top_radius=1.5):
    """下环(z=bottom) → 上环 → 顶点 的锥形冠；下环即 Margin。"""
    bottom = _ring(radius, bottom_z)
    top = _ring(top_radius, top_z)
    apex = np.array([[0.0, 0.0, apex_z]])
    vertices = np.vstack([bottom, top, apex])
    faces = []
    for index in range(SEGMENTS):
        nxt = (index + 1) % SEGMENTS
        b0, b1 = index, nxt
        t0, t1 = SEGMENTS + index, SEGMENTS + nxt
        faces += [[b0, b1, t1], [b0, t1, t0]]
        faces += [[t0, t1, 2 * SEGMENTS]]
    mesh = trimesh.Trimesh(vertices=vertices, faces=np.asarray(faces, np.int64), process=False)
    outward = np.asarray(mesh.triangles_center, float) * np.array([1.0, 1.0, 0.0])
    if float(np.mean(np.einsum("ij,ij->i", np.asarray(mesh.face_normals, float), outward))) < 0:
        mesh.faces = np.asarray(mesh.faces, np.int64)[:, [0, 2, 1]]
    return mesh


def _margin_curve():
    ring = _ring(2.0, 0.0)
    return np.vstack([ring, ring[:1]])


def _antagonist_plate(z, *, half=4.0, divisions=24):
    """一块**加密**的平板（真实对颌扫描是密网格；稀疏的 4 角平板会在柱里留下空洞）。"""
    grid = np.linspace(-half, half, divisions + 1)
    xx, yy = np.meshgrid(grid, grid)
    vertices = np.stack([xx.ravel(), yy.ravel(), np.full(xx.size, z)], axis=1)
    faces = []
    for row in range(divisions):
        for col in range(divisions):
            a = row * (divisions + 1) + col
            b, c, d = a + 1, a + divisions + 1, a + divisions + 2
            faces += [[a, c, d], [a, d, b]]
    plate = trimesh.Trimesh(vertices=vertices, faces=np.asarray(faces, np.int64), process=False)
    return FitSurface(plate, (-AXIS).tolist())


@pytest.fixture()
def design():
    return {"margin_json": json.dumps(_margin_curve().tolist()),
            "parameters_json": json.dumps({"occlusal_target_mm": 0.05}),
            "cutter_path": None}


def _patch_antagonist(monkeypatch, surface):
    monkeypatch.setattr(contact_measurement, "confirmed_antagonist_surface",
                        lambda *a, **k: (surface, {"revision_id": "test-bite"}, None))


@pytest.fixture()
def frame():
    return build_patient_frame(_margin_curve(), AXIS)


def test_carve_removes_interference_and_locks_the_cervical_band(design, frame, monkeypatch):
    mesh = _crown_mesh()
    _patch_antagonist(monkeypatch, _antagonist_plate(7.6))
    adapted, report = adapt_occlusal(design, mesh, frame, {"revision_id": "test-frame"})

    assert report["status"] == "OCCLUSAL_ADAPTED", report
    assert report["analysis_mesh_source"] == "final"
    assert report["vertex_count"] == len(mesh.vertices)
    occlusion = report["occlusion"]
    assert occlusion["before_min_distance_mm"] == pytest.approx(-0.4, abs=0.02)
    assert occlusion["after_min_distance_mm"] >= -0.1
    assert occlusion["penetrating_vertices_before"] > 0
    assert occlusion["penetrating_vertices_after"] == 0
    assert report["mesh_sha256_before"] != report["mesh_sha256_after"]

    protected = report["protected"]
    assert protected["locked_displacement_max_mm"] == 0.0
    assert protected["margin_band_displacement_max_mm"] == 0.0
    assert protected["intaglio_displacement_max_mm"] == 0.0
    assert protected["moved_vertex_count"] > 0

    assert adapted is not None
    # 颈缘（下环）逐点不动；顶点确实降下来
    assert np.allclose(np.asarray(adapted.vertices, float)[:_ring(2.0, 0.0).shape[0]],
                       np.asarray(mesh.vertices, float)[:_ring(2.0, 0.0).shape[0]])
    assert float(np.asarray(adapted.vertices, float)[-1, 2]) < float(np.asarray(mesh.vertices, float)[-1, 2])


def test_infeasible_drop_reports_failure_without_touching_the_crown(design, frame, monkeypatch):
    mesh = _crown_mesh()
    _patch_antagonist(monkeypatch, _antagonist_plate(1.0))
    adapted, report = adapt_occlusal(design, mesh, frame, {"revision_id": "test-frame"})

    assert adapted is None
    assert report["status"] == "OCCLUSAL_ADAPTATION_FAILED"
    assert "REQUIRED_DROP_EXCEEDS_OCCLUSAL_DOMAIN" in report["reasons"]
    assert report["required_drop"]["max_mm"] > report["domain"]["occlusal_domain_extent_mm"]
    assert report["occlusion"]["after_min_distance_mm"] is None


def test_missing_antagonist_is_an_input_gap_not_an_algorithm_error(design, frame, monkeypatch):
    mesh = _crown_mesh()
    monkeypatch.setattr(contact_measurement, "confirmed_antagonist_surface",
                        lambda *a, **k: (None, {}, {"status": "NOT_ASSESSED",
                                                   "reason": "OCCLUSION_REGISTRATION_UNVERIFIED"}))
    adapted, report = adapt_occlusal(design, mesh, frame, {"revision_id": None})

    assert adapted is None
    assert report["status"] == "OCCLUSAL_ADAPTATION_FAILED"
    assert report["reasons"] == ["ANALYSIS_INPUT_MISSING"]
    assert report["reason"] == "OCCLUSION_REGISTRATION_UNVERIFIED"
