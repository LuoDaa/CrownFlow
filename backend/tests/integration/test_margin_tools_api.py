"""边缘线工具的 API 回归：M1 自动候选 / 凸凹特征吸附 / 悬停剖面。

用**合成基台**（下粗上细圆柱并集，带明确的外缘棱与内角）走真实接口，
不碰任何真实病例库。三条契约被钉死：

1. ``/auto-margin-m1`` 是 **fail-loud**：没有可信闭环时 ``points`` 为空且
   ``status != AUTO_ACCEPTED``（绝不返回一条"看起来合法"的曲线）；
2. ``/margin-snap`` 的两种模式**行为不同**：SUPRA 吸到外缘棱 → 半径变大，
   SUB 停在凹角 → 半径基本不变；
3. ``/section-preview`` 返回过给定点的剖面折线（3D 与 2D 对应）。
"""
from __future__ import annotations

import io
import math

import numpy as np
import pytest
import trimesh
from fastapi.testclient import TestClient


def _step_die(max_edge=0.15):
    lo = trimesh.creation.cylinder(radius=3.0, height=1.0, segments=64)
    lo.apply_translation([0, 0, -0.5])
    up = trimesh.creation.cylinder(radius=2.2, height=1.0, segments=64)
    up.apply_translation([0, 0, 0.5])
    body = trimesh.boolean.union([lo, up], engine="manifold")
    if max_edge:
        v, f = trimesh.remesh.subdivide_to_size(body.vertices, body.faces, max_edge=max_edge)
        body = trimesh.Trimesh(v, f, process=True)
    return body


def _upload_die(client, case_id, mesh):
    stl = mesh.export(file_type="stl")
    r = client.post(f"/api/cases/{case_id}/assets/support",
                    files={"file": ("synthetic_die.stl", stl, "model/stl")})
    assert r.status_code == 200, r.text
    return r.json()


def _fixture(client_factory, mesh):
    client = client_factory()
    case = client.post("/api/cases", json={"name": "margin-tools"}).json()["id"]
    _upload_die(client, case, mesh)
    design = client.post("/api/designs",
                         json={"case_id": case, "target_fdi": 46,
                               "support_type": "prep", "draft": True}).json()["id"]
    return client, design


@pytest.fixture()
def dense_client(isolated_db):
    from app.main import app
    return TestClient(app)


def _ring(radius, n=48):
    return [[radius * math.cos(2 * math.pi * i / n),
             radius * math.sin(2 * math.pi * i / n), 0.0] for i in range(n)]


def test_section_preview_returns_profile_through_point(dense_client):
    client, design = _fixture(lambda: dense_client, _step_die())
    point = [3.0, 0.0, -0.3]
    r = client.post(f"/api/designs/{design}/section-preview",
                    json={"point": point, "span_mm": 8.0})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["polylines_3d"], "应至少切出一段剖面折线"
    assert len(body["polylines_3d"]) == len(body["polylines_uv"])
    for poly3d, poly_uv in zip(body["polylines_3d"], body["polylines_uv"]):
        assert len(poly3d) == len(poly_uv) >= 2
        assert all(len(p) == 3 for p in poly3d) and all(len(p) == 2 for p in poly_uv)
    u = np.asarray(body["basis"]["u"], float)
    assert abs(float(np.linalg.norm(u)) - 1.0) < 1e-6, "u 应为单位向量（就位轴）"
    assert np.allclose(np.asarray(body["plane"]["origin"], float), point)


def test_margin_snap_supra_moves_out_sub_stays(dense_client):
    client, design = _fixture(lambda: dense_client, _step_die(max_edge=0.2))
    loop = _ring(2.2)
    base = 2.2

    sub = client.post(f"/api/designs/{design}/margin-snap",
                      json={"points": loop, "mode": "SUB", "radius_mm": 1.0})
    supra = client.post(f"/api/designs/{design}/margin-snap",
                        json={"points": loop, "mode": "SUPRA", "radius_mm": 1.0})
    assert sub.status_code == 200 and supra.status_code == 200, (sub.text, supra.text)
    sub_body, supra_body = sub.json(), supra.json()

    def mean_radius(points):
        P = np.asarray(points, float)
        return float(np.mean(np.hypot(P[:, 0], P[:, 1])))

    assert sub_body["status"] == "APPLIED" and supra_body["status"] == "APPLIED"
    assert mean_radius(supra_body["points"]) - base > 0.3, "SUPRA 应吸到外缘棱"
    assert abs(mean_radius(sub_body["points"]) - base) < 0.1, "SUB 应停在凹角"
    assert supra_body["mode"] == "SUPRA" and sub_body["mode"] == "SUB"


def test_auto_margin_m1_is_fail_loud_and_reports_provenance(dense_client):
    """合成粗网格上引擎不保证给出可信闭环 —— 但契约必须成立。"""
    client, design = _fixture(lambda: dense_client, _step_die(max_edge=None))
    r = client.post(f"/api/designs/{design}/auto-margin-m1",
                    json={"mode": "SUPRA", "radius_mm": 1.0, "apply_snap": True})
    assert r.status_code == 200, r.text
    body = r.json()
    for key in ("points", "status", "reason", "config_id", "axis", "axis_source",
                "mode", "requires_decision", "message"):
        assert key in body, f"缺少 {key}"
    assert body["axis_source"] == "provisional_sidewall_fit", "未给轴时应如实标注降级来源"
    assert abs(float(np.linalg.norm(body["axis"])) - 1.0) < 1e-6
    if not body["points"]:
        assert body["status"] != "AUTO_ACCEPTED", "无闭环时不得标 AUTO_ACCEPTED"
        assert body["reason"], "fail-loud 必须给出原因"
    else:
        assert len(body["points"]) >= 12
