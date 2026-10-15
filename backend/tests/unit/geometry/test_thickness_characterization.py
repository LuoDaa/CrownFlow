"""Thickness 域测量的**特征测试**（B1 第二步）。

结论先行（B1 不改变行为，只把行为写下来）：**thickness 的"符号"与 Contact / Occlusion
都不是同一个量**，所以它保留自己的 ``ThicknessMeasurement`` 领域对象，只共用
validity / closest primitive：

| 维度 | Contact（`clearance_detail` + `FitSurface`） | Occlusion | Thickness（`SurfaceQuery`） |
|---|---|---|---|
| 符号来源 | 闭实体实体内外 / 人工外向法向 | 一次性质心定向 | **平滑法向**（最近面三顶点面积加权法向均值） |
| validity | 边界 + 折痕 | 仅边界安全 | **仅边界安全**（无边界 ⇒ 全部有效） |
| 额外如实计数 | `ambiguous_count` | `ambiguous` | **`flipped`**（单面法向与平滑法向反向的点） |
| 穿透量语义 | `max(0, −signed)`（真穿透深度） | `max(target_gap − signed, 0)`（相对目标的亏欠） | `signed − target`（相对目标的余量，负 = 冲突） |
| 输入契约 | 任意网格 | 任意网格 + 颈环（锁定区需要） | **外表面必须有开放颈缘**（否则无法定位 margin） |
| 相交（crossing） | 服务层 `surface_crossings` | **不负责** | **本模块负责**（`self_intersections` / `input_qc`） |

最后一行尤其重要：**"谁负责相交审计"在三个域里答案不同**，混起来读会得出错误结论。
"""

from __future__ import annotations

import numpy as np
import pytest
import trimesh

from app.geometry.crown_outer.contact import NeighborPatch
from app.geometry.crown_outer.patient_frame import PatientFrame
from app.geometry.crown_outer.thickness import (SurfaceQuery, ThicknessError,
                                                ThicknessParameters, audit_thickness,
                                                input_qc, self_intersections)

FRAME = PatientFrame(np.zeros(3), np.array([0.0, 0.0, 1.0]),
                     np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]))


def flat_grid(*, n: int = 5, spacing: float = 1.0, z: float = 0.0):
    """z=const 的平面网格（开放边界）。绕向决定面法向指向，测试里不假设方向。"""
    xs = np.arange(n) * spacing
    verts = np.array([[x, y, z] for x in xs for y in xs], dtype=float)
    faces = []
    for i in range(n - 1):
        for j in range(n - 1):
            a = i * n + j
            faces += [[a, a + 1, a + n + 1], [a, a + n + 1, a + n]]
    return verts, np.asarray(faces, dtype=np.int64)


def shell_cap(radius: float, *, subdivisions: int = 3) -> trimesh.Trimesh:
    """带开放颈缘的冠形壳（球冠）—— ``audit_thickness`` 的合法输入。"""
    mesh = trimesh.creation.icosphere(subdivisions=subdivisions, radius=radius)
    keep = np.asarray(mesh.triangles_center)[:, 2] > -radius * 0.55
    return mesh.submesh([np.flatnonzero(keep)], append=True, repair=False)


def open_ring_edges(mesh) -> int:
    f = np.asarray(mesh.faces)
    edges = np.sort(np.vstack((f[:, :2], f[:, 1:], f[:, [2, 0]])), axis=1)
    _, counts = np.unique(edges, axis=0, return_counts=True)
    return int((counts == 1).sum())


# ---------------------------------------------------------------- 1. 符号来源

def test_sign_uses_smoothed_normal_not_single_face_normal():
    """符号由**平滑法向**决定：翻一个三角形的绕向不会让整片符号翻错，且被如实计数。

    这就是 `thickness.py` 写明要用平滑法向的理由：Occlusion v1 留下约 2% 翻面，
    单面法向会让翻面处的符号整体翻错（实测 8301 采样点里 244 点属此类）。
    """
    verts, faces = flat_grid()
    sq = SurfaceQuery(verts, faces, boundary_safety_mm=0.0, max_query_edge_mm=0.0)
    probes = np.c_[verts[:, :2], np.full(len(verts), 0.5)]     # 平面上方 0.5 mm
    _p, _f, signed, _d, valid, flipped = sq.query(probes)

    assert valid.all()
    assert np.allclose(np.abs(signed), 0.5, atol=1e-9), "幅值 = 到平面的精确距离"
    assert (signed > 0).all() or (signed < 0).all(), "同一平面上的符号必须一致"
    assert not flipped.any(), "绕向一致时不得报翻面"

    broken = faces.copy()
    broken[0] = broken[0][::-1]
    sq2 = SurfaceQuery(verts, broken, boundary_safety_mm=0.0, max_query_edge_mm=0.0)
    _p2, _f2, signed2, _d2, valid2, flipped2 = sq2.query(probes)
    assert np.array_equal(np.sign(signed2), np.sign(signed)), \
        "翻一个三角形不得整体翻符号（平滑法向是面积加权）"
    assert flipped2.any(), "单面法向与平滑法向反向处必须如实标记"

    sq2.normal_sign = -1.0
    _p3, _f3, signed3, _d3, _v3, _fl3 = sq2.query(probes)
    assert np.allclose(signed3, -signed2), "normal_sign 乘子必须被尊重（外部定向用）"


def test_validity_is_boundary_safety_only():
    """validity 只剔除**开放边界安全距离内**的采样；不看实体内外、没有折痕掩码。"""
    verts, faces = flat_grid()
    sq = SurfaceQuery(verts, faces, boundary_safety_mm=0.5, max_query_edge_mm=0.0)
    probes = np.array([[0.05, 2.0, 0.2], [2.0, 2.0, 0.2]])    # 边界带内 / 中间
    _p, _f, _s, _d, valid, _fl = sq.query(probes)
    assert not bool(valid[0]), "边界带内判为无效"
    assert bool(valid[1]), "中间区域有效"

    inner = trimesh.creation.icosphere(subdivisions=2, radius=2.0)
    closed = SurfaceQuery(np.asarray(inner.vertices), np.asarray(inner.faces),
                          boundary_safety_mm=0.5, max_query_edge_mm=0.0)
    _p2, _f2, _s2, _d2, valid2, _fl2 = closed.query(np.asarray(inner.vertices))
    assert valid2.all(), "闭合网格没有开放边界 ⇒ 全部有效（宽松是刻意的）"


# ---------------------------------------------------------------- 2. 相交职责

def test_crossing_audit_is_owned_by_this_module():
    """与 Occlusion 相反：**相交审计属于 thickness 的职责**（`self_intersections`/`input_qc`）。"""
    a = trimesh.Trimesh([[-1, -1, 0], [1, -1, 0], [0, 1, 0]], [[0, 1, 2]], process=False)
    b = trimesh.Trimesh([[0, -0.5, -1], [0, -0.5, 1], [0, 0.5, 0]], [[0, 1, 2]], process=False)
    verts = np.vstack([a.vertices, b.vertices])
    faces = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int64)

    report = self_intersections(verts, faces)
    assert report["self_intersections"] >= 1, report
    assert report.get("self_intersections_truncated") is False
    assert input_qc(verts, faces)["nonmanifold_edges"] >= 0

    clean = trimesh.creation.icosphere(subdivisions=1, radius=1.0)
    assert self_intersections(np.asarray(clean.vertices),
                              np.asarray(clean.faces))["self_intersections"] == 0


# ---------------------------------------------------------------- 3. 输入契约

def test_outer_surface_must_have_an_open_cervical_ring():
    """外表面必须带开放颈缘 —— 否则无法定位 margin，必须报错而不是硬算。"""
    closed = trimesh.creation.icosphere(subdivisions=2, radius=3.0)
    with pytest.raises(ThicknessError) as err:
        audit_thickness(closed, closed, FRAME,
                        params=ThicknessParameters(min_thickness_mm=0.5, max_query_edge_mm=0.0))
    assert "颈缘" in str(err.value)

    cap = shell_cap(3.0)
    assert open_ring_edges(cap) > 0, "夹具本身要有开放颈缘"


# ---------------------------------------------------------------- 4. 薄判定与归因

def _thin_params(**kw) -> ThicknessParameters:
    base = dict(min_thickness_mm=0.5, boundary_safety_mm=0.0, max_query_edge_mm=0.0,
                ray_step_mm=0.05, ray_max_mm=1.2)
    base.update(kw)
    return ThicknessParameters(**base)


def test_thin_judgement_and_unverified_conflicts_are_reported_honestly():
    """壳厚 0.4 < 最小 0.5 ⇒ 报薄；**没有 patch 时两种冲突归因必须标"未验"**。"""
    outer, inner = shell_cap(3.0), shell_cap(2.6)
    res = audit_thickness(outer, inner, FRAME, params=_thin_params())

    assert res.summary["median_thickness_mm"] == pytest.approx(0.4, abs=0.02)
    assert res.summary["thin_vertex_count"] > 0
    assert res.summary["n_evaluated"] == res.summary["n_samples"]
    assert res.thin_area.get("CONTACT_THICKNESS_CONFLICT", 0.0) == 0.0
    assert res.thin_area.get("OCCLUSION_THICKNESS_CONFLICT", 0.0) == 0.0
    assert any("CONTACT_THICKNESS_CONFLICT" in n and "未验" in n for n in res.notes), res.notes
    assert any("OCCLUSION_THICKNESS_CONFLICT" in n and "未验" in n for n in res.notes), res.notes
    assert res.status != "PASS_ENGINEERING", "存在薄区时不得报通过"


def test_conflict_attribution_uses_target_relative_clearance():
    """归因口径：``signed − target < clearance_margin_mm``（严格小于）。

    把邻牙面放在"推进 deficit 后仍不足目标间隙"的距离上 ⇒ 必须归因成接触冲突；
    放远一点 ⇒ 归因回"可修"。两处都不动几何，只测。
    """
    outer, inner = shell_cap(3.0), shell_cap(2.6)
    deficit = 0.5 - 0.4                      # 需要推进的厚度缺口
    target = 0.02
    margin = 0.02

    def plate(bottom_z: float) -> NeighborPatch:
        box = trimesh.creation.box(extents=(4.0, 4.0, 0.4))
        box.apply_translation([0, 0, bottom_z + 0.2])
        return NeighborPatch(box.vertices, box.faces)

    def attribution(bottom_z: float) -> dict:
        res = audit_thickness(outer, inner, FRAME,
                              contact_patches={"mesial": plate(bottom_z)},
                              contact_target_mm=target,
                              params=_thin_params(clearance_margin_mm=margin))
        return res.thin_area

    near = attribution(3.0 + deficit + target + margin * 0.5)     # 差 margin/2 ⇒ 冲突
    far = attribution(3.0 + deficit + target + margin * 5)        # 远超 margin ⇒ 可修
    assert near.get("CONTACT_THICKNESS_CONFLICT", 0.0) > 0.0, near
    assert far.get("CONTACT_THICKNESS_CONFLICT", 0.0) == 0.0, far
    assert far.get("REPAIRABLE_THIN", 0.0) > 0.0, far


def test_thin_judgement_never_runs_on_invalid_samples():
    """薄判定只作用于**有效采样**：无效采样既不算薄、也不进分母。

    注意 validity 比的是"到**边界折线采样点**（0.2 mm 间距）的距离"，不是到连续曲线的
    精确距离；所以把安全距离拉到 50 mm 时是**绝大多数**而非全部采样被剔掉
    （实测 521 里剔 495）。断言用的是恒等式而不是具体条数，避免把这个实现细节写死。
    """
    outer, inner = shell_cap(3.0), shell_cap(2.6)
    loose = audit_thickness(outer, inner, FRAME, params=_thin_params(boundary_safety_mm=0.0))
    assert loose.summary["n_boundary_invalid"] == 0
    assert loose.summary["n_evaluated"] == loose.summary["n_samples"]

    strict = audit_thickness(outer, inner, FRAME, params=_thin_params(boundary_safety_mm=50.0))
    s = strict.summary
    assert s["n_boundary_invalid"] > 0
    assert s["n_evaluated"] == s["n_samples"] - s["n_boundary_invalid"], s
    assert s["thin_vertex_count"] <= s["n_evaluated"], "薄区不得超出有效采样"
    assert strict.status != "PASS_ENGINEERING"
