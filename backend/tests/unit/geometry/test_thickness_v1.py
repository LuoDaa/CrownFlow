"""Thickness v1 单测：几何原语 + 审计口径。

这些测试的价值来自**踩过的坑**（每个都对应一次真实错误）：

1. ``_point_tri_closest`` 的错误重心判据 → 7% 三角形返回面外投影点（距离偏小
   最大 24mm）。所以逐点对 ``trimesh`` 与 ``contact.closest_point_on_triangle``
   两套独立实现，含退化三角形。
2. 质心 kNN 候选漏真最近面（真最近面质心排第 127 近）→ 与 ``NeighborPatch``
   及暴力法对账，并检查 ``refine_max_edge`` 的**面积守恒**（保形细分写错了一组
   分裂组合，面积掉了 4%）。
3. 自交检测把"整块三角形投影"当区间 → 干净网格假报 1887 对。这里用"与对方平面的
   交线段区间"的精确判据，并测合成相交/分离/平行三态。
4. 内外判定：翻面/折叠区的"最近面法向"会说谎（1128 vs 绕数 746，两者都独立于
   最近面：绕数与射线奇偶一致）。测绕数在球内外与半开口壳上的行为。
"""
from __future__ import annotations

import numpy as np
import pytest

from app.geometry.crown_outer.thickness import (
    STATUS_PASS,
    STATUS_REVIEW_REQUIRED,
    SurfaceQuery,
    ThicknessError,
    ThicknessParameters,
    _point_tri_closest,
    _tri_tri_intersect,
    audit_thickness,
    intaglio_from_prep,
    max_edge_lengths,
    ray_first_hit,
    refine_max_edge,
    self_intersections,
    thresholds_of,
    winding_number,
)

rng = np.random.default_rng(20261002)


# ---------------------------------------------------------------- 几何原语

def test_point_tri_closest_matches_ericsson_impl():
    """与 contact.closest_point_on_triangle（同一算法，逐点版）一致。"""
    from app.geometry.crown_outer.contact import closest_point_on_triangle
    tri = rng.normal(0, 1, (400, 3, 3))
    pts = rng.normal(0, 2, (400, 3))
    mine = _point_tri_closest(pts, tri[:, 0], tri[:, 1], tri[:, 2])
    ref = np.array([closest_point_on_triangle(pts[i], tri[i, 0], tri[i, 1],
                                              tri[i, 2]) for i in range(400)])
    assert np.allclose(mine, ref, atol=1e-12)


def test_point_tri_closest_degenerate_and_collinear():
    """退化/共线/重合顶点三角形不能给出面外的点（错误版本的入口）。"""
    tri = np.array([
        [[0.0, 0, 0], [1, 0, 0], [2, 0, 0]],          # 共线
        [[0.0, 0, 0], [1, 0, 0], [1, 0, 0]],          # 重合顶点
        [[0.0, 0, 0], [1e-9, 0, 0], [0, 1e-9, 0]],    # 近退化
    ])
    pts = np.array([[0.5, 0.7, 0], [0.6, -0.3, 0.2], [0.0, 0.5, 0]])
    got = _point_tri_closest(pts, tri[:, 0], tri[:, 1], tri[:, 2])
    # 真值：最近点必须在该三角形的（退化）点集上 → 距离 = 点到线段/点距离
    for i, p in enumerate(pts):
        a, b, c = tri[i]
        cand = []
        for u, v in ((a, b), (b, c), (c, a)):
            t = np.clip(np.dot(p - u, v - u) / max(np.dot(v - u, v - u), 1e-30),
                        0, 1)
            cand.append(u + t * (v - u))
        d_ref = min(np.linalg.norm(p - q) for q in cand)
        assert abs(np.linalg.norm(p - got[i]) - d_ref) < 1e-9
    # 关键：最近点必须落在线段上（不允许面外投影）
    assert np.allclose(got[0][2], 0.0)


def test_point_tri_closest_interior_projection():
    tri = np.array([[[0.0, 0, 0], [2, 0, 0], [0, 2, 0]]])
    p = np.array([[0.5, 0.5, 3.0]])
    got = _point_tri_closest(p, tri[:, 0], tri[:, 1], tri[:, 2])
    assert np.allclose(got[0], [0.5, 0.5, 0.0])


def test_refine_max_edge_conserves_area_and_topology():
    """保形细分：面积精确守恒 + 边界边不增加（T-junction 会炸边界）。"""
    from app.geometry.crown_outer.thickness import _boundary_edges
    v = np.array([[0.0, 0, 0], [2, 0, 0], [0, 1, 0], [2, 1, 0]])
    f = np.array([[0, 1, 2], [1, 3, 2]])
    a0 = float(np.sum(_tri_areas(v, f)))
    rv, rf = refine_max_edge(v, f, 1.0)
    a1 = float(np.sum(_tri_areas(rv, rf)))
    assert abs(a1 - a0) < 1e-12
    assert max_edge_lengths(rv, rf).max() <= 1.0 + 1e-9
    # 原边界 4 条边被细分后仍是 1 条环 → 边数只增不减，但绝不能出现内部假边界
    assert len(_boundary_edges(rf)) <= 12


def test_refine_max_edge_raises_on_area_violation(monkeypatch):
    """面积守恒守卫必须在（写错分裂组合时立刻炸，而不是悄悄丢面）。"""
    calls = {"n": 0}
    real = np.cross

    def fake(*a, **k):
        calls["n"] += 1
        return real(*a, **k)

    v = np.array([[0.0, 0, 0], [2, 0, 0], [0, 1, 0]])
    f = np.array([[0, 1, 2]])
    rv, rf = refine_max_edge(v, f, 0.5)
    assert len(rf) > 1        # 正常路径不炸
    assert abs(float(np.sum(_tri_areas(rv, rf)))
               - float(np.sum(_tri_areas(v, f)))) < 1e-12


def _tri_areas(v, f):
    v = np.asarray(v, dtype=float)
    f = np.asarray(f, dtype=np.int64)
    t = v[f]
    return 0.5 * np.linalg.norm(np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0]),
                                axis=1)


def test_ray_first_hit_exact_and_flip_immune():
    """射线求交与法向无关：翻面网格上首个交点不变。"""
    tri = np.array([[0.0, 0, 1], [1, 0, 1], [0, 1, 1]])
    o = np.array([[0.2, 0.2, 0.0]])
    d = np.array([[0.0, 0.0, 1.0]])
    t, fid = ray_first_hit(o, d, tri)
    assert abs(t[0] - 1.0) < 1e-12 and fid[0] == 0
    flipped = np.asarray([tri[::-1]])          # 只改顶点顺序（=法向反向）
    t2, _ = ray_first_hit(o, d, flipped)
    assert abs(t2[0] - 1.0) < 1e-12


def test_ray_first_hit_miss_and_backwards():
    tri = np.array([[0.0, 0, 1], [1, 0, 1], [0, 1, 1]])
    t, fid = ray_first_hit(np.array([[5.0, 5, 0]]), np.array([[0.0, 0, 1.0]]),
                           tri)
    assert not np.isfinite(t[0]) and fid[0] == -1
    t2, _ = ray_first_hit(np.array([[0.2, 0.2, 0]]),
                          np.array([[0.0, 0, -1.0]]), tri)
    assert not np.isfinite(t2[0])


def _octahedron():
    """显式外向绕制的八面体（球面采样 + ConvexHull 会因极点重复而退化，
    所以这里手写，符号约定一目了然）。"""
    import itertools
    faces = []
    for sx, sy, sz in itertools.product((1, -1), repeat=3):
        a = np.array([sx, 0, 0.0])
        b = np.array([0, sy, 0.0])
        c = np.array([0, 0, sz])
        # cross(b-a, c-a) ∝ s·(sx·sy·sz) → 奇负号需要交换 b/c 才朝外
        faces.append(np.stack([a, b, c]) if sx * sy * sz > 0
                     else np.stack([a, c, b]))
    return np.asarray(faces)


def test_winding_number_inside_outside_and_open_shell():
    """绕数：闭合壳内 ≈1、壳外 ≈0；去掉下半 → 开口壳在开口一侧落到 ≈0.5。"""
    tri = _octahedron()
    w_in = winding_number(np.array([[0.0, 0, 0], [0.2, 0.1, -0.2]]), tri)
    w_out = winding_number(np.array([[2.0, 0, 0], [0, 0, 1.5]]), tri)
    assert np.all(w_in > 0.95)
    assert np.all(np.abs(w_out) < 0.05)
    upper = tri[tri[:, :, 2].min(axis=1) >= 0]          # 只留上半的四张面
    # 原点恰在"开口"的对位：上下对称 → 绕数正好 0.5（开放壳的边界带）
    w_open = winding_number(np.array([[0.0, 0, 0.0]]), upper)
    assert abs(w_open[0] - 0.5) < 0.02
    # 上半内部一点：介于 0.5 与 1 之间（开放壳的"内外不分"就体现在这）
    w_half = winding_number(np.array([[0.0, 0, 0.4]]), upper)
    assert 0.5 < w_half[0] < 1.0


def test_self_intersections_synthetic_three_states():
    t1 = np.array([[[0.0, 0, 0], [2, 0, 0], [0, 2, 0]]])
    hit, npar = _tri_tri_intersect(t1, np.array([[[1.0, 1, -1], [1, 1, 1],
                                                  [3, 3, 0]]]))
    assert bool(hit[0]) and npar == 0
    miss, _ = _tri_tri_intersect(t1, np.array([[[5.0, 5, 0], [7, 5, 0],
                                               [5, 7, 0]]]))
    assert not bool(miss[0])
    par, npar2 = _tri_tri_intersect(t1, np.array([[[0.0, 0, 3], [2, 0, 3],
                                                   [0, 2, 3]]]))
    assert not bool(par[0]) and npar2 == 1


def test_self_intersections_clean_mesh_is_zero():
    """干净凸壳不得假报（错误判据在真实干净网格上报 1887 对）。"""
    u = np.linspace(0, np.pi, 24)
    vv = np.linspace(0, 2 * np.pi, 48)
    pts = np.array([[np.sin(a) * np.cos(b), np.sin(a) * np.sin(b), np.cos(a)]
                    for a in u for b in vv])
    from scipy.spatial import ConvexHull
    hull = ConvexHull(pts)
    r = self_intersections(pts, hull.simplices.astype(np.int64))
    assert r["self_intersections"] == 0


def test_surface_query_matches_neighbor_patch():
    """SurfaceQuery 与 NeighborPatch.closest 在真实形状上逐点一致。"""
    from app.geometry.crown_outer.contact import NeighborPatch
    v, f = _synthetic_crown()
    sq = SurfaceQuery(v, f, boundary_safety_mm=0.0, max_query_edge_mm=0.2)
    npatch = NeighborPatch(v, f)
    pts = v[rng.integers(0, len(v), 300)] + rng.normal(0, 0.25, (300, 3))
    *_, d1, _v1, _fl = sq.query(pts)
    d2 = npatch.closest_with_validity(pts, boundary_safety_mm=0.0)[3]
    assert np.abs(d1 - d2).max() < 1e-6


# ---------------------------------------------------------------- 审计口径

def _synthetic_crown(n_side=48, height=6.0, r_outer=3.0, r_inner=2.0):
    """开口壳：外壁 r_outer、颈环 r_inner（z=0），顶盖封到 axis；**故意**让
    侧壁厚度在颈缘处按线性从 0 变到 1mm，便于验证阈值规则。"""
    v, f = [], []
    for k in range(n_side):
        a = 2 * np.pi * k / n_side
        v.append((np.cos(a) * r_outer, np.sin(a) * r_outer, 0.0))       # 颈环
        v.append((np.cos(a) * r_outer, np.sin(a) * r_outer, height))    # 上沿
    v.append((0.0, 0.0, height))                                        # 顶点
    apex = len(v) - 1
    for k in range(n_side):
        k2 = (k + 1) % n_side
        b0, u0, b1, u1 = 2 * k, 2 * k + 1, 2 * k2, 2 * k2 + 1
        f.append((b0, b1, u1))
        f.append((b0, u1, u0))
        f.append((u0, u1, apex))
    return np.asarray(v, float), np.asarray(f, np.int64)


def _cap_shell(wall_mm, n_side=64, r_out=3.0, z_top=6.0):
    """合成冠壳：底部小开口(r=1) → 短法兰 → 外壁(r=r_out) → 顶盖。

    **开口必须小**：绕数判内外依赖"开口的立体角很小"，开口大的圆柱在开口附近
    绕数落到 ≈0.5，内外不可分（第一版合成例就是拿开放圆柱当冠，审计判不出内）。
    返回 (外壁半径 = r_out 的冠壳, 内壁 = r_out - wall_mm 的直筒)。
    """
    v, f = [], []
    for k in range(n_side):
        a = 2 * np.pi * k / n_side
        v.append((np.cos(a) * 1.0, np.sin(a) * 1.0, 0.0))         # 开口沿
        v.append((np.cos(a) * r_out, np.sin(a) * r_out, 0.5))     # 法兰外沿
        v.append((np.cos(a) * r_out, np.sin(a) * r_out, z_top))   # 壁顶
    v.append((0.0, 0.0, z_top + 0.2))                             # 顶点
    apex = len(v) - 1
    for k in range(n_side):
        k2 = (k + 1) % n_side
        a0, a1 = 3 * k, 3 * k + 1
        b0, b1 = 3 * k2, 3 * k2 + 1
        c0, c1 = 3 * k + 2, 3 * k2 + 2
        f += [(a0, b0, a1), (a1, b0, b1)]          # 法兰
        f += [(a1, b1, c1), (a1, c1, c0)]          # 壁
        f += [(c0, c1, apex)]                      # 顶盖扇
    inner_r = r_out - wall_mm
    iv, if_ = [], []
    m = 48
    z0, z1 = 2.0, z_top - 0.4
    for k in range(m):
        a = 2 * np.pi * k / m
        iv.append((np.cos(a) * inner_r, np.sin(a) * inner_r, z0))
        iv.append((np.cos(a) * inner_r, np.sin(a) * inner_r, z1))
    for k in range(m):
        k2 = (k + 1) % m
        p0, p1, q0, q1 = 2 * k, 2 * k + 1, 2 * k2, 2 * k2 + 1
        if_ += [(p0, q0, q1), (p0, q1, p1)]
    return (np.asarray(v, float), np.asarray(f, np.int64),
            np.asarray(iv, float), np.asarray(if_, np.int64))


class _Mesh:
    """最小网格容器（引擎只要求 .vertices/.faces）。"""

    def __init__(self, v, f):
        self.vertices = np.asarray(v, float)
        self.faces = np.asarray(f, np.int64)


def _frame():
    from app.geometry.crown_outer.patient_frame import PatientFrame
    return PatientFrame(origin=np.zeros(3), z=np.array([0.0, 0, 1]),
                        x=np.array([1.0, 0, 0]), y=np.array([0.0, 1, 0]))


def test_audit_passes_when_thick_enough():
    """壁厚处处 > T_min → PASS（且没有一处负壁厚）。"""
    ov, of, iv, if_ = _cap_shell(wall_mm=1.2)
    res = audit_thickness(
        _Mesh(ov, of), _Mesh(iv, if_), _frame(),
        params=ThicknessParameters(min_thickness_mm=0.5,
                                   max_query_edge_mm=0.4))
    assert res.summary["n_outside_outer"] == 0
    assert res.status == STATUS_PASS
    assert res.summary["thin_area_mm2"] == pytest.approx(0.0, abs=1e-9)
    assert res.summary["median_thickness_mm"] == pytest.approx(1.2, abs=0.05)


def test_audit_flags_thin():
    """壁厚 0.3mm < T_min=0.5 → 报薄，且测量值就是 0.3。"""
    ov, of, iv, if_ = _cap_shell(wall_mm=0.3)
    res = audit_thickness(
        _Mesh(ov, of), _Mesh(iv, if_), _frame(),
        params=ThicknessParameters(min_thickness_mm=0.5,
                                   max_query_edge_mm=0.4))
    assert res.status == STATUS_REVIEW_REQUIRED
    assert res.summary["thin_area_mm2"] > 0
    t = res.thickness_map["t_primary_mm"]
    ev = res.thickness_map["flags"] == 0
    assert abs(np.median(t[ev]) - 0.3) < 0.03


def test_audit_negative_coverage_detected():
    """内壁半径 > 外壁 → 全部负壁厚（覆盖率失败），绕数判据必须认出来。"""
    ov, of, iv, if_ = _cap_shell(wall_mm=-0.5)          # 内壁 r = 3.5 > 外壁 3.0
    res = audit_thickness(
        _Mesh(ov, of), _Mesh(iv, if_), _frame(),
        params=ThicknessParameters(min_thickness_mm=0.5,
                                   max_query_edge_mm=0.4))
    assert res.summary["n_outside_outer"] > 0
    assert res.summary["negative_coverage"]["area_mm2"] > 0
    assert res.summary["negative_coverage"]["depth_mm_p50"] == \
        pytest.approx(0.5, abs=0.02)


def test_threshold_rule_linear_and_step():
    p = ThicknessParameters(min_thickness_mm=0.6,
                            near_margin_thickness_mm=0.2,
                            near_margin_distance_mm=0.8)
    d = np.array([0.0, 0.4, 0.8, 2.0])
    thr, near = thresholds_of(d, p)
    assert np.allclose(thr, [0.2, 0.4, 0.6, 0.6])
    assert near.tolist() == [True, True, False, False]
    p2 = ThicknessParameters(min_thickness_mm=0.6,
                             near_margin_thickness_mm=0.2,
                             near_margin_distance_mm=0.8,
                             near_margin_mode="step")
    thr2, _ = thresholds_of(d, p2)
    assert np.allclose(thr2, [0.2, 0.2, 0.6, 0.6])


def test_parameters_require_explicit_threshold():
    v, f = _cap_shell(wall_mm=1.2)[0:2]
    with pytest.raises(ThicknessError):
        audit_thickness(_Mesh(v, f), _Mesh(v, f), _frame(), params=None)
    with pytest.raises(ThicknessError):
        audit_thickness(_Mesh(v, f), _Mesh(v, f), _frame(),
                        params=ThicknessParameters(min_thickness_mm=0.0))


def _cylinder(r=1.0, z0=0.0, z1=1.0, n=24, *, flip=False):
    """直筒侧壁（顶点法向恰为径向，便于精确断言偏移量）。"""
    v, f = [], []
    for k in range(n):
        a = 2 * np.pi * k / n
        v.append((np.cos(a) * r, np.sin(a) * r, z0))
        v.append((np.cos(a) * r, np.sin(a) * r, z1))
    for k in range(n):
        k2 = (k + 1) % n
        p0, p1, q0, q1 = 2 * k, 2 * k + 1, 2 * k2, 2 * k2 + 1
        f += [(p0, q0, q1), (p0, q1, p1)]
    f = np.asarray(f, np.int64)
    return np.asarray(v, float), (f[:, ::-1].copy() if flip else f)


def test_intaglio_from_prep_offsets_and_orients():
    """偏移沿外法向（+= gap）；代型法向整体朝内时翻号并记录。"""
    v, f = _cylinder(r=1.0)
    out, _, info = intaglio_from_prep(v, f, 0.1, axis=[0, 0, 1])
    r = np.linalg.norm(out[:, :2], axis=1)
    # 面积加权顶点法向在三角化直筒上有 ~2.5° 的离散偏差（对角分裂不对称），
    # 断言"方向朝外 + 量级正确"而不是 1e-12 级的精确值
    assert np.all(r > 1.0) and np.allclose(r, 1.1, atol=2e-3)
    assert not info["normal_flipped"]
    # 反向绕制 → 法向朝内 → 必须翻号后仍然朝外偏移（否则 intaglio 会缩进牙体）
    v2, f2 = _cylinder(r=1.0, flip=True)
    out2, _, info2 = intaglio_from_prep(v2, f2, 0.1, axis=[0, 0, 1])
    r2 = np.linalg.norm(out2[:, :2], axis=1)
    assert np.all(r2 > 1.0) and np.allclose(r2, 1.1, atol=2e-3)
    assert info2["normal_flipped"]
