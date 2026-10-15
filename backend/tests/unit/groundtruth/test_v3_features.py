"""v3 特征提取的本地单元测试 —— 全合成，不碰任何真实数据。

锁住的规则：

  1. 局部系：Z=咬合、X=近中、Y=Z×X，且正交归一；缺失时明确 MISSING_LOCAL_FRAME，
     不允许偷偷 fallback 到世界 Z（正式特征绝不做这种 fallback）。
  2. prominence 的物理符号：中心凸起 > 0，洞 < 0，平面 ≈ 0 ——
     这就是 v3 区分 A/B/C 的全部依据。
  3. 单趟读取：hash 与 hashlib 对文件整体算的一致；采样格在 7mm 邻域有点。
  4. 径向曲线单调性 sanity：凸起的 z(r) 应先高后低。
"""

from __future__ import annotations

import hashlib
import math
import struct

import numpy as np

import pytest

from app.groundtruth.v3_features import (
    in_plane_basis,
    lattice_search,
    margin_pca_frame,
    point_in_polygon,
    point_to_polyline_dist,
    resolve_frame,
    CENTER_R,
    LOCAL_RADIUS,
    RING_R,
    SHOULDER_R,
    local_features,
    points_near,
    radial_profile_xy,
    read_stl_single_pass,
    tooth_local_frame,
)

Z_AXIS = [0.0, 0.0, 1.0]
X_AXIS = [1.0, 0.0, 0.0]


# ---------------------------------------------------------------- 夹具

def _tri_grid_surface(fn, half=10.0, n=40):
    """z=fn(x,y) 的高度场三角化。"""
    verts = []
    tris = []
    step = 2*half/n
    for i in range(n):
        for j in range(n):
            x0 = -half + i*step
            y0 = -half + j*step
            verts.append((x0, y0, fn(x0, y0)))
    idx = lambda i, j: i*n + j          # noqa: E731
    for i in range(n-1):
        for j in range(n-1):
            a, b = idx(i, j), idx(i+1, j)
            c, d = idx(i+1, j+1), idx(i, j+1)
            tris.append((verts[a], verts[b], verts[c]))
            tris.append((verts[a], verts[c], verts[d]))
    return tris


def _write_stl(path, tris):
    with open(path, "wb") as f:
        f.write(b"\0"*80)
        f.write(struct.pack("<I", len(tris)))
        for a, b, c in tris:
            f.write(struct.pack("<3f", 0.0, 0.0, 1.0))
            for p in (a, b, c):
                f.write(struct.pack("<3f", *p))
            f.write(struct.pack("<H", 0))


def prep_tris(cx=0.0, cy=0.0, *, stub_h=2.0, stub_r=3.0, floor=0.0):
    """中心有一个圆柱形预备体凸起的牙龈平面（A 类）。"""
    def fn(x, y):
        r = math.hypot(x-cx, y-cy)
        return floor + (stub_h if r <= stub_r else 0.0)
    return _tri_grid_surface(fn)


def socket_tris(cx=0.0, cy=0.0, *, depth=-2.0, hole_r=3.5):
    """中心有个**锥形**洞的平面（B 类：代型被抽走）。

    锥形而非平底+垂直壁：真实插座/预备体有斜度，r<1.5 的中心盘内
    会同时采到洞壁与洞底 —— 平底夹具测不出 center_spread 的差异。
    """
    def fn(x, y):
        r = math.hypot(x-cx, y-cy)
        if r >= hole_r:
            return 0.0
        return depth * (1.0 - r/hole_r)
    return _tri_grid_surface(fn)


def flat_tris():
    return _tri_grid_surface(lambda x, y: 0.0)


def _grid_of(tris, max_points=200_000):
    """把三角形顶点塞进 v3 的顶点格子（走公开的单趟读取更真实——经文件）。"""
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as d:
        p = Path(d)/"m.stl"
        _write_stl(p, tris)
        return read_stl_single_pass(str(p))


FRAME = None


def setup_function(_):
    global FRAME
    FRAME = tooth_local_frame(Z_AXIS, X_AXIS)


# ---------------------------------------------------------------- 局部系

def test_tooth_local_frame_is_orthonormal_with_y_cross():
    z, x, y = tooth_local_frame(Z_AXIS, X_AXIS)
    assert z == pytest.approx([0, 0, 1], abs=1e-9)
    assert x == pytest.approx([1, 0, 0], abs=1e-9)
    assert y == pytest.approx([0, 1, 0], abs=1e-9)


def test_tooth_local_frame_rejects_missing_or_degenerate():
    assert tooth_local_frame(None, X_AXIS) is None
    assert tooth_local_frame(Z_AXIS, None) is None
    assert tooth_local_frame([0, 0, 0], X_AXIS) is None


# ---------------------------------------------------------------- prominence 符号

def test_prominence_positive_for_prep_bump():
    g = _grid_of(prep_tris())
    f = local_features(g["buckets"], (0.0, 0.0, 2.0), FRAME)
    assert f is not None
    # 中心取的是 stub 顶面（z≈2），外环取的是平面（z≈0）
    assert f["prominence_median"] > 1.0
    assert f["radial_coverage"] > 0.9, "圆柱 stub 的凸起应当 360° 连续"
    assert f["shoulder_contrast"] > 0.5
    assert f["center_count"] > 0 and f["ring_count"] > 0


def test_prominence_negative_for_socket():
    g = _grid_of(socket_tris())
    # Center 在洞底（预备体被抽走后 <Center> 落在插座里）
    f = local_features(g["buckets"], (0.0, 0.0, -2.0), FRAME)
    assert f is not None
    assert f["prominence_median"] < -1.0
    assert f["shoulder_contrast"] < 0.0


def test_prominence_near_zero_for_flat():
    g = _grid_of(flat_tris())
    f = local_features(g["buckets"], (0.0, 0.0, 0.0), FRAME)
    assert f is not None
    assert abs(f["prominence_median"]) < 0.3
    assert abs(f["shoulder_contrast"]) < 0.3


def test_frame_sign_matters_but_module_does_not_hide_it():
    """z 轴翻转时 prominence 反号 —— 说明模块不自作聪明定"上"，
    方向语义由调用方（AxisOcclusalOfToothModel 的约定）决定并另行校准。"""
    g = _grid_of(prep_tris())
    flipped = ([ -v for v in Z_AXIS], [-v for v in X_AXIS])
    f1 = local_features(g["buckets"], (0.0, 0.0, 2.0),
                        tooth_local_frame(Z_AXIS, X_AXIS))
    f2 = local_features(g["buckets"], (0.0, 0.0, 2.0),
                        tooth_local_frame(*flipped))
    assert f1["prominence_median"] == pytest.approx(
        -f2["prominence_median"], abs=1e-6)


def test_missing_local_points_returns_none():
    g = _grid_of(prep_tris())
    assert local_features(g["buckets"], (500.0, 0.0, 0.0), FRAME) is None


# ---------------------------------------------------------------- 单趟读取

def test_single_pass_hash_matches_file_hash(tmp_path):
    p = tmp_path/"m.stl"
    _write_stl(p, prep_tris())
    got = read_stl_single_pass(str(p))
    raw = p.read_bytes()
    assert got["sha256"] == hashlib.sha256(raw).hexdigest()[:16]
    assert got["ok"] and got["faces"] > 0


def test_points_near_finds_local_points(tmp_path):
    p = tmp_path/"m.stl"
    _write_stl(p, prep_tris())
    got = read_stl_single_pass(str(p))
    pts = points_near(got["buckets"], (0.0, 0.0, 1.0), LOCAL_RADIUS)
    assert len(pts) > 50


# ---------------------------------------------------------------- 径向曲线

def test_radial_profile_shows_bump_then_drop():
    g = _grid_of(prep_tris())
    pts = points_near(g["buckets"], (0.0, 0.0, 0.0), LOCAL_RADIUS)
    rs, zs = radial_profile_xy(pts, (0.0, 0.0, 0.0),
                               tooth_local_frame(Z_AXIS, X_AXIS))
    assert rs and len(rs) == len(zs)
    # 中心带（r<1.5）应明显高于外环带（4<r<6）
    inner = [z for r, z in zip(rs, zs) if r < CENTER_R]
    outer = [z for r, z in zip(rs, zs) if RING_R[0] <= r < RING_R[1]]
    assert sum(inner)/len(inner) - sum(outer)/len(outer) > 1.0


def test_feature_bands_match_user_spec():
    assert CENTER_R == 1.5
    assert SHOULDER_R == (1.5, 3.5)
    assert RING_R == (4.0, 6.0)


# ============================================================ v3.1 新增

def _frame_for(axis_occ, axis_mes):
    return tooth_local_frame(axis_occ, axis_mes)


def test_sign_free_features_invariant_under_axis_flip():
    """v3.1 符号无关性的**实测边界**（重要，别删）：

    coverage/spread/z_range/|prom|/|shoulder| 严格不变；
    **rel_pos 不变号而是互补**（翻转后 = 1 − rel_pos）—— 它本质上度量
    "中心处于局部高度哪一端"，没有外部"上方向"就无法区分 Prep 与插座。
    实测 <Axis> 符号跨病例不一致（见 docs/LOCALIZATION_QA.md），
    所以 rel_pos 只能与 QA2 标签统计联用，不能单独当判据。
    """
    g = _grid_of(prep_tris())
    f1 = local_features(g["buckets"], (0.0, 0.0, 2.0),
                        tooth_local_frame(Z_AXIS, X_AXIS))
    f2 = local_features(g["buckets"], (0.0, 0.0, 2.0),
                        tooth_local_frame([-v for v in Z_AXIS],
                                          [-v for v in X_AXIS]))
    for key in ("radial_coverage", "radial_asymmetry",
                "z_range", "center_spread", "ring_spread",
                "prominence_abs", "shoulder_contrast_abs"):
        v1, v2 = f1[key], f2[key]
        if v1 is None or v2 is None:
            assert v1 == v2, key
        else:
            assert v1 == pytest.approx(v2, abs=1e-9), \
                f"{key} 不满足符号无关：{v1} vs {v2}"
    assert f1["rel_pos"] == pytest.approx(1.0 - f2["rel_pos"], abs=1e-9), \
        "rel_pos 翻转后应互补"
    assert f1["prominence_median"] == pytest.approx(
        -f2["prominence_median"], abs=1e-9)


def test_rel_pos_extremes_for_prep_and_socket():
    g_prep = _grid_of(prep_tris())
    f_prep = local_features(g_prep["buckets"], (0.0, 0.0, 2.0), FRAME)
    assert f_prep["rel_pos"] > 0.85, \
        f"stub 顶应处于局部高度上端：{f_prep['rel_pos']}"
    assert f_prep["center_spread"] < 1.0, "stub 顶面平坦：center_spread 应小"

    g_soc = _grid_of(socket_tris())
    f_soc = local_features(g_soc["buckets"], (0.0, 0.0, -2.0), FRAME)
    # 洞底 + 洞壁混合：rel_pos 应明显低于 prep（不硬断言 0，壁采样会抬高中位数）
    assert f_soc["rel_pos"] < f_prep["rel_pos"] - 0.3
    assert f_soc["center_spread"] > f_prep["center_spread"], \
        "插座壁跨越全深度：center_spread 应大于 stub 顶"


def test_margin_pca_frame_matches_tooth_frame_on_synthetic():
    """margin 环的 PCA 法向应恢复 z 轴；两帧下的特征一致（符号无关量）。"""
    ring = [(5.0*math.cos(a), 5.0*math.sin(a), 0.0)
            for a in np.linspace(0, 2*math.pi, 60)]
    frame = margin_pca_frame(ring)
    assert frame is not None
    z, _x, _y = frame
    assert abs(z[2]) == pytest.approx(1.0, abs=1e-6), "平面环的法向应为 ±Z"
    g = _grid_of(prep_tris())
    f1 = local_features(g["buckets"], (0.0, 0.0, 2.0),
                        tooth_local_frame(Z_AXIS, X_AXIS))
    f2 = local_features(g["buckets"], (0.0, 0.0, 2.0), frame)
    for key in ("rel_pos", "radial_coverage", "z_range", "prominence_abs"):
        assert f1[key] == pytest.approx(f2[key], abs=0.25), key


def test_resolve_frame_priority_and_unavailable():
    class T:
        axis_occlusal_of_tooth_model = None
        axis_mesial = None
    t = T()
    frame, src = resolve_frame(t, [(5*math.cos(a), 5*math.sin(a), 0.0)
                                   for a in np.linspace(0, 6.28, 40)])
    assert src == "MARGIN_PCA" and frame is not None
    frame, src = resolve_frame(t, [])
    assert src == "UNAVAILABLE" and frame is None
    t.axis_occlusal_of_tooth_model = Z_AXIS
    t.axis_mesial = X_AXIS
    frame, src = resolve_frame(t, [])
    assert src == "TOOTH_FRAME" and frame is not None


# ============================================================ v3.2 格搜索

def _margin_ring(cx=0.0, cy=0.0, r=5.0, n=60):
    return [(cx + r*math.cos(a), cy + r*math.sin(a), 0.0)
            for a in np.linspace(0, 2*math.pi, n)]


def test_lattice_finds_offset_prep_peak():
    """**v3.2 的立项依据**：Prep 轴与 <Center> 偏移 6mm 时，
    固定 Center 圆盘 prom≈0，格搜索必须在偏移处找到峰。"""
    g = _grid_of(prep_tris(cx=4.0, cy=3.0))
    f_center = local_features(g["buckets"], (0.0, 0.0, 0.0), FRAME)
    # 前提复现：Center 处几乎无凸起
    assert f_center["prominence_abs"] < 0.5
    # margin 是**该预备体自己的**颈缘环：以预备体为中心（真实语义）
    peaks = lattice_search(g["buckets"], (0.0, 0.0, 0.0), Z_AXIS,
                           margin_points=_margin_ring(cx=4.0, cy=3.0, r=5.0))
    assert peaks, "应至少找到一个峰"
    top = peaks[0]
    # 格坐标 (du,dv) 沿 in_plane_basis(Z) 的 e1/e2；世界偏移 (4,3) 要换基
    e1, e2 = in_plane_basis(Z_AXIS)
    wx = top["du"]*e1[0] + top["dv"]*e2[0]
    wy = top["du"]*e1[1] + top["dv"]*e2[1]
    assert math.hypot(wx-4.0, wy-3.0) <= 1.0, \
        f"峰位（换回世界系）应落在偏移预备体处：({wx:.1f},{wy:.1f})"
    assert top["prominence_abs"] > 1.5
    assert top["in_margin_polygon"] is True, "环内预备体峰应落在 margin 多边形内"


def test_lattice_nms_returns_separated_peaks():
    """两个相距 8mm 的预备体 -> 两个峰，且峰间距离 >= NMS 距离。"""
    g = _grid_of(_two_preps())
    peaks = lattice_search(g["buckets"], (0.0, 0.0, 0.0), Z_AXIS)
    assert len(peaks) >= 2
    d = math.hypot(peaks[0]["du"]-peaks[1]["du"],
                   peaks[0]["dv"]-peaks[1]["dv"])
    assert d >= 3.0


def _two_preps():
    tris = prep_tris(cx=-4.0, cy=0.0)
    more = prep_tris(cx=4.0, cy=0.0)
    return tris + more


def test_lattice_flat_surface_yields_no_strong_peak():
    g = _grid_of(flat_tris())
    peaks = lattice_search(g["buckets"], (0.0, 0.0, 0.0), Z_AXIS)
    if peaks:
        assert peaks[0]["prominence_abs"] < 0.5, "平面上不应有显著峰"


def test_margin_fields_absent_when_no_margin():
    g = _grid_of(prep_tris())
    peaks = lattice_search(g["buckets"], (0.0, 0.0, 0.0), Z_AXIS,
                           margin_points=None)
    assert peaks
    assert peaks[0]["dist_to_margin_mm"] is None
    assert peaks[0]["in_margin_polygon"] is None


def test_point_in_polygon_and_polyline_dist():
    ring3 = _margin_ring(r=5.0)
    ring = [(x, y) for x, y, _z in ring3]        # 2D 折线（格平面投影）
    assert point_in_polygon(0.0, 0.0, ring) is True
    assert point_in_polygon(9.0, 0.0, ring) is False
    assert point_in_polygon(0.0, 0.0, ring3[:2]) is None          # 点太少
    assert point_to_polyline_dist(6.0, 0.0, ring) == pytest.approx(1.0, abs=1e-6)
    assert point_to_polyline_dist(0.0, 0.0, []) is None
