"""边缘线「磁力」特征吸附的单元测试（对应 exocad Supragingival / Subgingival）。

夹具全部用 trimesh 构造（法向一致、可比解析推导）：

1. **球**（处处凸，二面角 ~2.8°）→ 全部为负，且低于 8° 特征阈值 ⇒ 不吸附、REVIEW_REQUIRED；
2. **环面** → 内圈（凹）为正、外圈（凸）为负（±11.2°）；
3. **台阶代型**（下粗上细两个圆柱并集）→ 外缘棱 −90°、内角 +90°；
   细分到 ~0.15 mm 面后，SUPRA 把边缘点吸到外缘棱（半径 2.20 → ~2.88），
   SUB 停在凹角（半径基本不变）。

这些断言把两件事钉死：**符号约定**（凸为负/凹为正、与网格绕向无关）与
**两种模式的行为差异**（不是"都能吸附"就算过）。
"""
from __future__ import annotations

import math

import numpy as np
import pytest
import trimesh

from app.geometry.margin_feature_snap import (
    MODE_SUB, MODE_SUPRA, STATUS_APPLIED, STATUS_REVIEW_REQUIRED,
    face_concavity_deg, snap_to_feature,
)

#: 台阶代型的几何常量（下面多处引用，改一处即可）
R_WIDE, R_NARROW = 3.0, 2.2


def _step_die(segments=64, max_edge=None):
    """下粗（r=3.0）上细（r=2.2）两个圆柱并集 → 外缘棱（凸）+ 内角（凹）。"""
    lo = trimesh.creation.cylinder(radius=R_WIDE, height=1.0, segments=segments)
    lo.apply_translation([0, 0, -0.5])
    up = trimesh.creation.cylinder(radius=R_NARROW, height=1.0, segments=segments)
    up.apply_translation([0, 0, 0.5])
    body = trimesh.boolean.union([lo, up], engine="manifold")
    if max_edge:
        v, f = trimesh.remesh.subdivide_to_size(body.vertices, body.faces, max_edge=max_edge)
        body = trimesh.Trimesh(v, f, process=True)
    return body


def _ring(radius, n=48, z=0.0):
    return [[radius * math.cos(2 * math.pi * i / n),
             radius * math.sin(2 * math.pi * i / n), z] for i in range(n)]


def _mean_radius(points):
    P = np.asarray(points, float)
    return float(np.mean(np.hypot(P[:, 0], P[:, 1])))


# --------------------------------------------------------------- 1. 符号约定

def test_sphere_is_convex_everywhere():
    """球面处处凸 → 全部为负（凸），且量级只有 ~3°（低于特征阈值）。"""
    sphere = trimesh.creation.icosphere(subdivisions=4, radius=3.0)
    conc = face_concavity_deg(sphere.triangles, ref_point=np.zeros(3))
    assert float(np.median(conc)) < 0, "球面应为凸（负）"
    assert float(np.max(conc)) < 0
    assert abs(float(np.median(conc))) < 8.0, "细分球面的二面角应在特征阈值以下"


def test_torus_inner_concave_outer_convex():
    torus = trimesh.creation.torus(major_radius=3.0, minor_radius=0.9,
                                   major_segments=96, minor_segments=32)
    cents = torus.triangles.mean(axis=1)
    rad = np.hypot(cents[:, 0], cents[:, 1])
    conc = face_concavity_deg(torus.triangles, ref_point=np.zeros(3))
    inner = rad < np.percentile(rad, 25)
    outer = rad > np.percentile(rad, 75)
    assert float(np.median(conc[inner])) > 0, "环面内圈应为凹（正）"
    assert float(np.median(conc[outer])) < 0, "环面外圈应为凸（负）"


def test_step_die_sign_convention_and_winding_independence():
    die = _step_die()
    cents = die.triangles.mean(axis=1)
    rad = np.hypot(cents[:, 0], cents[:, 1])
    band = np.abs(cents[:, 2]) < 0.4
    conc = face_concavity_deg(die.triangles, ref_point=np.zeros(3))
    wide = band & (rad > 2.9)
    narrow = band & (rad < 2.35)
    assert float(np.median(conc[wide])) < -80, "外缘棱应为凸（≈ −90°）"
    assert float(np.median(conc[narrow])) > 80, "内角应为凹（≈ +90°）"

    # 绕向翻转后（法向朝内）符号必须不变 —— 这正是 ref_point 修正的作用
    flipped = die.copy()
    flipped.invert()
    conc_flip = face_concavity_deg(flipped.triangles, ref_point=np.zeros(3))
    assert float(np.median(conc_flip[wide])) < -80
    assert float(np.median(conc_flip[narrow])) > 80


# --------------------------------------------------------------- 2. 吸附行为

def test_supra_snaps_to_outer_crest_sub_stays_in_concave_corner():
    die = _step_die(max_edge=0.15)          # 密度接近真实扫描导数级
    loop = _ring(R_NARROW)                  # 起点在细颈根部（凹角所在环）
    base = _mean_radius(loop)

    supra = snap_to_feature(die.triangles, loop, None, MODE_SUPRA, radius_mm=1.0)
    sub = snap_to_feature(die.triangles, loop, None, MODE_SUB, radius_mm=1.0)

    assert supra["status"] == STATUS_APPLIED and sub["status"] == STATUS_APPLIED
    assert supra["n_moved"] >= int(0.6 * len(loop))

    # SUPRA：吸到外缘棱 → 半径显著变大
    assert _mean_radius(supra["points"]) - base > 0.4
    assert supra["mean_move_mm"] > 0.3
    # SUB：本来就在凹角上 → 基本不动
    assert abs(_mean_radius(sub["points"]) - base) < 0.1
    assert sub["mean_move_mm"] < 0.1


def test_smooth_convex_surface_reports_review_required_and_keeps_points():
    """球面没有 8° 以上的特征 → 不吸附、如实报 REVIEW_REQUIRED，点原样返回。"""
    sphere = trimesh.creation.icosphere(subdivisions=4, radius=3.0)
    loop = _ring(3.0)
    res = snap_to_feature(sphere.triangles, loop, None, MODE_SUPRA, radius_mm=1.0)
    assert res["status"] == STATUS_REVIEW_REQUIRED
    assert res["n_moved"] == 0 and res["reason"]
    assert np.allclose(np.asarray(res["points"], float), np.asarray(loop, float))


def test_radius_and_mode_validation():
    die = _step_die(segments=32)
    loop = _ring(R_NARROW, n=12)
    for bad in (0.0, 0.05, 6.0):
        with pytest.raises(ValueError):
            snap_to_feature(die.triangles, loop, None, MODE_SUPRA, radius_mm=bad)
    with pytest.raises(ValueError):
        snap_to_feature(die.triangles, loop, None, "MIDDLE", radius_mm=1.0)
    with pytest.raises(ValueError):
        snap_to_feature(die.triangles, [[0, 0, 0], [1, 0, 0]], None, MODE_SUPRA, radius_mm=1.0)
    with pytest.raises(ValueError):
        snap_to_feature([], loop, None, MODE_SUPRA, radius_mm=1.0)
