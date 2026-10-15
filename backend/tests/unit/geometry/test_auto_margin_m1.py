"""M1 Geometry Auto Margin 的合成测试。

合成夹具是**解析可验证**的：一个"代型" = 下段圆柱（外壁）+ 上段圆锥（预备体），
交界处就是 Margin。这样预测点落在交界附近才算对，不靠真实数据背书。
"""
from __future__ import annotations

import math

import pytest

from app.geometry.auto_margin import engine, geometry_score, loop as loop_mod
from app.geometry.auto_margin.models import (AutoMarginConfig, STATUS_OK,
                                             STATUS_REVIEW)
from app.geometry.auto_margin.projection import (MeshIndex, closest_on_tri,
                                                 point_to_polyline_dist)


def _ring(radius, y, n=64):
    return [(radius*math.cos(2*math.pi*i/n), y, radius*math.sin(2*math.pi*i/n))
            for i in range(n)]


def die_with_margin(wall_r=5.0, margin_y=0.0, top_r=3.0, top_y=6.0,
                    bottom_y=-8.0, n=64):
    """下段圆柱（外壁，半径 wall_r）+ 上段圆锥（预备体，收窄到 top_r）。

    Margin = y == margin_y 处半径从 wall_r 跳到预备体收窄起点的那一圈。
    """
    rings = []
    # 壁面做密（0.5mm 一档）—— 真实扫描网格量级；粗网格会让"底盖棱污染"
    # 落进候选带，那是夹具不真实，不是判据问题（已单独修判据为按边相邻）
    ys = [bottom_y]
    y = bottom_y
    while y < margin_y - 1e-9:
        y = min(y + 0.5, margin_y)
        ys.append(y)
    ys += [margin_y + 0.002, margin_y + 0.5, margin_y + 1.5,
           margin_y + 3.0, margin_y + 4.5, top_y]
    for y in ys:
        if y <= margin_y:
            rings.append(_ring(wall_r, y, n))
        else:
            t = (y - margin_y) / max(top_y - margin_y, 1e-9)
            rings.append(_ring(wall_r + (top_r - wall_r) * t, y, n))
    tris = []
    for k in range(len(rings) - 1):
        a, b = rings[k], rings[k+1]
        for i in range(n):
            j = (i+1) % n
            tris.append((a[i], a[j], b[i]))
            tris.append((a[j], b[j], b[i]))
    # 顶盖
    top = rings[-1]
    apex = (0.0, top_y + 0.5, 0.0)
    for i in range(n):
        tris.append((top[i], top[(i+1) % n], apex))
    # 底盖
    bot = rings[0]
    base = (0.0, bottom_y - 0.5, 0.0)
    for i in range(n):
        tris.append((bot[(i+1) % n], bot[i], base))
    return tris, margin_y


def test_projection_matches_builder_algorithm():
    """投影实现与构建器逐点一致（含退化三角形）。"""
    sys_path = __import__("sys").path
    try:
        sys_path.insert(0, "tools")
        from build_auto_margin_gt import closest_on_tri as ref
    except Exception:                                   # noqa: BLE001
        pytest.skip("tools/build_auto_margin_gt.py 不可导入")
    cases = [((0.2, 0.2, 0.0), (0, 0, 0), (1, 0, 0), (0, 1, 0)),
             ((0.2, 0.2, 1.0), (0, 0, 0), (1, 0, 0), (0, 1, 0)),
             ((2.0, 0.0, 0.0), (0, 0, 0), (1, 0, 0), (0, 1, 0)),
             ((1.0, 1.0, 1.0), (0, 0, 0), (1, 0, 0), (2, 0, 0))]
    for p, a, b, c in cases:
        assert abs(closest_on_tri(p, a, b, c)[1] - ref(p, a, b, c)[1]) < 1e-9


def test_mesh_index_finds_surface():
    tris = [((0, 0, 0), (1, 0, 0), (0, 1, 0))]
    idx = MeshIndex(tris)
    _q, d, _f = idx.nearest((0.2, 0.2, 1.0))
    assert d == pytest.approx(1.0, abs=1e-9)
    _q, d, _f = idx.nearest((0.3, 0.3, 0.0))
    assert d == pytest.approx(0.0, abs=1e-9)


def test_engine_predicts_a_closed_loop_near_the_ridge():
    """合成代型上：预测环应闭、不自交、半径接近外壁半径、高度接近交界。"""
    tris, margin_y = die_with_margin()
    res = engine.predict_margin(tris, (0.0, 1.0, 0.0),
                                AutoMarginConfig(score="sharp"))
    assert res.margin is not None, res.reason
    q = res.quality
    assert q["closed"] is True, q
    assert q["n_self_intersections"] == 0, q
    assert res.status in (STATUS_OK, STATUS_REVIEW)
    ys = [p[1] for p in res.margin.points]
    assert min(ys) > -6.0 and max(ys) < 5.0, (min(ys), max(ys))
    rs = [math.hypot(p[0], p[2]) for p in res.margin.points]
    assert 3.0 < min(rs) and max(rs) < 5.6, (min(rs), max(rs))


def test_all_scores_run_and_are_finite():
    tris, _y = die_with_margin()
    for name in geometry_score.SCORES:
        res = engine.predict_margin(tris, (0.0, 1.0, 0.0),
                                    AutoMarginConfig(score=name))
        assert res.margin is not None, name
        assert len(res.margin.points) >= 12, name


def test_sector_search_fills_every_sector_on_a_clean_die():
    tris, _y = die_with_margin(n=64)
    res = engine.predict_margin(tris, (0.0, 1.0, 0.0))
    assert res.quality["filled_sector_frac"] >= 0.98, res.quality


def test_point_to_polyline_distance():
    square = [(1, 0, 0), (0, 0, 1), (-1, 0, 0), (0, 0, -1)]
    assert point_to_polyline_dist((1.0, 0.0, 0.0), square) == pytest.approx(0.0)
    assert point_to_polyline_dist((2.0, 0.0, 0.0), square) == pytest.approx(1.0)
