"""② transition 三模式的合成夹具（用户 2026-10-01 §③）。

Fixture B：**表面连续但面质心跳动** —— 相邻扇区候选的 Δh 超过 V2 常数门，
但在 mesh 上其实局部相连。要求：
    V2_FROZEN → 禁止；MESH_AWARE / HYBRID → 允许。

Fixture "可达 ≠ 合理"：两点 Euclidean 很近（隔着一道缝），但**只能绕远**才连得上。
要求：MESH_AWARE / HYBRID 都不能把它当成正常局部 transition。
"""
from __future__ import annotations

import math

from app.geometry.auto_margin import features, prep_frame
from app.geometry.auto_margin.features import FaceStat
from app.geometry.auto_margin.graph_search import (SurfaceDistance,
                                                   transition_cost)
from app.geometry.auto_margin.models import AutoMarginConfig

V2 = AutoMarginConfig(transition_mode="V2_FROZEN")
MA = AutoMarginConfig(transition_mode="MESH_AWARE")
HY = AutoMarginConfig(transition_mode="HYBRID")


def _band(radius=5.0, rows=(-1.0, 0.0, 1.0), n=64, skip_deg=None):
    """圆柱带：rows 为各圈高度；skip_deg=(from,to) 时挖掉一段做成"缝"。"""
    rings = []
    for h in rows:
        rings.append([(radius*math.cos(2*math.pi*i/n), h,
                       radius*math.sin(2*math.pi*i/n)) for i in range(n)])
    tris = []
    for k in range(len(rings) - 1):
        a, b = rings[k], rings[k+1]
        for i in range(n):
            j = (i + 1) % n
            if skip_deg:
                ang = math.degrees(2*math.pi*i/n)
                if skip_deg[0] <= ang < skip_deg[1]:
                    continue
            tris.append((a[i], a[j], b[i]))
            tris.append((a[j], b[j], b[i]))
    return tris


def _stats_and_surface(tris, axis=(0.0, 1.0, 0.0)):
    cents = [((t[0][0]+t[1][0]+t[2][0])/3.0, (t[0][1]+t[1][1]+t[2][1])/3.0,
              (t[0][2]+t[1][2]+t[2][2])/3.0) for t in tris]
    frame = prep_frame.build_frame(cents, axis)
    stats, _adj = features.face_stats(tris, frame)
    eadj = features.build_edge_adjacency(tris)
    surf = SurfaceDistance(tris, eadj, {s.fid: s.centroid for s in stats})
    return stats, surf, frame


# ============================================================ Fixture B
def test_v2_gate_blocks_surface_continuous_transition():
    """面质心 Δh 超过 v2 常数门，但在网格上局部连续。

    → V2_FROZEN 禁止；MESH_AWARE / HYBRID 允许。这正对应 Blind v1 里
    6/15 的"GT 候选每个扇区都在、却过不了常数门"。
    """
    tris = _band()
    stats, surf, _frame = _stats_and_surface(tris)
    a = max(stats, key=lambda s: s.h)
    b = min(stats, key=lambda s: s.h)
    dh = abs(b.h - a.h)
    disc = features.face_scale_mm(stats) * V2.gate_disc_factor
    assert dh > V2.max_dh_mm + disc, (
        "夹具前提：Δh=%.3f 需超过 v2 有效门 %.3f" % (dh, V2.max_dh_mm + disc))
    d_geo, _n = surf.geo(a.fid, b.fid, MA.surface_search_radius_mm)
    assert d_geo is not None, "夹具前提：局部半径内两面应能连通"
    assert transition_cost("V2_FROZEN", a, b, V2, surf, 0.25) is None, \
        "v2 常数门应禁止这个转移"
    c_ma = transition_cost("MESH_AWARE", a, b, MA, surf, 0.25)
    c_hy = transition_cost("HYBRID", a, b, HY, surf, 0.25)
    assert c_ma is not None, "MESH_AWARE 应认出它们是局部连续的"
    assert c_hy is not None, "HYBRID 应允许（Δh 在宽松 sanity 内）"


# ============================================================ 可达 ≠ 合理
def test_far_detour_is_rejected():
    """隔缝相邻（Euclid 近）却只能绕远 → 必须禁止（不是合理局部 transition）。"""
    tris = _band(skip_deg=(170.0, 190.0))
    stats, surf, _frame = _stats_and_surface(tris)
    left = sorted([s for s in stats if 150 < math.degrees(s.theta) < 170],
                  key=lambda s: s.theta)[-1]
    right = sorted([s for s in stats if 190 < math.degrees(s.theta) < 210],
                   key=lambda s: s.theta)[0]
    d_euc = math.dist(left.centroid, right.centroid)
    d_geo, _n = surf.geo(left.fid, right.fid, 60.0)
    assert d_euc < 3.0, "夹具前提：跨缝两点 Euclidean 应很近"
    assert d_geo is not None, "夹具前提：绕远路径应在 60mm 半径内可达"
    assert d_geo > MA.max_detour_ratio * d_euc, (
        "夹具前提：绕行应远超 detour 上限（geo=%.2f euc=%.2f）" % (d_geo, d_euc))
    assert transition_cost("MESH_AWARE", left, right, MA, surf, 0.25) is None
    assert transition_cost("HYBRID", left, right, HY, surf, 0.25) is None


def test_local_radius_makes_distant_pair_unreachable():
    """局部半径的意义：超出半径的转移直接不可达（不许绕半圈）。"""
    tris = _band()
    stats, surf, _frame = _stats_and_surface(tris)
    stats_sorted = sorted(stats, key=lambda s: s.theta)
    a, b = stats_sorted[0], stats_sorted[len(stats_sorted) // 2]
    d_geo, _n = surf.geo(a.fid, b.fid, MA.surface_search_radius_mm)
    assert d_geo is None, "对面侧的面不应在 %.1fmm 局部半径内连通" % \
        MA.surface_search_radius_mm
    assert transition_cost("MESH_AWARE", a, b, MA, surf, 0.25) is None


def test_hybrid_keeps_loose_sanity_gate():
    """HYBRID 仍要挡"跨几毫米"的离谱跳跃，但不承担平滑职责。"""
    tris = _band()
    stats, surf, _frame = _stats_and_surface(tris)
    a = max(stats, key=lambda s: s.h)
    b = min(stats, key=lambda s: s.h)
    far = FaceStat(fid=b.fid, centroid=b.centroid, normal=b.normal, area=b.area,
                   h=b.h - 10.0, r=b.r, theta=b.theta, nx=b.nx,
                   sharp_deg=b.sharp_deg, radial_slope=b.radial_slope)
    assert abs(far.h - a.h) > HY.sanity_dh_mm
    assert transition_cost("HYBRID", a, far, HY, surf, 0.25) is None, \
        "Δh=%.1fmm 应被宽松 sanity 挡住" % abs(far.h - a.h)
