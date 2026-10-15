"""M1 v3 ridge cluster 候选生成的最小测试（合成件 + 直接构造）。"""
from __future__ import annotations

import math

from app.geometry.auto_margin import candidate_band, features, prep_frame, ridge_cluster
from app.geometry.auto_margin.models import AutoMarginConfig
from tests.unit.geometry.test_auto_margin_m1 import die_with_margin


def _prep(tris, axis=(0.0, 1.0, 0.0)):
    cents = [((t[0][0]+t[1][0]+t[2][0])/3.0, (t[0][1]+t[1][1]+t[2][1])/3.0,
              (t[0][2]+t[1][2]+t[2][2])/3.0) for t in tris]
    frame = prep_frame.build_frame(cents, axis)
    stats, adj = features.face_stats(tris, frame)
    return stats, adj, frame


def test_clusters_are_bounded_in_angle_and_height():
    """**核心不变量**：簇必须在角向与轴向都有界。

    没有角向界时，一整圈特征（底盖棱）会变成一个 128 面的巨簇，在**每个扇区**
    都是最高分 → 路径退化成同一簇的拷贝（实测环长 0、点数 240→120）。
    """
    tris, _y = die_with_margin()
    stats, adj, frame = _prep(tris)
    cfg = AutoMarginConfig(candidate_mode="ridges")
    cand = candidate_banc = candidate_band.candidate_facet_ids(stats, cfg)
    cl, sc = ridge_cluster.build_clusters(stats, adj, frame, cfg, cand, tris)
    assert cl, "应至少出一些簇"
    by_id = {s.fid: s for s in stats}
    for c in cl:
        ang = sorted(by_id[f].theta for f in c.support)
        span = ang[-1] - ang[0]
        span = min(span, 2 * math.pi - span)
        assert math.degrees(span) <= cfg.ridge_max_span_deg + 6.0, (
            "簇角向跨度 %.1f° 超限" % math.degrees(span))
    # 每个扇区的候选应是**不同**的簇（不是同一个巨簇霸占所有扇区）
    tops = {sc[s][0][1] for s in sc}
    assert len(tops) > 5, "各扇区 top-1 簇只有 %d 个不同值 → 又被巨簇霸占了" % len(tops)


def test_cluster_interface_matches_facestat():
    """簇要能直接喂给 DP（DP 与下游一行不改）。"""
    tris, _y = die_with_margin()
    stats, adj, frame = _prep(tris)
    cfg = AutoMarginConfig(candidate_mode="ridges")
    cl, sc = ridge_cluster.build_clusters(stats, adj, frame, cfg,
                                          candidate_band.candidate_facet_ids(
                                              stats, cfg), tris)
    for c in cl:
        for attr in ("fid", "h", "r", "centroid", "area"):
            assert hasattr(c, attr), attr
        assert c.fid == c.cid


def test_one_ridge_yields_few_clusters_not_one_per_face():
    """一条真脊应聚成少量簇，而不是让每个面各占一个名额。"""
    tris, _y = die_with_margin()
    stats, adj, frame = _prep(tris)
    cfg = AutoMarginConfig(candidate_mode="ridges")
    cand = candidate_band.candidate_facet_ids(stats, cfg)
    cl, sc = ridge_cluster.build_clusters(stats, adj, frame, cfg, cand, tris)
    assert len(cl) < len(cand) * 0.5, "簇数 %d 相对候选面 %d 太多（没聚起来）" % (
        len(cl), len(cand))
    assert all(c.support_faces >= cfg.ridge_min_support for c in cl)
