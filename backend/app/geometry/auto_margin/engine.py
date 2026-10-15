"""M1 引擎：代型网格 + 就位轴 → 预测的 Margin 闭环。

    frame → 面特征 → 候选带 → 每扇区 Top-K 候选
          → [v1: argmax | v2: 环状 DP]
          → 角向排序 → 平滑 → 重采样 → 投影回网格 → QC

**全程不读 GT Margin**（GT 只在 tools/run_auto_margin_m1.py 的 evaluate 里）。
v1 与 v2 **只差路径选择那一步**，其余完全相同 —— 这样对照才能回答
"提升是不是来自整圈一致性"。
"""

from __future__ import annotations

import dataclasses
import time

from . import (candidate_band, features, graph_search, loop as loop_mod,
               qc_rules, ridge_cluster)
from . import prep_frame, projection, quality, spline
import math

from .models import (AutoMarginConfig, AutoMarginResult, PredictedMargin,
                     STATUS_FAILED, STATUS_REVIEW_REQUIRED,
                     frozen_config_id)


def predict_margin(tris, axis, config=None, return_candidates=False,
                   stats=None, adj=None, frame=None):
    """``stats/adj/frame`` 可由调用方传入以复用（同一网格跑多组配置时省一遍
    特征计算；**不影响结果**，特征只依赖网格与轴）。"""
    t0 = time.time()
    cfg = config or AutoMarginConfig()
    res = AutoMarginResult()
    if len(tris) < 100:
        res.reason = "mesh_too_small=%d" % len(tris)
        res.elapsed_s = time.time() - t0
        return res
    # 坐标系只用**网格自身质心 + 给定的轴**建立（纯几何，无 GT）
    if frame is None:
        cents = [((t[0][0]+t[1][0]+t[2][0])/3.0, (t[0][1]+t[1][1]+t[2][1])/3.0,
                  (t[0][2]+t[1][2]+t[2][2])/3.0) for t in tris]
        frame = prep_frame.build_frame(cents, axis)
    if stats is None:
        stats, adj = features.face_stats(tris, frame)
    # 跳变硬门的离散度项：候选是"面"，精度下界就是网格面尺度（纯几何、无 GT）
    cfg = dataclasses.replace(cfg,
                              _face_scale_mm=features.face_scale_mm(stats))
    cand = candidate_band.candidate_facet_ids(stats, cfg)
    if not cand:
        res.reason = "empty_candidate_band"
        res.elapsed_s = time.time() - t0
        return res
    if cfg.candidate_mode == "ridges":
        # v3：候选 = ridge 簇（一条脊一个候选），DP 与下游完全复用
        cl_stats, sector_cands = ridge_cluster.build_clusters(
            stats, adj, frame, cfg, cand, tris)
        dp_stats = cl_stats
        full = {sec: [(sc, cid) for sc, cid in items]
                for sec, items in sector_cands.items()}
        res.quality["n_clusters"] = len(cl_stats)
    else:
        pair = candidate_band.sector_candidates(stats, adj, frame, cfg, cand,
                                                tris, return_full=True)
        sector_cands, full = pair
        dp_stats = stats
    if len(sector_cands) < 12:
        res.status = STATUS_FAILED
        res.reason = "candidate_coverage_insufficient(%d sectors)" % len(
            sector_cands)
        res.elapsed_s = time.time() - t0
        return res
    notes_extra = []
    e1 = e2 = None
    surf = None
    if cfg.method == "dp" and cfg.transition_mode != "V2_FROZEN":
        # 只在非冻结模式下构建（局部表面测地；纯几何、无 GT）
        eadj = features.build_edge_adjacency(tris)
        surf = graph_search.SurfaceDistance(
            tris, eadj, {s.fid: s.centroid for s in stats})
    if cfg.method == "dp":
        path, e1, e2 = graph_search.cyclic_dp_path(sector_cands, dp_stats,
                                                   frame, cfg, surf)
        if not path:
            if cfg.fallback_mode == "ARGMAX":
                # 仅研发诊断：回退并**如实标注**（生产默认不走这条）
                path = graph_search.argmax_path(sector_cands, dp_stats)
                notes_extra.append("dp_infeasible_fallback_argmax")
            else:
                # **生产语义（fail-loud）**：绝不输出"看起来合法"的曲线
                res.status = STATUS_REVIEW_REQUIRED
                res.reason = "NO_FEASIBLE_CYCLE"
                res.quality = {"n_sectors_filled": len(sector_cands),
                               "n_clusters": res.quality.get("n_clusters")}
                res.elapsed_s = round(time.time() - t0, 1)
                res.config_id = frozen_config_id(cfg)
                res.score_method = cfg.score.upper()
                res.transition_method = cfg.transition_mode
                return res
    else:
        path = graph_search.argmax_path(sector_cands, dp_stats)
    pts = graph_search.sector_points(path, dp_stats)
    pts = loop_mod.order_by_theta(pts, frame)
    sm = spline.circular_smooth(pts, cfg.smooth_window)
    rs = spline.resample_closed(sm, cfg.resample_n)
    index = projection.MeshIndex(tris)
    proj, disp = projection.project_points(rs, index)
    _raw_status, q, notes = quality.assess(proj, frame, cfg, raw_points=pts,
                                           disp=disp)
    if "n_clusters" in res.quality:      # 候选生成统计别被 QC 覆盖掉
        q["n_clusters"] = res.quality["n_clusters"]
    # ---- 置信度 / 歧义（全部几何量，不看 GT）
    ranks = []
    for sec, fid in path.items():
        items = sector_cands.get(sec) or []
        for idx, (_sc, cid) in enumerate(items):
            if cid == fid:
                ranks.append(idx + 1)
                break
    q["chosen_rank_mean"] = (round(sum(ranks) / len(ranks), 2) if ranks
                             else None)
    q["chosen_rank_p95"] = (sorted(ranks)[min(len(ranks) - 1,
                                              int(.95 * (len(ranks) - 1)))]
                            if ranks else None)
    q["chosen_rank_max"] = max(ranks) if ranks else None
    # 最大转移步长（3D，沿选中的环）
    pts_raw = graph_search.sector_points(path, dp_stats)
    steps = [math.dist(pts_raw[i], pts_raw[(i + 1) % len(pts_raw)])
             for i in range(len(pts_raw))]
    q["max_transition_mm"] = round(max(steps), 4) if steps else None
    q["mean_transition_mm"] = (round(sum(steps) / len(steps), 4) if steps
                               else None)
    res.loop_energy = round(e1, 6) if e1 is not None else None
    # 真 second-best：屏蔽 P1 候选邻域后重解（结构性不同），只记录、不拒绝
    if cfg.method == "dp" and path and e1 is not None:
        _p2, e2s, sep = graph_search.structurally_distinct_second(
            sector_cands, dp_stats, frame, cfg, path, surface=surf)
        if e2s is not None:
            res.second_energy = round(e2s, 6)
            res.energy_gap = round((e2s - e1) / (abs(e1) + 1e-9), 6)
            q["second_separation_mm"] = (round(sep, 4) if sep is not None
                                         else None)
            disc = getattr(cfg, "_face_scale_mm", 0.3) or 0.3
            q["ambiguous_second_cycle"] = bool(sep is not None
                                               and sep <= 3.0 * disc)
    q["energy_gap"] = res.energy_gap
    # 状态由 **GT-independent QC 规则**统一裁决（qc_rules，冻结候选版）
    status, qc_reasons = qc_rules.decide_status(feasible=True, quality=q,
                                                config=cfg)
    notes = list(notes_extra) + list(qc_reasons) + list(notes)
    margin = PredictedMargin(points=proj, n_sectors_filled=len(path),
                             n_sectors=cfg.n_sectors,
                             closed=bool(q.get("closed")), notes=notes)
    res.status = status
    res.margin = margin
    res.quality = q
    res.reason = ";".join(notes)
    res.elapsed_s = round(time.time() - t0, 1)
    res.config_id = frozen_config_id(cfg)
    res.score_method = cfg.score.upper()
    res.transition_method = cfg.transition_mode
    res.candidate_rank_stats = {"mean": q.get("chosen_rank_mean"),
                                "p95": q.get("chosen_rank_p95"),
                                "max": q.get("chosen_rank_max")}
    res.confidence = {k: q.get(k) for k in
                      ("energy_gap", "second_separation_mm",
                       "ambiguous_second_cycle", "chosen_rank_mean",
                       "chosen_rank_max", "filled_sector_frac",
                       "max_transition_mm", "closed",
                       "n_self_intersections", "proj_disp_p95_mm")}
    if return_candidates:
        # 供 M1 的候选召回诊断（"GT 附近到底在不在 Top-K 里"）
        res.candidates = {"topk": sector_cands, "full": full,
                          "chosen": path, "stats": dp_stats,
                          "mode": cfg.candidate_mode}
    return res
