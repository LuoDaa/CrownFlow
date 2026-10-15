"""候选带 + 每扇区 Top-K 候选（v2 用）。

M1 v1 只用 `candidate_facet_ids`（一条 GT-free 的轴向先验带）；v2 在此之上
按扇区保留 **Top-K** 候选，交给 `graph_search.cyclic_dp_path` 做整圈优化。

候选打分在**行内归一化**（除以该行候选分数的 p95）——这样 λ 的量级跨病例可比，
且归一化不改变 argmax（v1 的结果不受影响）。
"""

from __future__ import annotations

import math

from . import geometry_score


def band_limits(stats, trim_frac):
    hs = [s.h for s in stats]
    lo, hi = min(hs), max(hs)
    span = hi - lo
    return lo + trim_frac * span, hi - trim_frac * span


def candidate_facet_ids(stats, config):
    """返回候选面 id 列表（GT-free）。"""
    h_lo, h_hi = band_limits(stats, config.band_trim_frac)
    return [s.fid for s in stats if h_lo <= s.h <= h_hi]


def _row_p95(vals):
    if not vals:
        return 1.0
    v = sorted(vals)
    return v[min(len(v) - 1, int(0.95 * (len(v) - 1)))] or 1.0


def sector_candidates(stats, adj, frame, config, candidate_ids, tris,
                      return_full=False):
    """→ {sector: [(score_norm, fid), ...] 降序，最多 top_k 个}

    扇区归属按**面的角向跨度**登记（只按质心归属会让粗网格漏扇区 —— 实测
    filled_frac 0.93 → 1.00）。同一面可同时进入相邻扇区的候选池。
    """
    from .graph_search import _sectors_spanned
    score_fn = geometry_score.score_of(config.score)
    by_id = {s.fid: s for s in stats}
    raw = {}
    for fid in candidate_ids:
        sc = score_fn(by_id[fid], stats, adj)
        if sc > 0.0:
            raw[fid] = sc
    if not raw:
        return {}
    norm = _row_p95(list(raw.values()))
    pool = {}
    for fid, sc in raw.items():
        for sec in _sectors_spanned(by_id[fid], frame, tris[fid],
                                    config.n_sectors):
            pool.setdefault(sec, []).append((sc / norm, fid))
    out = {}
    for sec, items in pool.items():
        items.sort(key=lambda x: (-x[0], -by_id[x[1]].area))
    for sec, items in pool.items():
        out[sec] = items[:config.top_k]
    if return_full:
        # 全部候选（按分数降序）——给"GT 附近是否在 Top-K"的诊断用，
        # 只有它在**比 Top-K 更大**的池子上排名，这个诊断才有意义（否则恒真）
        return out, {sec: items for sec, items in pool.items()}
    return out
