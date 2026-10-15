"""M1 v3：Ridge Cluster 候选生成。

**只替换候选生成器**，DP / 硬门 / 平滑 / v2 配置一概不动（用户 2026-09-30 规格）。

v2 的问题（已被证据锁定）：候选是"每扇区按分数排 Top-K 的面"。一条强脊可能占掉
十几个高分面，把真正 Margin 那条较弱脊**挤出 Top-K** —— 召回诊断显示大误差扇区里
GT 最近候选的排名只在 Top-5 占 4~14%。

v3 的改法：

    该扇区局部表面
      → 沿高度方向做局部 profile
      → 找**所有**局部 ridge / local maxima（多信号，不用固定阈值）
      → 相邻面合并成 ridge cluster（一条真脊 = 一个 cluster）
      → 每个 cluster 出 1 个代表候选
      → 交给**同一个** cyclic DP

多信号（因为 Margin 有些地方不是锐肩，fuzzy/chamfer 处 sharp 很小）：

    sharp      按边二面角
    turn       |nx − 邻域均值 nx|      （法向转向）
    rbreak     |径向斜率 − 邻域均值|     （径向剖面折点）
    curv       sharp / 面尺度           （曲率代理）

**判据是"局部极大"而不是"超过某阈值"**：只有当某面在它**轴向上下两侧**的邻居里
都是该信号的局部极大时，它才成为 ridge seed。这样 fuzzy Margin 也能出候选。

代表点取 cluster 内按信号强度加权的质心，并记录 peak/mean score、宽度、支撑面数。
"""

from __future__ import annotations

import math

from .graph_search import _sectors_spanned


def _signal(st, stats, adj, name):
    if name == "sharp":
        return st.sharp_deg
    if name == "turn":
        ns = adj.get(st.fid) or ()
        if not ns:
            return 0.0
        m = sum(stats[g].nx for g in ns) / len(ns)
        return abs(m - st.nx) * 100.0
    if name == "rbreak":
        ns = adj.get(st.fid) or ()
        if not ns:
            return 0.0
        m = sum(stats[g].radial_slope for g in ns) / len(ns)
        return abs(st.radial_slope - m)
    if name == "curv":
        return st.sharp_deg
    raise ValueError("unknown ridge signal %r" % name)


#: 参与 ridge 判定的信号（用户规格：不要只靠 sharp）
RIDGE_SIGNALS = ("sharp", "turn", "rbreak")


def ridge_seeds(stats, adj, sig, dh):
    """局部极大判据：某面在两个**轴向方向**的邻居里都是局部极大。

    ``dh`` 是"上下邻面"的最小轴向间隔（取面尺度量级）。没有低于/高于它的邻居时
    该方向视为不成立（避免把网格边界当脊）。
    """
    seeds = set()
    for s in stats:
        ns = adj.get(s.fid) or ()
        lo = [stats[g] for g in ns if stats[g].h < s.h - dh]
        hi = [stats[g] for g in ns if stats[g].h > s.h + dh]
        if not lo or not hi:
            continue
        v = sig[s.fid]
        if v <= 0.0:
            continue
        if v >= max(sig[g.fid] for g in lo) and v >= max(sig[g.fid] for g in hi):
            seeds.add(s.fid)
    return seeds


class RidgeCluster:
    """一条 ridge 的代表候选（DP 只看到它，而不是十几个高分面）。"""

    # 接口与 FaceStat 对齐：`fid` = `cid`，因此 DP/sector_points 等**一行不用改**
    __slots__ = ("cid", "fid", "h", "r", "centroid", "area", "peak_score",
                 "mean_score", "width_mm", "support_faces", "sharp_deg",
                 "turn", "sectors", "signals", "support")

    def __init__(self, cid):
        self.cid = cid
        self.fid = cid
        self.h = 0.0
        self.r = 0.0
        self.centroid = (0.0, 0.0, 0.0)
        self.area = 0.0
        self.peak_score = 0.0
        self.mean_score = 0.0
        self.width_mm = 0.0
        self.support_faces = 0
        self.sharp_deg = 0.0
        self.turn = 0.0
        self.sectors = set()
        self.signals = ()
        self.support = []        # 支撑面 id 列表（= 该脊实际跨过的面）


def build_clusters(stats, adj, frame, config, candidate_ids, tris):
    """→ (cand_stats, {sector: [(score_norm, cid), ...]})

    ``cand_stats`` 是 RidgeCluster 列表，接口与 FaceStat 子集一致
    （``h`` / ``r`` / ``centroid`` / ``area``），因此 **cyclic_dp_path 一行不改**。
    """
    allow = set(candidate_ids)
    sig = {}
    for name in RIDGE_SIGNALS:
        sig[name] = {s.fid: _signal(s, stats, adj, name) for s in stats}
    # 面尺度量级 → 局部极大的"上下邻面"间隔
    areas = sorted(s.area for s in stats)
    dh = max(0.15, (2.0 * areas[len(areas) // 2]) ** 0.5 * 0.5)
    seeds = {}
    for name in RIDGE_SIGNALS:
        for fid in ridge_seeds(stats, adj, sig[name], dh):
            if fid in allow:
                seeds.setdefault(fid, set()).add(name)
    if not seeds:
        return [], {}
    # --- 连通聚类（按边相邻，仅 seeds）+ 按轴向跨度过大时按最大间隙切分
    out = []
    seen = set()
    for fid in seeds:
        if fid in seen:
            continue
        # **有界区域生长**：轴向跨度与角向跨度同时受限，否则一整圈特征（底盖棱）
        # 会变成一个巨簇并在每个扇区都排第一（实测：环长退化为 0、点数 240→120）
        comp, stack = [], [(fid, stats[fid].theta)]
        seen.add(fid)
        th_max = math.radians(config.ridge_max_span_deg)
        h_lim = config.ridge_split_span_factor * dh
        while stack:
            u, th0 = stack.pop()
            comp.append(u)
            for g in adj.get(u, ()):
                if g not in seeds or g in seen:
                    continue
                dth = abs(stats[g].theta - th0)
                dth = min(dth, 2.0 * math.pi - dth)      # 跨 0/2π
                if dth > th_max:
                    continue
                hs = [stats[f].h for f in comp] + [stats[g].h]
                if max(hs) - min(hs) > h_lim:
                    continue
                seen.add(g)
                stack.append((g, th0))
        for part in _split_by_axial_gap(comp, stats, dh, config):
            c = RidgeCluster(len(out))
            tot_w = 0.0
            cx = cy = cz = 0.0
            hs, rs, sc = [], [], []
            for f in part:
                s0 = stats[f]
                w = s0.area
                cx += s0.centroid[0] * w
                cy += s0.centroid[1] * w
                cz += s0.centroid[2] * w
                tot_w += w
                hs.append(s0.h)
                rs.append(s0.r)
                sc.append(max(sig[n][f] for n in seeds[f]))
            if tot_w <= 0.0:
                continue
            c.centroid = (cx / tot_w, cy / tot_w, cz / tot_w)
            c.area = tot_w
            c.h = sum(hs) / len(hs)
            c.r = sum(rs) / len(rs)
            c.peak_score = max(sc)
            c.mean_score = sum(sc) / len(sc)
            c.width_mm = max(hs) - min(hs)
            c.support_faces = len(part)
            c.sharp_deg = max(stats[f].sharp_deg for f in part)
            c.turn = max(sig["turn"][f] for f in part)
            c.signals = tuple(sorted({n for f in part for n in seeds[f]}))
            c.support = list(part)
            out.append(c)
    # --- 代表性候选：过滤过小簇，赋扇区
    keep = [c for c in out if c.support_faces >= config.ridge_min_support]
    # **重编号**：DP 用 `stats[cid]` 索引候选表，所以 cid 必须等于在 keep 里的下标
    for i, c in enumerate(keep):
        c.cid = i
        c.fid = i
    by_id = {s.fid: s for s in stats}
    cands = {}
    for c in keep:
        # 一条脊按其支撑面的**角向跨度**登记到各扇区（与 v2 同规则）
        for f in c.support:
            for sec in _sectors_spanned(by_id[f], frame, tris[f],
                                        config.n_sectors):
                cands.setdefault(sec, []).append((c.peak_score, c.cid))
                c.sectors.add(sec)
    # 行内归一化（与 v2 口径一致），每扇区取前 ridge_top_k 个簇
    allsc = [x[0] for v in cands.values() for x in v]
    norm = _row_p95(allsc)
    sector_cands = {}
    for sec, items in cands.items():
        # 同一簇可能经多个支撑面进入同一扇区 → 去重
        best = {}
        for sc0, cid in items:
            if cid not in best or sc0 > best[cid]:
                best[cid] = sc0
        ranked = sorted(best.items(), key=lambda kv: -kv[1])
        sector_cands[sec] = [(sc0 / norm, cid) for cid, sc0 in
                             ranked[:config.ridge_top_k]]
    return keep, sector_cands


def _split_by_axial_gap(comp, stats, dh, config, depth=0):
    """把轴向跨度过大的连通 seed 分量**递归**切成多条脊。

    为什么必须递归（实测踩到）：合成件上一条沿整面墙的 seed 链跨了 5.67mm，
    一次只切两半仍留下大簇；那个巨簇在**每个扇区**都是最高分，于是"环"退化成
    同一簇的 120 个拷贝（重采样因总长为 0 直接早退，点数 240→120）。
    真脊是"曲线"：在某角度位置，它的面在轴向上只跨一两个面宽。
    """
    limit = config.ridge_split_span_factor * dh
    hs = sorted(stats[f].h for f in comp)
    span = hs[-1] - hs[0]
    if span <= limit or len(comp) < 4 or depth >= 6:
        return [comp]
    gap, at = 0.0, -1
    for i in range(len(hs) - 1):
        d = hs[i + 1] - hs[i]
        if d > gap:
            gap, at = d, i
    thr = hs[at] + gap * 0.5 if at >= 0 else None
    if thr is None:
        return [comp]
    a = [f for f in comp if stats[f].h <= thr]
    b = [f for f in comp if stats[f].h > thr]
    if not a or not b:
        return [comp]
    return (_split_by_axial_gap(a, stats, dh, config, depth + 1)
            + _split_by_axial_gap(b, stats, dh, config, depth + 1))


def _row_p95(vals):
    if not vals:
        return 1.0
    v = sorted(vals)
    return v[min(len(v) - 1, int(0.95 * (len(v) - 1)))] or 1.0
