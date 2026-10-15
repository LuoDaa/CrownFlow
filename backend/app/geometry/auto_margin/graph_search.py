"""在网格上搜索闭环。

两个方法，**同一套候选、同一套特征**，只差"是否整圈一起优化"：

* ``argmax_path``      —— M1 v1：每扇区独立取最高分。
* ``cyclic_dp_path``   —— M1 v2：扇区×Top-K 候选 格点上的**环状**最小代价路径。

v2 的能量（用户 2026-09-30 规格）：

    E(P) = Σ_i E_data(c_i)
         + λ_h Σ ρ(h_i − h_{i−1})
         + λ_r Σ ρ(r_i − r_{i−1})
         + λ_p Σ ρ(‖p_i − p_{i−1}‖)

    E_data = −归一化几何分数；ρ = Huber（拐点 huber_delta_mm）

并带**跳变硬门**：|Δh| > max_dh_mm 或 |Δr| > max_dr_mm 的转移直接禁止
（阈值由 train 的 GT 相邻扇区分布推出，见 tools/derive_m1_gates.py）。

**闭环是求出来的，不是把开放路径首尾缝起来的**：枚举 sector 0 的每个候选作为
起点，跑线性 DP，最后强制回到该起点，取全局最优。

v1 的弱点（本文件存在的理由）：一个扇区内若有多条脊（底盖棱、解剖脊），
独立 argmax 会在少数扇区跳到别的脊 —— 实测中位 0.07mm 已很好，但 p95 1.2~1.8mm。
"""

from __future__ import annotations

import math

from . import geometry_score


def _sectors_spanned(st, frame, tri, n_sectors):
    """面按**角向跨度**覆盖的扇区集合。

    只按质心归属会让粗糙网格漏扇区（实测合成件 filled_frac 仅 0.93：
    脊面质心落在相邻扇区）。按顶点角向跨度登记后回到 ~1.0，真实密集扫描网格
    同样受益（不漏空、也不人为补点）。
    """
    ths = [frame.coords(v)[2] for v in tri]
    lo, hi = min(ths), max(ths)
    step = 2.0 * math.pi / n_sectors
    if hi - lo > math.pi:            # 跨 0/2π 的面：反向取跨度
        lo, hi = hi, lo + 2.0 * math.pi
    i0 = int(lo / step)
    i1 = int(hi / step)
    out = {(i % n_sectors) for i in range(i0, i1 + 1)}
    if not out:
        out = {int(st.theta / step) % n_sectors}
    return out


def argmax_path(sector_cands, stats):
    """v1：扇区 → 该扇区最高分面（候选池已按分数降序，取第一个）。"""
    return {sec: cands[0][1] for sec, cands in sector_cands.items() if cands}


def sector_points(sector_faces, stats):
    """扇区 → 面质心点列（按扇区号排序，天然闭环顺序）。"""
    by_id = {s.fid: s for s in stats}
    return [by_id[sector_faces[sec]].centroid
            for sec in sorted(sector_faces)]


def _huber(x, delta):
    x = abs(x)
    if x <= delta:
        return 0.5 * x * x
    return delta * (x - 0.5 * delta)


def cyclic_dp_path(sector_cands, stats, frame, config, surface=None):
    """环状最小代价路径 → (path, E1, E2)（每个出现过的扇区恰好一个）。

    ``path`` = {sector: fid}；``E1`` = 最优环能量；``E2`` = **起点候选不同**的
    次优环能量（用于置信度/歧义：两者接近说明病例本身有歧义 → REVIEW）。
    不可行时返回 ``({}, None, None)``。

    ``sector_cands`` = {sector: [(score_norm, fid), ...] 降序}。
    找不到任何可行闭环（全被硬门挡住）时返回 ``{}``，由调用方回退并报 REVIEW。
    """
    secs = sorted(sector_cands)
    n = len(secs)
    if n < 12:
        return {}, None, None
    cands = [[fid for _s, fid in sector_cands[s]] for s in secs]
    snorm = [[sc for sc, _f in sector_cands[s]] for s in secs]
    H = [[stats[fid].h for fid in cs] for cs in cands]
    R = [[stats[fid].r for fid in cs] for cs in cands]
    P = [[stats[fid].centroid for fid in cs] for cs in cands]
    lam_h, lam_r, lam_p = config.lambda_h, config.lambda_r, config.lambda_p
    dlt = config.huber_delta_mm
    # 有效硬门 = train 冻结的 base + 离散度项（网格面尺度）
    disc = getattr(config, "_face_scale_mm", 0.0) * config.gate_disc_factor
    # 每扇区名义弧长（测地代价的期望值）：2π·r̄ / n_sectors
    rs_all = [stats[fid].r for cs in cands for fid in cs]
    expected_step = (2.0 * math.pi * (sum(rs_all) / max(len(rs_all), 1))
                     / max(config.n_sectors, 1))

    def step_cost(i, k, j, l):
        # **统一入口**：可行性 + 代价由 transition_mode 决定（三模式共用平滑项）
        return transition_cost(config.transition_mode, stats[cands[i][k]],
                               stats[cands[j][l]], config, surface,
                               expected_step)

    results = []          # (total, start, backs, last) —— 每个起点一条
    k0 = len(cands[0])
    for start in range(k0):
        dp = [float("inf")] * k0
        dp[start] = -snorm[0][start]
        backs = []
        for i in range(1, n):
            ki, kp = len(cands[i]), len(cands[i - 1])
            ndp = [float("inf")] * ki
            bk = [-1] * ki
            for l in range(ki):
                bestv, bestk = float("inf"), -1
                for k in range(kp):
                    if dp[k] == float("inf"):
                        continue
                    c = step_cost(i - 1, k, i, l)
                    if c is None:
                        continue
                    v = dp[k] + c
                    if v < bestv:
                        bestv, bestk = v, k
                if bestk >= 0:
                    ndp[l] = bestv - snorm[i][l]
                    bk[l] = bestk
            dp = ndp
            backs.append(bk)
        for l in range(len(cands[n - 1])):
            if dp[l] == float("inf"):
                continue
            c = step_cost(n - 1, l, 0, start)
            if c is None:
                continue
            results.append((dp[l] + c, start, [list(b) for b in backs], l))
    if not results:
        return {}, None, None
    results.sort(key=lambda x: x[0])
    best_total, start, backs, last = results[0]
    path = {secs[n - 1]: cands[n - 1][last]}
    cur = last
    for i in range(n - 1, 0, -1):
        cur = backs[i - 1][cur]
        if cur < 0:
            return {}, None, None
        path[secs[i - 1]] = cands[i - 1][cur]
    if path[secs[0]] != cands[0][start]:
        return {}, None, None
    # 次优：**起点候选不同**的另一条环（结构性不同的解，用于能量间隙）
    second = None
    for tot, st2, _bk, _ls in results[1:]:
        if st2 != start:
            second = tot
            break
    return path, best_total, second

def structurally_distinct_second(sector_cands, stats, frame, config, best_path,
                                 ban_radius_mm=None, surface=None):
    """**结构性不同的**次优环 → (path2, E2, separation_mm)。

    为什么不是"屏蔽最优解的一条边"（用户 2026-10-01 §5）：那样得到的几乎还是
    同一条 Margin（只是某个扇区换了个邻居候选），能量差没有歧义意义。
    这里改成把 P1 在每个扇区选中的候选**及其 3D 邻域**整片禁用后重解，
    并额外给出 ``separation_mm`` = 两条环逐扇区对应点距离的中位数 —— 只有
    separation 足够大时，能量差才代表"存在另一条竞争 Margin"。

    **只用于记录/置信度，不参与拒绝**（阈值未验证前不许用来卡病例）。
    """
    if not best_path or not sector_cands:
        return {}, None, None
    disc = getattr(config, "_face_scale_mm", 0.3) or 0.3
    ban = ban_radius_mm if ban_radius_mm is not None else max(1.0, 3.0 * disc)
    banned = set()
    for sec, fid in best_path.items():
        if fid >= len(stats):
            continue
        p0 = stats[fid].centroid
        for _sc, cid in sector_cands.get(sec, ()):
            if cid < len(stats) and math.dist(stats[cid].centroid, p0) <= ban:
                banned.add((sec, cid))
    reduced = {}
    for sec, items in sector_cands.items():
        keep = [(sc, cid) for sc, cid in items if (sec, cid) not in banned]
        if keep:
            reduced[sec] = keep
    if len(reduced) < 12:
        return {}, None, None
    p2, e2, _e3 = cyclic_dp_path(reduced, stats, frame, config, surface)
    if not p2:
        return {}, None, None
    dists = sorted(math.dist(stats[best_path[s0]].centroid, stats[p2[s0]].centroid)
                   for s0 in (set(best_path) & set(p2))
                   if best_path[s0] < len(stats) and p2[s0] < len(stats))
    sep = dists[len(dists) // 2] if dists else None
    return p2, e2, sep


# ============================================================ ② transition 层
# 用户 2026-10-01 规格：只替换"是否允许连接 + transition cost"，DP 主体不动。
#   V2_FROZEN  固定 Δh/Δr + face_scale（Champion baseline，永远保留）
#   MESH_AWARE 真实 face-dual 图上的**局部**表面连通性
#   HYBRID     mesh 连通性为主 + 很宽松的 h/r sanity
import heapq


class SurfaceDistance:
    """face-dual 图上的**局部**最短表面路径（带缓存）。

    * 边权 = 两个面质心的 3D 距离
    * 搜索被 ``radius_mm`` 截断 —— 只在局部邻域内找路径
    * ``detour`` = d_geo / d_euclid，用于识别"绕远才连通"的伪相邻
    """

    def __init__(self, tris, adj, centroids):
        self.tris = tris
        self.adj = adj
        self.cen = centroids
        self._cache = {}

    def geo(self, a, b, radius_mm):
        """→ (d_geo 或 None, 访问面数)。None = 局部半径内不可达。"""
        key = (a, b, radius_mm)
        hit = self._cache.get(key)
        if hit is not None:
            return hit
        dist = {a: 0.0}
        pq = [(0.0, a)]
        out = (None, 0)
        while pq:
            d, f = heapq.heappop(pq)
            if f == b:
                out = (d, len(dist))
                break
            if d > radius_mm:
                continue
            cf = self.cen.get(f)
            if cf is None:
                continue
            for g in self.adj.get(f, ()):
                cg = self.cen.get(g)
                if cg is None:
                    continue
                w = math.dist(cf, cg)
                nd = d + w
                if nd <= radius_mm and nd < dist.get(g, float("inf")):
                    dist[g] = nd
                    heapq.heappush(pq, (nd, g))
        self._cache[key] = out
        return out


def transition_cost(mode, st_a, st_b, cfg, surf, expected_step_mm):
    """→ (cost 或 None)。None = 该转移被**禁止**。

    三种模式共用同一套平滑项（λh/λr/λp + Huber）；差别只在"可行性"与
    MESH_AWARE/HYBRID 额外加的测地代价。
    """
    dh = abs(st_b.h - st_a.h)
    dr = abs(st_b.r - st_a.r)
    d_euc = math.dist(st_a.centroid, st_b.centroid)
    base = (cfg.lambda_h * _huber(dh, cfg.huber_delta_mm)
            + cfg.lambda_r * _huber(dr, cfg.huber_delta_mm)
            + cfg.lambda_p * _huber(d_euc, cfg.huber_delta_mm))
    if mode == "V2_FROZEN":
        disc = (getattr(cfg, "_face_scale_mm", 0.0) or 0.0) * cfg.gate_disc_factor
        if dh > cfg.max_dh_mm + disc or dr > cfg.max_dr_mm + disc:
            return None
        return base
    # HYBRID：先用很宽松的 sanity 挡明显离谱的跳跃
    if mode == "HYBRID":
        if dh > cfg.sanity_dh_mm or dr > cfg.sanity_dr_mm:
            return None
    if surf is None:
        return None
    d_geo, _n = surf.geo(st_a.fid, st_b.fid, cfg.surface_search_radius_mm)
    if d_geo is None:
        return None                      # 局部不可达
    detour = d_geo / max(d_euc, 1e-6)
    if detour > cfg.max_detour_ratio:
        return None                      # 可达但要绕远 → 不是合理局部 transition
    exp = max(expected_step_mm, 1e-6)
    return base + cfg.lambda_geo * _huber(d_geo / exp, cfg.huber_delta_mm)
