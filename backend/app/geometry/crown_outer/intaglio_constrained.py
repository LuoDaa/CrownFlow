"""J.1 —— Intaglio-Constrained Crown Outer（外表面被内表面 + 最小厚度包络约束）。

用户 2026-10-02 规格：

    输入：Intaglio + FDI anatomy template + Cervical loop + MinThickness field
    输出：Outer
    核心约束：
      1. 共享颈缘边界（颈环钉死，位移精确 0）
      2. Outer 包住 Intaglio
      3. 局部厚度 ≥ required thickness
      4. 解剖偏差尽量小
      5. 无翻面
      6. 无自交

## 为什么不是 "Outer = Intaglio + normal·T"

牙尖/窝沟/邻接/颈缘都是凹面 —— 法向 offset 会自交，且凹面处法向本身不可靠
（Occlusion 留下的翻面会让"局部法向"直接说谎）。所以改用**距离场包络**：

    Envelope = { x : s(x) < T(x) }     s = 到 Intaglio 的**带符号**距离（材料侧为正）
    约束     : 任何 Outer 点都必须落在 Envelope 之外，即 s(p) ≥ T(p)

即"内表面外面套一层最小材料厚度保护壳"。它是**隐式定义**的（距离场水平集），
**构造上不可能自交**。

**必须用带符号距离**：外表面若整片落进预备体内部（真实例的覆盖率失败），最近距离
照样很大（= 到内表面的距离），只有符号能区分"在外侧太近"与"根本在预备体里面"。

## 为什么要**两个方向**都测（本模块的核心教训）

顶点级约束**兜不住面内部**：一个粗三角形三个顶点都 ≥T，它中间仍可能离内表面更近
（合成例实测 0.42 vs 要求 0.50）。而**验收口径是内表面侧的 t_closest**
（Thickness Audit 从内表面采样点测距）。所以每次迭代同时做两件事：

    (a) 顶点侧：Outer 顶点 → 内表面距离 < T 的，推到 cp + n̂·(T+safety)
    (b) 内表面侧（**验收口径**）：Intaglio 采样点 → Outer 距离 < T 的，
        把它最近的 Outer 顶点推到 q + n̂(q)·(T+safety)

并且工作副本先做**保形细分**（几何不变），把两个方向的离散误差压到远小于 T。

## 求解

沿用已验证的**局部带状 Laplacian**（Contact v2 / I.2 / Occlusion 同一套）：

```
违反包络的顶点（seed）→ 钉在 fixed 集里的目标位（α 由 line search 定）
过渡环（seed 邻域 BFS，避开颈环）→ 自由变量，Laplacian 平滑混合
        ↓
接受条件 = 双向最大 deficit 严格下降 AND 无新翻面 AND 无自交
           AND 锁定集（颈环+刀口带+远场）位移精确 0
```

**解剖偏差最小化**：只有违反包络的顶点被推，且只推到"刚好达标 + safety"。

工具/验收：``tools/run_outer_v2_case.py``（H.1 → J.1 → Thickness Audit 复验）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace

import numpy as np
from scipy.spatial import cKDTree

from .laplacian import LocalBandError, solve_local_band
from .thickness import (
    SurfaceQuery,
    ThicknessParameters,
    _boundary_edges,
    geometry_safety_gate,
    refine_max_edge,
    thresholds_of,
    vertex_normals,
)

STATUS_OK = "OK"
STATUS_REVIEW_REQUIRED = "REVIEW_REQUIRED"
STATUS_CONFLICT = "CONSTRAINT_CONFLICT"
STATUS_FAILED = "FAILED"


class OuterV2Error(Exception):
    pass


@dataclass
class OuterV2Params:
    """``thickness`` 必须与 Thickness Audit 用的是同一份设计规则（同一个对象）。"""

    thickness: ThicknessParameters
    #: 目标 = T + safety（离散化 + 面内部余量，见模块 docstring）
    safety_mm: float = 0.05
    #: 工作副本细分到多大边长（几何不变；压两个方向的离散误差）
    max_work_edge_mm: float = 0.35
    #: 内表面查询网格细分（Audit 默认同值）
    max_query_edge_mm: float = 0.35
    max_outer_iterations: int = 12
    line_search_alphas: tuple = (1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125,
                                 0.015625)
    transition_rings: int = 4
    deficit_tol_mm: float = 0.01
    #: 颈缘刀口带（外表面侧）：边界钉死，这里推会造唇边
    knife_edge_mm: float = 0.10
    #: 内表面侧的排除带：与 Audit 的 boundary_safety_mm 同值。
    #  颈缘处两面**按构造**重合（共享颈缘边界）→ 壁厚必然 →0，在那里强制 T
    #  等于跟"共享边界"这条约束打架，而且规则本身在 d=0 处给的是近颈缘值不适用；
    #  第一版只用 knife_edge=0.1 排除 → 卡在 z=0 的内表面环上（deficit 恒 0.5002）。
    intaglio_exclude_mm: float = 0.60
    #: 硬门。**翻面用面积加权**：大面翻向 = 折叠（必须 0）；数值级微面翻向
    #  不是折叠 —— 计数口径会把后者当缺陷，于是大变形时每一步都被拒（真实例
    #  实测 α=0.0625 仅 1 个微面翻向就被拒，一步都走不了）。计数门设为 None 关闭，
    #  面积门设极小值；自交门保持 0（那才是真折叠的判据）。
    max_flip_ratio: float | None = None
    max_flipped_area_ratio: float | None = 1e-4
    max_self_intersections: int = 0


@dataclass
class OuterV2Result:
    status: str = STATUS_FAILED
    mesh: object | None = None
    iterations: int = 0
    alphas_accepted: list = field(default_factory=list)
    history: list = field(default_factory=list)
    envelope_before: dict = field(default_factory=dict)
    envelope_after: dict = field(default_factory=dict)
    #: 解剖偏差 = **原始 anatomy 顶点**位移统计（细分只加新点，原顶点仍是子集）
    anatomy_deviation: dict = field(default_factory=dict)
    safety: dict = field(default_factory=dict)
    work_faces: int = 0
    notes: list = field(default_factory=list)

    def summary(self) -> dict:
        return {"status": self.status, "iterations": self.iterations,
                "alphas": self.alphas_accepted,
                "envelope_before": self.envelope_before,
                "envelope_after": self.envelope_after,
                "anatomy_deviation": self.anatomy_deviation,
                "safety_ok": self.safety.get("ok"),
                "safety_violations": self.safety.get("violations"),
                "safety_metrics": self.safety.get("metrics"),
                "work_faces": self.work_faces}


def _loop_indices(faces) -> np.ndarray:
    be = _boundary_edges(np.asarray(faces, dtype=np.int64))
    if len(be) == 0:
        raise OuterV2Error("anatomy 网格没有开放边界 —— 颈环缺失")
    return np.asarray(sorted({int(v) for e in be for v in e}), dtype=np.int64)


def build_intaglio_constrained_outer(anatomy_mesh, intaglio_mesh,
                                     params: OuterV2Params) -> OuterV2Result:
    """把解剖模板（已摆位、颈环已对齐）约束到 Intaglio 的最小厚度包络之外。"""
    if params is None or params.thickness is None:
        raise OuterV2Error("OuterV2Params.thickness（设计规则）必须显式给出")
    tp = params.thickness
    res = OuterV2Result()
    av_in = np.asarray(anatomy_mesh.vertices, dtype=np.float64)
    af_in = np.asarray(anatomy_mesh.faces, dtype=np.int64)
    iv = np.asarray(intaglio_mesh.vertices, dtype=np.float64)
    ifa = np.asarray(intaglio_mesh.faces, dtype=np.int64)
    if len(af_in) == 0 or len(ifa) == 0:
        raise OuterV2Error("输入网格为空")

    # ---- 工作副本：保形细分（几何不变）。顶点级约束兜不住面内部，
    #      细分把"任意面内点到最近顶点"的距离上界压到 max_work_edge 以下。
    av, af = refine_max_edge(av_in, af_in, params.max_work_edge_mm)
    res.work_faces = len(af)
    res.notes.append("工作副本：%d 面 → %d 面（最长边 ≤ %.3fmm，几何不变）"
                     % (len(af_in), len(af), params.max_work_edge_mm))

    loop_idx = _loop_indices(af)
    loop_set = np.zeros(len(av), dtype=bool)
    loop_set[loop_idx] = True
    margin_tree = cKDTree(av[loop_idx])
    dm_out = margin_tree.query(av, k=1)[0]
    thr_out, _ = thresholds_of(dm_out, tp)
    # 颈环永不作为 seed（ev 里排除）；阈值给 0 而不是 inf —— inf*0 会出 NaN
    thr_out = np.where(loop_set, 0.0, thr_out)

    # ---- 内表面侧（验收口径）
    sq_in = SurfaceQuery(iv, ifa, boundary_safety_mm=0.6,
                         k_candidates=24,
                         max_query_edge_mm=params.max_query_edge_mm)
    vn_in = vertex_normals(iv, ifa)                    # 内表面外法向（材料侧）
    dm_in = cKDTree(av[loop_idx]).query(iv, k=1)[0]
    thr_in, _ = thresholds_of(dm_in, tp)
    # 内表面侧只对"离开刀口带"的点强制（颈缘处两面重合，本来就没有材料）
    in_evaluable = dm_in > params.intaglio_exclude_mm

    # ---- 顶点侧到内表面的距离
    sq_out = None

    def _push_dir(v, cp, sdist, n_hat):
        """推的方向：**径向**（背离最近点）而不是法向。

        法向投影 `cp + n̂·T` 在**凸棱/尖角**上是个不动点 —— 顶点法向在棱上被
        平滑成倾斜方向，|cp + n̂·T − 棱| = T·cosθ < T，于是每轮"目标=当前位置"，
        deficit 永远降不下去（实测卡在 0.5002 一动不动）。径向推到 |v−cp| = T
        则对光顺面与棱角都给出恰好 T（cp 仍是最近点）。
        例外：v 落在预备体**内部**（s<0）时径向方向朝里，必须走法向穿出去。
        """
        rad = v - cp
        rlen = np.linalg.norm(rad, axis=1)
        use_n = (sdist < 0.0) | (rlen < 1e-9)
        d = np.where(use_n, 1.0, np.maximum(rlen, 1e-12))
        return np.where(use_n[:, None], n_hat, rad / d[:, None])

    def measure(v):
        """双向测量 → (顶点侧 deficit, 内表面侧 deficit, 逐顶点目标建议)。

        ⚠️ 内表面侧必须对**当前位形**测距：`sq_out` 建一次、每步用
        ``update_vertices`` 同步（第一版建一次就不管了 → 永远在跟初始 anatomy
        比距离 → 所有 α 都"未下降"，假 CONSTRAINT_CONFLICT）。
        """
        q_cp, q_f, q_s, _q_d, _q_v, _q_fl = sq_in.query(v)
        n_hat = sq_in.closest_normal(q_f)
        def_v = np.maximum(thr_out + params.safety_mm - q_s, 0.0)
        tgt_v = q_cp + _push_dir(v, q_cp, q_s, n_hat)             * (thr_out + params.safety_mm)[:, None]
        sq_out.update_vertices(v)
        d_out = sq_out.query(iv)[3]
        def_q = np.where(in_evaluable,
                         np.maximum(thr_in + params.safety_mm - d_out, 0.0),
                         0.0)
        return def_v, def_q, q_s, tgt_v, n_hat, q_s

    # 内表面点 → 最近工作副本顶点的映射（每轮重算：顶点在动）
    movable = np.flatnonzero(~loop_set)          # 颈环永不作 seed（共享边界）

    def map_q_to_v(v, def_q, tgt_base):
        """内表面侧：对每个违反点 q，把它**局部球内所有可动顶点**推出去。

        为什么不是"只推最近的那个顶点"：球半径 r 里的三角形，只要三个顶点都在
        r 球外，整个三角形就在球外（球的凸性）—— 所以按球推能给出**面级**保证。
        只推最近顶点则会在"最贴 q 的那块面属于别的位置"时变成不动点（实测：
        q 的最近顶点已经恰好 0.6，但 0.135 处有另一片面卡着 → deficit 恒
        0.4651 永不下降）。
        颈环永不在 movable 里（共享颈缘边界）。
        """
        ids = np.flatnonzero(def_q > params.deficit_tol_mm)
        if len(ids) == 0:
            return ids, np.zeros(0, dtype=np.int64), tgt_base
        tree = cKDTree(v[movable])
        r_ball = thr_in[ids] + params.safety_mm + params.max_work_edge_mm
        balls = tree.query_ball_point(iv[ids], r=r_ball)
        tgt = tgt_base.copy()
        push = np.zeros(len(v))
        for k, qid in enumerate(ids):
            vs = movable[np.asarray(balls[k], dtype=np.int64)]
            if len(vs) == 0:
                continue
            q = iv[qid]
            rad = v[vs] - q
            ln = np.maximum(np.linalg.norm(rad, axis=1), 1e-12)
            need = ln < thr_in[qid] + params.safety_mm - 1e-9
            # 只推"真的在包络里"的顶点：球里已经 ≥T 的顶点不动 —— 否则会把
            # 顶盖顶点朝轴心方向推（对 q 的径向在远处几乎是切向），把别处压坏
            # （实测 deficit 反而从 0.586 涨到 0.702）。
            vs, rad, ln = vs[need], rad[need], ln[need]
            if len(vs) == 0:
                continue
            cand = q + rad / ln[:, None] * (thr_in[qid] + params.safety_mm)
            d = np.linalg.norm(cand - v[vs], axis=1)
            upd = d > push[vs]
            tgt[vs[upd]] = cand[upd]
            push[vs[upd]] = d[upd]
        sel = push > 0
        return ids, np.flatnonzero(sel), tgt      # tgt 是全长度数组

    sq_out = SurfaceQuery(av, af, boundary_safety_mm=0.6,
                          k_candidates=24, max_query_edge_mm=0.0)
    q0 = sq_in.query(av)
    n0 = int((~loop_set & (dm_out >= params.knife_edge_mm)).sum())
    if n0 and int((q0[2][~loop_set & (dm_out >= params.knife_edge_mm)] < 0)
                  .sum()) > 0.5 * n0:
        raise OuterV2Error(
            "anatomy 多数顶点落在 Intaglio 内侧 —— 输入或内表面法向方向不对，"
            "**不自动翻号**，fail loud（自动翻号会把方向错误藏起来）")

    def report(v, def_v, def_q, s_signed):
        ev = ~loop_set & (dm_out >= params.knife_edge_mm)
        so = float(def_v[ev].sum())
        si = float(def_q.sum())
        return {"n_violating_outer": int((def_v[ev]
                                          > params.deficit_tol_mm).sum()),
                "n_violating_intaglio": int((def_q
                                             > params.deficit_tol_mm).sum()),
                "max_deficit_mm": float(max(def_v[ev].max() if n0 else 0.0,
                                            def_q.max() if len(def_q) else 0.0)),
                # 落在预备体内部（负壁厚）的顶点数：J.1 的首要目标
                "n_inside_prep": int((s_signed[ev] < 0).sum()),
                "sum_deficit_outer_mm": so,
                "sum_deficit_intaglio_mm": si,
                # 接受判据用**总违规量**而不是 max：max 可能被单个病态顶点
                # （局部凹槽互拉/输入本身不覆盖）钉死，于是每个 α 都被拒、
                # 整体一步都走不了（实测卡在 0.787 一动不动）。max 照报不误。
                "energy_mm": so + si}

    def_v0, def_q0, sv0, _t, _n, _ss = measure(av)
    res.envelope_before = report(av, def_v0, def_q0, sv0)
    res.history.append({"iteration": 0, **res.envelope_before})
    if res.envelope_before["max_deficit_mm"] <= params.deficit_tol_mm:
        res.notes.append("起始就没有包络违反（anatomy 已包住 intaglio + T）")

    adj: dict[int, set] = {}
    for a, b, c in af:
        for u, v in ((int(a), int(b)), (int(b), int(c)), (int(c), int(a))):
            adj.setdefault(u, set()).add(v)
            adj.setdefault(v, set()).add(u)

    best_v = av.copy()
    best_max = res.envelope_before["max_deficit_mm"]
    best_energy = res.envelope_before["energy_mm"]
    last_measure = (def_v0, def_q0, sv0, _t, _n, _ss)   # 接受态的量测缓存
    conflict = False
    it = 0
    for it in range(1, params.max_outer_iterations + 1):
        if best_max <= params.deficit_tol_mm:
            res.notes.append("第 %d 轮：双向最大 deficit %.4f ≤ tol %.4f，收敛"
                             % (it - 1, best_max, params.deficit_tol_mm))
            break
        if best_energy <= 1e-9:
            res.notes.append("第 %d 轮：总违规量 ~0，收敛" % (it - 1))
            break
        def_v, def_q, s_in, tgt_v, _n_hat, _s_signed = last_measure
        ev = ~loop_set & (dm_out >= params.knife_edge_mm)
        a_ids = np.flatnonzero(ev & (def_v > params.deficit_tol_mm))
        active = np.zeros(len(best_v), dtype=bool)
        active[a_ids] = True
        # 目标只对 seed 写；其余顶点保持原位（否则"锁定集"会被整片改写 —— 实测
        # 每个 α 都报"锁定集被移动 0.099"，因为 tgt_v 是**全顶点**的投影结果）
        tgt = best_v.copy()
        tgt[a_ids] = tgt_v[a_ids]
        q_ids, q_near, q_tgt = map_q_to_v(best_v, def_q, tgt)
        tgt = q_tgt                            # 内表面侧 = 验收口径 → 优先级更高
        active[q_near] = True
        act_ids = np.flatnonzero(active)
        if len(act_ids) == 0:
            break
        # 自由区 = seed 邻域 BFS（避开颈环）；seed 本身钉在 fixed 集的目标位
        free: set = set()
        cur_s = set(int(i) for i in act_ids)
        seen = set(cur_s) | set(int(i) for i in np.flatnonzero(loop_set))
        for _ in range(params.transition_rings):
            nxt = set()
            for v_ in cur_s:
                for w_ in adj.get(v_, ()):
                    if w_ not in seen:
                        nxt.add(w_)
            free |= nxt
            seen |= nxt
            cur_s = nxt
            if not cur_s:
                break
        free_ids = np.asarray(sorted(free), dtype=np.int64)
        fixed_ids = np.setdiff1d(np.arange(len(best_v)), free_ids)
        # 真正"不许动"的是 locked = fixed ∖ seeds（颈环 + 刀口带 + 远场）。
        locked_ids = np.setdiff1d(fixed_ids, act_ids)
        accepted = False
        for alpha in params.line_search_alphas:
            xt = tgt.copy()
            xt[act_ids] = best_v[act_ids] + alpha * (tgt[act_ids]
                                                     - best_v[act_ids])
            if len(free_ids) == 0:
                v_cand = xt.copy()
            else:
                try:
                    v_cand, _info = solve_local_band(best_v, af, free_ids,
                                                     fixed_ids, xt)
                except LocalBandError as exc:
                    res.notes.append("α=%.3f 局部求解失败: %s" % (alpha, exc))
                    continue
            moved_locked = np.abs(v_cand - best_v)[locked_ids].max() \
                if len(locked_ids) else 0.0
            if moved_locked != 0.0:
                res.notes.append("α=%.3f 锁定集被移动（%.2e）—— 拒绝"
                                 % (alpha, moved_locked))
                continue
            dv2, dq2, sv2, _t2, _n2, _ss2 = measure(v_cand)
            rep2 = report(v_cand, dv2, dq2, sv2)
            if rep2["energy_mm"] >= best_energy - 1e-12:
                res.notes.append("α=%.3f 总违规未下降（%.3f→%.3f mm，max "
                                 "%.4f→%.4f）—— 拒绝"
                                 % (alpha, best_energy, rep2["energy_mm"],
                                    best_max, rep2["max_deficit_mm"]))
                continue
            gate = geometry_safety_gate(
                v_cand, af, reference_vertices=av,
                max_flip_ratio=params.max_flip_ratio,
                max_flipped_area_ratio=params.max_flipped_area_ratio,
                max_self_intersections=params.max_self_intersections,
                expect_open_boundary=True)
            if not gate["ok"]:
                res.notes.append("α=%.3f 几何安全门未过：%s —— 拒绝"
                                 % (alpha, gate["violations"]))
                continue
            best_v = v_cand
            best_max = rep2["max_deficit_mm"]
            best_energy = rep2["energy_mm"]
            last_measure = (dv2, dq2, sv2, _t2, _n2, _ss2)
            res.alphas_accepted.append(float(alpha))
            accepted = True
            res.history.append({
                "iteration": it, "alpha": float(alpha),
                "n_active": int(len(act_ids)), "n_free": int(len(free_ids)),
                **rep2,
                "self_intersections": gate["metrics"].get(
                    "self_intersections"),
                "flipped_ratio": gate["metrics"].get("flipped_ratio")})
            break
        if not accepted:
            conflict = True
            res.notes.append("第 %d 轮：所有 α 都被拒（双向 deficit %.4f 未下降"
                             "或几何安全门未过）—— 停止，不硬推到折叠"
                             % (it, best_max))
            break

    def_v_f, def_q_f, sv_f, _t_f, _n_f, _ss_f = measure(best_v)
    res.envelope_after = report(best_v, def_v_f, def_q_f, sv_f)
    res.safety = geometry_safety_gate(
        best_v, af, reference_vertices=av,
        max_flip_ratio=params.max_flip_ratio,
        max_flipped_area_ratio=params.max_flipped_area_ratio,
        max_self_intersections=params.max_self_intersections,
        expect_open_boundary=True)
    disp = np.linalg.norm(best_v[:len(av_in)] - av_in, axis=1)
    moved = disp > 1e-9
    loop_in = np.zeros(len(av_in), dtype=bool)
    loop_in[:] = False
    res.anatomy_deviation = {
        "n_moved_vertices": int(moved.sum()),
        "moved_ratio": float(moved.mean()),
        "max_mm": float(disp.max()),
        "p95_mm": float(np.quantile(disp, .95)),
        "median_moved_mm": float(np.median(disp[moved])) if moved.any() else 0.0,
    }
    res.iterations = it if (conflict or best_max > params.deficit_tol_mm) \
        else max(0, it - 1)
    # 语义：CONFLICT = **一步都没能走**（第一个 α 就被总违规/几何安全门拒），
    # 那是真的推不动；只要推进过但留有残余 → REVIEW_REQUIRED + 如实报残余。
    moved_any = len(res.alphas_accepted) > 0
    if conflict and not moved_any:
        res.status = STATUS_CONFLICT
    elif res.envelope_after["max_deficit_mm"] <= params.deficit_tol_mm:
        res.status = STATUS_OK
    else:
        res.status = STATUS_REVIEW_REQUIRED
        res.notes.append("仍有残余：最大 deficit %.4fmm / 总违规 %.3fmm"
                         "（%d 外顶点 + %d 内表面点）—— 如实上报，不假称达标"
                         % (res.envelope_after["max_deficit_mm"],
                            res.envelope_after["energy_mm"],
                            res.envelope_after["n_violating_outer"],
                            res.envelope_after["n_violating_intaglio"]))
    if res.status == STATUS_OK and not res.safety.get("ok", False):
        res.status = STATUS_REVIEW_REQUIRED
        res.notes.append("几何安全门未过：%s" % res.safety.get("violations"))
    res.mesh = SimpleNamespace(vertices=best_v, faces=af)
    return res
