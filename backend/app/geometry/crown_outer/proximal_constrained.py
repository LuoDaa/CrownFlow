"""H.2.1 —— Proximal-Constrained CrownOuter Deformation。

用户 2026-10-09 规格：`8671440` 已证伪"placement 预缩放"这条路（改初始值没用，
Margin hard constraints + Laplacian 会把预缩放吸收掉）。所以 H.2.1 改的是
**优化问题本身**：

    旧 H.1   : Margin 硬约束 + anatomy 微分坐标 + 弱身体/咬合锚
    H.2.1    : Margin 硬约束 + **proximal clearance 约束（新增 rows）**
               + anatomy 微分坐标 + protected anatomy anchors

守死的回归条件：**默认关闭时与 H.1 数值等价**（`solve_laplacian` 里只有
`proximal_vertex_indices` 非空才加行；本模块不被调用就完全不走）。

四条实现纪律（用户规格 §2–§5）：

1. **平滑的 Neighbor Clearance Envelope**：不用 raw face normal（三角形切换处跳变
   会把锯齿目标塞进系统），改用"最近顶点 + 该顶点的平滑法向"。
2. **单边不等式（只推不拉）**：active set = `signed_clearance < clearance − band`
   的顶点；已经合法的 proximal anatomy 不进约束，避免做成"邻牙反模"。
3. **active-set 迭代**：每轮重测 → 更新 active → **从同一份摆位网格重解**
   （margin 硬约束每轮精确满足、anatomy 微分坐标每轮保持）→ 每轮过几何安全门。
4. **Margin 最高优先级**：proximal 权重（默认 300）远低于 margin（1e4）；
   与硬约束重叠的顶点直接剔除（硬约束赢）。
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from .contact import ContactParameters, NeighborPatch, _region_mask
from .contact_refine import occ_table_mask
from .deform_gate import deformation_gate
from .laplacian import DeformInputs, solve_laplacian
from .models import CrownOuterConfig
from .patient_frame import PatientFrame
from .proximal_fit import ProximalFitParams


class NeighborClearanceQuery:
    """**唯一**的邻牙间隙口径：约束构建 / 穿透审计 / Contact Gate 三处共用。

    直接转发到 ``NeighborPatch.closest()`` —— 精确最近三角面（子 0.01mm）
    + **该三角面的法向** × ``normal_sign``，与 Contact 引擎、K0 审计**同源**。

    为什么必须唯一（用户 2026-10-09）：上一版自建了一层"最近顶点 + 平滑法向"
    envelope，结果是**内部 target residual = 0.0000，而审计那把尺仍读到
    1.06/1.08mm** —— 解到自己的目标也不改善穿透，权重扫描因此变成空操作。
    任何符号/法向/口径修正从现在起只改这一处。
    """

    def __init__(self, patch: NeighborPatch, boundary_safety_mm: float = 0.6,
                 obliquity_min: float = 0.90):
        self.patch = patch
        #: 与 Contact v2/I.2 **同一口径**：最近点落在开放 patch 边界安全带内
        #: 时 signed 不可靠（项目实测边界附近出现 −4mm 伪 signed），语义 =
        #: ``AMBIGUOUS_PATCH_EDGE``，**不得作为接触证据**。
        self.boundary_safety_mm = float(boundary_safety_mm)
        #: 近点法向可靠性阈值（``AMBIGUOUS_CREASE``）：``|dot(p−c, n̂)| / euclid``
        #: 低于它的点，最近点落在三角形**共享边/顶点**（扫描网格的折痕）上，
        #: 法向投影在 µm 级位移下可跳变 ~0.1mm（实测 —— 一个深穿 1.2mm 的顶点
        #: 被读成 0.03mm，8.5µm 位移后读数跳到 −1.13mm）。这类点不得进入
        #: deficit/约束/Gate 决策。0.90 ≈ 26°。
        self.obliquity_min = float(obliquity_min)

    def closest(self, points):
        return self.patch.closest(points)

    def closest_valid(self, points):
        """→ (cp, fids, signed, dist, valid)；valid=False 见 ``boundary_safety_mm``。"""
        return self.patch.closest_with_validity(
            points, boundary_safety_mm=self.boundary_safety_mm)

    def _robust_sc(self, signed, dist):
        """**决策幅值口径**：``sign(dot) · euclid``（幅值单规则，连续）。

        保留为兼容包装；真正实现已收进 `NeighborPatch.clearance_detail`
        （用户决策 ③：全项目唯一实现）。
        """
        signed = np.asarray(signed, np.float64)
        dist = np.asarray(dist, np.float64)
        return np.sign(signed) * dist

    def clearance_detail(self, points):
        """→ (signed, valid, cp, fids, parts)：**唯一口径**（见 NeighborPatch）。"""
        return self.patch.clearance_detail(
            points, boundary_safety_mm=self.boundary_safety_mm,
            obliquity_min=self.obliquity_min)

    def signed_clearance_raw(self, points):
        """裸读数（取证用）：``dot(p−c, n̂)`` 原样返回，不做稳健化。"""
        q, fids, signed, _dist = self.patch.closest(points)
        n = self.patch.normals[fids] * float(self.patch.normal_sign)
        return signed, q, n

    def signed_clearance(self, points):
        """→ (signed, q, n̂)：正 = 邻牙外侧（安全），负 = 穿进邻牙。

        幅值口径 = ``sign(dot)·euclid``（**单规则**，见
        ``signed_clearance_decision`` 的说明：dot 在 crease 点会随获胜面切换
        跳变，µm 位移可造成 0.1mm 级伪回归）。
        """
        signed, _valid, cp, fids, _parts = self.clearance_detail(points)
        n = self.patch.normals[fids] * float(self.patch.normal_sign)
        return signed, cp, n

    def signed_clearance_valid(self, points):
        """→ (signed, q, n̂, valid)：**决策路径必须用这个版本**（稳健 + 有效）。

        ``valid`` = 边界有效（``AMBIGUOUS_PATCH_EDGE``）**且** 近点法向可靠
        （``AMBIGUOUS_CREASE``：``|dot|/euclid ≥ obliquity_min``）。用户口径纪律：
        有效性在 H.1 基线上一冻到底（``frozen_editable_domain``），不随形变漂移。
        """
        signed, valid, cp, fids, _parts = self.clearance_detail(points)
        n = self.patch.normals[fids] * float(self.patch.normal_sign)
        return signed, cp, n, valid

    def signed_clearance_decision(self, points):
        """**决策幅值口径**：``sign(dot) · euclid`` —— 幅值一律用 euclid。

        为什么幅值必须单规则（用户口径纪律：同一个量只有一把尺）：dot 在 crease
        点处随**获胜面切换**跳变；若 before/after 各自用 "dot 或 euclid"
        （按斜切阈值切换），会出现 +0.36mm 的纯口径跳变（实测 id=1606：
        dot 从 −0.68 跳到 −0.32，deficit 伪恶化 +78µm）。euclid 在同一冻结
        顶点集上连续、单调；符号在接触区稳定（冻结 valid 集要求 H.1 斜切 ≥ 阈值）。

        → (sc, q, fids)；``sc`` 正 = 邻牙外侧。
        """
        sc, _valid, cp, fids, _parts = self.clearance_detail(points)
        return sc, cp, fids

    def target(self, points, clearance_mm: float):
        """→ (q + n̂·clearance, q, n̂, signed)。"""
        signed, q, n = self.signed_clearance(points)
        return q + n * float(clearance_mm), q, n, signed


def proximal_candidates(v0, faces, frame: PatientFrame, margin_idx,
                        p: ProximalFitParams, root_idx=None):
    """四类锁定区之外的、朝近/远中那一半的顶点（proximal candidate）。

    ``root_idx``：根部顶点必须**排除** —— 它们在求解器里是硬锚（不可动），
    留在 active set 里只会把目标残差恒久抬高、并把权重扫描变成空操作
    （实测：566 个 active 里 101 个是 root，残差 0.82mm 全来自它们）。
    """
    loop_set = np.zeros(len(v0), dtype=bool)
    loop_set[np.asarray(margin_idx, np.int64)] = True
    cparams = ContactParameters(contact_band_fraction=p.band_fraction,
                                cervical_lock_mm=p.cervical_lock_mm,
                                occlusal_anchor_percentile=(
                                    p.occlusal_anchor_percentile),
                                occ_table_normal_dot=p.occ_table_normal_dot)
    occ_t = occ_table_mask(v0, faces, frame, cparams)
    d_loop = cKDTree(v0[np.asarray(margin_idx, np.int64)]).query(v0, k=1)[0]
    cervical = (d_loop < p.cervical_lock_mm) & ~loop_set
    root_set = np.zeros(len(v0), dtype=bool)
    if root_idx is not None and len(root_idx):
        root_set[np.asarray(root_idx, np.int64)] = True
    blocked = loop_set | occ_t | cervical | root_set
    cand = {}
    for side in ("mesial", "distal"):
        cand[side] = _region_mask(v0, frame, np.asarray(margin_idx, np.int64),
                                  cparams, side) & ~blocked
    return cand, {"occ_table": occ_t, "cervical": cervical, "loop_set": loop_set,
                  "root": root_set}


def build_active_constraints(v, faces, frame, patches, margin_idx, p,
                             qs=None, *, activation_band_mm: float = 0.02,
                             root_idx=None, frozen_sign=None,
                             dom=None):
    """一轮 active set（用户 2026-10-09 §2/§3）。

    返回 ``(idx, tgt, stats, per_side)``；``return_domain=True`` 时附带冻结的
    ``audit_ids / editable_ids`` 与排除原因分账。

    口径纪律：
    * **符号冻结**：``frozen_sign`` 给出后用它的值查询，绝不再改 patch 的
      ``normal_sign``（审计只读）；
    * **两个量分开**：``penetration = max(0, −signed)``（真穿透）与
      ``clearance_deficit = max(0, clearance − signed)``（目标间隙欠缺）——
      旧代码写 ``max(0, −signed − clearance)``，既不是穿透也不是欠缺。
    """
    v = np.asarray(v, np.float64)
    if qs is None:
        qs = {s: NeighborClearanceQuery(patches[s]) for s in patches}
    adj_all: dict = {}
    for a_, b_, c_ in faces:
        for u_, v_ in ((a_, b_), (b_, c_), (c_, a_)):
            adj_all.setdefault(int(u_), set()).add(int(v_))
            adj_all.setdefault(int(v_), set()).add(int(u_))
    if frozen_sign is not None:
        for s_ in qs:
            qs[s_].patch.normal_sign = float(frozen_sign[s_])
    # **冻结约束候选域**（用户 2026-10-09 §2）：候选 ID 一律来自 dom（在 H.1 基线上
    # 冻结），每轮只有"间隙状态"决定激活 —— 不再按当前几何重算 proximal_candidates
    # （否则形变后的法向/区域分类会改约束资格 = 集合漂移）。
    if dom is not None:
        cand = {s: np.zeros(len(v), dtype=bool) for s in ("mesial", "distal")}
        for s_ in ("mesial", "distal"):
            cand[s_][dom["per_side"][s_]] = True
        zones = {}
    else:
        cand, zones = proximal_candidates(v, faces, frame, margin_idx, p,
                                          root_idx=root_idx)
    d = float(p.target_clearance_mm)
    rows_idx, rows_tgt, rows_w, stats, per_side = [], [], [], {}, {}
    for side in ("mesial", "distal"):
        ids = np.flatnonzero(cand[side])
        n_amb = 0
        if dom is not None:
            vd = (dom.get("valid_ids") or {}).get(side)
            if vd is not None and len(vd) == len(dom["per_side"][side]):
                vd = np.asarray(vd, bool)
                n_amb = int((~vd).sum())
                ids = dom["per_side"][side][vd]   # AMBIGUOUS_PATCH_EDGE 不建约束
        if not len(ids):
            stats[side] = {"side": side, "n_candidate": 0, "n_active": 0,
                           "n_ambiguous": n_amb}
            per_side[side] = (np.zeros(0, np.int64), np.zeros((0, 3)))
            continue
        tgt_side, q, n, sc = qs[side].target(v[ids], d)
        # 激活/统计用**决策幅值口径**（sign·euclid，单规则连续）；
        # target 位置仍用已定向法向 n（与 Contact 引擎同源）。
        sc, _q2, _f2 = qs[side].signed_clearance_decision(v[ids])
        pen = np.maximum(0.0, -sc)
        deficit = np.maximum(0.0, d - sc)
        act = deficit > activation_band_mm
        stats[side] = {
            "side": side,
            "n_candidate": int(len(ids)), "n_active": int(act.sum()),
            "n_ambiguous": n_amb,
            "clearance_p50_mm": float(np.median(sc)),
            "clearance_min_mm": float(sc.min()),
            "penetration_p95_mm": float(np.quantile(pen, .95)),
            "penetration_max_mm": float(pen.max()),
            "penetration_area_mm2": _vertex_area(v, faces)[ids][pen > 0].sum(),
            "deficit_p95_mm": float(np.quantile(deficit, .95)),
            "deficit_max_mm": float(deficit.max()),
        }
        sel = ids[act] if act.any() else np.zeros(0, np.int64)
        per_side[side] = (sel, tgt_side[act] if act.any() else np.zeros((0, 3)))
        if len(sel):
            rows_idx.append(sel)
            rows_tgt.append(tgt_side[act])
            rows_w.append(np.ones(len(sel)))
    if not rows_idx:
        out = (np.zeros(0, np.int64), np.zeros((0, 3)), stats, per_side)
        return out
    idx = np.concatenate(rows_idx)
    tgt = np.vstack(rows_tgt)
    w_row = np.concatenate(rows_w)
    uniq, inv = np.unique(idx, return_inverse=True)
    if len(uniq) != len(idx):
        acc = np.zeros((len(uniq), 3))
        cnt = np.zeros(len(uniq))
        for k, u in enumerate(inv):
            acc[u] += tgt[k]
            cnt[u] += 1
        tgt = acc / cnt[:, None]
        idx = uniq
        w_row = np.ones(len(uniq))       # 重复顶点按全强度（取中点目标）
    return (idx, tgt, stats, per_side, w_row)


def _vertex_area(v, faces):
    """每顶点当量面积（面积加权 1/3），用于"穿透面积"这类报告。"""
    tri = np.asarray(v, np.float64)[np.asarray(faces, np.int64)]
    a = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0],
                                      tri[:, 2] - tri[:, 0]), axis=1)
    out = np.zeros(len(v))
    for k in range(3):
        np.add.at(out, np.asarray(faces, np.int64)[:, k], a / 3.0)
    return out


def exclude_reasons(v, faces, frame, margin_idx, p, root_idx=None):
    """``audit_ids − editable_ids`` 的分账（用户 §2）：按原因给出顶点数与穿透面积。

    审计域**不缩到候选集**：锁定区仍需被审计，只是不允许动。
    """
    loop_set = np.zeros(len(v), dtype=bool)
    loop_set[np.asarray(margin_idx, np.int64)] = True
    cparams = ContactParameters(contact_band_fraction=p.band_fraction,
                                cervical_lock_mm=p.cervical_lock_mm,
                                occlusal_anchor_percentile=(
                                    p.occlusal_anchor_percentile),
                                occ_table_normal_dot=p.occ_table_normal_dot)
    occ_t = occ_table_mask(v, faces, frame, cparams)
    d_loop = cKDTree(v[np.asarray(margin_idx, np.int64)]).query(v, k=1)[0]
    cervical = (d_loop < p.cervical_lock_mm) & ~loop_set
    root_set = np.zeros(len(v), dtype=bool)
    if root_idx is not None and len(root_idx):
        root_set[np.asarray(root_idx, np.int64)] = True
    land = np.zeros(len(v), dtype=bool)
    for side in ("mesial", "distal"):
        land |= _region_mask(v, frame, np.asarray(margin_idx, np.int64),
                             cparams, side)
    return {"margin": loop_set, "cervical": cervical, "occlusal": occ_t,
            "root": root_set, "not_proximal_land": ~(land | loop_set
                                                     | cervical | occ_t
                                                     | root_set)}


def solve_proximal_constrained(placed_v, faces, margin_idx, margin_targets,
                               frame, patches, *, root_idx=None, occ_idx=None,
                               config=None, params=None, max_iterations: int = 5,
                               activation_band_mm: float = 0.02,
                               reference_v=None,
                               alphas=(1.0, 0.5, 0.25, 0.125, 0.0625),
                               max_backtracks: int = 5,
                               deficit_max_tol_mm: float = 1e-6,
                               protection_rings: int = 4,
                               protection_decay: float = 0.5,
                               width_field=None,
                               width_field_only: bool = False) -> dict:
    """H.2.1 主入口。``placed_v`` = H.1 摆位后的模板顶点（含 margin 目标）。

    ``reference_v``：几何门的**参考网格**，应为 **H.1 的输出**。
    ⚠️ 不能用 `placed` 当参考：margin 拟合本身就要把颈环动到患者颈缘（实测
    2.8mm），拿 `placed` 比会把 H.1 自己的合规形变当成"H.2.1 锁定位移/翻面"，
    数字全是假的。锁定区检查交给求解器自报的 margin 残差。
    """
    cfg = config or CrownOuterConfig()
    p = params or ProximalFitParams()
    v0 = np.asarray(placed_v, np.float64)
    faces = np.asarray(faces, np.int64)
    margin_idx = np.asarray(margin_idx, np.int64)
    margin_targets = np.asarray(margin_targets, np.float64)
    root = np.asarray(root_idx if root_idx is not None else [], np.int64)
    occ = np.asarray(occ_idx if occ_idx is not None else [], np.int64)
    qs = {s: NeighborClearanceQuery(patches[s]) for s in patches}
    adj_all: dict = {}
    for a_, b_, c_ in faces:
        for u_, v_ in ((a_, b_), (b_, c_), (c_, a_)):
            adj_all.setdefault(int(u_), set()).add(int(v_))
            adj_all.setdefault(int(v_), set()).add(int(u_))
    # ⚠️ **法向符号在开跑前冻结**：审计只读，不许在测量时再改 patch 的 normal_sign
    # （否则"约束用一套符号、审计用另一套"，又是假收敛的来源）。
    frozen_sign = {s: float(patches[s].normal_sign) for s in patches}

    # ⚠️ 返回语义（用户 2026-10-09 §1）：
    #   accepted_v 初值 = reference_v（合法 H.1），**不是 placed**；
    #   被拒绝的候选**不更新** accepted_v，只留诊断；
    #   `placed` 只是求解的线性化中心，不是答案。
    vref = np.asarray(reference_v, np.float64) if reference_v is not None else v0
    accepted = vref.copy()
    # **冻结可编辑域**（在 H.1 基线上算一次）：每轮 deficit 都在同一批顶点上量，
    # 不能只比"动态 active 子集"（换一批点指标就能变好）。
    dom = frozen_editable_domain(vref, faces, frame, patches, margin_idx, p,
                                 root_idx=root, qs=qs, frozen_sign=frozen_sign)
    history = []
    prev_active = None
    prev_idx = None
    last_candidate = None
    for it in range(max_iterations):
        if width_field_only:
            # **只做宽度重塑**（B 层的独立一步）：不叠加局部包络约束，
            # 否则"全局重塑 + 局部修正"的合成步会大到一个合法 α 都找不到。
            idx_a = np.zeros(0, np.int64)
            tgt_a = np.zeros((0, 3))
            w_row_a = np.zeros(0)
            stats, per_side = {}, {s_: (np.zeros(0, np.int64),
                                        np.zeros((0, 3)))
                                   for s_ in ("mesial", "distal")}
        else:
            idx_a, tgt_a, stats, per_side, w_row_a = build_active_constraints(
                accepted, faces, frame, patches, margin_idx, p, qs=qs,
                activation_band_mm=activation_band_mm, root_idx=root,
                frozen_sign=frozen_sign, dom=dom)
        # 集合漂移取证（用户 §1）：与上一轮 active 对比
        prev_set = set(int(i) for i in prev_idx) if prev_idx is not None else set()
        cur_set = set(int(i) for i in idx_a)
        set_drift = {"n_prev": len(prev_set), "n_cur": len(cur_set),
                     "added": len(cur_set - prev_set),
                     "removed": len(prev_set - cur_set),
                     "same": prev_set == cur_set}
        prev_idx = idx_a
        # 冻结保护带（一次定好、不随迭代变化）：钉在**当前 accepted 位置**、权重衰减
        idx, tgt, w_row = idx_a, tgt_a, w_row_a
        # 保护带与 active 是否存在**无关** —— 宽度重塑（B）时 active 为空但
        # 宽度场的边界同样需要软锚，否则边界位移突变会折叠（实测 B 层 α 全灭）。
        if len(dom["protection_ids"]) and (len(idx) or width_field is not None):
            idx = np.concatenate([idx, dom["protection_ids"]])
            tgt = np.vstack([tgt, accepted[dom["protection_ids"]]])
            w_row = np.concatenate([w_row, dom["protection_w"]])
        # ---- H.2.2-B：MD 宽度场（与 proximal clearance 同一套 rows，一次求解）
        if width_field is not None:
            from .md_width_field import (MDWidthFieldParams,
                                         build_md_width_targets)
            wf_res = width_field["result"]
            wf_fp = width_field.get("params") or MDWidthFieldParams()
            widx, wtgt, wwt = build_md_width_targets(
                accepted, faces, frame, margin_idx, dom, wf_res, p, wf_fp)
            if len(widx):
                keep = ~np.isin(widx, idx)      # 与 envelope 行重叠的先并到 envelope
                idx = np.concatenate([idx, widx[keep]])
                tgt = np.vstack([tgt, wtgt[keep]])
                w_row = np.concatenate([w_row, wwt[keep]])
            # **锁定区在 B 步内钉住**（用户 R8：cervical/occlusal 是绝对锁区）：
            # occlusal 锚权重只有 0.3，宽度行会把咬合面耦合拖走（实测 α=1 时
            # 0.70mm、翻面 254 个）。这里加与 margin 同量级（1e4）的软钉。
            for _rn in ("cervical", "occlusal"):
                _rm = (dom.get("regions") or {}).get(_rn)
                if _rm is None:
                    continue
                _rid = np.flatnonzero(np.asarray(_rm, bool))
                _rid = _rid[~np.isin(_rid, idx)]
                if not len(_rid):
                    continue
                idx = np.concatenate([idx, _rid])
                tgt = np.vstack([tgt, accepted[_rid]])
                w_row = np.concatenate([w_row, np.full(len(_rid), 1e4 / 300.0)])
        v_new, info = solve_laplacian(DeformInputs(
            vertices_placed=v0, faces=faces,
            margin_vertex_indices=margin_idx, margin_targets=margin_targets,
            root_vertex_indices=root, occlusal_vertex_indices=occ, config=cfg,
            proximal_vertex_indices=idx, proximal_targets=tgt,
            proximal_row_weights=w_row))
        resid_fixed = np.linalg.norm(v_new[idx] - tgt, axis=1)
        d_clear = float(p.target_clearance_mm)
        d_prev = deficit_on_domain(accepted, dom, d_clear)
        p95_prev, max_prev = combined_deficit(d_prev)

        def _check(trial):
            """每个 trial 必须同时满足：零新增自交（相对上一步）+ 零累计翻面
            （相对 H.1）+ 锁定区位移 0 + 冻结域 deficit 改善且 max 不恶化。"""
            rec = {}
            moved_t = np.flatnonzero(np.abs(trial - accepted).max(axis=1) > 1e-12)
            focus_t = np.flatnonzero(np.isin(faces, moved_t).any(axis=1))
            g_self = deformation_gate(
                accepted, trial, faces, scale_mm=p.gate_scale_mm,
                focus_faces=np.flatnonzero(np.isin(faces, moved_t).any(axis=1)),
                locked_ids=None)
            moved_c = np.flatnonzero(np.abs(trial - vref).max(axis=1) > 1e-12)
            g_cum = deformation_gate(
                vref, trial, faces, scale_mm=p.gate_scale_mm,
                focus_faces=np.flatnonzero(np.isin(faces, moved_c).any(axis=1)),
                locked_ids=None)
            # severe 级自交（面积 ≥ (scale/2)²，与项目 micro 口径一致）
            from .thickness import self_intersections as _si
            thr_sev = max((p.gate_scale_mm / 2.0) ** 2, 1e-9)
            ev_a = _si(accepted, faces, severe_min_area_mm2=thr_sev,
                       severe_max_normal_dot=0.999, focus_faces=focus_t,
                       return_pairs=True)
            ev_t = _si(trial, faces, severe_min_area_mm2=thr_sev,
                       severe_max_normal_dot=0.999, focus_faces=focus_t,
                       return_pairs=True)
            pairs_sev = _new_pairs_severe(ev_a, ev_t)
            ev_self_sev = len(pairs_sev)
            rec_evidence = _pair_evidence(ev_t, pairs_sev)
            mid = np.asarray(dom["margin_ids"], np.int64)
            lock_disp = float(np.max(np.linalg.norm(
                trial[mid] - np.asarray(margin_targets, float), axis=1)))
            # **逐区位移**（用户 §3）：margin/cervical/occlusal 硬锁（位移≈0），
            # buccal/lingual 是加性保护（按既定 p95 允许值验收）。
            region_disp = {}
            for rname, rmask in dom["regions"].items():
                ids_r = np.flatnonzero(rmask)
                if not len(ids_r):
                    region_disp[rname] = {"n": 0}
                    continue
                d_r = np.linalg.norm(trial[ids_r] - accepted[ids_r], axis=1)
                region_disp[rname] = {
                    "n": int(len(ids_r)),
                    "p95_mm": float(np.quantile(d_r, .95)),
                    "max_mm": float(d_r.max())}
            # 硬锁区判据用**项目既定限值**：solve_laplacian 的 margin 残差上限
            # 是 0.01mm → 这里同值。写成 1e-6 会把 1~24µm 的耦合残差（物理上就是 0）
            # 判成违规，等于自己造一个永远 False 的门（实测 it0 的 α=1/16 本来合法）。
            hard_ok = all(region_disp[k].get("max_mm", 0.0)
                          <= float(p.hard_region_tol_mm)
                          for k in ("margin", "cervical", "occlusal"))
            bl95 = region_disp.get("buccal_lingual", {}).get("p95_mm", 0.0)
            soft_ok = bl95 <= float(p.buccal_lingual_p95_allow_mm)
            dd = deficit_on_domain(trial, dom, d_clear)
            p95_now, max_now = combined_deficit(dd)
            improved = (p95_now < p95_prev - 1e-9) and (max_now <= max_prev + deficit_max_tol_mm)
            rec.update({
                "deficit_p95_before": round(p95_prev, 5),
                "deficit_p95_after": round(p95_now, 5),
                "deficit_max_before": round(max_prev, 5),
                "deficit_max_after": round(max_now, 5),
                "improved": bool(improved),
                "new_self_intersections": int(
                    (g_self.metrics.get("self_intersections_focus") or {}).get("new", 0)),
                "cumulative_flips_severe": int(
                    (g_cum.metrics.get("flipped") or {}).get("n_flipped_severe", 0)),
                "locked_displacement_mm": round(lock_disp, 6),
                "gate_self_ok": bool(g_self.ok),
                "gate_cum_ok": bool(g_cum.ok),
                "per_side_deficit": dd,
            })
            # 自交判据用 **severe**（面积分级）：微面/近相切对按项目口径记 micro、
            # 照常上抛但不作为硬违规（与 J.2 / Phase K 同一套尺）。
            new_sev = int(ev_self_sev)
            rec["new_self_intersections_severe"] = new_sev
            rec["selfint_evidence"] = rec_evidence
            rec["new_self_intersections_all"] = int(
                (g_self.metrics.get("self_intersections_focus") or {}).get("new", 0))
            rec["region_displacement"] = region_disp
            rec["hard_regions_ok"] = bool(hard_ok)
            rec["buccal_lingual_within_allow"] = bool(soft_ok)
            # α 逐点 deficit 变化（固定顶点集：冻结可编辑域，用户 §2）
            d_before = deficit_on_domain(accepted, dom, d_clear)
            def _sum(st):
                return float(sum(v["deficit_p95"] * v.get("n", 0)
                                 for v in st.values() if v.get("n")))
            rec["per_side_deficit_delta"] = {
                s_: {"p95": round(dd[s_]["deficit_p95"]
                                  - d_before[s_]["deficit_p95"], 5),
                     "max": round(dd[s_]["deficit_max"]
                                  - d_before[s_]["deficit_max"], 5),
                     "sum": round(_sum({s_: dd[s_]}) - _sum({s_: d_before[s_]}), 4)}
                for s_ in dd}
            ok = (new_sev == 0
                  and rec["cumulative_flips_severe"] == 0
                  and lock_disp <= 1e-6
                  and hard_ok and soft_ok
                  and improved)
            if not ok:
                why = []
                if rec["new_self_intersections"]:
                    why.append("new_selfint=%d" % rec["new_self_intersections"])
                if rec["cumulative_flips_severe"]:
                    why.append("cum_flips=%d" % rec["cumulative_flips_severe"])
                if lock_disp > 1e-9:
                    why.append("locked_disp=%.2e" % lock_disp)
                # **取证必须完整**（不打印 = 看不见失败模式）：锁定区/颊舌超限
                # 也逐项写出，否则"所有 α 全灭"时无法归因是哪个判据挡的。
                if not hard_ok:
                    bad = [(k, region_disp[k].get("max_mm", 0.0))
                           for k in ("margin", "cervical", "occlusal")
                           if region_disp[k].get("max_mm", 0.0)
                           > float(p.hard_region_tol_mm)]
                    why.append("锁定区超限(%s)" % "、".join(
                        "%s max=%.4f>%.4f" % (k, v_, p.hard_region_tol_mm)
                        for k, v_ in bad))
                if not soft_ok:
                    why.append("颊舌 p95=%.4f>%.4f"
                               % (bl95, p.buccal_lingual_p95_allow_mm))
                if not improved:
                    if p95_now >= p95_prev - 1e-9:
                        why.append("deficit p95 未改善(%.4f→%.4f)"
                                   % (p95_prev, p95_now))
                    else:
                        why.append("deficit max 恶化(%.4f→%.4f, tol %.1e)"
                                   % (max_prev, max_now, deficit_max_tol_mm))
                rec["reject"] = "；".join(why)
            return ok, rec

        trial_v, alpha_used, trials = pick_alpha(
            accepted, v_new, alphas=alphas, check=_check,
            max_backtracks=max_backtracks)
        diag = {}
        for side in ("mesial", "distal"):
            sel, _t = per_side[side]
            if not len(sel):
                continue
            sc_b, _q1, _n1 = qs[side].signed_clearance(vref[sel])
            sc_a, _q2, _n2 = qs[side].signed_clearance(
                (v_new if trial_v is None else trial_v)[sel])
            _c0, f0, _s0, _d0 = qs[side].closest(vref[sel])
            _c1, f1, _s1, _d1 = qs[side].closest(
                (v_new if trial_v is None else trial_v)[sel])
            diag[side] = {
                "n_constrained": int(len(sel)),
                "fixed_target_residual_mm": {
                    "p50": float(np.median(resid_fixed)),
                    "p95": float(np.quantile(resid_fixed, .95)),
                    "max": float(resid_fixed.max())},
                "penetration_before_p95_mm": float(np.quantile(np.maximum(0.0, -sc_b), .95)),
                "penetration_after_p95_mm": float(np.quantile(np.maximum(0.0, -sc_a), .95)),
                "deficit_before_p95_mm": float(np.quantile(np.maximum(0.0, d_clear - sc_b), .95)),
                "deficit_after_p95_mm": float(np.quantile(np.maximum(0.0, d_clear - sc_a), .95)),
                "closest_face_changed_ratio": float(np.mean(f0 != f1)),
                "normal_sign": float(qs[side].patch.normal_sign),
            }
        rec = {"iteration": it, "n_active": int(len(idx_a)),
               "set_drift": set_drift, "stats": stats,
               "solver": info,
               "fixed_target_residual_mm": {
                   "p50": float(np.median(resid_fixed)),
                   "p95": float(np.quantile(resid_fixed, .95)),
                   "max": float(resid_fixed.max())},
               "requery": diag, "alpha_trials": trials,
               "alpha": alpha_used,
               "deficit_frozen_domain": {"before": d_prev, "after":
                                         deficit_on_domain(
                                             trial_v if trial_v is not None else accepted,
                                             dom, d_clear)},
               "n_moved": int(len(np.flatnonzero(
                   np.abs((trial_v if trial_v is not None else accepted)
                          - vref).max(axis=1) > 1e-12))),
               "patch_normal_sign": frozen_sign}
        history.append(rec)
        last_candidate = v_new
        if trial_v is None:
            rec["blocked"] = True
            rec["note"] = ("所有 α 均无法在保持几何安全的前提下改善冻结域 deficit "
                           "→ 保留上一次 accepted，迭代阻断（不强行展开）")
            break
        accepted = trial_v            # ← α 回溯后的合法小步（不是整步候选）
        active_now = {s: stats[s]["n_active"] for s in stats}
        if prev_active == active_now and alpha_used == 1.0:
            rec["converged"] = True
            break
        prev_active = active_now
    return {"accepted_v": accepted, "last_candidate_v": last_candidate,
            "history": history, "queries": qs, "frozen_sign": frozen_sign}


def edge_length_change_p95(v0, v1, faces) -> float:
    """边长变化 p95（形变自然度的代理指标：即使没翻面，也可能拉得很难看）。"""
    f = np.asarray(faces, np.int64)
    e = np.vstack([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]])
    e = np.unique(np.sort(e, axis=1), axis=0)
    a = np.linalg.norm(np.asarray(v0, float)[e[:, 0]] - np.asarray(v0, float)[e[:, 1]],
                       axis=1)
    b = np.linalg.norm(np.asarray(v1, float)[e[:, 0]] - np.asarray(v1, float)[e[:, 1]],
                       axis=1)
    ok = a > 1e-9
    rel = np.abs(b[ok] / a[ok] - 1.0)
    return float(np.quantile(rel, .95))


# --------------------------------------------------------------------------
# per-iteration α 回溯（用户 2026-10-09 收口轮）
# --------------------------------------------------------------------------

def frozen_editable_domain(v_ref, faces, frame, patches, margin_idx, p,
                           root_idx=None, *, qs=None, frozen_sign=None):
    """**在 H.1 基线上**冻结可编辑域与符号（一次算出，全程复用）。

    返回 dict：``editable_ids`` / ``qs`` / ``sign`` / ``margin_ids``。
    审计域（audit_ids）= 全部冠表面顶点，不在这里裁剪。
    """
    v = np.asarray(v_ref, np.float64)
    if qs is None:
        qs = {s: NeighborClearanceQuery(patches[s]) for s in patches}
    adj_all: dict = {}
    for a_, b_, c_ in faces:
        for u_, v_ in ((a_, b_), (b_, c_), (c_, a_)):
            adj_all.setdefault(int(u_), set()).add(int(v_))
            adj_all.setdefault(int(v_), set()).add(int(u_))
    if frozen_sign is not None:
        for s in qs:
            qs[s].patch.normal_sign = float(frozen_sign[s])
    cand, _zones = proximal_candidates(v, faces, frame, margin_idx, p,
                                       root_idx=root_idx)
    editable = np.flatnonzero(cand["mesial"] | cand["distal"])
    sides = {s: np.flatnonzero(cand[s]) for s in ("mesial", "distal")}
    # **保护/参与带**（用户 2026-10-09）：活动集外扩 N 环，**一次冻结**。
    # 环上顶点是"钉在当前位置、权重衰减"的软锚 —— **不是硬锁**，可以被求解移动
    # （会逐区报告位移）。不能把"未添加邻牙目标"当成硬锁。
    prot = {}
    seen = set(int(i) for i in editable) | set(int(i) for i in margin_idx)
    frontier = set(int(i) for i in editable)
    for ring in range(1, max(1, p.protection_rings) + 1):
        nxt = set()
        for u in frontier:
            for w_ in adj_all.get(u, ()):
                if w_ not in seen:
                    nxt.add(w_)
        seen |= nxt
        if not nxt:
            break
        for i in nxt:
            prot[int(i)] = float(p.protection_decay ** ring)
        frontier = nxt
    prot_ids = np.asarray(sorted(prot), dtype=np.int64)
    prot_w = (np.asarray([prot[int(i)] for i in prot_ids], np.float64)
              if len(prot_ids) else np.zeros(0))
    # **有效性在 H.1 基线上冻结**（用户口径纪律：判据必须在固定参考系上算并锁住）：
    # 边界伪读数点不进入 deficit/约束决策；valid 子集一次冻结、全程复用，
    # 不允许随形变漂移（否则"换一批点指标就能变好"）。
    valid_ids = {}
    n_amb = {}
    for s_ in ("mesial", "distal"):
        ids_ = sides[s_]
        if len(ids_):
            _sc, _q, _n, _vd = qs[s_].signed_clearance_valid(v[ids_])
            valid_ids[s_] = np.asarray(_vd, bool)
            n_amb[s_] = int((~valid_ids[s_]).sum())
        else:
            valid_ids[s_] = np.zeros(0, bool)
            n_amb[s_] = 0
    # 分区集合（用于逐区位移验收）
    reasons = exclude_reasons(v, faces, frame, margin_idx, p, root_idx=root_idx)
    lat = (v - frame.origin) @ np.asarray(frame.y, float)
    bl = np.abs(lat) > 0.5
    regions = {
        "margin": reasons["margin"],
        "cervical": reasons["cervical"],
        "occlusal": reasons["occlusal"],
        "buccal_lingual": bl & ~reasons["occlusal"] & ~reasons["margin"]
        & ~reasons["cervical"],
    }
    return {"editable_ids": editable, "per_side": sides, "qs": qs,
            "sign": {s: float(qs[s].patch.normal_sign) for s in qs},
            "margin_ids": np.asarray(margin_idx, np.int64),
            "protection_ids": prot_ids, "protection_w": prot_w,
            "regions": regions,
            "valid_ids": valid_ids, "n_ambiguous_side": n_amb}


def deficit_on_domain(v, dom, clearance_mm: float) -> dict:
    """在**冻结的可编辑域**上量 deficit（= max(0, clearance − signed)）。

    用户口径：不能只比较"动态 active 子集"的 p95 —— 那样每轮换一批顶点，
    指标可以靠换人变好。这里每轮都量同一个 frozen 域。
    同时报 penetration（max(0, −signed)），两者不混用。
    """
    v = np.asarray(v, np.float64)
    out = {}
    for s, ids in dom["per_side"].items():
        if len(ids) == 0:
            out[s] = {"n": 0}
            continue
        # 有效性冻结在 H.1（dom["valid_ids"]）：before/after 用同一批顶点，
        # AMBIGUOUS_PATCH_EDGE 点（边界伪 signed）不计入.
        vd = (dom.get("valid_ids") or {}).get(s)
        n_amb = 0
        if vd is not None and len(vd) == len(ids):
            n_amb = int((~np.asarray(vd, bool)).sum())
            ids = ids[np.asarray(vd, bool)]
        if len(ids) == 0:
            # 全部被有效性过滤（如 patch 裁得贴边时）—— 该侧无证据，如实记 0，
            # 不能对空数组取分位（曾崩 numpy quantile）。
            out[s] = {"n": 0, "n_ambiguous": n_amb}
            continue
        sc, _q, _f = dom["qs"][s].signed_clearance_decision(v[ids])
        deficit = np.maximum(0.0, float(clearance_mm) - sc)
        pen = np.maximum(0.0, -sc)
        out[s] = {
            "n": int(len(ids)),
            "n_ambiguous": n_amb,
            "deficit_p50": float(np.median(deficit)),
            "deficit_p95": float(np.quantile(deficit, .95)),
            "deficit_max": float(deficit.max()),
            "penetration_p95": float(np.quantile(pen, .95)),
            "penetration_max": float(pen.max()),
        }
    return out


def combined_deficit(stats: dict) -> tuple[float, float]:
    """(p95_max_over_sides, max_over_sides) —— 用于"改善且 max 不恶化"的判定。"""
    p95 = max((s.get("deficit_p95", 0.0) for s in stats.values()), default=0.0)
    mx = max((s.get("deficit_max", 0.0) for s in stats.values()), default=0.0)
    return float(p95), float(mx)


def pick_alpha(accepted_v, candidate_v, *, alphas, check, max_backtracks=None):
    """**纯函数**：α 回溯选择（用户 2026-10-09 §1–§2）。

    ``check(trial)`` → ``(ok: bool, record: dict)`` 由调用方给出（几何门 +
    冻结域 deficit 改善 + 锁定区位移）。返回
    ``(trial_v | None, alpha | None, trials: list[dict])``：
    **全部 α 失败时 trial_v=None** —— 调用方必须保留上一次 accepted（不强行展开）。
    """
    a = np.asarray(accepted_v, np.float64)
    c = np.asarray(candidate_v, np.float64)
    if c.shape != a.shape:
        raise ValueError("candidate 与 accepted 形状不一致")
    seq = list(alphas if max_backtracks is None else alphas[:max_backtracks])
    trials = []
    for alpha in seq:
        trial = a + float(alpha) * (c - a)
        ok, rec = check(trial)
        rec = dict(rec)
        rec["alpha"] = float(alpha)
        rec["accepted"] = bool(ok)
        trials.append(rec)
        if ok:
            return trial, float(alpha), trials
    return None, None, trials


def _new_pairs_severe(ev_after_base, ev_trial):
    """trial 相对 base 的**新增 severe 自交对**（面积已由 severe_* 过滤）。"""
    pa = ev_trial.get("self_intersections_severe_pairs_list")
    if pa is None or len(pa) == 0:
        return []
    pb = ev_after_base.get("self_intersections_severe_pairs_list")
    if pb is None or len(pb) == 0:
        return list(pa)
    s0 = {tuple(map(int, x)) for x in np.asarray(pb)}
    return [tuple(map(int, x)) for x in np.asarray(pa)
            if tuple(map(int, x)) not in s0]


def _pair_evidence(ev, pairs):
    """新增 severe 自交对的证据：面积、法向角、交线段长（用于数值级 vs 真折叠定性）。"""
    if not pairs:
        return []
    diag = {tuple(map(int, d["faces"])): d
            for d in (ev.get("self_intersections_severe_diag")
                      or ev.get("self_intersections_diag") or [])}
    out = []
    for fa, fb in pairs[:8]:
        d = diag.get((int(fa), int(fb))) or diag.get((int(fb), int(fa))) or {}
        out.append({"faces": [int(fa), int(fb)],
                    "area_mm2": d.get("area_mm2"),
                    "abs_normal_dot": d.get("abs_normal_dot"),
                    "centroid_gap_mm": d.get("centroid_gap_mm")})
    return out
