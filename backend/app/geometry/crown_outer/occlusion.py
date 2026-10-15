"""Occlusion v1 —— Collision Resolution Engine（咬合碰撞消除）。

职责（用户 2026-10-02 规格，v1 **只做 A**）：

    Contact/I.2 输出 Crown + Antagonist patch + occlusal handoff
        ↓ Antagonist SDF
    识别真实咬合干涉
        ↓ SDF 决定目标位置（沿对颌法向投影到 target clearance）
    只开放 **Occlusal editable cap**（法向判据的咬合面）+ 少量过渡环
        ↓ local-band Laplacian 平滑
    monotone line search
        ↓ Occlusion QC（含 Contact 保持性复测——只测，不重跑 solver）

硬锁（由构造保证，非弱锚）：

    Cervical / Margin                 绝对不动
    Mesial / Distal contact core      绝对不动（一切非咬合面顶点都在锁定集）
    过渡环不出咬合面（BFS 被锁定集挡住）

不做（放后面）：接触面积优化、窝沟解剖重塑、动态咬合、"太松抬高牙尖"（B 类）。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import trimesh

from ..measurement_core import (DOMAIN_OCCLUSION, OCCLUSION_CONTRACT, SignSource,
                                domain_report, stats as _stats_of)
from .contact import NeighborPatch, _boundary_loop
from .contact_i2 import I2Params
from .contact_refine import occ_table_mask
from .laplacian import solve_local_band
from .patient_frame import PatientFrame


class OcclusionError(Exception):
    pass


@dataclass
class OcclusionParams:
    #: 咬合面定义（与 Contact v2 同一法向判据，**不得退回 z≥p75**）。
    occlusal_anchor_percentile: float = 75.0
    occ_table_normal_dot: float = 0.6
    #: occ_table_mask 内部也用（与 Contact 同语义；本模块里颈环反正整体锁定）。
    cervical_lock_mm: float = 0.3
    target_gap_mm: float = 0.05
    inside_reach_mm: float = 2.0
    boundary_safety_mm: float = 0.6
    #: 咬合面内的过渡环数（BFS 被"非咬合面即锁定"挡住，不会侵入 proximal）。
    transition_rings: int = 3
    max_outer_iterations: int = 10
    line_search_alphas: tuple = (1.0, 0.5, 0.25, 0.125, 0.0625)
    #: 收敛：occlusal p95 ≤ tolerance。
    pen_tolerance_mm: float = 0.05
    min_relative_improvement: float = 0.02
    #: Contact 保持性容差：每侧 contact pen p95 允许的恶化上限（mm）。
    contact_preservation_tol_mm: float = 0.02


@dataclass
class OcclusionResult:
    status: str = "FAILED"
    mesh: object | None = None
    iterations: int = 0
    alphas_accepted: list = field(default_factory=list)
    history: list = field(default_factory=list)
    #: Contact 保持性（before/after 的近远中穿透统计）
    contact_preservation: dict = field(default_factory=dict)
    #: 位移统计（咬合面顶点）
    occlusal_displacement: dict = field(default_factory=dict)
    margin_max_displacement_mm: float = 0.0
    proximal_core_max_displacement_mm: float = 0.0
    n_flipped_faces: int = 0
    modified_vertex_count: int = 0
    #: B1：本域测量契约标签（域 / 符号来源 / 有效性 / 穿透含义 / 是否负责相交审计）
    domain_report: dict = field(default_factory=dict)
    #: O1：本次使用的符号来源（SOLID_CONTAINMENT / REVIEWED_NORMAL_LOCAL / UNSIGNED_ONLY）
    sign_method: str = ""
    #: O1：signed 是否可信（False ⇒ 不得下穿透结论）
    signed_available: bool = False
    notes: list = field(default_factory=list)


def _loop_of(vertices, faces):
    loop = _boundary_loop(vertices, faces)
    if loop is None:
        raise OcclusionError("冠网格没有可识别的颈环边界")
    return np.asarray(loop, dtype=np.int64)


def _classify(vertices, faces, frame, p):
    """→ (occ_table, locked)。**一切非咬合面顶点都锁**（proximal core 语义）。"""
    loop_idx = _loop_of(vertices, faces)
    loop_set = np.zeros(len(vertices), dtype=bool)
    loop_set[loop_idx] = True
    occ_t = occ_table_mask(vertices, faces, frame, p)
    locked = ~occ_t                      # 含颈环、全部邻接壁、底部
    return loop_idx, loop_set, occ_t, locked


def _contact_stats(vertices, faces, patches, frame, p, *, outward=None):
    """Contact 保持性复测（与 I.2 measure 同口径；**只测不修**）。

    O1：符号同样来自共享 primitive。未声明外向法向的开放裁片 ⇒ 该侧只报幅值
    （``signed_available=False``），不参与"穿透是否恶化"的判定 —— 那种判定需要符号。
    """
    from .contact_i2 import _half_domains, _zones as _i2_zones
    loop_idx, loop_set, occ_t, cervical, editable = _i2_zones(
        vertices, faces, frame, p)
    domains = _half_domains(vertices, frame, loop_set)
    out = {}
    cn = vertex_normals_of(vertices, faces)
    for side, patch in patches.items():
        wids = np.flatnonzero(domains[side])
        source = _sign_source_for(patch, (outward or {}).get(side))
        _cp, _fids, signed, _dist, valid = source.query(vertices[wids])
        if not source.signed_available:
            out[side] = {"count": 0, "p95": 0.0, "max": 0.0, "min_gap": 0.0,
                         "signed_available": False, "sign_method": source.method}
            continue
        is_edit = editable[wids]
        sel = is_edit & valid & (signed < 0) & \
            (np.abs(signed) <= p.inside_reach_mm)
        ids = np.flatnonzero(sel)
        pen = np.maximum(p.target_gap_mm - signed[ids], 0.0) if len(ids) else \
            np.array([0.0])
        out[side] = {"count": int(len(ids)),
                     "p95": float(np.quantile(pen, .95)) if len(ids) else 0.0,
                     "max": float(pen.max()) if len(ids) else 0.0,
                     "signed_available": True, "sign_method": source.method,
                     # signed 是子集坐标（长度=len(wids)）—— 必须用布尔掩码
                     # 而不是全局 id 索引（子集/全局坐标混淆，I.2 踩过同一坑）
                     "min_gap": float(signed[is_edit & valid].min())
                     if (is_edit & valid).any() else 0.0}
    return out


def _sign_source_for(reference, outward=None) -> SignSource:
    """把入参收敛成共享符号 primitive。

    接受 ``FitSurface``（已含符号来源）、``SignSource``、或裸 ``NeighborPatch`` +
    调用方声明的 ``outward``。**裸 patch 且未声明法向 ⇒ UNSIGNED_ONLY**（不猜）。
    """
    if isinstance(reference, SignSource):
        return reference
    source = getattr(reference, "source", None)
    if isinstance(source, SignSource):
        return source
    if hasattr(reference, "normal_sign") and hasattr(reference, "normals"):
        return SignSource.for_patch(reference, outward)          # 裸 NeighborPatch
    return SignSource.for_mesh(reference, outward)               # 裸 trimesh（或同形对象）


def _shortfall_only(source: SignSource, points, p: OcclusionParams) -> dict:
    """无符号时只报 clearance shortfall（用 unsigned 幅值 + 目标间隙）。"""
    _cp, _fids, _signed, distance, valid = source.query(points)
    distance = np.asarray(distance, float)
    valid = np.asarray(valid, bool)
    shortfall = np.maximum(p.target_gap_mm - distance, 0.0)
    if not valid.any():
        return {"max_mm": 0.0, "p95_mm": 0.0, "count": 0}
    return {"max_mm": float(shortfall[valid].max()),
            "p95_mm": float(np.quantile(shortfall[valid], .95)),
            "count": int((shortfall[valid] > 0).sum())}


def vertex_normals_of(vertices, faces):
    n = np.cross(vertices[faces[:, 1]] - vertices[faces[:, 0]],
                 vertices[faces[:, 2]] - vertices[faces[:, 0]])
    L = np.linalg.norm(n, axis=1)
    L[L == 0] = 1.0
    vn = np.zeros_like(vertices)
    for k in range(3):
        np.add.at(vn, faces[:, k], n / L[:, None])
    Ln = np.linalg.norm(vn, axis=1)
    Ln[Ln == 0] = 1.0
    return vn / Ln[:, None]


def solve_occlusion_collision(crown_mesh, patient_frame: PatientFrame,
                              antagonist_patch,
                              contact_patches: dict | None = None,
                              params: OcclusionParams | None = None,
                              *, antagonist_outward=None,
                              contact_outward: dict | None = None
                              ) -> OcclusionResult:
    """Occlusion v1 主入口：对颌 SDF 投影 + 咬合面内 local band + line search。

    **符号来源（O1，2026-10-13）**：不再使用"一次性质心投票"。

    - 对颌是**闭合可定向实体** → 实体内外判定（与 Contact 同一个 `SignSource`）；
    - 对颌是**开放裁片 + 调用方声明的外向法向**（`antagonist_outward`，或直接传
      `auto_fit_adapter.FitSurface`）→ 局部可信的 signed；
    - 两者都没有 → ``UNSIGNED_ONLY``：**不提供 signed、不得下穿透结论**，
      状态直接 ``NOT_ASSESSED``，且不动几何。

    域内语义不变：``deficit = max(target_gap - signed, 0)``（目标相对亏欠），
    与 Contact 的 ``penetration = max(-signed, 0)`` 是两件事。
    """
    p = params or OcclusionParams()
    res = OcclusionResult()
    vertices = np.asarray(crown_mesh.vertices, dtype=np.float64).copy()
    faces = np.asarray(crown_mesh.faces, dtype=np.int64)
    loop_idx, loop_set, occ_t, locked = _classify(vertices, faces,
                                                  patient_frame, p)
    locked_ids = set(int(i) for i in np.flatnonzero(locked))
    contact_patches = contact_patches or {}
    adj = {}
    for a, b, c3 in faces:
        for uu, vv in ((int(a), int(b)), (int(b), int(c3)), (int(c3), int(a))):
            adj.setdefault(uu, set()).add(vv)
            adj.setdefault(vv, set()).add(uu)

    # O1：符号一律来自共享 primitive（闭合实体用实体内外判定；开放裁片要已声明法向）。
    antagonist_source = _sign_source_for(antagonist_patch, antagonist_outward)
    res.sign_method = antagonist_source.method
    res.signed_available = bool(antagonist_source.signed_available)
    # 契约里的 `sign_method` 是**允许的候选清单**；本次实际用的是 `active_sign_method`
    # （两者同名会让人读错，这正是 B1/O1 想消除的那类混淆）。
    res.domain_report = domain_report(
        DOMAIN_OCCLUSION, active_sign_method=antagonist_source.method,
        active_sign=antagonist_source.as_report(),
        note="符号来源与 Contact 共享（SOLID_CONTAINMENT / REVIEWED_NORMAL_LOCAL）；无符号则不下穿透结论")
    if not res.signed_available:
        # 没有可信符号时不猜、不动、不假装评估过。
        res.status = "NOT_ASSESSED"
        res.notes.append(
            "对颌参考面没有可信符号来源（既非闭合实体，也没有调用方声明的外向法向）"
            " ⇒ SIGN_UNAVAILABLE_REFERENCE_HAS_NO_OUTWARD_NORMAL："
            "只允许报 clearance shortfall，不得把间隙不足说成穿透；本轮不做任何形变")
        res.mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        res.history.append({"iteration": 0, "count": 0, "p50": 0.0, "p95": 0.0, "max": 0.0,
                            "sum": 0.0, "ambiguous": 0, "sign_method": res.sign_method,
                            "signed_available": False, "signed_min_mm": None,
                            "penetration_mm": None,
                            "clearance_shortfall_mm": _shortfall_only(
                                antagonist_source, vertices[occ_t], p)})
        return res

    contact_before = (_contact_stats(vertices, faces, contact_patches,
                                     patient_frame, p)
                      if contact_patches else {})
    n0 = np.cross(vertices[faces[:, 1]] - vertices[faces[:, 0]],
                  vertices[faces[:, 2]] - vertices[faces[:, 0]])

    def occl_measure(v):
        """咬合面干涉测量（editable = occ_table 全体；子集坐标=occ_table id）。

        两个量**分开报**（O1）：
        - ``penetration_mm``：真穿透深度 ``-signed``（需要 signed<0 才有值）；
        - ``clearance_shortfall_mm``：``max(target_gap - signed, 0)``（目标相对亏欠）。
        无符号来源时只报后者的幅值版本，前者恒为 ``None``。
        """
        ids = np.flatnonzero(occ_t)
        cp, fids, signed, dist, valid = antagonist_source.query(v[ids])
        signed_ok = np.asarray(signed, float)
        shortfall = np.where(valid, np.maximum(p.target_gap_mm - signed_ok, 0.0), 0.0)
        hit = valid & (signed_ok < 0) & (dist <= p.inside_reach_mm)
        shared = _stats_of(shortfall[hit])          # 共用统计 primitive（p50/p95/max）
        depth = np.maximum(-signed_ok[hit], 0.0)
        depth_stats = _stats_of(depth)
        return {"ids": ids, "closest": cp, "fids": fids, "valid": valid,
                "signed": signed_ok, "dist": dist,
                "stats": {"count": int(hit.sum()),
                          "p50": (shared or {}).get("p50_mm", 0.0),
                          "p95": (shared or {}).get("p95_mm", 0.0),
                          "max": (shared or {}).get("max_mm", 0.0),
                          "sum": float(shortfall[hit].sum())},
                "signed_available": True,
                "signed_min_mm": float(signed_ok[valid].min()) if valid.any() else None,
                "penetration_mm": ({"max_mm": (depth_stats or {}).get("max_mm", 0.0),
                                    "p95_mm": (depth_stats or {}).get("p95_mm", 0.0),
                                    "count": int(hit.sum())} if hit.any() else
                                   {"max_mm": 0.0, "p95_mm": 0.0, "count": 0}),
                "clearance_shortfall_mm": {"max_mm": float(shortfall[valid].max()) if valid.any() else 0.0,
                                           "p95_mm": float(np.quantile(shortfall[valid], .95)) if valid.any() else 0.0,
                                           "count": int((shortfall[valid] > 0).sum()) if valid.any() else 0},
                "ambiguous": int(np.sum(~valid))}

    m0 = occl_measure(vertices)
    res.history.append({"iteration": 0, **m0["stats"], "ambiguous": m0["ambiguous"],
                        "sign_method": res.sign_method, "signed_available": True,
                        "signed_min_mm": m0["signed_min_mm"],
                        "penetration_mm": m0["penetration_mm"],
                        "clearance_shortfall_mm": m0["clearance_shortfall_mm"]})
    best_v = vertices.copy()
    best = m0["stats"]

    def flip_of(v):
        n1 = np.cross(v[faces[:, 1]] - v[faces[:, 0]],
                      v[faces[:, 2]] - v[faces[:, 0]])
        den = np.linalg.norm(n0, axis=1) * np.linalg.norm(n1, axis=1)
        okd = den > 1e-18
        return float(np.mean(np.einsum("ij,ij->i", n0[okd], n1[okd])
                             / den[okd] < 0)) if okd.any() else 0.0
    base_flip = flip_of(vertices)

    for it in range(1, p.max_outer_iterations + 1):
        if best["p95"] <= p.pen_tolerance_mm:
            res.notes.append("occlusal p95=%.4f ≤ tolerance，收敛" % best["p95"])
            break
        m = occl_measure(best_v)
        hit_ids = m["ids"][m["valid"] & (m["signed"] < 0)
                           & (m["dist"] <= p.inside_reach_mm)]
        hit_pos = np.flatnonzero(m["valid"] & (m["signed"] < 0)
                                 & (m["dist"] <= p.inside_reach_mm))
        if len(hit_pos) == 0:
            res.notes.append("无咬合干涉顶点，停止")
            break
        # 拉出方向取自**符号来源的 patch**（其 normal_sign 已按实体/已声明法向定好）
        nhat = antagonist_source.patch.normals[m["fids"][hit_pos]] \
            * antagonist_source.patch.normal_sign
        # 目标 = closest + n̂·target_gap（沿对颌法向拉出到 clearance）
        tgt = m["closest"][hit_pos] \
            + nhat * p.target_gap_mm
        # 过渡环：从干涉顶点 BFS，**被锁定集挡住 → 不出咬合面**
        seed = set(int(i) for i in hit_ids)
        seen = set(seed) | locked_ids
        free, cur = set(), set(seed)
        for _ in range(p.transition_rings):
            nxt = set()
            for v_ in cur:
                for w_ in adj.get(v_, ()):
                    if w_ not in seen:
                        nxt.add(w_)
            free |= nxt
            seen |= nxt
            cur = nxt
            if not cur:
                break
        # 注意：整片咬合面都在穿透时过渡环为空（没有可混合的自由顶点）——
        # 这不是停止条件，而是"刚性应用目标"的退化情形；line search 的翻面门
        # 会自然约束步长（实测首跑在这里直接停，p95 纹丝不动）。
        free_ids = np.asarray(sorted(free), dtype=np.int64)
        fixed_ids = np.setdiff1d(np.arange(len(vertices)), free_ids)
        accepted = False
        for alpha in p.line_search_alphas:
            xt = best_v.copy()
            xt[hit_ids] = best_v[hit_ids] + alpha * (tgt - best_v[hit_ids])
            if len(free_ids) == 0:
                v_cand = xt.copy()
            else:
                try:
                    v_cand, _info = solve_local_band(best_v, faces, free_ids,
                                                     fixed_ids, xt)
                except Exception as exc:          # noqa: BLE001
                    res.notes.append("α=%.3f 局部求解失败: %s" % (alpha, exc))
                    continue
            moved_locked = np.abs(v_cand - best_v)[np.asarray(
                sorted(locked_ids), dtype=np.int64)].max() if locked_ids else 0.0
            if moved_locked != 0.0:
                res.notes.append("α=%.3f 锁定区被移动（%.2e）—— 拒绝"
                                 % (alpha, moved_locked))
                continue
            if flip_of(v_cand) > base_flip + 0.002 + 0.004 * it:
                res.notes.append("α=%.3f 翻面恶化——拒绝" % alpha)
                continue
            m_cand = occl_measure(v_cand)
            if m_cand["stats"]["p95"] < best["p95"] - 1e-9 and \
                    m_cand["stats"]["max"] <= best["max"] + 1e-9:
                # **Contact 保持性复测**（只测不修）
                if contact_patches:
                    cs = _contact_stats(v_cand, faces, contact_patches,
                                        patient_frame, p)
                    for side in cs:
                        if cs[side]["p95"] > contact_before.get(
                                side, {}).get("p95", 0.0) \
                                + p.contact_preservation_tol_mm:
                            res.notes.append(
                                "α=%.3f Contact %s 恶化（%.3f→%.3f）—— 拒绝"
                                % (alpha, side,
                                   contact_before[side]["p95"], cs[side]["p95"]))
                            break
                    else:
                        pass
                    if any(cs[side]["p95"] > contact_before.get(
                            side, {}).get("p95", 0.0)
                            + p.contact_preservation_tol_mm for side in cs):
                        continue
                best_v = v_cand
                best = m_cand["stats"]
                accepted = True
                res.alphas_accepted.append(alpha)
                res.history.append({"iteration": it, "alpha": alpha, **best,
                                    "ambiguous": m_cand["ambiguous"],
                                    "sign_method": res.sign_method, "signed_available": True,
                                    "signed_min_mm": m_cand["signed_min_mm"],
                                    "penetration_mm": m_cand["penetration_mm"],
                                    "clearance_shortfall_mm": m_cand["clearance_shortfall_mm"]})
                break
        if not accepted:
            res.notes.append("所有 α 均无法单调改善，停止")
            break
        if len(res.history) > 1:
            prev = res.history[-2]["p95"]
            if prev > 0 and (prev - best["p95"]) / prev \
                    < p.min_relative_improvement:
                res.notes.append("相对改善 < ε，停止")
                break
        res.iterations = it

    # ---- 结果
    res.mesh = trimesh.Trimesh(vertices=best_v, faces=faces, process=False)
    m_fin = occl_measure(best_v)
    res.contact_preservation = {"before": contact_before,
                                "after": (_contact_stats(
                                    best_v, faces, contact_patches,
                                    patient_frame, p)
                                    if contact_patches else {})}
    d_occ = np.linalg.norm(best_v[occ_t] - vertices[occ_t], axis=1) \
        if len(occ_t) else np.array([0.0])
    res.occlusal_displacement = {"p50": float(np.quantile(d_occ, .5)),
                                 "p95": float(np.quantile(d_occ, .95)),
                                 "max": float(d_occ.max())}
    res.margin_max_displacement_mm = float(np.max(np.linalg.norm(
        best_v[loop_idx] - vertices[loop_idx], axis=1)))
    core_idx = np.asarray(sorted(locked_ids), dtype=np.int64)
    d_core = (np.linalg.norm(best_v[core_idx] - vertices[core_idx], axis=1)
              if len(core_idx) else np.array([0.0]))
    res.proximal_core_max_displacement_mm = float(d_core.max())
    res.n_flipped_faces = int(round(flip_of(best_v) * len(faces)))
    res.modified_vertex_count = int(np.sum(
        np.linalg.norm(best_v - vertices, axis=1) > 1e-9))
    res.status = ("OK" if m_fin["stats"]["p95"] <= p.pen_tolerance_mm
                  else "REVIEW_REQUIRED")
    # B1：把本域的测量契约显式标注进结果 —— 跨域读数字的人必须先看到"这是目标相对亏欠，
    # 不是 Contact 的穿透深度"，也看到"相交审计不在本域"（那是 thickness/services 的活）。
    res.domain_report = domain_report(
        DOMAIN_OCCLUSION,
        active_sign_method=res.sign_method,
        active_sign=antagonist_source.as_report(),
        final_stats=dict(m_fin["stats"]), ambiguous=m_fin["ambiguous"],
        iterations=res.iterations, status=res.status)
    res.notes.append("本域口径：%s；有效性=%s；相交审计不属本域（见 measurement_core）"
                     % (OCCLUSION_CONTRACT.penetration_meaning, OCCLUSION_CONTRACT.validity))
    return res
