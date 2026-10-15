"""Contact I.2：SDF 几何投影 + 局部带 Laplacian blend + 单调 line search。

归因（docs/CONTACT_ENGINE_V1.md I.2 节）锁定的范围：

    只解决 PROXIMAL_EDITABLE 穿透。
    OCCLUSAL_LOCKED → 移交 Occlusion Engine（handoff，**Contact 不许修**）。
    CERVICAL_LOCKED → 永久锁定。

架构（用户 2026-10-02 规格）：

    SDF 决定目标位置，Laplacian 只负责形变传播与平滑。

    每轮：
      重新分类（occ_table / 颈环带 / editable）→ 重测 signed 距离
      → editable 穿透顶点的 correction = 沿**邻牙 patch 法向**拉回 target_gap
      → line search α ∈ {1, 1/2, 1/4, ...}：local-band solve 后
         **只接受 editable 穿透单调下降**的步（且锁定区位移为 0、翻面不恶化）
      → 收敛（进 tolerance / 相对改善 < ε / 无 α 可行）即停

不再有固定 0.35mm trust step 与"两轮无改善即 rollback"（归因证明它们把合法
的 1~2mm 整面收窄直接判死）。累计形变不设人为上限，只如实报告分布
（proximal_shrink_*），供"UPSTREAM_CROWN_TOO_WIDth"判定。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy.spatial import cKDTree

from .contact import NeighborPatch, _boundary_loop
from .contact_refine import occ_table_mask, vertex_normals
from .deform_gate import deformation_gate
from .laplacian import solve_local_band
from .patient_frame import PatientFrame


class ContactI2Error(Exception):
    """I.2 输入/执行错误。"""


@dataclass
class I2Params:
    #: 目标接触间隙（与 Contact v1/v2 同义）。
    target_gap_mm: float = 0.05
    #: "真穿透"的欧氏距离上界。
    inside_reach_mm: float = 2.0
    #: patch 边界安全带（小于它 → 距离不可信，跳过并计数）。
    boundary_safety_mm: float = 0.6
    #: 颈缘锁带宽度。
    cervical_lock_mm: float = 0.3
    #: 咬合面定义（与 v2 一致，法向判据）。
    occlusal_anchor_percentile: float = 75.0
    occ_table_normal_dot: float = 0.6
    #: 过渡环数（顶点 BFS）。
    transition_rings: int = 3
    #: 外层迭代上限（不再"两轮即 rollback"）。
    max_outer_iterations: int = 10
    #: line search 步长序列。
    line_search_alphas: tuple = (1.0, 0.5, 0.25, 0.125, 0.0625)
    #: 收敛：editable p95 ≤ tolerance。
    pen_tolerance_mm: float = 0.05
    #: 停止：相对改善 < ε。
    min_relative_improvement: float = 0.02
    #: 几何硬门的尺度（mm）：翻面/自交按 (scale/2)² 面积分级 micro。
    #: 取网格尺度（≈体素），**不是临床阈值**；构造参数，manifest 记 provenance。
    gate_scale_mm: float = 0.10


@dataclass
class I2Result:
    status: str = "FAILED"
    mesh: object | None = None
    iterations: int = 0
    alphas_accepted: list = field(default_factory=list)
    #: 每轮的 editable/occlusal/cervical 穿透统计（含 before）
    history: list = field(default_factory=list)
    handoff_to_occlusion: dict = field(default_factory=dict)
    proximal_shrink: dict = field(default_factory=dict)
    margin_max_displacement_mm: float = 0.0
    occlusal_p95_displacement_mm: float = 0.0
    n_flipped_faces: int = 0
    #: 有可修穿透、但没有任何 α 能"改善 + 几何安全" → 交给上层判 conflict
    constraint_conflict: bool = False
    notes: list = field(default_factory=list)


def _loop_of(vertices, faces):
    loop = _boundary_loop(vertices, faces)
    if loop is None:
        raise ContactI2Error("冠网格没有可识别的颈环边界")
    return np.asarray(loop, dtype=np.int64)


def _zones(vertices, faces, frame, p):
    """→ (loop_idx, loop_set, occ_table, cervical_band, editable)。"""
    loop_idx = _loop_of(vertices, faces)
    loop_set = np.zeros(len(vertices), dtype=bool)
    loop_set[loop_idx] = True
    occ_t = occ_table_mask(vertices, faces, frame, p)
    d_loop = cKDTree(vertices[loop_idx]).query(vertices, k=1)[0]
    cervical = (d_loop < p.cervical_lock_mm) & ~loop_set
    editable = ~loop_set & ~occ_t & ~cervical
    return loop_idx, loop_set, occ_t, cervical, editable


def _half_domains(vertices, frame, loop_set):
    """按近远中半域划分测量域（与 v1/v2 同义：sign·u ≥ −0.5）。"""
    rel = vertices - frame.origin
    u = rel @ frame.x
    return {"mesial": (u >= -0.5) & ~loop_set,
            "distal": (-u >= -0.5) & ~loop_set}


def _pen_stats(signed, ids, p):
    pen = np.maximum(p.target_gap_mm - signed, 0.0)
    if len(ids) == 0:
        return {"count": 0, "p50": 0.0, "p95": 0.0, "max": 0.0}
    return {"count": int(len(ids)), "p50": float(np.median(pen)),
            "p95": float(np.quantile(pen, .95)), "max": float(pen.max())}


def measure(vertices, faces, patches, domains, editable, occ_t, cervical,
            frame, p, vertices_ref=None):
    """逐侧测量：editable / occlusal / cervical 三账 + patch 歧义计数。

    口径 = ``NeighborPatch.clearance_detail``（全项目唯一，用户决策 ③）。
    ``vertices_ref``：**有效性掩码在参考几何上一次冻结**（用户纪律：判据必须在
    固定参考系上算并锁住）—— 投影目标 signed 始终来自当前几何，但"哪些点算
    接触证据"由参考几何决定，避免掩码随形变漂移。
    """
    out = {}
    for side, patch in patches.items():
        # **不在这里定向**：冠深陷邻牙时"区域法向投票"会翻错（I.1 时代已知），
        # 且逐轮重定向会让符号在迭代间漂移。定向在 solve 入口做**一次**
        # （质心判据，见 _orient_toward_crown_centroid）。
        wids = np.flatnonzero(domains[side])
        signed, valid, cp, fids, parts = patch.clearance_detail(vertices[wids])
        if vertices_ref is not None:
            _s2, valid_ref, _c2, _f2, parts = patch.clearance_detail(
                np.asarray(vertices_ref, float)[wids])
            valid = valid_ref
        amb = int(np.sum(~parts["valid_boundary"]))
        amb_crease = int(np.sum(~parts["valid_crease"]))
        # 注意：cp/signed/valid 等都是**子集坐标**（长度 = len(wids)），
        # wids 才是全局 id。曾写成 g=wids 再 g[wids] —— 数组自索引
        # （wids 长度 177、值到 385）当场越界，报错的 "size 177" 是 wids 自己。
        pen = np.maximum(p.target_gap_mm - signed, 0.0)
        pen_v = np.where(valid & (signed < 0), pen, 0.0)

        def stats(mask):
            sel = mask & valid
            ids = np.flatnonzero(sel)
            if len(ids) == 0:
                return {"count": 0, "p50": 0.0, "p95": 0.0, "max": 0.0,
                        "sum": 0.0}
            return {"count": int(len(ids)),
                    "p50": float(np.median(pen_v[ids])),
                    "p95": float(np.quantile(pen_v[ids], .95)),
                    "max": float(pen_v[ids].max()),
                    "sum": float(pen_v[ids].sum())}

        is_edit = editable[wids]
        is_occ = occ_t[wids]
        is_cer = cervical[wids]
        out[side] = {
            "ambiguous_patch_edge": amb,
            "ambiguous_crease": amb_crease,
            "editable": stats(is_edit & (signed < 0)),
            "occlusal": stats(is_occ & (signed < 0)),
            "cervical": stats(is_cer & (signed < 0)),
            # 投影目标（只对 editable 穿透顶点）
            "ids": wids[is_edit & valid & (signed < 0)],
            "closest": cp[is_edit & valid & (signed < 0)],
            "fids": fids[is_edit & valid & (signed < 0)],
        }
    return out


def solve_proximal_projection(crown_mesh, patient_frame: PatientFrame,
                              mesial_patch: NeighborPatch,
                              distal_patch: NeighborPatch,
                              params: I2Params | None = None) -> I2Result:
    """I.2 主入口：SDF 投影定目标，local-band Laplacian 传播，line search 接受。"""
    p = params or I2Params()
    res = I2Result()
    vertices = np.asarray(crown_mesh.vertices, dtype=np.float64).copy()
    faces = np.asarray(crown_mesh.faces, dtype=np.int64)
    loop_idx, loop_set, occ_t, cervical, editable = _zones(
        vertices, faces, patient_frame, p)
    locked = loop_set | occ_t
    locked_ids = set(int(i) for i in np.flatnonzero(locked))
    patches = {"mesial": mesial_patch, "distal": distal_patch}
    # 一次性定向（质心判据）：冠质心在邻牙体外 —— 与穿透深度无关，
    # 多数投票在"整面深陷"时必然翻错（实测 I.2 首跑全 0 穿透的根因）。
    centroid = vertices.mean(axis=0)
    for patch in patches.values():
        fc = patch.vertices[patch.faces].mean(axis=1)
        w = np.einsum("ij,ij->i", centroid - fc, patch.normals)
        patch.normal_sign = 1.0 if w.sum() > 0 else -1.0
    domains = _half_domains(vertices, patient_frame, loop_set)
    adj = {}
    for a, b, c3 in faces:
        for uu, vv in ((int(a), int(b)), (int(b), int(c3)), (int(c3), int(a))):
            adj.setdefault(uu, set()).add(vv)
            adj.setdefault(vv, set()).add(uu)

    m0 = measure(vertices, faces, patches, domains, editable, occ_t, cervical,
                 patient_frame, p, vertices_ref=vertices)
    res.history.append({
        "iteration": 0,
        "mesial_editable": m0["mesial"]["editable"],
        "distal_editable": m0["distal"]["editable"],
        "mesial_occlusal": m0["mesial"]["occlusal"],
        "distal_occlusal": m0["distal"]["occlusal"],
        "mesial_cervical": m0["mesial"]["cervical"],
        "distal_cervical": m0["distal"]["cervical"],
    })

    def combined_pen(m):
        return max(m["mesial"]["editable"]["p95"],
                   m["distal"]["editable"]["p95"])

    best_v = vertices.copy()
    best_pen = combined_pen(m0)
    n0 = np.cross(vertices[faces[:, 1]] - vertices[faces[:, 0]],
                  vertices[faces[:, 2]] - vertices[faces[:, 0]])

    for it in range(1, p.max_outer_iterations + 1):
        m = measure(best_v, faces, patches, domains, editable, occ_t, cervical,
                    patient_frame, p, vertices_ref=vertices)
        cur_pen = combined_pen(m)
        if cur_pen <= p.pen_tolerance_mm:
            res.notes.append("editable p95=%.4f ≤ tolerance，收敛" % cur_pen)
            break
        # 投影目标（沿邻牙 patch 法向）
        ids_all, tgt_all = [], []
        for side in ("mesial", "distal"):
            ids = m[side]["ids"]
            if len(ids) == 0:
                continue
            nhat = patches[side].normals[m[side]["fids"]] \
                * patches[side].normal_sign
            tgt = m[side]["closest"] + nhat * p.target_gap_mm
            ids_all.append(np.asarray(ids, dtype=np.int64))
            tgt_all.append(tgt)
        if not ids_all:
            res.notes.append("无 editable 穿透顶点，停止")
            break
        ids = np.concatenate(ids_all)
        tgt = np.vstack(tgt_all)
        # 过渡环：从目标顶点 BFS，锁定区绝不进入
        seed = set(int(i) for i in ids)
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
        if not free:
            res.notes.append("过渡环为空，停止")
            break
        free_ids = np.asarray(sorted(free), dtype=np.int64)
        fixed_ids = np.setdiff1d(np.arange(len(vertices)), free_ids)

        # 几何硬门的**聚焦集合**（本步可能动的面）：修改区 ∪ 过渡环的 1-ring。
        # 该集合在每个 α 之间不变 → baseline（形变前同口径自交）只需要算一次。
        focus_v = np.zeros(len(vertices), dtype=bool)
        focus_v[ids] = True
        focus_v[free_ids] = True
        focus_faces = np.flatnonzero(focus_v[faces].any(axis=1))
        base = deformation_gate(best_v, best_v, faces, scale_mm=p.gate_scale_mm,
                                focus_faces=focus_faces,
                                locked_ids=locked_ids)
        accepted = False
        for alpha in p.line_search_alphas:
            xt = best_v.copy()
            xt[ids] = best_v[ids] + alpha * (tgt - best_v[ids])
            try:
                v_cand, _info = solve_local_band(best_v, faces, free_ids,
                                                 fixed_ids, xt)
            except Exception as exc:              # noqa: BLE001
                res.notes.append("α=%.3f 局部求解失败: %s" % (alpha, exc))
                res.history.append({"iteration": it, "alpha": alpha,
                                    "accepted": False,
                                    "reject": "solve_failed: %s" % str(exc)[:80]})
                continue
            # **硬几何门**（用户 2026-10-08 规格）：零新增自交 / 零非流形 /
            # 零翻面（面积 ≥ (scale/2)²）/ 锁定区位移精确 0。
            # 旧的"翻面别太多（+0.2% tolerance）"已删除 —— 它正是
            # "为了清零穿透换来局部折叠"的入口。
            gate = deformation_gate(best_v, v_cand, faces,
                                    scale_mm=p.gate_scale_mm,
                                    focus_faces=focus_faces,
                                    locked_ids=locked_ids,
                                    baseline_pairs=base.pairs)
            m_cand = measure(v_cand, faces, patches, domains, editable, occ_t,
                             cervical, patient_frame, p, vertices_ref=vertices)
            new_pen = combined_pen(m_cand)
            improved = new_pen < cur_pen - 1e-9
            rec = {
                "iteration": it, "alpha": alpha,
                "editable_p95_before": round(float(cur_pen), 5),
                "editable_p95_after": round(float(new_pen), 5),
                "mesial_editable": m_cand["mesial"]["editable"],
                "distal_editable": m_cand["distal"]["editable"],
                "mesial_occlusal": m_cand["mesial"]["occlusal"],
                "distal_occlusal": m_cand["distal"]["occlusal"],
                "mesial_cervical": m_cand["mesial"]["cervical"],
                "distal_cervical": m_cand["distal"]["cervical"],
                "gate": gate.as_dict(),
                "improved": bool(improved),
                "accepted": False,
            }
            if not improved:
                rec["reject"] = "editable p95 未下降"
            elif not gate.ok:
                rec["reject"] = "geometry gate: " + gate.note
            else:
                best_v = v_cand
                best_pen = new_pen
                accepted = True
                rec["accepted"] = True
                res.alphas_accepted.append(alpha)
            res.history.append(rec)
            if accepted:
                break
        if not accepted:
            # 有可修的穿透、却没有任何 α 同时满足"改善 + 几何安全"
            res.constraint_conflict = True
            res.notes.append(
                "所有 α 均无法在**保持几何安全**的前提下单调改善 → "
                "CONTACT_CONSTRAINT_CONFLICT（拒绝以折叠换穿透清零）")
            break
        prev = res.history[-2] if len(res.history) > 1 else None
        if prev is not None:
            prev_pen = max(prev["mesial_editable"]["p95"],
                           prev["distal_editable"]["p95"])
            if prev_pen > 0 and (prev_pen - best_pen) / prev_pen \
                    < p.min_relative_improvement:
                res.notes.append("相对改善 < ε，停止")
                break
        res.iterations = it

    # ---- 结果组装
    import trimesh
    res.mesh = trimesh.Trimesh(vertices=best_v, faces=faces, process=False)
    m_final = measure(best_v, faces, patches, domains, editable, occ_t,
                      cervical, patient_frame, p, vertices_ref=vertices)
    res.handoff_to_occlusion = {
        "note": "OCCLUSAL_LOCKED 穿透 Contact 不修，移交 Occlusion Engine",
        "mesial": m_final["mesial"]["occlusal"],
        "distal": m_final["distal"]["occlusal"],
    }
    # proximal 收窄量分布（editable 顶点从 v_ref 起的位移，沿半域方向的分量）
    shrink = []
    rel = (best_v - vertices) @ patient_frame.x
    for side, sign in (("mesial", 1.0), ("distal", -1.0)):
        sel = np.flatnonzero(editable & domains[side])
        shrink.append({"side": side,
                       "p50": float(np.quantile(sign * rel[sel], .5)),
                       "p95": float(np.quantile(sign * rel[sel], .95)),
                       "max": float((sign * rel[sel]).max()),
                       "moved_fraction": float(np.mean(np.abs(rel[sel]) > 1e-9))})
    res.proximal_shrink = {"per_side": shrink}
    res.margin_max_displacement_mm = float(np.max(np.linalg.norm(
        best_v[loop_idx] - vertices[loop_idx], axis=1)))
    dt = np.linalg.norm(best_v[occ_t] - vertices[occ_t], axis=1) \
        if len(occ_t) else np.array([0.0])
    res.occlusal_p95_displacement_mm = float(np.percentile(dt, 95))
    n1 = np.cross(best_v[faces[:, 1]] - best_v[faces[:, 0]],
                  best_v[faces[:, 2]] - best_v[faces[:, 0]])
    den = np.linalg.norm(n0, axis=1) * np.linalg.norm(n1, axis=1)
    okd = den > 1e-18
    res.n_flipped_faces = int(np.sum(
        np.einsum("ij,ij->i", n0[okd], n1[okd]) / den[okd] < 0))
    if res.constraint_conflict:
        res.status = "CONSTRAINT_CONFLICT"
    else:
        res.status = ("OK" if combined_pen(m_final) <= p.pen_tolerance_mm
                      else "REVIEW_REQUIRED")
    return res



