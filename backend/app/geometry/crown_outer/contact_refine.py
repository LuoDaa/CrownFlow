"""I.1 Contact Penetration Refinement —— Contact v1 之后的稳定局部精修。

用户规格（提交 I.1）的六条要点，逐条对应：

1. **固定 reference，不再累计形变**：``V_ref`` = H.1 冠，永不改变。
   每轮只用当前 ``V_best`` 重新测最近面，Laplacian 求解始终以 ``V_ref``
   为基准（delta 坐标来自 H.1 的解剖）。上一版迭代发散的根因就是
   锚点跟着已变形的网格漂移。
2. **Oriented Distance 而不是封闭体 inside/outside**：``φ(x) = (x−q)·n``
   每轮重新找 ``q, n``。邻牙是裁出来的**开放曲面 patch**，没有严格的
   inside/outside，传统 SDF 的正负号在 patch 边界附近不可靠。
3. **patch 边界单独处理**：最近点落在开放边界安全带内 → ``valid=False``，
   语义 ``AMBIGUOUS_PATCH_EDGE``，不参与投影（这是 −4mm 伪 signed 的根源）。
4. **阻尼 + trust region**：``Δx = α(g−φ)n``，α 默认 0.4，单轮
   ``max_step_mm`` 夹紧；重新测最近三角形，最多 ``max_iterations`` 轮。
5. **收敛用几何量而不是 status**：penetration_max / p95、contact_gap_error、
   翻转率不恶化、Margin/Occlusal 位移 ≈ 0。连续两轮穿透不降 → 停止并
   ``REVIEW_REQUIRED``，绝不硬迭代。
6. **rollback**：候选先评估，只有整体目标改善才提交；否则退回上一轮。
   这样即使局部最近面切换得很差，也不会再现上一版
   ``翻转 32% / lingual +4.97mm`` 的崩坏。

**不上 IPC**：IPC Toolkit 的 barrier potential / CCD 适合把 Contact+Occlusion
统一成严格碰撞优化器，但当前只是修冠的局部接触，引入它会让工程复杂度
上一个量级。留给以后按需评估。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .contact import (
    CONTACT_OCCLUSAL_ANCHOR,
    ContactError,
    ContactResult,
    NeighborPatch,
    SideContactReport,
    _boundary_loop,
    _region_mask,
)
from .laplacian import DeformInputs, solve_laplacian
from .models import STATUS_FAILED, STATUS_OK, STATUS_REVIEW_REQUIRED, \
    CrownOuterConfig
from .patient_frame import PatientFrame


@dataclass
class RefineParams:
    """I.1 精修参数。**全部 PROVISIONAL**，由合成 fixture + 真实例扫参数定。"""

    #: 阻尼系数 α（规格建议 0.3~0.5）。
    damping: float = 0.4  # PROVISIONAL
    #: 单轮最大位移（trust region）。
    max_step_mm: float = 0.35  # PROVISIONAL
    #: 最大轮数（规格建议 5~10）。
    max_iterations: int = 8  # PROVISIONAL
    #: 最近点距 patch 边界的安全带（小于它 → AMBIGUOUS_PATCH_EDGE）。
    boundary_safety_mm: float = 0.6  # PROVISIONAL
    #: 目标接触间隙。
    target_gap_mm: float = 0.05  # PROVISIONAL
    #: 接触区带宽比例（沿用 v1 语义）。
    contact_band_fraction: float = 0.6
    #: 颈缘锁带。
    cervical_lock_mm: float = 0.3
    #: 咬合区分位（_region_mask 的轴向 cut，与 CrownOuter 同义）。
    occlusal_anchor_percentile: float = 75.0
    #: "真穿透"的欧氏距离上界。
    inside_reach_mm: float = 2.0
    #: 收敛阈值。
    penetration_tol_mm: float = 0.05  # PROVISIONAL
    #: **Contact v2：局部带状求解**（默认开）。自由区 = 接触带 + 过渡环；
    #: 咬合/颈缘/远场是固定边界 —— 位移由构造精确为 0（不再是弱锚）。
    local_band: bool = True
    #: 接触带外扩的过渡环数（顶点 BFS）。
    transition_rings: int = 3
    contact_gap_tol_mm: float = 0.10  # PROVISIONAL
    #: contact_gap_error 的统计带：只统计"离邻牙最近处 + band"内的顶点。
    #: 带太宽（曾用 1.0mm）会把本来该有间隙的顶点也算进来，误差永远不收敛。
    contact_band_mm: float = 0.4  # PROVISIONAL
    #: 接受新轮次时允许的翻转率恶化量。
    flip_worsen_tol: float = 0.002  # PROVISIONAL

    def as_dict(self) -> dict:
        return {k: getattr(self, k) for k in sorted(self.__dataclass_fields__)
                } | {"_provisional": True}


#: 咬合锚区定义：冠部 z 高于该分位视为咬合区（与 CrownOuter 同义）。
OCCLUSAL_ANCHOR_PERCENTILE = 75.0


def vertex_normals(vertices, faces) -> np.ndarray:
    """面积加权顶点法向（用于 patch 定向）。"""
    tri = vertices[faces]
    fn = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    out = np.zeros_like(vertices)
    for k in range(3):
        np.add.at(out, faces[:, k], fn)
    n = np.linalg.norm(out, axis=1)
    n[n < 1e-18] = 1.0
    return out / n[:, None]


def _loop_and_zones(vertices, faces, patient_frame, params):
    loop = _boundary_loop(vertices, faces)
    if loop is None:
        raise ContactError("冠网格没有可识别的颈环边界")
    loop_idx = np.asarray(loop, dtype=np.int64)
    loop_set = np.zeros(len(vertices), dtype=bool)
    loop_set[loop_idx] = True
    crown = ~loop_set
    z_thr = np.percentile(vertices[crown, 2], OCCLUSAL_ANCHOR_PERCENTILE)
    occ = np.flatnonzero(crown & (vertices[:, 2] >= z_thr))
    return loop_idx, loop_set, occ


def _side_measure(vertices, patch, mask, params, *, wide=None):
    """单侧测量：ori大距离 + 边界有效性 + 接触区。"""
    m = RefineParams() if params is None else params
    wide = mask if wide is None else wide
    wids = np.flatnonzero(wide)
    if len(wids) == 0:
        return None
    # **统一口径**（用户 2026-10-09 决策 ③）：幅值 euclid + 边界/crease 有效性；
    # `dist ≡ |signed|`，`valid` 同时含两类歧义掩码。
    signed, valid, cp, fids, _parts = patch.clearance_detail(
        vertices[wids], boundary_safety_mm=m.boundary_safety_mm)
    dist = np.abs(np.asarray(signed, np.float64))
    nhat = patch.normals[fids] * patch.normal_sign
    in_band = mask[wids]
    phi = signed
    inside = valid & (phi < 0) & (dist <= m.inside_reach_mm)
    out = {
        "ids": wids, "cp": cp, "fids": fids, "phi": phi, "dist": dist,
        "valid": valid, "nhat": nhat, "in_band": in_band, "inside": inside,
        "penetration_max_mm": float(-np.min(phi[inside]))
        if inside.any() else 0.0,
        "n_ambiguous_edge": int((~np.asarray(valid, bool)).sum()),
    }
    if inside.any():
        pen_vals = -phi[inside]
        out["penetration_p95_mm"] = float(np.percentile(pen_vals, 95))
    else:
        out["penetration_p95_mm"] = 0.0
    # 接触区带（用于 gap error）
    pos = phi[valid & (phi >= 0)]
    pos_min = float(pos.min()) if len(pos) else np.inf
    band = in_band & valid & (dist <= pos_min + m.contact_band_mm)
    if band.any():
        out["contact_gap_error_p95_mm"] = float(np.percentile(
            np.abs(phi[band] - m.target_gap_mm), 95))
    else:
        out["contact_gap_error_p95_mm"] = 0.0
    out["n_contact_band"] = int(band.sum())
    out["band"] = band
    return out


def evaluate(vertices, faces, patient_frame, patches, masks, wides,
             params, v_ref, loop_idx, occ) -> dict:
    """几何量评估（收敛与 rollback 的唯一依据，不看 status）。"""
    m = params
    stats = {}
    for side, patch in patches.items():
        st = _side_measure(vertices, patch, masks[side], m, wide=wides[side])
        stats[side] = st
    pen_max = max((s["penetration_max_mm"] for s in stats.values()
                   if s), default=0.0)
    pen_p95 = max((s["penetration_p95_mm"] for s in stats.values()
                   if s), default=0.0)
    gap_err = max((s["contact_gap_error_p95_mm"] for s in stats.values()
                   if s), default=0.0)
    # 翻转率（相对 V_ref）
    n0 = np.cross(v_ref[faces[:, 1]] - v_ref[faces[:, 0]],
                  v_ref[faces[:, 2]] - v_ref[faces[:, 0]])
    n1 = np.cross(vertices[faces[:, 1]] - vertices[faces[:, 0]],
                  vertices[faces[:, 2]] - vertices[faces[:, 0]])
    den = np.linalg.norm(n0, axis=1) * np.linalg.norm(n1, axis=1)
    okd = den > 1e-18
    flipped = float(np.mean(
        np.einsum("ij,ij->i", n0[okd], n1[okd]) / den[okd] < 0)) if okd.any() \
        else 0.0
    return {
        "penetration_max_mm": pen_max,
        "penetration_p95_mm": pen_p95,
        "contact_gap_error_p95_mm": gap_err,
        "flipped_ratio": flipped,
        "margin_disp_max_mm": float(np.max(np.linalg.norm(
            vertices[loop_idx] - v_ref[loop_idx], axis=1))),
        "occlusal_disp_p95_mm": float(np.percentile(
            np.linalg.norm(vertices[occ] - v_ref[occ], axis=1), 95))
        if len(occ) else 0.0,
        "sides": stats,
    }


def _key(metrics):
    """改善判据的字典序键（越小越好）。"""
    return (round(metrics["penetration_max_mm"], 4),
            round(metrics["contact_gap_error_p95_mm"], 4))


def _improves(new, best, params) -> bool:
    if new["flipped_ratio"] > best["flipped_ratio"] + params.flip_worsen_tol:
        return False
    return _key(new) < _key(best)


def _converged(m, params) -> bool:
    return (m["penetration_max_mm"] <= params.penetration_tol_mm
            and m["penetration_p95_mm"] <= params.penetration_tol_mm
            and m["contact_gap_error_p95_mm"] <= params.contact_gap_tol_mm)


def _vertex_adjacency(faces, n_vertices):
    adj = [set() for _ in range(n_vertices)]
    for a, b, c in faces:
        adj[a].update((b, c))
        adj[b].update((a, c))
        adj[c].update((a, b))
    return adj


def _transition_rings(seeds, locked, adj, n_rings):
    """接触带**外侧**的过渡环（不含种子本身；颈环/咬合绝不进入）。

    语义（Contact v2）：接触带顶点是**固定**的（硬设到目标位），只有外侧
    过渡环是自由变量 —— 第一版曾把种子也放进自由区，而目标只作用于固定
    顶点，结果没有任何力把接触区拉向目标，穿透纹丝不动（合成测试抓到）。
    """
    # 注意：locked 是**布尔数组**；set(locked) 会迭代出 {True, False} 而不是
    # 索引（实测踩到：锁定排除因此静默失效，过渡环侵入了颈环与咬合区）。
    locked_set = set(int(i) for i in np.flatnonzero(locked))
    cur = set(int(i) for i in seeds) - locked_set
    seen = set(cur) | locked_set
    out = set()
    for _ in range(n_rings):
        nxt = set()
        for v in cur:
            for w in adj[v]:
                if w not in seen:
                    nxt.add(w)
        out |= nxt
        seen |= nxt
        cur = nxt
        if not cur:
            break
    return out


def refine_contacts(crown_h1, patient_frame: PatientFrame,
                    mesial_neighbor: NeighborPatch,
                    distal_neighbor: NeighborPatch,
                    params: RefineParams | None = None,
                    *,
                    start_from=None) -> ContactResult:
    """I.1 主入口。``crown_h1``：H.1 冠（**V_ref，永不改变**）。

    ``start_from``：初始 ``V_best``（一般传 Contact v1 的输出网格）；
    缺省用 ``crown_h1`` 自身。
    """
    p = params or RefineParams()
    res = ContactResult(parameters=p.as_dict())
    try:
        v_ref = np.asarray(crown_h1.vertices, dtype=np.float64).copy()
        faces = np.asarray(crown_h1.faces, dtype=np.int64)
        loop_idx, loop_set, occ = _loop_and_zones(
            v_ref, faces, patient_frame, p)
        u_lat = (v_ref - patient_frame.origin) @ patient_frame.x
        patches = {"mesial": mesial_neighbor, "distal": distal_neighbor}
        masks, wides = {}, {}
        for s in ("mesial", "distal"):
            mask = _region_mask(v_ref, patient_frame, loop_idx, p, s) \
                & ~loop_set
            masks[s] = mask
            sign = 1.0 if s == "mesial" else -1.0
            wides[s] = (sign * u_lat >= -0.5) & ~loop_set
        cn = vertex_normals(v_ref, faces)
        for side, patch in patches.items():
            ids = np.flatnonzero(masks[side])
            if len(ids) == 0:
                ids = np.flatnonzero(~loop_set)
            patch.orient_against_crown(v_ref[ids], cn[ids])

        occ_metric = occ
        if p.local_band:
            # 指标与锁定用**同一**咬合面定义（法向判据）；纯 z 分位的旧 occ
            # 含邻接壁，会把接触修正误读成咬合拖动。
            occ_metric = np.flatnonzero(occ_table_mask(v_ref, faces,
                                                       patient_frame, p))
        v_best = (np.asarray(start_from.vertices, dtype=np.float64).copy()
                  if start_from is not None else v_ref.copy())
        best = evaluate(v_best, faces, patient_frame, patches, masks, wides,
                        p, v_ref, loop_idx, occ_metric)
        history = [{"iteration": 0, **{k: v for k, v in best.items()
                                      if k != "sides"}}]
        no_progress = 0
        it_used = 0

        for it in range(p.max_iterations):
            # ---- 从 V_best 重新测最近面，算阻尼步长
            need_ids, need_tgt, penalties = [], [], []
            for side in ("mesial", "distal"):
                st = _side_measure(v_best, patches[side], masks[side], p,
                                   wide=wides[side])
                if st is None:
                    continue
                active = st["valid"] & (
                    (st["phi"] < p.target_gap_mm)
                    | (st["in_band"] & (st["dist"] <= float(
                        np.min(st["dist"])) + p.contact_band_mm)))
                if not active.any():
                    continue
                order = np.flatnonzero(active)
                delta = p.damping * (p.target_gap_mm
                                     - st["phi"][order])[:, None] \
                    * st["nhat"][order]
                norm = np.linalg.norm(delta, axis=1)
                too_big = norm > p.max_step_mm
                if too_big.any():
                    delta[too_big] *= (p.max_step_mm
                                       / norm[too_big])[:, None]
                need_ids.append(st["ids"][order])
                need_tgt.append(v_best[st["ids"][order]] + delta)
                penalties.append(float(norm.max()) if len(norm) else 0.0)
            if not need_ids:
                break
            ids = np.concatenate(need_ids)
            tgt = np.vstack(need_tgt)
            if len(ids) < 8:
                # 活跃集过小：没有可修的量，视为收敛
                #（solve_laplacian 需要 ≥8 个边界顶点，硬凑没意义）。
                res.notes.append("活跃约束仅 %d 个顶点，停止迭代" % len(ids))
                break

            # ---- 永远以 V_ref 为基准求解（不累计形变）
            if p.local_band:
                # **Contact v2**：自由区 = 接触带 + 过渡环；颈环与咬合区是
                # 固定边界（位移由构造精确为 0）。接触带顶点的目标由本侧
                # need 集给出；两侧的 need 合并成一次求解。
                locked = loop_set.copy()
                # 咬合锁定 = 高 z **且法向朝轴**（咬合面）；径向法向的高 z
                # 顶点是邻接壁，必须允许修正 —— 只按 z 分位会把接触区锁死。
                locked[occ_table_mask(v_ref, faces, patient_frame, p)] = True
                adj = _vertex_adjacency(faces, len(v_ref))
                seeds = set()
                for arr in need_ids:
                    seeds.update(int(i) for i in arr)
                free = _transition_rings(seeds, locked, adj,
                                         p.transition_rings)
                if not free:
                    v_cand = v_ref.copy()
                else:
                    free_ids = np.asarray(sorted(free), dtype=np.int64)
                    fixed_ids = np.setdiff1d(np.arange(len(v_ref)), free_ids)
                    xt = v_ref.copy()
                    # 接触带顶点的硬目标（两侧合并）。**锁定区（颈环 ∪ 咬合）
                    # 的目标一律忽略**：落在锁定区的接触种子本迭代不动
                    # （fail-safe）——咬合接触是 Occlusion Engine 的职责，
                    # 这里绝不允许为修邻接而移动咬合/颈缘顶点。
                    for arr_ids, arr_tgt in zip(need_ids, need_tgt):
                        ids_arr = np.asarray(arr_ids, dtype=np.int64)
                        movable = ~locked[ids_arr]
                        xt[ids_arr[movable]] =                             np.asarray(arr_tgt, dtype=np.float64)[movable]
                    from app.geometry.crown_outer.laplacian import                         solve_local_band
                    v_cand, _info = solve_local_band(
                        v_ref, faces, free_ids, fixed_ids, xt)
            else:
                v_cand, _info = solve_laplacian(DeformInputs(
                    vertices_placed=v_ref, faces=faces,
                    margin_vertex_indices=ids, margin_targets=tgt,
                    root_vertex_indices=loop_idx,
                    occlusal_vertex_indices=occ,
                    config=CrownOuterConfig(
                        weight_occlusal_anchor=CONTACT_OCCLUSAL_ANCHOR,
                        max_constraint_residual_mm=0.05)))

            cand = evaluate(v_cand, faces, patient_frame, patches, masks,
                            wides, p, v_ref, loop_idx, occ_metric)
            it_used = it + 1
            if _improves(cand, best, p):
                # 阻尼：只取阻尼系数比例的改善量（trust region 已夹紧单步）
                v_best, best = v_cand, cand
                history.append({"iteration": it_used, "accepted": True,
                                **{k: v for k, v in cand.items()
                                   if k != "sides"}})
                no_progress = 0
            else:
                history.append({"iteration": it_used, "accepted": False,
                                **{k: v for k, v in cand.items()
                                   if k != "sides"}})
                no_progress += 1
                if no_progress >= 2:
                    res.notes.append(
                        "连续 %d 轮无改善，停止迭代（rollback）" % no_progress)
                    break
            if _converged(best, p):
                break

        # ---- 组装结果
        import trimesh
        res.mesh = trimesh.Trimesh(vertices=v_best, faces=faces,
                                   process=False)
        res.margin_max_displacement_mm = best["margin_disp_max_mm"]
        res.occlusal_p95_displacement_mm = best["occlusal_disp_p95_mm"]
        n0 = np.cross(v_ref[faces[:, 1]] - v_ref[faces[:, 0]],
                      v_ref[faces[:, 2]] - v_ref[faces[:, 0]])
        n1 = np.cross(v_best[faces[:, 1]] - v_best[faces[:, 0]],
                      v_best[faces[:, 2]] - v_best[faces[:, 0]])
        den = np.linalg.norm(n0, axis=1) * np.linalg.norm(n1, axis=1)
        okd = den > 1e-18
        res.n_flipped_faces = int(np.sum(
            np.einsum("ij,ij->i", n0[okd], n1[okd]) / den[okd] < 0))
        for side in ("mesial", "distal"):
            rep = SideContactReport(side=side)
            st = best["sides"].get(side)
            if st is not None:
                rep.n_region_vertices = int(masks[side].sum())
                rep.penetration_mm = st["penetration_max_mm"]
                rep.min_distance_mm = float(np.min(st["dist"]))
                rep.contact_area_mm2 = float(st["n_contact_band"]) * (
                    _area_per_vertex(v_ref, faces))
                rep.vertices_modified = int(np.sum(
                    np.linalg.norm(v_best - v_ref, axis=1) > 1e-6))
                rep.status = "OK" if st["penetration_max_mm"] <= \
                    p.penetration_tol_mm else "PENETRATION_REMAINS"
            setattr(res, side, rep)

        conv = _converged(best, p)
        notes = list(res.notes)
        if not conv:
            notes.append(
                "未收敛：penetration_max=%.3f 阈值 %.3f，gap_err_p95=%.3f "
                "阈值 %.3f" % (best["penetration_max_mm"],
                               p.penetration_tol_mm,
                               best["contact_gap_error_p95_mm"],
                               p.contact_gap_tol_mm))
        if best["margin_disp_max_mm"] > 0.01:
            notes.append("margin 位移 %.4f 超过 0.01"
                         % best["margin_disp_max_mm"])
        if best["occlusal_disp_p95_mm"] > 0.3:
            notes.append("咬合区位移 p95 %.3f 偏大"
                         % best["occlusal_disp_p95_mm"])
        res.notes = notes
        res.status = STATUS_OK if (conv and not notes) else \
            STATUS_REVIEW_REQUIRED
        res.parameters = p.as_dict() | {"iterations_used": it_used}
        res.history = history
        return res
    except Exception as exc:                     # noqa: BLE001
        res.status = STATUS_FAILED
        res.notes = res.notes + [f"{type(exc).__name__}: {exc}"]
        raise


def _area_per_vertex(vertices, faces) -> float:
    tri = vertices[faces]
    total = float(0.5 * np.linalg.norm(
        np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]),
        axis=1).sum())
    return total / max(len(vertices), 1)

def occ_table_mask(vertices, faces, patient_frame, params=None):
    """咬合"表面"掩码：高 z 且法向朝就位轴。

    磨牙邻接壁在牙尖高度也有高 z，但法向径向 —— 只按 z 分位会把邻接接触区
    误锁死（实测真实例 2.0mm 残留穿透全在那里，修不动）。
    """
    m = params or RefineParams()
    import numpy as np
    vertices = np.asarray(vertices, dtype=np.float64)
    loop_idx, loop_set, _occ = _loop_and_zones(vertices, np.asarray(faces),
                                               patient_frame, m)
    crown = ~loop_set
    z_thr = np.percentile(vertices[crown, 2], m.occlusal_anchor_percentile)
    vn = vertex_normals(vertices, np.asarray(faces))
    az = np.abs(vn @ patient_frame.z)
    thr = getattr(m, "occ_table_normal_dot", 0.6)
    return crown & (vertices[:, 2] >= z_thr) & (az >= thr)
