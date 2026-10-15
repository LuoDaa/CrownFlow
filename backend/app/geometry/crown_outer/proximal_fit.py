"""K0 —— Neighbor-Constrained Anatomy Prior（近远中约束的解剖先验）。

用户 2026-10-09 规格（路线①）：Contact 的 `CONTACT_CONSTRAINT_CONFLICT`（几何全绿、
editable 残留 p95 1.02mm、任何 α 都折叠）**不是 Contact 的错**，是上游 H.1 的
近远中整体宽出约 1mm。所以在

    H.1 Anatomy Prior  →  **K0**  →  J.2 Frozen  →  Contact Hardened

之间加一层。**J.2 算法不动、Contact Gate 不放宽**，只换 J.2 的 anatomy 输入。

K0 的三条纪律（都来自用户规格）：

1. **先测后修**：先出账（穿透面积/p50/p95/max + 需要的收窄量 p50/p95/max），
   让数据说话，再动几何。
2. **不是"投到邻牙表面"**：目标是 **Neighbor Clearance Envelope**
   = 邻牙面 + target clearance；anatomy 不得穿进去，但只在 proximal fitting zone
   生效，其余（occlusal / buccal / lingual / margin）尽量保持 H.1。
3. **低频整体收窄，不逐顶点硬拉**：近中/远中各自看成一个 patch，
   沿 **±AxisMesial** 做整体 inward displacement，transition band 平滑衰减到 0
   （区域内部零应变 —— 这正是 Contact 逐点投影折叠的教训）。
4. **硬锁四类区域**：margin/cervical、occlusal core、buccal core、lingual core；
   自由区只有 proximal wall + 少量 transition ring。
5. 每一步都过 `deform_gate` 的硬几何门（新增自交 0 / 非流形不新增 / 翻面 0 /
   锁定区位移 0），与 Contact/Occlusion 同一套口径。
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np
from scipy.spatial import cKDTree

from .contact import (
    ContactParameters,
    NeighborPatch,
    _boundary_loop,
    _region_mask,
)
from .contact_refine import occ_table_mask
from .deform_gate import deformation_gate
from .patient_frame import PatientFrame

STATUS_OK = "OK"
STATUS_REVIEW = "REVIEW_REQUIRED"
STATUS_CONFLICT = "UPSTREAM_SPACE_CONFLICT"
STATUS_INFEASIBLE = "PROXIMAL_FIT_INFEASIBLE"


class ProximalFitError(ValueError):
    pass


@dataclass
class ProximalFitParams:
    #: 目标间隙（= Neighbor Clearance Envelope 的 clearance）。构造参数，
    #: 与 Contact 的 target_gap 同义；**不是临床建议**。
    target_clearance_mm: float = 0.05
    #: 近远中带宽比例（沿用 Contact 已验证的语义）。
    band_fraction: float = 0.6
    #: 颈缘锁带宽。
    cervical_lock_mm: float = 0.3
    #: 咬合面判据（法向 + z 分位），与 Contact 一致。
    occlusal_anchor_percentile: float = 75.0
    occ_table_normal_dot: float = 0.6
    #: transition band 宽度（顶点 BFS 环数）—— 应变摊开的地方。
    transition_rings: int = 6
    #: α line search。
    line_search_alphas: tuple = (1.0, 0.5, 0.25, 0.125, 0.0625)
    #: 几何门尺度（mm）。
    gate_scale_mm: float = 0.10
    #: 单侧总收窄上限：超过它说明"塞不进去"，要报 UPSTREAM_SPACE_CONFLICT。
    max_shrink_mm: float = 4.0
    #: 计算 required_shrink 时，|n̂·â| 低于它就认为"沿轴移动改不了这个点的间隙"。
    axial_normal_dot_min: float = 0.50
    #: 位移场的 Laplacian 平滑：轮数与步长（低频化的关键参数）。
    field_smooth_iters: int = 12
    field_smooth_lambda: float = 0.5
    #: 保护/参与带（活动集外扩环数 + 权重衰减）；一次冻结，**不是硬锁**。
    protection_rings: int = 4
    protection_decay: float = 0.5
    #: 加性保护区的 p95 位移允许值（Phase K 既定口径）。
    buccal_lingual_p95_allow_mm: float = 0.30
    #: 硬锁区（margin/cervical/occlusal）的逐区 max 位移上限 = 项目既定
    #: margin 约束残差上限（solve_laplacian.max_constraint_residual_mm）。
    hard_region_tol_mm: float = 0.01


@dataclass
class ProximalFitResult:
    status: str = STATUS_REVIEW
    mesh: object | None = None
    audit: dict = field(default_factory=dict)
    fit: dict = field(default_factory=dict)
    notes: list = field(default_factory=list)

    def as_dict(self) -> dict:
        return {"status": self.status, "audit": self.audit, "fit": self.fit,
                "notes": self.notes}


def _proximal_masks(vertices, faces, frame: PatientFrame, loop_idx,
                    p: ProximalFitParams):
    """四类锁定 + 两侧可编辑 proximal wall（沿用已验证的法向/带宽语义）。"""
    cparams = ContactParameters(target_gap_mm=p.target_clearance_mm,
                                contact_band_fraction=p.band_fraction,
                                cervical_lock_mm=p.cervical_lock_mm,
                                occlusal_anchor_percentile=(
                                    p.occlusal_anchor_percentile),
                                occ_table_normal_dot=p.occ_table_normal_dot)
    loop_set = np.zeros(len(vertices), dtype=bool)
    loop_set[loop_idx] = True
    occ = occ_table_mask(vertices, faces, frame, cparams)
    d_loop = cKDTree(vertices[loop_idx]).query(vertices, k=1)[0]
    cervical = (d_loop < p.cervical_lock_mm) & ~loop_set
    masks = {}
    for side in ("mesial", "distal"):
        band = _region_mask(vertices, frame, loop_idx, cparams, side)
        masks[side] = band & ~loop_set & ~occ & ~cervical
    locked = loop_set | occ | cervical
    return masks, locked, {"loop_set": loop_set, "occlusal": occ,
                           "cervical": cervical}


def _side_audit(vertices, patch: NeighborPatch, mask, axis_sign,
                p: ProximalFitParams) -> dict:
    """单侧账：穿透统计 + 需要的轴向收窄量。

    轴向模型：沿 â（= mesial 的 −x / distal 的 +x）移动 δ 会让
    signed 距离按 (n̂·â) 的速率增加，因此
        δ_required = (clearance − signed) / (n̂·â)   （n̂·â > 阈值时）
    —— 只在沿轴真能改善间隙的地方算，其余如实标 `axial_ineffective`。
    """
    ids = np.flatnonzero(mask)
    out = {"side": patch.name if hasattr(patch, "name") else "",
           "n_editable_prior": int(len(ids))}
    if len(ids) == 0:
        out["status"] = "NO_REGION"
        return out
    # **统一口径**（用户 2026-10-09 决策 ③）：幅值 euclid + 边界/crease 有效性。
    # 审计是"对当前几何的一次新鲜测量"，参考系即自身 —— 冻结纪律只约束
    # **决策路径**（H.2.x 的 before/after 比较），不约束单次快照审计。
    signed, valid, cp, fids, _parts = patch.clearance_detail(vertices[ids])
    dist = np.abs(np.asarray(signed, np.float64))
    valid = np.asarray(valid, bool)
    inside = (signed < 0) & (dist <= 2.0) & valid
    pen = np.maximum(0.0, -signed[inside] - p.target_clearance_mm)
    # 顶点当量面积（面积加权）
    out["penetration_mm"] = {
        "n": int(inside.sum()),
        "p50": float(np.median(pen)) if len(pen) else 0.0,
        "p95": float(np.quantile(pen, .95)) if len(pen) else 0.0,
        "max": float(pen.max()) if len(pen) else 0.0,
        "sum": float(pen.sum()) if len(pen) else 0.0,
    }
    n̂ = patch.normals[fids] * patch.normal_sign
    â = np.asarray(axis_sign, dtype=float)[None, :]
    proj = np.einsum("ij,ij->i", n̂, np.broadcast_to(â, n̂.shape))
    need = (p.target_clearance_mm - signed)
    # ⚠️ 只统计**真穿透点**需要的收窄量：把"本来就没穿、need≤0"的点算进来，
    # p50 会被拉成 0（实测 mesial 中位数 0.0 就是这么来的），看不出"整面宽 1mm"。
    eff = inside & (np.abs(proj) >= p.axial_normal_dot_min)
    delta = np.zeros(len(ids))
    delta[eff] = np.clip(need[eff] / proj[eff], 0.0, None)
    out["required_shrink_mm"] = {
        "n_penetrating": int(inside.sum()),
        "n_axial_effective": int(eff.sum()),
        "n_axial_ineffective": int((inside & ~eff).sum()),
        "p50": float(np.quantile(delta[eff], .5)) if eff.any() else 0.0,
        "p75": float(np.quantile(delta[eff], .75)) if eff.any() else 0.0,
        "p95": float(np.quantile(delta[eff], .95)) if eff.any() else 0.0,
        "max": float(delta[eff].max()) if eff.any() else 0.0,
    }
    out["status"] = "OK"
    return out


def measure_proximal_prior(anatomy_mesh, frame: PatientFrame,
                           mesial: NeighborPatch, distal: NeighborPatch,
                           p: ProximalFitParams | None = None) -> dict:
    """**只测不改**：K0 的第一轮账（用户规格 §3）。"""
    p = p or ProximalFitParams()
    v = np.asarray(anatomy_mesh.vertices, np.float64)
    f = np.asarray(anatomy_mesh.faces, np.int64)
    loop = _boundary_loop(v, f)
    if loop is None:
        raise ProximalFitError("冠网格没有可识别的颈环边界")
    loop_idx = np.asarray(loop, np.int64)
    # 与引擎同口径（质心法）；见 tools/run_h2_1_case.py 的说明
    from .contact import NeighborPatch as _NP
    _centroid = np.asarray(v, float).mean(axis=0)
    for patch in (mesial, distal):
        _fc = patch.vertices[patch.faces].mean(axis=1)
        _w = np.einsum("ij,ij->i", _centroid - _fc, patch.normals)
        patch.normal_sign = 1.0 if _w.sum() > 0 else -1.0
    masks, locked, zones = _proximal_masks(v, f, frame, loop_idx, p)
    ax = np.asarray(frame.x, float)
    audit = {
        "n_vertices": int(len(v)),
        "n_locked": int(locked.sum()),
        "target_clearance_mm": p.target_clearance_mm,
        "axis_mesial": [float(x) for x in ax],
    }
    for side, sign in (("mesial", -ax), ("distal", ax)):
        audit[side] = _side_audit(v, mesial if side == "mesial" else distal,
                                  masks[side], sign, p)
        audit[side]["side"] = side
        audit[side]["n_region"] = int(masks[side].sum())
    return audit


def fit_proximal_prior(anatomy_mesh, frame: PatientFrame,
                       mesial: NeighborPatch, distal: NeighborPatch,
                       p: ProximalFitParams | None = None,
                       *, max_rounds: int = 4) -> ProximalFitResult:
    """低频整体收窄 + 硬几何门 + α line search。

    每轮：测 → 取单侧**一个标量** δ_side（required_shrink 的分位，
    即"整面收多少"）→ w(v) 在 proximal core 为 1、transition band 平滑衰减到 0
    → v' = v ∓ w·(α·δ_side)·â → 硬几何门 → 接受/拒绝。
    """
    p = p or ProximalFitParams()
    res = ProximalFitResult()
    v0 = np.asarray(anatomy_mesh.vertices, np.float64)
    f = np.asarray(anatomy_mesh.faces, np.int64)
    loop = _boundary_loop(v0, f)
    if loop is None:
        raise ProximalFitError("冠网格没有可识别的颈环边界")
    loop_idx = np.asarray(loop, np.int64)
    # 与引擎同口径（pipeline._orient_patches 的质心法）；orient_against_crown
    # 的符号与引擎相反，换用等于换尺（实测 H.1 基线 1.04/1.11 → 1.78/1.83）。
    _centroid = np.asarray(v0, float).mean(axis=0)
    for patch in (mesial, distal):
        _fc = patch.vertices[patch.faces].mean(axis=1)
        _w = np.einsum("ij,ij->i", _centroid - _fc, patch.normals)
        patch.normal_sign = 1.0 if _w.sum() > 0 else -1.0
    patches = {"mesial": mesial, "distal": distal}
    ax = np.asarray(frame.x, float)

    # 邻接图（transition band 的 BFS）
    adj: dict[int, set] = {}
    for a, b, c in f:
        for uu, vv in ((a, b), (b, c), (c, a)):
            adj.setdefault(int(uu), set()).add(int(vv))
            adj.setdefault(int(vv), set()).add(int(uu))

    masks, locked, zones = _proximal_masks(v0, f, frame, loop_idx, p)
    res.audit = measure_proximal_prior(
        type("M", (), {"vertices": v0, "faces": f})(), frame, mesial, distal, p)
    locked_ids = set(int(i) for i in np.flatnonzero(locked))

    cur = v0.copy()
    rounds = []
    for rnd in range(max_rounds):
        masks, _, _ = _proximal_masks(cur, f, frame, loop_idx, p)
        need = {}
        for side in ("mesial", "distal"):
            a = _side_audit(cur, patches[side], masks[side],
                            -ax if side == "mesial" else ax, p)
            need[side] = a
        # 停：两侧都不再需要收窄（p95 ≤ tolerance）
        if all(need[s].get("required_shrink_mm", {}).get("p95", 0.0)
               <= 0.05 for s in need):
            rounds.append({"round": rnd, "stop": "p95 收窄需求已 ≤ 0.05mm"})
            break
        moved_any = False
        for side in ("mesial", "distal"):
            req = need[side].get("required_shrink_mm") or {}
            delta_side = float(req.get("p75", 0.0))
            if delta_side <= 0.05:
                continue
            if delta_side > p.max_shrink_mm:
                res.status = STATUS_CONFLICT
                res.notes.append(
                    "%s 需要收窄 %.2fmm（> %.1f）—— 空间不够，不是收窄能解决的"
                    % (side, delta_side, p.max_shrink_mm))
                res.mesh = type(anatomy_mesh)(vertices=cur, faces=f,
                                              process=False)
                res.fit = {"rounds": rounds}
                return res
            core = masks[side]
            # transition band：从 core 外扩 ring 圈，权重随 BFS 层数线性衰减
            w = np.zeros(len(cur))
            w[core] = 1.0
            frontier = set(int(i) for i in np.flatnonzero(core))
            seen = set(frontier) | locked_ids
            for ring in range(1, p.transition_rings + 1):
                nxt = set()
                for u in frontier:
                    for vv in adj.get(u, ()):
                        if vv not in seen:
                            nxt.add(vv)
                seen |= nxt
                if not nxt:
                    break
                fall = 1.0 - ring / (p.transition_rings + 1.0)
                for vv in nxt:
                    w[vv] = max(w[vv], fall)
                frontier = nxt
            w[list(locked_ids)] = 0.0          # 锁定区永不参与
            direction = (-ax if side == "mesial" else ax)
            # ---- 位移场：v1 是"整面沿 ±AxisMesial 平移"，实测不够（mesial
            # 1.129→1.074 / distal 1.114→1.098；账里 distal 只有 76/371 个
            # 穿透点沿轴有效）。改成：逐点需求 δ_v 沿**邻牙法向**算出来，
            # 再在 free 区做若干轮 Laplacian 平滑 —— 幅值低频、方向贴合法向。
            ids_side = np.flatnonzero(masks[side])
            cp_s, fids_s, signed_s, _d_s = patches[side].closest(cur[ids_side])
            n̂_s = patches[side].normals[fids_s] * patches[side].normal_sign
            # 目标位移与 Contact 引擎同构：tgt = closest + n̂·clearance，
            # Δ = tgt − v（**穿出去**的方向）。符号写反 = 越推越深，实测
            # distal 各 α 都"penetration 未降"就是这个。方向逐点，幅值低频。
            tgt_s = cp_s + n̂_s * p.target_clearance_mm
            dv = tgt_s - cur[ids_side]
            mag = np.linalg.norm(dv, axis=1)
            field = np.zeros(len(cur))
            field[ids_side] = mag
            nrm_full = np.zeros_like(cur)
            ok_m = mag > 1e-12
            if ok_m.any():
                dirn = np.zeros_like(dv)
                dirn[ok_m] = dv[ok_m] / mag[ok_m][:, None]
                nrm_full[np.asarray(ids_side)[ok_m]] = dirn[ok_m]
            free_ids = np.flatnonzero(w > 0)
            for _ in range(p.field_smooth_iters):
                acc = np.zeros(len(cur))
                cnt = np.zeros(len(cur))
                for u in free_ids:
                    nb = [x for x in adj.get(int(u), ()) if w[x] > 0]
                    if not nb:
                        continue
                    acc[u] = sum(field[x] for x in nb) / len(nb)
                    cnt[u] = 1.0
                upd = cnt > 0
                field[upd] = ((1 - p.field_smooth_lambda) * field[upd]
                              + p.field_smooth_lambda * acc[upd])
            field *= w
            accepted = False
            for alpha in p.line_search_alphas:
                disp = alpha * field
                xt = cur + disp[:, None] * nrm_full   # +Δ：往 clearance envelope 里退
                moved = np.flatnonzero(np.abs(xt - cur).max(axis=1) > 1e-12)
                focus = np.flatnonzero(np.isin(f, moved).any(axis=1))
                # ⚠️ **对原始网格 v0 累计判定**：逐步比"当前状态"会累积出折叠
                # （实测每步都过、最终 21 片 severe flip）。
                gate = deformation_gate(v0, xt, f, scale_mm=p.gate_scale_mm,
                                        focus_faces=focus,
                                        locked_ids=locked_ids)
                # 目标：neighbor penetration ↓（同一套 envelope 复测）
                after = _side_audit(xt, patches[side], masks[side],
                                    direction, p)
                pen_before = need[side]["penetration_mm"]["p95"]
                pen_after = after["penetration_mm"]["p95"]
                improved = pen_after < pen_before - 1e-6
                rec = {"round": rnd, "side": side, "alpha": float(alpha),
                       "delta_side_mm": round(float(field.max()), 4),
                       "penetration_p95_before": round(pen_before, 4),
                       "penetration_p95_after": round(pen_after, 4),
                       "gate": gate.as_dict(), "accepted": False}
                if improved and gate.ok:
                    cur = xt
                    rec["accepted"] = True
                    accepted = True
                    moved_any = True
                    rounds.append(rec)
                    break
                rec["reject"] = ("penetration 未降" if not improved
                                 else "geometry gate: " + gate.note)
                rounds.append(rec)
            if not accepted:
                res.notes.append(
                    "%s：第 %d 轮所有 α 都无法在几何安全前提下降低穿透 —— "
                    "该侧到此为止（如实记录，不折叠）" % (side, rnd))
        if not moved_any:
            break
    # 收尾：整体几何复核（自交/非流形/翻面 + 四类锁定区位移）
    moved = np.flatnonzero(np.abs(cur - v0).max(axis=1) > 1e-12)
    focus = np.flatnonzero(np.isin(f, moved).any(axis=1))
    final_gate = deformation_gate(v0, cur, f, scale_mm=p.gate_scale_mm,
                                  focus_faces=focus, locked_ids=locked_ids)
    res.fit = {
        "rounds": rounds,
        "n_moved_vertices": int(len(moved)),
        "max_displacement_mm": float(np.max(np.linalg.norm(
            cur[moved] - v0[moved], axis=1))) if len(moved) else 0.0,
        "final_gate": final_gate.as_dict(),
        "buccal_lingual_deviation_mm": _off_axis_deviation(v0, cur, f, frame,
                                                           zones, ax),
    }
    res.mesh = type(anatomy_mesh)(vertices=cur, faces=f, process=False)
    if final_gate.ok and res.status != STATUS_CONFLICT:
        res.status = STATUS_OK
    return res


def _off_axis_deviation(v0, v1, faces, frame: PatientFrame, zones, ax) -> dict:
    """非近远中区域（颊舌 + margin + 咬合 core）的偏差 —— 应接近 0。"""
    d = np.linalg.norm(v1 - v0, axis=1)
    lat = (v0 - frame.origin) @ np.asarray(frame.y, float)   # buccal/lingual 轴
    sel = np.abs(lat) > 0.5
    out = {"buccal_lingual_p95_mm": float(np.quantile(d[sel], .95))
           if sel.any() else None,
           "buccal_lingual_max_mm": float(d[sel].max()) if sel.any() else None}
    for k, mask in zones.items():
        out["%s_max_mm" % k] = float(d[mask].max()) if mask.any() else 0.0
    return out
