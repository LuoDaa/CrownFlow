"""H.2.2-B —— 同 FDI 参数化宽度重塑：**MD body width field**。

用户 2026-10-09 规格（后备路线 B）：换模板已证伪（`73f8cb6`：同 FDI 三个基型
全在 ~1.0–1.13mm，因为患者可用总宽 ≈9.2mm 比库存任何模板 10.9–11.4mm 都小
1.7–2.2mm）。所以要在**同一模板**上做参数化宽度重塑，并且：

* **不是**再回到 uniform scale（H.2 已证伪：被 margin 拟合吸收）
* **不是** aniso scale（H.2 v2 已证伪：只让半宽动 0.24mm）
* 而是 **MD body width field**：按**高度分层的宽度比** `s_md(t)`，
  逐顶点沿 ±MD 轴把截面缩到可用宽度，**margin/颈缘不动、颊舌尺寸不动**、
  咬合区渐弱（避免把 cusp 横向压扁）。

数据来源与 H.2.1 同一个 primitive（`NeighborClearanceQuery`）：
每层高度取该层 proximal 墙上的 deficit p95 → `W_avail(t) = W_template(t) − Δu(t)`。

产物是可直接塞进 `solve_laplacian` 的一组 (indices, targets, row_weights)，
与 proximal clearance 约束**同一套机制**（同一次求解里共存）。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .proximal_fit import ProximalFitParams


@dataclass
class MDWidthFieldParams:
    #: 分层层数（沿就位轴把冠体等分；颈缘以下/保护带不参与）
    n_bands: int = 6
    #: 保护带：归一化高度低于它的顶点权重为 0（保 margin 与颈缘）。
    t_protect: float = 0.10
    #: 顶部渐弱起点与最小权重（避免 cusp 横向被压扁）。
    t_taper_from: float = 0.70
    top_weight: float = 0.30
    #: 宽度比下限（防塌陷）；也是"塞不进去"的判据阈值之一。
    s_min: float = 0.80
    #: 行权重（与 proximal clearance 约束同一量纲，低于 margin 的 1e4）。
    row_weight: float = 300.0
    #: 每层不足的最小统计人数（少于此不采信该层，用邻近层插值）。
    min_band_samples: int = 8


@dataclass
class MDWidthFieldResult:
    s_of_t: list = field(default_factory=list)      # [(t_center, s_md), ...]
    band_table: list = field(default_factory=list)  # 每层明细
    n_targets: int = 0
    indices: object = None
    targets: object = None
    weights: object = None
    notes: list = field(default_factory=list)

    def as_dict(self) -> dict:
        return {"s_of_t": self.s_of_t, "band_table": self.band_table,
                "n_targets": self.n_targets, "notes": self.notes}


def _height_param(v, frame, margin_idx):
    """归一化高度 t ∈ [0,1]：沿就位轴从颈缘质心量到冠部顶端。"""
    axis = np.asarray(frame.z, float)
    origin = np.asarray(v, float)[np.asarray(margin_idx, np.int64)].mean(axis=0)
    z = (np.asarray(v, float) - origin) @ axis
    z0 = float(z[np.asarray(margin_idx, np.int64)].mean())
    z1 = float(z.max())
    return (z - z0) / max(z1 - z0, 1e-9)


def measure_md_width_field(v_ref, faces, frame, margin_idx, dom,
                           p: ProximalFitParams | None = None,
                           fp: MDWidthFieldParams | None = None
                           ) -> MDWidthFieldResult:
    """在**基线网格**上量出分层宽度比 s_md(t)（只测，不改几何）。"""
    p = p or ProximalFitParams()
    fp = fp or MDWidthFieldParams()
    v = np.asarray(v_ref, np.float64)
    x = np.asarray(frame.x, float)
    res = MDWidthFieldResult()
    t = _height_param(v, frame, margin_idx)
    u = (v - np.asarray(v, float)[np.asarray(margin_idx, np.int64)].mean(axis=0)) @ x
    # MD 中心线：按层取 u 的中位数（比全局质心稳）
    edges = np.linspace(0.0, 1.0, fp.n_bands + 1)
    d = float(p.target_clearance_mm)
    centers, ss = [], []
    for k in range(fp.n_bands):
        lo, hi = edges[k], edges[k + 1]
        band = (t >= lo) & (t < hi)
        tc = 0.5 * (lo + hi)
        if not band.any():
            res.band_table.append({"t": tc, "n": 0, "s_md": None})
            continue
        uc = float(np.median(u[band]))
        half = float(np.quantile(np.abs(u[band] - uc), 0.98))   # 该层 MD 半宽
        need, ns = [], 0
        for side in ("mesial", "distal"):
            ids = dom["per_side"][side]
            # 边界伪 signed 点（AMBIGUOUS_PATCH_EDGE，冻结在 H.1）不参与层内统计
            vd = (dom.get("valid_ids") or {}).get(side)
            if vd is not None and len(vd) == len(ids):
                okv = np.zeros(len(v), bool)
                okv[ids[np.asarray(vd, bool)]] = True
                sel = ids[band[ids] & okv[ids]]
            else:
                sel = ids[band[ids]]
            ns += len(sel)
            if len(sel) < fp.min_band_samples:
                continue
            sc, _q, _f = dom["qs"][side].signed_clearance_decision(v[sel])
            deficit = np.maximum(0.0, d - sc)
            need.append(float(np.quantile(deficit, 0.95)))
        rec = {"t": round(tc, 3), "n": int(band.sum()), "n_proximal": int(ns),
               "md_half_width_mm": round(half, 4),
               "deficit_p95_mm": round(max(need), 4) if need else None}
        if need and half > 1e-6:
            s = (half - max(need)) / half
            s = float(np.clip(s, fp.s_min, 1.0))
            rec["s_md_raw"] = round((half - max(need)) / half, 4)
            rec["s_md"] = round(s, 4)
            centers.append(tc)
            ss.append(s)
        else:
            rec["s_md"] = None
        res.band_table.append(rec)
    if not centers:
        res.notes.append("没有任何层拿到可用 deficit → 宽度场退化为 1.0（不动）")
        res.s_of_t = [(0.0, 1.0), (1.0, 1.0)]
        return res
    # 保护带与顶部渐弱：t ≤ t_protect 不参与；t_taper_from 之上按 top_weight 缩放
    centers = np.asarray(centers)
    ss = np.asarray(ss)
    res.s_of_t = [(0.0, 1.0)] + [(float(a), float(b))
                                 for a, b in zip(centers, ss)] + [(1.0, float(ss[-1]))]
    res.notes.append("保护带 t≤%.2f 权重 0；顶部 t≥%.2f 权重降到 %.2f"
                     % (fp.t_protect, fp.t_taper_from, fp.top_weight))
    return res


def s_at(t, s_of_t) -> np.ndarray:
    """分段线性插值 s_md(t)。"""
    t = np.asarray(t, np.float64)
    xs = np.asarray([a for a, _ in s_of_t], np.float64)
    ys = np.asarray([b for _, b in s_of_t], np.float64)
    order = np.argsort(xs)
    return np.interp(t, xs[order], ys[order])


def build_md_width_targets(v_ref, faces, frame, margin_idx, dom,
                           field_res: MDWidthFieldResult,
                           p: ProximalFitParams | None = None,
                           fp: MDWidthFieldParams | None = None):
    """→ (indices, targets, weights)：逐顶点沿 ±MD 缩到 s_md(t)。

    保护带（t ≤ t_protect）、margin、**cervical/occlusal 锁定区**权重为 0；
    顶部按 (1→top_weight) 线性渐弱。**只动 MD 分量**，颊舌尺寸不变。
    """
    p = p or ProximalFitParams()
    fp = fp or MDWidthFieldParams()
    v = np.asarray(v_ref, np.float64)
    x = np.asarray(frame.x, float)
    origin = v[np.asarray(margin_idx, np.int64)].mean(axis=0)
    t = _height_param(v, frame, margin_idx)
    u = (v - origin) @ x
    s = s_at(t, field_res.s_of_t)
    # 顶部渐弱：t ≥ t_taper_from 时权重 (1 → top_weight)
    tw = np.ones(len(v))
    hi = t >= fp.t_taper_from
    tw[hi] = 1.0 - (1.0 - fp.top_weight) * (t[hi] - fp.t_taper_from) \
        / max(1.0 - fp.t_taper_from, 1e-9)
    w = tw.copy()
    w[t <= fp.t_protect] = 0.0
    loop_set = np.zeros(len(v), dtype=bool)
    loop_set[np.asarray(margin_idx, np.int64)] = True
    w[loop_set] = 0.0
    # **硬锁区权重 0**（用户 R8/R11）：margin/cervical/occlusal 按项目口径是锁定区
    # （逐区验收 max ≤ 0.01mm）。宽度场行集必须先把这三区剔除，否则 300 权重的
    # 宽度行会把颈缘/咬合顶点直接拖走 —— 实测这就是 B 层"每个 α 都 hard✗"的根因。
    regs = dom.get("regions") or {}
    for rname in ("margin", "cervical", "occlusal"):
        rm = regs.get(rname)
        if rm is not None:
            w[np.flatnonzero(np.asarray(rm, bool))] = 0.0
    # 咬合面下缘淡出到 0：occ_table_mask 的顶点全部在 t ≥ t_stop 之上。
    # 不淡出的话 t=0.75~0.9 的行（全权重）会把表下缘拉出折痕（实测 α=1 时
    # 咬合区被耦合拖动 0.70mm、翻面 254 个）。
    occ_m = regs.get("occlusal")
    if occ_m is not None:
        occ_ids = np.flatnonzero(np.asarray(occ_m, bool))
        if len(occ_ids) >= 8:
            t_stop = float(np.quantile(t[occ_ids], 0.02))
            span = max(t_stop - fp.t_taper_from, 1e-9)
            bnd = np.clip((t_stop - t) / span, 0.0, 1.0)
            w = w * bnd
    delta_u = (s - 1.0) * u * w
    idx = np.flatnonzero(np.abs(delta_u) > 1e-9)
    tgt = np.asarray(v[idx], float) + (delta_u[idx])[:, None] * x[None, :]
    wt = np.full(len(idx), fp.row_weight / 300.0)      # 与 proximal 同量纲基准
    return idx, tgt, wt
