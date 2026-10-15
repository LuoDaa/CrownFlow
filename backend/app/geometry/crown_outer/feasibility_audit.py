"""J.3 Constraint-Integrated CrownOuter —— **第一件：Feasibility Audit**。

用户 2026-10-09 决策 ②（架构立项）：基础 CrownOuter 的约束模型要从
``Anatomy ∪ MinThickness Envelope`` 升级成约束可行域：

    MinimumThicknessVolume（下界）
        ∩  AllowedNeighborSpace（上界）
        ∩  AllowedAntagonistSpace（上界）
        ∩  Cervical/Margin（边界条件）
        +  Anatomy Prior
        ↓
    Final feasible anatomy

并且在**生成之前**就判断可行性（不再"先生成 → 再反复 deformation → 最后才发现
空间不够"）：

* 为了满足 MinThickness，外表面**至少要**到哪里？
  ``min_required_outer(p) = intaglio(p) + T_min · n̂(p)``
  （沿代型/内表面外法向 —— 与厚度引擎的 ``t_ray`` 同一方向定义）
* 为了不撞邻牙/对颌，外表面**最多**允许到哪里？
  ``max_allowed_outer(p)``：``neighbor 带符号间隙 ≥ 目标间隙``

若 ``min_required_outer`` 已经侵入允许空间（``signed < target_clearance``）：

    LOCAL_CONSTRAINT_CONFLICT

聚合到病例级：若有非噪声量的采样点局部冲突 → ``CONSTRAINT_CONFLICT``。

本模块**只测量、不生成、不求解**；几秒级返回。FDI46（cf_b1b73bae221211b1）
是第一个应该在秒级被识别出冲突的病例。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .contact import NeighborPatch
from .thickness import intaglio_from_prep, winding_number


@dataclass
class FeasibilityParams:
    #: 材料最小厚度（必须由调用方给：病例设计参数 / 材料 / 加工规则）
    min_thickness_mm: float = 0.6
    #: 邻牙目标间隙（DistanceToNeighbor；exocad 里 0 = 目标零距离邻接）
    neighbor_clearance_mm: float = 0.05
    #: 对颌目标间隙（DistanceToAntagonist）
    antagonist_clearance_mm: float = 0.10
    #: 采样上限（面积加权均匀抽稀；不抽稀时用全部顶点）
    max_samples: int = 6000
    #: 局部冲突的噪声下限：单点侵入深度小于它不计入冲突统计
    noise_floor_mm: float = 0.05
    #: 判病例级冲突所需的最少冲突采样点数
    min_conflict_samples: int = 20
    #: 外法向定向探针步长（winding 探针用）
    orient_probe_mm: float = 0.15


@dataclass
class FeasibilityResult:
    status: str = "OK"                     # OK / CONSTRAINT_CONFLICT / INPUT_INVALID
    n_samples: int = 0
    n_orientation_ambiguous: int = 0
    neighbor: dict = field(default_factory=dict)
    antagonist: dict = field(default_factory=dict)
    notes: list = field(default_factory=list)

    def as_dict(self) -> dict:
        return {"status": self.status, "n_samples": self.n_samples,
                "n_orientation_ambiguous": self.n_orientation_ambiguous,
                "neighbor": self.neighbor, "antagonist": self.antagonist,
                "notes": self.notes}


def orient_outward(vertices, faces, *, probe_mm: float = 0.15,
                   max_probe_faces: int = 800):
    """外法向定向（用项目自己的 winding_number 做壳外/壳内探针）。

    → ``(faces_fixed, vnormals, n_ambiguous)``。winding 查询是 O(点数×面数)，
    做满 4 万面要分钟级，与"生成前秒级判定"的用途不符 —— 探针只在**固定种子
    抽样子集**面上做；探针不明确的面用"背离网格质心"的凸性启发式兜底并如实计数。
    """
    v = np.asarray(vertices, np.float64)
    f = np.asarray(faces, np.int64)
    tri = v[f]
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    ln = np.linalg.norm(n, axis=1)
    ok = ln > 1e-12
    n = n / np.maximum(ln, 1e-12)[:, None]
    cen = tri.mean(axis=1)
    cand = np.flatnonzero(ok)
    if len(cand) > max_probe_faces:
        rng = np.random.default_rng(20261009)
        sub = np.sort(rng.choice(cand, size=max_probe_faces, replace=False))
    else:
        sub = cand
    w_out = winding_number(cen[sub] + n[sub] * probe_mm, tri)
    w_in = winding_number(cen[sub] - n[sub] * probe_mm, tri)
    good = (w_out < 0.5) & (w_in > 0.5)           # 该面已朝外（壳外 winding≈0）
    bad = (w_out > 0.5) & (w_in < 0.5)            # 该面朝内
    known = np.zeros(len(f), bool)
    known[sub] = good | bad
    f_good = np.zeros(len(f), bool)
    f_good[sub] = good
    f_bad = np.zeros(len(f), bool)
    f_bad[sub] = bad
    flip = f_bad.copy()
    unknown = ~known
    centroid = v.mean(axis=0)
    outward = np.einsum("ij,ij->i", n, cen - centroid) > 0
    flip[unknown] = ~outward[unknown]
    f2 = f.copy()
    f2[flip] = f2[flip][:, [0, 2, 1]]
    f2 = f2[ok]
    vn = _accumulate_vertex_normals(v, f2)
    return f2, vn, int(unknown.sum())


def _accumulate_vertex_normals(v, f):
    vn = np.zeros_like(v)
    tri = v[f]
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])   # 面积加权
    for k in range(3):
        np.add.at(vn, f[:, k], n)
    ln = np.linalg.norm(vn, axis=1)
    return vn / np.maximum(ln, 1e-12)[:, None]


def sample_points(v, f, *, max_samples: int):
    """面积加权均匀抽稀（返回顶点 id 数组；固定种子 = 确定性）。"""
    tri = v[f]
    area = 0.5 * np.linalg.norm(
        np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    varea = np.zeros(len(v))
    for k in range(3):
        np.add.at(varea, f[:, k], area / 3.0)
    ids = np.flatnonzero(varea > 0)
    if len(ids) <= max_samples:
        return ids
    p = varea[ids] / varea[ids].sum()
    rng = np.random.default_rng(20261009)     # 固定种子：确定性
    pick = rng.choice(ids, size=max_samples, replace=False, p=p)
    return np.sort(pick)


def _score_against(pts_min, n_pts, patch: NeighborPatch, clearance: float):
    """把"最小必需外表面"点投到允许空间上判冲突。"""
    signed, valid, cp, fids, _parts = patch.clearance_detail(pts_min)
    deficit = np.maximum(0.0, float(clearance) - signed)
    hard = valid & (deficit > 0.0)
    out = {
        "n_queried": int(len(pts_min)),
        "n_invalid": int((~valid).sum()),
        "n_conflict": int(hard.sum()),
        "n_conflict_above_floor": 0,
        "deficit_p50_mm": 0.0, "deficit_p95_mm": 0.0, "deficit_max_mm": 0.0,
        "headroom_p95_mm": None,
        "conflict_centroid": None,
    }
    if len(pts_min):
        head = signed - float(clearance)
        out["headroom_p95_mm"] = float(np.quantile(head[valid], .95)) \
            if valid.any() else None
    if hard.any():
        d = deficit[hard]
        out["deficit_p50_mm"] = float(np.median(d))
        out["deficit_p95_mm"] = float(np.quantile(d, .95))
        out["deficit_max_mm"] = float(d.max())
        out["conflict_centroid"] = np.round(
            pts_min[hard].mean(axis=0), 3).tolist()
    return out, hard, deficit


def feasibility_audit(intaglio_vertices, intaglio_faces, patches: dict,
                      *, params: FeasibilityParams | None = None,
                      prepared_from=None) -> FeasibilityResult:
    """**J.3 可行性审计**（不生成、不求解）。

    ``patches``：``{"mesial": NeighborPatch|None, "distal": ..., "antagonist": ...}``
    （None = 该方向空间未提供，如实记录为未审计）。
    ``intaglio_vertices/faces``：内表面（代型 + cement gap）；法向沿"朝冠材料"
    方向（= 代型外法向），与厚度引擎 ``t_ray`` 同一方向定义。
    """
    p = params or FeasibilityParams()
    res = FeasibilityResult()
    v = np.asarray(intaglio_vertices, np.float64)
    f0 = np.asarray(intaglio_faces, np.int64)
    if len(v) < 100 or len(f0) < 100:
        res.status = "INPUT_INVALID"
        res.notes.append("内表面网格过小（n_v=%d n_f=%d）" % (len(v), len(f0)))
        return res
    f, vn, n_amb = orient_outward(v, f0, probe_mm=p.orient_probe_mm)
    keep_f = f
    res.n_orientation_ambiguous = n_amb
    if n_amb:
        res.notes.append("外法向定向：%d 个面的 winding 探针未取样/不分明，已用凸性启发式兜底（凹坑区可能不准，如实计数）" % n_amb)
    ids = sample_points(v, keep_f, max_samples=p.max_samples)
    res.n_samples = int(len(ids))
    n_pts = np.asarray(vn[ids], float)
    base = np.asarray(v[ids], float)
    min_outer = base + float(p.min_thickness_mm) * n_pts

    for key, clearance in (("mesial", p.neighbor_clearance_mm),
                           ("distal", p.neighbor_clearance_mm),
                           ("antagonist", p.antagonist_clearance_mm)):
        patch = (patches or {}).get(key)
        if patch is None:
            continue
        sc, hard, deficit = _score_against(min_outer, n_pts, patch, clearance)
        sc["clearance_target_mm"] = float(clearance)
        sc["conflict_kind"] = ("NEIGHBOR_MIN_THICKNESS_CONFLICT"
                               if key != "antagonist"
                               else "ANTAGONIST_MIN_THICKNESS_CONFLICT")
        above = hard & (deficit > float(p.noise_floor_mm))
        sc["n_conflict_above_floor"] = int(above.sum())
        sc["deficit_above_floor_max_mm"] = (float(deficit[above].max())
                                            if above.any() else 0.0)
        if key == "antagonist":
            res.antagonist = sc
        else:
            res.neighbor[key] = sc

    n_bad = max([(res.neighbor.get(k, {}) or {}).get("n_conflict_above_floor", 0)
                 for k in ("mesial", "distal")] + [0])
    n_bad_ant = (res.antagonist or {}).get("n_conflict_above_floor", 0)
    if n_bad >= int(p.min_conflict_samples) or n_bad_ant >= int(
            p.min_conflict_samples):
        res.status = "CONSTRAINT_CONFLICT"
        res.notes.append(
            "LOCAL_CONSTRAINT_CONFLICT：最小必需外表面（内表面 + %.2fmm）"
            "侵入允许空间 —— 邻牙最多 %d 点 / 对颌 %d 点（噪声下限 %.2fmm 之上）"
            % (p.min_thickness_mm, n_bad, n_bad_ant, p.noise_floor_mm))
    else:
        res.notes.append("未发现超噪局部冲突（邻牙 %d / 对颌 %d 点）"
                         % (n_bad, n_bad_ant))
    return res
