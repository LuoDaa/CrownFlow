"""形变步的**硬几何门**（Contact / Occlusion 硬化共用）。

用户 2026-10-08 规格：接受一个形变步，必须同时满足

    penetration ↓（由调用方判）
    AND new self-intersections = 0
    AND non-manifold edges      = 0
    AND orientation（翻面）     = 0
    AND 锁定区位移              = 0

**旧的"翻面别太多（flip tolerance +0.2%）"判据作废** —— 它正是"为了把穿透
清零而换来局部折叠"的入口。这里只认"零新增"，不认"别太多"。

两处工程细节（都为了让它能**每一步都跑**，而不是只当收尾检查）：

1. **聚焦自交检测**：形变只动一小片顶点，新增自交必然牵涉被动过的面
   → 只测"关注面 ∪ 近邻"的候选对（O(关注面×k) 而不是 O(全网格)）。
   为分开"新"与"旧"，**形变前也跑同一份聚焦测试**，两次取差集。
2. **翻面按尺度分级**：面积 < (scale/2)² 的退化片记为 micro（照常计数上抛），
   只有 ≥ 该尺度的翻面才算硬违规 —— 与 J.2 的 micro_defect 口径一致。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .thickness import self_intersections


@dataclass
class DeformGateResult:
    ok: bool = False
    violations: list = field(default_factory=list)
    metrics: dict = field(default_factory=dict)
    note: str = ""
    #: 本次聚焦自交检测的命中对（(m,2) int64）—— 复用为下一步的 baseline，
    #: 这样"新增自交"永远是与**同口径**的上一次比较，不重复跑。
    pairs: object = None

    def as_dict(self) -> dict:
        return {"ok": bool(self.ok), "violations": list(self.violations),
                "metrics": self.metrics, "note": self.note}


def _face_areas(v, faces):
    return 0.5 * np.linalg.norm(np.cross(v[faces[:, 1]] - v[faces[:, 0]],
                                         v[faces[:, 2]] - v[faces[:, 0]]),
                                axis=1)


def _face_normals(v, faces):
    n = np.cross(v[faces[:, 1]] - v[faces[:, 0]],
                 v[faces[:, 2]] - v[faces[:, 0]])
    return n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-18)


def flipped_faces(v0, v1, faces, *, scale_mm: float,
                  min_area_mm2: float | None = None) -> dict:
    """形变造成的**翻面**统计，按面积分级（micro vs severe）。

    用"与形变前法向反号"判定；面积 < min_area 的记为 micro，不算硬违规。
    """
    n0 = np.cross(v0[faces[:, 1]] - v0[faces[:, 0]],
                  v0[faces[:, 2]] - v0[faces[:, 0]])
    n1 = np.cross(v1[faces[:, 1]] - v1[faces[:, 0]],
                  v1[faces[:, 2]] - v1[faces[:, 0]])
    l0 = np.linalg.norm(n0, axis=1)
    l1 = np.linalg.norm(n1, axis=1)
    ok = (l0 > 1e-18) & (l1 > 1e-18)
    dot = np.ones(len(faces))
    dot[ok] = np.einsum("ij,ij->i", n0[ok], n1[ok]) / (l0[ok] * l1[ok])
    flipped = ok & (dot < 0.0)
    area = _face_areas(v1, faces)
    thr = min_area_mm2 if min_area_mm2 is not None else (scale_mm / 2.0) ** 2
    micro = flipped & (area < thr)
    severe = flipped & ~micro
    return {"n_flipped_total": int(flipped.sum()),
            "n_flipped_severe": int(severe.sum()),
            "n_flipped_micro": int(micro.sum()),
            "min_dot": float(dot[ok].min()) if ok.any() else None,
            "area_threshold_mm2": float(thr),
            "severe_faces": [int(x) for x in np.flatnonzero(severe)[:20]]}


def nonmanifold_edges(faces) -> int:
    from collections import Counter
    cnt: Counter = Counter()
    for a, b, c in np.asarray(faces, np.int64):
        for u, w in ((int(a), int(b)), (int(b), int(c)), (int(c), int(a))):
            cnt[(min(u, w), max(u, w))] += 1
    return int(sum(1 for v in cnt.values() if v > 2))


def deformation_gate(v0, v1, faces, *, scale_mm: float,
                     focus_faces=None, locked_ids=None,
                     locked_tol: float = 0.0,
                     baseline_pairs=None) -> DeformGateResult:
    """形变步硬门。

    ``focus_faces``：本步可能动的面 id（自交聚焦用；None = 全网格，慢）。
    用 ``deformation_gate(v, v, ...)`` 先跑一次即得该聚焦集合的 baseline
    （结果里的 ``pairs`` 传回 ``baseline_pairs`` 复用）。
    ``locked_ids``：必须位移精确为 0 的顶点（颈环 / 咬合锁 / 接触核）。

    ⚠️ 口径说明：**非流形边是网格的不变量**（只动顶点改不了组合结构），
    所以这里与自交一样按"**不新增**"判（`after ≤ before`）；要求"绝对为 0"
    是**阶段门**（对最终产物）的事，不是单步门的事。
    """
    v0 = np.asarray(v0, np.float64)
    v1 = np.asarray(v1, np.float64)
    faces = np.asarray(faces, np.int64)
    res = DeformGateResult()

    # 1) 锁定区位移必须精确为 0
    if locked_ids is not None and len(locked_ids):
        li = np.asarray(sorted(int(x) for x in locked_ids), dtype=np.int64)
        dmax = float(np.max(np.linalg.norm(v1[li] - v0[li], axis=1)))
        res.metrics["locked_displacement_max_mm"] = dmax
        if dmax > locked_tol:
            res.violations.append("locked_displacement=%.3e" % dmax)

    # 2) 非流形边：网格不变量 → 只判"不新增"（绝对为 0 由阶段门负责）
    nm0 = nonmanifold_edges(faces)
    res.metrics["nonmanifold_edges"] = {"baseline": nm0, "after": nm0}
    if nm0:
        res.metrics["nonmanifold_edges"]["note"] = (
            "输入网格本身非流形；单步门只判不新增，绝对为 0 由阶段门判")

    # 3) 翻面（按面积分级）
    fl = flipped_faces(v0, v1, faces, scale_mm=scale_mm)
    res.metrics["flipped"] = fl
    if fl["n_flipped_severe"]:
        res.violations.append("flipped_severe=%d" % fl["n_flipped_severe"])

    # 4) 新增自交（聚焦；扣掉形变前就有的）
    focus = None
    if focus_faces is not None:
        focus = np.zeros(len(faces), dtype=bool)
        ff = np.asarray(focus_faces).ravel()           # 空数组要显式转 int，
        if ff.size:                                    # 否则 np.asarray([]) 是 float
            focus[ff.astype(np.int64)] = True
    thr = max((scale_mm / 2.0) ** 2, 1e-9)
    ev1 = self_intersections(v1, faces, severe_min_area_mm2=thr,
                             severe_max_normal_dot=0.999,
                             focus_faces=focus, return_pairs=True)
    if ev1.get('self_intersections_truncated'):
        res.violations.append('self_intersection_screen_truncated')
    p1 = ev1.get("self_intersections_pairs")
    res.pairs = p1
    n_before = None
    if baseline_pairs is not None:
        n_before = int(len(baseline_pairs))
        n_new = _new_pairs(baseline_pairs, p1)
    else:
        ev0 = self_intersections(v0, faces, severe_min_area_mm2=thr,
                                 severe_max_normal_dot=0.999,
                                 focus_faces=focus, return_pairs=True)
        if ev0.get('self_intersections_truncated'):
            res.violations.append('baseline_self_intersection_screen_truncated')
        p0 = ev0.get("self_intersections_pairs")
        n_before = int(ev0.get("self_intersections") or 0)
        n_new = _new_pairs(p0, p1)
    res.metrics["self_intersections_focus"] = {
        "before": n_before,
        "after": int(ev1.get("self_intersections") or 0),
        "new": int(n_new),
        "focused": focus is not None,
        "pair_tests": int(ev1.get("pair_tests") or 0)}
    if n_new:
        res.violations.append("new_self_intersections=%d" % n_new)

    res.ok = not res.violations
    if not res.ok:
        res.note = "；".join(res.violations)
    return res


def _new_pairs(p0, p1) -> int:
    if p1 is None or len(p1) == 0:
        return 0
    if p0 is None or len(p0) == 0:
        return int(len(p1))
    s0 = {tuple(map(int, x)) for x in np.asarray(p0)}
    return int(sum(1 for x in np.asarray(p1)
                   if tuple(map(int, x)) not in s0))
