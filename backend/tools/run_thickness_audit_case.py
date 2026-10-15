#!/usr/bin/env python3
"""Thickness v1 真实例验收：Contact v2/I.2 → Occlusion v1 → **壁厚审计**（只测）。

    H.1 冠 → Contact/I.2 → crown_occluded.stl   （已冻结的上游产物）
                                 ↓
    代型 prep.stl（align_prep_to_design）→ intaglio = prep + CementGapThickness
                                 ↓
    审计：分区账 / 双测法 / 输入 QC / 冲突归因（接触+对颌 clearance）

阈值 provenance 全部来自病例 ``.modelInfo`` 设计参数（die_ref.json），
不写任何临床默认值。输出报告 PHI-free（case_uuid + 统计量 + 逐顶点量 CSV）。

用法::

    python3 tools/run_thickness_audit_case.py \
        --case-dir <...>/crown_outer_cases/cf_b1b73bae221211b1 \
        --outer <...>/occlusion_v1_case/crown_occluded.stl \
        --reference <...>/contact_v1_case/h1_crown.stl \
        --out <...>/occlusion_v1_case
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.geometry.crown_outer.contact import NeighborPatch          # noqa: E402
from app.geometry.crown_outer.patient_frame import PatientFrame     # noqa: E402
from app.geometry.crown_outer.thickness import (                    # noqa: E402
    REGION_ROWS,
    ThicknessParameters,
    audit_thickness,
    intaglio_from_prep,
)


def patient_frame_from_meta(meta) -> PatientFrame:
    """与 run_contact_v1_case 同一约定：X = AxisMesial ⊥ Z（不用 PCA）。"""
    margin = np.asarray(meta["margin_points"], dtype=float)
    z = np.asarray(meta["insertion_axis"], dtype=float)
    z = z / np.linalg.norm(z)
    m = np.asarray(meta["axis_mesial"], dtype=float)
    m = m - (m @ z) * z
    x = m / np.linalg.norm(m)
    y = np.cross(z, x)
    y = y / np.linalg.norm(y)
    return PatientFrame(origin=margin.mean(axis=0), z=z, x=x, y=y)


def _f(params, key, default=None):
    v = params.get(key)
    if v is None or v in ("NaN", ""):
        return default
    try:
        return float(v)
    except ValueError:
        return default


def _parity_check(outer, points) -> dict:
    """封底成实体 + 射线奇偶判内外（**不用任何法向**的独立复核）。"""
    import trimesh
    from collections import Counter
    ov = np.asarray(outer.vertices, float)
    of = np.asarray(outer.faces, np.int64)
    cnt: Counter = Counter()
    for a, b, c in of:
        for u, w in ((a, b), (b, c), (c, a)):
            cnt[(min(int(u), int(w)), max(int(u), int(w)))] += 1
    bedges = [e for e, m in cnt.items() if m == 1]
    adj: dict = {}
    for u, w in bedges:
        adj.setdefault(u, []).append(w)
        adj.setdefault(w, []).append(u)
    start = bedges[0][0]
    seq, prev, cur = [start], None, start
    while True:
        nxt = [w for w in adj[cur] if w != prev]
        if not nxt:
            break
        prev, cur = cur, nxt[0]
        if cur == start:
            break
        seq.append(cur)
    cen = ov[seq].mean(axis=0)
    cap = np.asarray([(seq[i], seq[(i + 1) % len(seq)], len(ov))
                      for i in range(len(seq))], dtype=np.int64)
    solid = trimesh.Trimesh(np.vstack([ov, cen]), np.vstack([of, cap]),
                            process=False)
    inside = solid.contains(np.asarray(points, float))
    return {"watertight": bool(solid.is_watertight),
            "n_points": int(len(points)),
            "outside": int((~inside).sum()),
            "note": ("封底平面会切过非平面颈缘，开口带附近的点两种方法都可疑；"
                     "两个数并列给出，不取其一")}


def _gt_reference(gt_path, iv, meta, tmap) -> dict:
    """audit-only：同一套 intaglio 点对 GT 冠**外表面**的壁厚（只用外表面子网格）。"""
    import trimesh
    from app.geometry.crown_outer.gt_compare import extract_outer_surface
    from app.geometry.crown_outer.thickness import SurfaceQuery, winding_number
    gt = trimesh.load(gt_path, force="mesh")
    axis = np.asarray(meta["insertion_axis"], float)
    axis = axis / np.linalg.norm(axis)
    mc = np.asarray(meta["margin_points"], float).mean(0)
    outer_f = extract_outer_surface(gt, axis, mc)
    sub = gt.submesh([outer_f], append=True, repair=False)   # 自洽子网格
    sq = SurfaceQuery(np.asarray(sub.vertices, float),
                      np.asarray(sub.faces, np.int64),
                      boundary_safety_mm=0.6, max_query_edge_mm=None)
    _cp, _f, _s, d, _v, _fl = sq.query(iv)
    w = winding_number(iv, sq.tri_original)
    out = w < 0.5
    t_ours = np.asarray(tmap["t_primary_mm"], float)
    close = d < 5.0
    return {"gt_n_outside": int(out.sum()),
            "gt_p05": float(np.quantile(d, .05)),
            "gt_p50": float(np.quantile(d, .50)),
            "gt_p95": float(np.quantile(d, .95)),
            "n_comparable": int(close.sum()),
            "diff_p05": float(np.quantile(t_ours[close] - d[close], .05)),
            "diff_p50": float(np.median(t_ours[close] - d[close])),
            "diff_p95": float(np.quantile(t_ours[close] - d[close], .95)),
            "note": "audit-only：GT 冠外表面只作参照，不参与任何求解"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case-dir", required=True)
    ap.add_argument("--outer", required=True,
                    help="Contact/I.2 + Occlusion 之后的外表面 STL")
    ap.add_argument("--reference", default=None,
                    help="Occlusion 之前的外表面（算翻面率用）")
    ap.add_argument("--prep", default=None, help="默认 case-dir/prep.stl")
    ap.add_argument("--out", default=None)
    ap.add_argument("--tmin", type=float, default=None,
                    help="覆盖最小壁厚（默认取病例设计参数 MinThickness）")
    ap.add_argument("--sweep", default=None,
                    help="逗号分隔的 T_min 扫描（只影响判定，不重测几何）")
    a = ap.parse_args()

    import trimesh
    case_dir = a.case_dir
    out_dir = a.out or os.path.dirname(os.path.abspath(a.outer))
    os.makedirs(out_dir, exist_ok=True)
    meta = json.load(open(os.path.join(case_dir, "case.json"),
                          encoding="ascii"))
    ref = json.load(open(os.path.join(case_dir, "die_ref.json"),
                         encoding="ascii"))
    prm = ref.get("design_parameters", {})
    frame = patient_frame_from_meta(meta)

    prep_path = a.prep or os.path.join(case_dir, "prep.stl")
    prep = trimesh.load(prep_path, force="mesh")
    outer = trimesh.load(a.outer, force="mesh")
    reference = trimesh.load(a.reference, force="mesh") if a.reference else None
    mesial = NeighborPatch(
        *(lambda m: (m.vertices, m.faces))(
            trimesh.load(os.path.join(case_dir, "mesial_neighbor_patch.stl"),
                         force="mesh")))
    distal = NeighborPatch(
        *(lambda m: (m.vertices, m.faces))(
            trimesh.load(os.path.join(case_dir, "distal_neighbor_patch.stl"),
                         force="mesh")))
    ant_path = os.path.join(case_dir, "antagonist_patch.stl")
    antagonist = NeighborPatch(
        *(lambda m: (m.vertices, m.faces))(
            trimesh.load(ant_path, force="mesh"))) \
        if os.path.exists(ant_path) else None

    # 一次性定向（质心判据，与 I.2 / Occlusion 同法）
    centroid = np.asarray(outer.vertices, float).mean(axis=0)
    for patch in (mesial, distal, antagonist):
        if patch is None:
            continue
        fc = patch.vertices[patch.faces].mean(axis=1)
        w = np.einsum("ij,ij->i", centroid - fc, patch.normals)
        patch.normal_sign = 1.0 if w.sum() > 0 else -1.0

    gap = _f(prm, "CementGapThickness", 0.0) or 0.0
    iv, ifaces, info = intaglio_from_prep(
        np.asarray(prep.vertices, float), np.asarray(prep.faces, np.int64),
        gap, axis=np.asarray(meta["insertion_axis"], float))
    print("intaglio = prep + %.3fmm gap；法向翻转=%s（轴向壁 median dot=%s）"
          % (gap, info["normal_flipped"], info["axial_wall_median_dot"]))

    tmin = a.tmin if a.tmin is not None else _f(prm, "MinThickness")
    if tmin is None:
        raise SystemExit("病例没有 MinThickness 设计参数，必须用 --tmin 显式给")
    params = ThicknessParameters(
        min_thickness_mm=tmin,
        near_margin_thickness_mm=_f(prm, "MinThicknessNearPreparationMargin"),
        near_margin_distance_mm=_f(
            prm, "MinThicknessFullDistanceToPreparationMargin"),
        near_margin_mode="linear",
        cement_gap_mm=gap,
    )
    class _M:                      # 最小网格容器（引擎只要 vertices/faces）
        pass
    im = _M()
    im.vertices, im.faces = iv, ifaces
    res = audit_thickness(
        outer, im, frame,
        axis_mesial=np.asarray(meta.get("axis_mesial"), float),
        axis_buccal=np.asarray(meta.get("axis_buccal"), float),
        contact_patches={"mesial": mesial, "distal": distal},
        antagonist_patch=antagonist,
        contact_target_mm=_f(prm, "DistanceToNeighbor", 0.0) or 0.0,
        antagonist_target_mm=_f(prm, "DistanceToAntagonist", 0.1) or 0.0,
        reference_mesh=reference,
        params=params)

    print("\n== 输入 QC ==")
    for k, v in res.input_qc.items():
        print("   %-32s %s" % (k, v))
    print("\n== 区域表（T_min=%.2f，近颈缘 %.2f→%.2f @%.2fmm，%s）=="
          % (params.min_thickness_mm, params.near_margin_thickness_mm or 0,
             params.min_thickness_mm, params.near_margin_distance_mm or 0,
             params.near_margin_mode))
    hdr = ("Region", "n", "min", "p05", "median", "p95",
           "<Tmin area mm2", "worst deficit", "outside", "ambig")
    print("| " + " | ".join(hdr) + " |")
    print("|" + "---|" * len(hdr))
    for name in REGION_ROWS:
        r = res.regions[name]
        if not r.get("n"):
            print("| %s | 0 | | | | | | | | |" % name)
            continue
        print("| %s | %d | %.3f | %.3f | %.3f | %.3f | %.4f | %.3f | %d | %d |"
              % (name, r["n"], r["min"], r["p05"], r["median"], r["p95"],
                 r["thin_area_mm2"], r["worst_deficit_mm"],
                 r["n_outside_outer"], r["n_ambiguous"]))
    print("\n== 汇总 ==")
    for line in res.summary_lines().splitlines():
        print(line)
    print(json.dumps(res.summary, indent=1, ensure_ascii=False))
    for note in res.notes:
        print("note:", note)

    if a.sweep:
        print("\n== T_min 扫描（同一套 t，只换阈值）==")
        tprim = res.thickness_map["t_primary_mm"]
        area = res.thickness_map["vertex_area_mm2"]
        ev = res.thickness_map["flags"] != 4
        print("| T_min | thin vertices | thin area mm2 |")
        print("|---|---|---|")
        for t in [float(x) for x in a.sweep.split(",")]:
            thin = ev & (tprim < t)
            print("| %.2f | %d | %.4f |" % (t, thin.sum(), area[thin].sum()))

    # ---- 独立复核 1：封底 + 射线奇偶（不依赖任何法向）
    gtc = _parity_check(outer, iv)
    print("\n== 独立复核：封底 + 射线奇偶判内外 ==")
    print("   绕数判在冠外 %d 点；射线奇偶判在冠外 %d 点（差异 = 开口带附近）"
          % (res.summary["n_outside_outer"], gtc["outside"]))
    rep_extra = {"parity_check": gtc}

    # ---- 独立复核 2（audit-only GT）：同一套 intaglio 对 GT 冠外表面
    gt_path = os.path.join(case_dir, "final_crown.stl")
    if os.path.exists(gt_path):
        g = _gt_reference(gt_path, iv, meta, res.thickness_map)
        rep_extra["gt_reference"] = g
        print("\n== GT 参照（audit-only，不参与求解）==")
        print("   GT 冠：负壁厚 %d 点；t p05=%.3f p50=%.3f p95=%.3f"
              % (g["gt_n_outside"], g["gt_p05"], g["gt_p50"], g["gt_p95"]))
        print("   我们的冠：负壁厚 %d 点；t p05=%.3f p50=%.3f p95=%.3f"
              % (res.summary["n_outside_outer"],
                 res.summary["p05_thickness_mm"] or 0,
                 res.summary["median_thickness_mm"] or 0,
                 float(np.quantile(res.thickness_map["t_primary_mm"], .95))))
        print("   同点差（我们 - GT）p50=%.3fmm（负 = 我们更薄）"
              % g["diff_p50"])

    rep = {"case_uuid": meta.get("case_uuid"), "status": res.status,
           "params": params.__dict__, "input_qc": res.input_qc,
           "regions": res.regions, "thin_area_by_kind_mm2": res.thin_area,
           "summary": res.summary, "notes": res.notes,
           "independent_checks": rep_extra}
    with open(os.path.join(out_dir, "thickness_audit_report.json"), "w",
              encoding="ascii") as fh:
        json.dump(rep, fh, indent=1)
    tm = res.thickness_map
    with open(os.path.join(out_dir, "thickness_map.csv"), "w", newline="",
              encoding="ascii") as fh:
        w = csv.writer(fh)
        w.writerow(["x", "y", "z", "t_closest_mm", "t_ray_mm", "t_primary_mm",
                    "threshold_mm", "deficit_mm", "d_margin_mm", "region",
                    "kind", "flags", "clearance_mm", "blockers",
                    "vertex_area_mm2"])
        for i in range(len(tm["vertices"])):
            w.writerow(["%.5f" % c for c in tm["vertices"][i]] +
                       ["%.5f" % tm["t_closest_mm"][i],
                        "%.5f" % tm["t_ray_mm"][i],
                        "%.5f" % tm["t_primary_mm"][i],
                        "%.5f" % tm["threshold_mm"][i],
                        "%.5f" % tm["deficit_mm"][i],
                        "%.5f" % tm["d_margin_mm"][i],
                        str(tm["region"][i]), str(tm["kind"][i]),
                        int(tm["flags"][i]),
                        ("%.5f" % tm["clearance_mm"][i]
                         if np.isfinite(tm["clearance_mm"][i]) else ""),
                        int(tm["blockers"][i]),
                        "%.6f" % tm["vertex_area_mm2"][i]])
    print("\n写出 %s / thickness_map.csv" % out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
