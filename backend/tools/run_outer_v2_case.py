#!/usr/bin/env python3
"""J.1 真实例验收：H.1（解剖）→ J.1（Intaglio 约束）→ Thickness Audit 复验。

用户 2026-10-02 规格：J.1 的验收就是"六条约束是否成立"，所以本工具**同时**：

1. 跑 J.1（`build_intaglio_constrained_outer`）；
2. 跑 Thickness Audit（**同一套设计规则对象**，不与构造脱节）；
3. 出四条硬事实：负壁厚（覆盖率）、薄区、翻面、自交；以及解剖偏差。

输出 PHI-free：case_uuid + 统计量 + 逐顶点 CSV。

用法::

    python3 tools/run_outer_v2_case.py \
        --case-dir <...>/crown_outer_cases/cf_b1b73bae221211b1 \
        --anatomy <...>/contact_v1_case/h1_crown.stl \
        --out <...>/outer_v2_j1_case
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.geometry.crown_outer.intaglio_constrained import (   # noqa: E402
    OuterV2Params,
    build_intaglio_constrained_outer,
)
from app.geometry.crown_outer.patient_frame import PatientFrame  # noqa: E402
from app.geometry.crown_outer.thickness import (                  # noqa: E402
    REGION_ROWS,
    ThicknessParameters,
    audit_thickness,
    intaglio_from_prep,
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_thickness_audit_case import _f, patient_frame_from_meta  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case-dir", required=True)
    ap.add_argument("--anatomy", required=True, help="H.1 冠（开放壳）")
    ap.add_argument("--out", required=True)
    ap.add_argument("--safety-mm", type=float, default=0.02)
    ap.add_argument("--max-iterations", type=int, default=12)
    a = ap.parse_args()

    import trimesh
    os.makedirs(a.out, exist_ok=True)
    meta = json.load(open(os.path.join(a.case_dir, "case.json"),
                          encoding="ascii"))
    ref = json.load(open(os.path.join(a.case_dir, "die_ref.json"),
                         encoding="ascii"))
    prm = ref.get("design_parameters", {})
    frame: PatientFrame = patient_frame_from_meta(meta)
    axis = np.asarray(meta["insertion_axis"], float)

    anatomy = trimesh.load(a.anatomy, force="mesh")
    prep = trimesh.load(os.path.join(a.case_dir, "prep.stl"), force="mesh")
    gap = _f(prm, "CementGapThickness", 0.0) or 0.0
    iv, ifaces, _info = intaglio_from_prep(
        np.asarray(prep.vertices, float), np.asarray(prep.faces, np.int64),
        gap, axis=axis)

    class _M:
        pass
    im = _M()
    im.vertices, im.faces = iv, ifaces
    tmin = _f(prm, "MinThickness")
    if tmin is None:
        raise SystemExit("病例没有 MinThickness，必须显式给（本工具不猜）")
    rule = ThicknessParameters(
        min_thickness_mm=tmin,
        near_margin_thickness_mm=_f(prm, "MinThicknessNearPreparationMargin"),
        near_margin_distance_mm=_f(
            prm, "MinThicknessFullDistanceToPreparationMargin"),
        cement_gap_mm=gap)

    # ---- J.1
    res = build_intaglio_constrained_outer(
        anatomy, im, OuterV2Params(thickness=rule, safety_mm=a.safety_mm,
                                   max_outer_iterations=a.max_iterations))
    print("== J.1 ==")
    print(json.dumps(res.summary(), indent=1, ensure_ascii=False))
    for n in res.notes[-6:]:
        print("note:", n)
    out_mesh = _M()
    out_mesh.vertices = np.asarray(res.mesh.vertices, float)
    out_mesh.faces = np.asarray(res.mesh.faces, np.int64)
    trimesh.Trimesh(out_mesh.vertices, out_mesh.faces, process=False).export(
        os.path.join(a.out, "crown_outer_v2_j1.stl"))

    # ---- Audit 复验（同一设计规则；anatomy 作参考网格算翻面率）
    print("\n== Thickness Audit 复验（J.1 输出）==")
    aud = audit_thickness(
        trimesh.Trimesh(out_mesh.vertices, out_mesh.faces, process=False),
        im, frame,
        axis_mesial=np.asarray(meta.get("axis_mesial"), float),
        axis_buccal=np.asarray(meta.get("axis_buccal"), float),
        reference_mesh=trimesh.Trimesh(np.asarray(anatomy.vertices, float),
                                       np.asarray(anatomy.faces, np.int64),
                                       process=False),
        params=rule)
    print(aud.summary_lines())
    s = aud.summary
    print("负壁厚(覆盖率失败): %d 点 / %.3f mm² | 薄区 %.3f mm² | 自交 %s | 翻面 %.4f%%"
          % (s["n_outside_outer"], s["negative_coverage"]["area_mm2"],
             s["thin_area_mm2"],
             aud.input_qc.get("self_intersections"),
             100.0 * (aud.input_qc.get("flipped_ratio") or 0.0)))
    print("区域表（T_min=%.2f）:" % rule.min_thickness_mm)
    print("| Region | n | min | p05 | median | thin mm2 | outside |")
    print("|---|---|---|---|---|---|---|")
    for name in REGION_ROWS:
        r = aud.regions[name]
        if not r.get("n"):
            print("| %s | 0 | | | | | |" % name)
            continue
        print("| %s | %d | %.3f | %.3f | %.3f | %.3f | %d |"
              % (name, r["n"], r["min"], r["p05"], r["median"],
                 r["thin_area_mm2"], r["n_outside_outer"]))

    rep = {"case_uuid": meta.get("case_uuid"),
           "j1": res.summary(), "j1_notes": res.notes,
           "audit": {"status": aud.status, "summary": aud.summary,
                     "input_qc": aud.input_qc, "regions": aud.regions,
                     "notes": aud.notes},
           "rule": rule.__dict__}
    with open(os.path.join(a.out, "outer_v2_j1_report.json"), "w",
              encoding="ascii") as fh:
        json.dump(rep, fh, indent=1)
    tm = aud.thickness_map
    with open(os.path.join(a.out, "thickness_map_v2.csv"), "w", newline="",
              encoding="ascii") as fh:
        w = csv.writer(fh)
        w.writerow(["x", "y", "z", "t_closest_mm", "t_ray_mm", "t_primary_mm",
                    "threshold_mm", "deficit_mm", "d_margin_mm", "region",
                    "kind", "flags", "clearance_mm", "blockers",
                    "vertex_area_mm2"])
        for i in range(len(tm["vertices"])):
            w.writerow(["%.5f" % c for c in tm["vertices"][i]] +
                       ["%.5f" % tm[k][i] for k in
                        ("t_closest_mm", "t_ray_mm", "t_primary_mm",
                         "threshold_mm", "deficit_mm", "d_margin_mm")] +
                       [str(tm["region"][i]), str(tm["kind"][i]),
                        int(tm["flags"][i]),
                        ("%.5f" % tm["clearance_mm"][i]
                         if np.isfinite(tm["clearance_mm"][i]) else ""),
                        int(tm["blockers"][i]),
                        "%.6f" % tm["vertex_area_mm2"][i]])
    print("\n写出 %s（crown_outer_v2_j1.stl + report + thickness_map_v2.csv）"
          % a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
