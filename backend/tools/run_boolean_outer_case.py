#!/usr/bin/env python3
"""J.2.1-B 真实例验收：Closed Envelope ∪ Closed Anatomy → Boolean → 裁 Body → 缝 Collar。

用户 2026-10-08 规格：这一轮**只验 Outer**，不跑 Contact / Occlusion。输出
用户指定的验收表（components / boundary loops / margin 精度 / 非流形 / 朝向 /
自交 / 负覆盖 / 位移 / envelope 违规 / 偏差分账）+ 面级来源标签统计。

用法::

    python3 tools/run_boolean_outer_case.py \
        --case-dir <...>/crown_outer_cases/cf_b1b73bae221211b1 \
        --out <...>/j21b_case            # 含真实几何，勿进 Git
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.geometry.crown_outer.boolean_envelope_outer import (   # noqa: E402
    BooleanOuterParams,
    build_boolean_outer,
)
from app.geometry.crown_outer.thickness import (                # noqa: E402
    ThicknessParameters,
    intaglio_from_prep,
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_thickness_audit_case import _f                          # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--anatomy", default=None,
                    help="H.1 冠（开放壳）；缺省 <case-dir>/../_anatomy_h1.stl")
    ap.add_argument("--intaglio", default=None,
                    help="intaglio STL；缺省按 prep + CementGap 现算")
    ap.add_argument("--voxel-mm", type=float, default=0.10)
    ap.add_argument("--d1-mm", type=float, default=0.60)
    ap.add_argument("--guard-mm", type=float, default=0.25)
    a = ap.parse_args()

    import trimesh
    os.makedirs(a.out, exist_ok=True)
    meta = json.load(open(os.path.join(a.case_dir, "case.json"),
                          encoding="ascii"))
    ref = json.load(open(os.path.join(a.case_dir, "die_ref.json"),
                         encoding="ascii"))
    prm = ref.get("design_parameters", {})
    axis = np.asarray(meta["insertion_axis"], float)
    mpts = np.asarray(meta["margin_points"], float)

    if not a.anatomy:
        raise SystemExit("必须给 --anatomy（H.1 冠的 STL；不要在工具里猜路径）")
    anatomy = trimesh.load(a.anatomy, force="mesh")
    if a.intaglio:
        inner = trimesh.load(a.intaglio, force="mesh")
    else:
        prep = trimesh.load(os.path.join(a.case_dir, "prep.stl"), force="mesh")
        gap = _f(prm, "CementGapThickness", 0.0) or 0.0
        iv, ifa, _info = intaglio_from_prep(
            np.asarray(prep.vertices, float), np.asarray(prep.faces, np.int64),
            gap, axis=axis)
        inner = trimesh.Trimesh(iv, ifa, process=False)

    tmin = _f(prm, "MinThickness")
    if tmin is None:
        raise SystemExit("病例没有 MinThickness，必须显式给（本工具不猜）")
    rule = ThicknessParameters(
        min_thickness_mm=tmin,
        near_margin_thickness_mm=_f(prm, "MinThicknessNearPreparationMargin"),
        near_margin_distance_mm=_f(
            prm, "MinThicknessFullDistanceToPreparationMargin"),
        cement_gap_mm=_f(prm, "CementGapThickness", 0.0) or 0.0)

    res = build_boolean_outer(
        anatomy, inner, mpts, axis,
        BooleanOuterParams(thickness=rule, voxel_mm=a.voxel_mm,
                           d1_mm=a.d1_mm, guard_mm=a.guard_mm))
    print(json.dumps(res.summary(), indent=1, ensure_ascii=False))
    for n in res.notes:
        print("note:", n)

    # ---- 用户规格的验收表（只验 Outer）
    print("\n== J.2-B 第一轮验收表（Outer only）==")
    print("| QC | 要求 | 实测 |")
    print("|---|---|---|")
    cq = res.clip_qc or {}
    bq = res.boolean_qc or {}
    st = (res.safety or {}).get("topology") or {}
    rows = [
        ("connected components", "1", st.get("components")),
        ("boundary loops", "1", st.get("boundary_loops")),
        ("boundary = prescribed Margin",
         "接近 collar 精度",
         "p95 %s / max %s mm；原 margin 顶点 %s"
         % (st.get("main_loop_to_margin_p95_mm"),
            st.get("main_loop_to_margin_max_mm"),
            st.get("main_loop_original_margin_vertices"))),
        ("non-manifold edges", "0",
         (res.safety or {}).get("metrics", {}).get("nonmanifold_edges")),
        ("orientation inconsistent", "0",
         (res.safety or {}).get("metrics", {}).get("winding_inconsistent_edges")),
        ("self-intersections", "0（micro 另计）",
         "%s（severe %s / micro %s；面积阈值 %s mm²）"
         % ((res.safety or {}).get("metrics", {}).get("self_intersections"),
            ((res.safety or {}).get("self_intersection_severity") or {}).get("severe"),
            ((res.safety or {}).get("self_intersection_severity") or {}).get("micro_defect"),
            ((res.safety or {}).get("self_intersection_severity") or {}).get("criteria", {}).get("min_area_mm2"))),
        ("degenerate faces", "记录",
         (res.safety or {}).get("metrics", {}).get("degenerate_faces")),
        ("negative coverage", "0", (res.coverage or {}).get("n_negative")),
        ("envelope violation（ramp 外）", "≈0（数值容差）",
         (res.coverage or {}).get("n_violating_beyond_ramp")),
        ("envelope violation（ramp band 内，如实推迟）",
         "记录", (res.coverage or {}).get("n_violating_in_ramp_band")),
        ("intaglio 位移", "0", "Intaglio 不在本阶段输出里（构造里 immutable）"),
        ("cervical loop 位移", "0",
         "collar 底环 = 原 margin 顶点（见上）"),
    ]
    for k, want, got in rows:
        print("| %s | %s | %s |" % (k, want, got))
    print("\n偏差分账：")
    for k, v in (res.deviation_split or {}).items():
        print("  %-22s %s" % (k, v))
    print("\n来源面积占比：%s"
          % json.dumps((res.provenance or {}).get("area_frac"),
                       ensure_ascii=False))
    print("guard band：%s" % json.dumps(bq.get("guard_band"), ensure_ascii=False))
    print("ring gate：%s" % json.dumps((res.stitch_qc or {}).get("ring_gate"),
                                      ensure_ascii=False))
    sr = (res.stitch_qc or {}).get("seam_repair")
    if sr:
        print("J.2.1-C seam repair：自交 %s→%s（severe %s→%s）；Phase1 %s / "
              "Phase2(DP) %s"
              % (sr.get("n_hits_before"), sr.get("n_hits_after"),
                 sr.get("n_severe_before"), sr.get("n_severe_after"),
                 json.dumps(sr.get("phase1"), ensure_ascii=False),
                 json.dumps(sr.get("phase2_dp"), ensure_ascii=False)))
    print("clip：%s" % json.dumps(cq, ensure_ascii=False))

    if res.mesh is not None:
        trimesh.Trimesh(np.asarray(res.mesh.vertices, float),
                        np.asarray(res.mesh.faces, np.int64),
                        process=False).export(os.path.join(a.out, "outer_j2b.stl"))
    with open(os.path.join(a.out, "outer_j2b_report.json"), "w",
              encoding="ascii") as fh:
        json.dump({"case_uuid": meta.get("case_uuid"),
                   "summary": res.summary(), "notes": res.notes,
                   "coverage": res.coverage,
                   "deviation_split": res.deviation_split,
                   "rule": rule.__dict__}, fh, indent=1)
    print("\n写出 %s" % a.out)
    return 0 if res.status == "OK" else 1


if __name__ == "__main__":
    raise SystemExit(main())
