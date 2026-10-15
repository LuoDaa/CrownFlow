#!/usr/bin/env python3
"""H.2 —— Neighbor-Aware Anatomy Placement：H.1 vs H.2 的**先验层**验收。

用户 2026-10-09 规格（路线①+③）：H.1 保留为 frozen baseline；H.2 = 邻牙感知的
**各向异性放置**（近远中按邻牙可用间隙定宽，颊舌/咬合不变），**不做自由形变**。

本轮只做三件事：
  1. Neighbor-space audit → available MD width
  2. Anisotropic placement → s_md / s_bl / s_z
  3. Acceptance report → H.1 vs H.2 的近远中穿透 / 几何 / 解剖偏差

**不看 Contact**（Contact 只在 H.2 → J.2 → K0 之后跑）。

用法::

    CROWNFLOW_TOOTH_LIBRARY_ROOT=<牙库根> \
    python tools/run_h2_placement_case.py --case-dir <病例包> --out <目录>
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.geometry.crown_outer.contact import NeighborPatch            # noqa: E402
from app.geometry.crown_outer.proximal_fit import (                   # noqa: E402
    ProximalFitParams,
    measure_proximal_prior,
)
from app.services.crown_outer_pipeline import (                       # noqa: E402
    _h1_template,
    patient_frame_from_meta,
)

#: s_md 下限（构造参数，**不是临床阈值**）：先允许最多收 15%，分布看清再调。
MD_CLAMP_MIN = 0.85
#: 近远中缩放在冠部顶保留的比例（0 = 回到 H.1 的完全衰减）。
#: H.2 要求中段按邻牙空间收窄、咬合区避免压扁 cusp → 取 0.35。
MD_BODY_KEEP = 0.35


def _patch(path):
    import trimesh
    m = trimesh.load(str(path), force="mesh")
    return NeighborPatch(np.asarray(m.vertices, float),
                         np.asarray(m.faces, np.int64))


def _geometry_qc(mesh) -> dict:
    from app.geometry.crown_outer.thickness import (geometry_safety_gate,
                                                    self_intersections)
    v = np.asarray(mesh.vertices, float)
    f = np.asarray(mesh.faces, np.int64)
    ev = self_intersections(v, f, severe_min_area_mm2=0.0025,
                            severe_max_normal_dot=0.999)
    g = geometry_safety_gate(v, f, max_flip_ratio=None,
                             max_self_intersections=None,
                             expect_open_boundary=True)
    return {"self_intersections": int(ev.get("self_intersections") or 0),
            "nonmanifold_edges": int(g["metrics"].get("nonmanifold_edges") or 0),
            "winding_inconsistent_edges": int(
                g["metrics"].get("winding_inconsistent_edges") or 0),
            "boundary_edges": int(g["metrics"].get("boundary_edges") or 0),
            "degenerate_faces": int(g["metrics"].get("degenerate_faces") or 0)}


def _half_width_md(mesh, frame, crown_mask=None) -> float:
    """冠体的近远中半宽（患者 MD 轴上的半跨）。"""
    v = np.asarray(mesh.vertices, float)
    lat = (v - frame.origin) @ np.asarray(frame.x, float)
    return float((lat.max() - lat.min()) / 2.0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--data-dir", default=None,
                    help="仓库 data/ 目录（放 H.1 STL 用）；缺省用 --out")
    ap.add_argument("--clearance-mm", type=float, default=0.05)
    a = ap.parse_args()

    import trimesh
    from app.geometry.crown_outer.generator import generate_crown_outer_v0
    from app.geometry.crown_outer.models import CrownOuterConfig

    os.makedirs(a.out, exist_ok=True)
    meta = json.load(open(os.path.join(a.case_dir, "case.json"),
                          encoding="ascii"))
    frame = patient_frame_from_meta(meta)
    margin = np.asarray(meta["margin_points"], float)
    axis = np.asarray(meta["insertion_axis"], float)
    mes = _patch(os.path.join(a.case_dir, "mesial_neighbor_patch.stl"))
    dis = _patch(os.path.join(a.case_dir, "distal_neighbor_patch.stl"))
    pf = ProximalFitParams(target_clearance_mm=a.clearance_mm)

    template, lib = _h1_template({}, int(meta["fdi"]))
    print("模板库 %s（FDI %d）" % (lib, meta["fdi"]))

    def run(md_scale, tag, md_body_keep=None):
        res = generate_crown_outer_v0(
            template, margin, axis,
            config=CrownOuterConfig(body_scale_decoupling=True),
            md_scale=md_scale, md_body_keep=md_body_keep)
        m = res.mesh
        audit = measure_proximal_prior(m, frame, mes, dis, pf)
        rec = {
            "tag": tag, "uniform_scale": res.uniform_scale,
            "anisotropic_scale": res.mesh_qc.get("anisotropic_scale"),
            "margin_error_p95_mm": res.margin_error_p95_mm,
            "displacement_by_region": res.displacement_by_region,
            "geometry": _geometry_qc(m),
            "half_width_md_mm": _half_width_md(m, frame),
            "penetration": {s: audit[s]["penetration_mm"]
                            for s in ("mesial", "distal")},
        }
        return res, m, audit, rec

    # ---- H.1（frozen baseline）
    res1, m1, aud1, rec1 = run(None, "H.1")
    print("\n== H.1 ==")
    print("uniform_scale=%.5f  half_width_md=%.3fmm  margin p95=%.4fmm"
          % (rec1["uniform_scale"], rec1["half_width_md_mm"],
             rec1["margin_error_p95_mm"] or 0.0))
    for s in ("mesial", "distal"):
        p = rec1["penetration"][s]
        print("  %-7s pen p95=%.4f max=%.4f (n=%d)" % (s, p["p95"], p["max"], p["n"]))
    print("  geometry:", json.dumps(rec1["geometry"], ensure_ascii=False))

    # ---- Neighbor-space audit → s_md
    hw = rec1["half_width_md_mm"]
    need = max(rec1["penetration"]["mesial"]["p95"],
               rec1["penetration"]["distal"]["p95"])
    s_md_raw = 1.0 - need / max(hw, 1e-9)
    s_md = float(np.clip(s_md_raw, MD_CLAMP_MIN, 1.0))
    print("\n== Neighbor-space audit ==")
    print("crown half-width(MD)=%.3fmm  需要收窄(取两侧较深)=%.4fmm" % (hw, need))
    print("s_md_raw=%.5f → clamp[%.2f,1] → **s_md=%.5f**" % (s_md_raw, MD_CLAMP_MIN, s_md))

    # ---- H.2（各向异性放置）
    res2, m2, aud2, rec2 = run(s_md, "H.2", md_body_keep=MD_BODY_KEEP)
    print("\n== H.2 ==")
    print("scale_vec=%s  half_width_md=%.3fmm  margin p95=%.4fmm"
          % (rec2["anisotropic_scale"], rec2["half_width_md_mm"],
             rec2["margin_error_p95_mm"] or 0.0))
    for s in ("mesial", "distal"):
        p1, p2 = rec1["penetration"][s], rec2["penetration"][s]
        print("  %-7s pen p95 %.4f→%.4f  max %.4f→%.4f  (n %d→%d)"
              % (s, p1["p95"], p2["p95"], p1["max"], p2["max"], p1["n"], p2["n"]))
    print("  geometry:", json.dumps(rec2["geometry"], ensure_ascii=False))
    print("  区域位移:", json.dumps(rec2["displacement_by_region"],
                                    ensure_ascii=False))

    print("\n== H.1 vs H.2（先验层验收，不看 Contact）==")
    print("| 指标 | H.1 | H.2 |")
    print("|---|---:|---:|")
    print("| md scale | %.5f | %.5f |" % (rec1["uniform_scale"], s_md))
    for s in ("mesial", "distal"):
        print("| %s penetration p95 | %.4f | %.4f |"
              % (s, rec1["penetration"][s]["p95"], rec2["penetration"][s]["p95"]))
    print("| margin fit p95 | %.4f | %.4f |"
          % (rec1["margin_error_p95_mm"] or 0.0, rec2["margin_error_p95_mm"] or 0.0))
    print("| geometry (自交/非流形/朝向) | %d/%d/%d | %d/%d/%d |"
          % (rec1["geometry"]["self_intersections"],
             rec1["geometry"]["nonmanifold_edges"],
             rec1["geometry"]["winding_inconsistent_edges"],
             rec2["geometry"]["self_intersections"],
             rec2["geometry"]["nonmanifold_edges"],
             rec2["geometry"]["winding_inconsistent_edges"]))

    m1.export(os.path.join(a.out, "anatomy_h1.stl"))
    m2.export(os.path.join(a.out, "anatomy_h2.stl"))
    with open(os.path.join(a.out, "h2_placement_report.json"), "w",
              encoding="utf-8") as fh:
        json.dump({"h1": rec1, "h2": rec2,
                   "audit": {"half_width_md_mm": hw, "required_shrink_mm": need,
                             "s_md_raw": s_md_raw, "s_md": s_md,
                             "clamp": [MD_CLAMP_MIN, 1.0],
                             "note": "s_md 由邻牙可用间隙推得；clamp 是构造参数"}},
                  fh, indent=1, ensure_ascii=False)
    print("\n写出 %s" % a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
