r"""第一颗 FDI46 真实回归的执行脚本（方案 §16 清单 1–9，在本地跑，需 numpy/scipy/trimesh）。

    python backend/tools/run_crown_outer_v0.py \
        --case-dir "D:\code\牙冠自动生成\数据\crown_outer_cases\cf_d310209b58909a45" \
        --out "D:\code\牙冠自动生成\数据\crown_outer_results\cf_d310209b58909a45"

产出：
    crown_outer_v0.stl     变形后的冠部外表面（开放面片，底边 = 患者颈缘）
    result.json            placement / correspondence / QC / GT 指标 / 配置快照
    gt_outer.stl           从 Final Crown 分离出的外表面 GT（acceptance-only）
    renders/*.png          placed / deformed / GT 三视图（复用 qa_render）
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.geometry.crown_outer import (  # noqa: E402
    CrownOuterConfig,
    PatientCrownContext,
    extract_outer_surface,
    generate_crown_outer_v0,
    load_crown_template,
    patient_margin_perimeter,
    sample_surface_points,
    surface_distance_metrics,
)
from app.importers.tooth_library import discover_libraries  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--library", default="炼金王")
    ap.add_argument("--gt", action="store_true",
                    help="同时做 Final Crown GT 回归比较")
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    meta = json.load(open(os.path.join(a.case_dir, "case.json"),
                          encoding="ascii"))
    ctx = PatientCrownContext(
        case_uuid=meta["case_uuid"], fdi=meta["fdi"],
        margin_points=np.asarray(meta["margin_points"]),
        insertion_axis=np.asarray(meta["insertion_axis"]),
        center=np.asarray(meta["center"]) if meta.get("center") else None,
        final_crown_gt_path=os.path.join(a.case_dir, "final_crown.stl")
        if a.gt else None)

    # ---- 1. 模板
    libs = {l.name: l for l in discover_libraries()}
    lib = libs[a.library]
    template = load_crown_template(lib, ctx.fdi)
    print("模板:", json.dumps(template.summary(), ensure_ascii=False))

    # ---- 2-7. 生成
    config = CrownOuterConfig()
    result = generate_crown_outer_v0(template, ctx.margin_points,
                                     ctx.insertion_axis, config=config)
    print("状态:", result.status, result.notes)
    stl_path = os.path.join(a.out, "crown_outer_v0.stl")
    result.mesh.export(stl_path)

    # ---- 8-9. GT 回归
    gt_metrics = None
    if a.gt:
        import trimesh
        crown = trimesh.load(ctx.final_crown_gt_path, force="mesh")
        gt_faces = extract_outer_surface(crown, ctx.insertion_axis,
                                         np.asarray(ctx.margin_points).mean(0))
        gt_outer = crown.submesh([gt_faces], append=True, repair=False)
        gt_outer.export(os.path.join(a.out, "gt_outer.stl"))
        print("GT 外表面: %d/%d 面" % (len(gt_faces), len(crown.faces)))

        pred_pts, pred_nrm = sample_surface_points(result.mesh, n=20000)
        gt_pts, gt_nrm = sample_surface_points(gt_outer, n=20000)
        gt_metrics = surface_distance_metrics(pred_pts, gt_pts, pred_nrm,
                                              gt_nrm)
        result.gt_metrics = gt_metrics
        print("GT 指标:", json.dumps(gt_metrics, indent=1))

    out_json = {
        "case_uuid": ctx.case_uuid, "fdi": ctx.fdi,
        "library": a.library,
        "template": template.summary(),
        "result": result.summary(),
        "config": config.as_dict(),
        "patient_margin_perimeter_mm":
            round(patient_margin_perimeter(ctx.margin_points), 3),
        "n_patient_margin_points": int(len(ctx.margin_points)),
        "gt_metrics": gt_metrics,
    }
    with open(os.path.join(a.out, "result.json"), "w", encoding="ascii") as fh:
        json.dump(out_json, fh, indent=1)
    print("\n写出:", a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
