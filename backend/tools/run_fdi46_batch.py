"""提交 H：FDI46 批量回归（18 例，固定配置，**禁止逐例调参**）。

    python backend/tools/run_fdi46_batch.py \
        --cases "D:\code\牙冠自动生成\数据\crown_outer_cases" \
        --out "D:\code\牙冠自动生成\数据\crown_outer_results\fdi46_batch"

统一：Template=炼金王 FDI46、N=256、同一 placement/solver/QC
（CrownOuterConfig() 默认值即冻结配置，改任何一项都视为新版本）。

输出 ``fdi46_regression.csv``（每例一行，字段按用户规格）+
``batch_summary.json``（跨例聚合：分区域带符号误差的中位数分布 ——
系统性偏差就从这里读）。
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.geometry.crown_outer import (  # noqa: E402
    CrownOuterConfig,
    generate_crown_outer_v0,
    load_crown_template,
    sample_surface_points,
    surface_distance_metrics,
)
from app.geometry.crown_outer.gt_compare import (  # noqa: E402
    anatomical_regions,
    region_signed_stats,
    signed_surface_error,
)
from app.importers.tooth_library import discover_libraries  # noqa: E402

CSV_COLUMNS = [
    "case_uuid", "status", "uniform_scale",
    "margin_error_median", "margin_error_p95", "margin_error_max",
    "cervical_displacement_median", "cervical_displacement_p95",
    "axial_displacement_median", "axial_displacement_p95",
    "occlusal_displacement_median", "occlusal_displacement_p95",
    "flipped_face_ratio", "degenerate_faces", "connected_components",
    "boundary_consistency",
    "pred_to_gt_median", "pred_to_gt_p95",
    "gt_to_pred_median", "gt_to_pred_p95",
    "chamfer", "hausdorff",
    "normal_angle_median", "normal_angle_p95",
    "signed_overall_median", "signed_overall_p95",
    "signed_cervical_median", "signed_cervical_p95",
    "signed_mesial_median", "signed_mesial_p95",
    "signed_distal_median", "signed_distal_p95",
    "signed_buccal_median", "signed_buccal_p95",
    "signed_lingual_median", "signed_lingual_p95",
    "signed_occlusal_median", "signed_occlusal_p95",
    "elapsed_s",
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--library", default="炼金王")
    ap.add_argument("--fdi", type=int, default=46)
    ap.add_argument("--h1", action="store_true", help="启用冠体缩放解耦")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    libs = {l.name: l for l in discover_libraries()}
    template = load_crown_template(libs[a.library], a.fdi)   # 只加载一次
    config = CrownOuterConfig(body_scale_decoupling=a.h1)    # 全病例共用
    print("配置（全部病例共用）:", json.dumps(config.as_dict()))

    case_dirs = sorted(
        d for d in os.listdir(a.cases)
        if d.startswith("cf_") and
        os.path.isfile(os.path.join(a.cases, d, "case.json")))
    print("病例 %d 例" % len(case_dirs))

    rows = []
    t_all = time.time()
    for i, cid in enumerate(case_dirs):
        cdir = os.path.join(a.cases, cid)
        meta = json.load(open(os.path.join(cdir, "case.json"),
                              encoding="ascii"))
        t0 = time.time()
        row = {k: "" for k in CSV_COLUMNS}
        row["case_uuid"] = cid
        try:
            from app.geometry.crown_outer import PatientCrownContext
            ctx = PatientCrownContext(
                case_uuid=cid, fdi=a.fdi,
                margin_points=np.asarray(meta["margin_points"]),
                insertion_axis=np.asarray(meta["insertion_axis"]),
                final_crown_gt_path=os.path.join(cdir, "final_crown.stl"))
            res = generate_crown_outer_v0(template, ctx.margin_points,
                                          ctx.insertion_axis, config=config)
            row["status"] = res.status
            row["uniform_scale"] = res.uniform_scale
            row["margin_error_median"] = res.margin_error_median_mm
            row["margin_error_p95"] = res.margin_error_p95_mm
            row["margin_error_max"] = res.margin_error_max_mm
            for reg in ("cervical", "axial", "occlusal"):
                d = res.displacement_by_region.get(reg, {})
                row[reg + "_displacement_median"] = d.get("median_mm")
                row[reg + "_displacement_p95"] = d.get("p95_mm")
            mq = res.mesh_qc.get("mesh", {})
            row["flipped_face_ratio"] = mq.get("flipped_ratio")
            row["degenerate_faces"] = mq.get("n_degenerate_faces")
            row["connected_components"] = mq.get("n_components")
            row["boundary_consistency"] = mq.get("boundary_consistent")

            # ---- GT 回归 + 带符号分区域
            import trimesh
            from app.geometry.crown_outer import extract_outer_surface
            crown = trimesh.load(ctx.final_crown_gt_path, force="mesh")
            mc = np.asarray(ctx.margin_points).mean(axis=0)
            gt_faces = extract_outer_surface(crown, ctx.insertion_axis, mc)
            gt_outer = crown.submesh([gt_faces], append=True, repair=False)
            gt_outer.export(os.path.join(a.out, cid + "_gt_outer.stl"))
            suffix = "_crown_outer_h1.stl" if a.h1 else "_crown_outer_v0.stl"
            res.mesh.export(os.path.join(a.out, cid + suffix))

            pred_pts, pred_nrm = sample_surface_points(res.mesh, n=15000)
            gt_pts, gt_nrm = sample_surface_points(gt_outer, n=15000)
            metrics = surface_distance_metrics(pred_pts, gt_pts,
                                               pred_nrm, gt_nrm)
            signed = signed_surface_error(pred_pts, gt_pts, gt_nrm)
            metrics["signed_overall"] = {
                "median": round(float(np.median(signed)), 4),
                "p95": round(float(np.percentile(np.abs(signed), 95)), 4),
            }
            regions = anatomical_regions(
                pred_pts, mc, ctx.insertion_axis,
                meta.get("axis_mesial"), meta.get("axis_buccal"))
            metrics["signed_by_region"] = region_signed_stats(
                pred_pts, signed, regions)
            res.gt_metrics = metrics

            row["pred_to_gt_median"] = metrics["pred_to_gt_mm"]["median"]
            row["pred_to_gt_p95"] = metrics["pred_to_gt_mm"]["p95"]
            row["gt_to_pred_median"] = metrics["gt_to_pred_mm"]["median"]
            row["gt_to_pred_p95"] = metrics["gt_to_pred_mm"]["p95"]
            row["chamfer"] = metrics["chamfer_mean_mm"]
            row["hausdorff"] = metrics["hausdorff_mm"]
            row["normal_angle_median"] = metrics["normal_angle_deg"]["median"]
            row["normal_angle_p95"] = metrics["normal_angle_deg"]["p95"]
            so = metrics["signed_overall"]
            row["signed_overall_median"] = so["median"]
            row["signed_overall_p95"] = so["p95"]
            for reg in ("cervical", "mesial", "distal", "buccal", "lingual",
                        "occlusal"):
                rs = metrics["signed_by_region"].get(reg, {})
                row["signed_" + reg + "_median"] = rs.get("signed_median_mm")
                row["signed_" + reg + "_p95"] = rs.get("signed_p95_mm")
        except Exception as exc:                     # noqa: BLE001
            row["status"] = "FAILED"
            row["elapsed_s"] = round(time.time() - t0, 1)
            print("[%2d] %s FAILED: %s: %s" % (i, cid[:14],
                                               type(exc).__name__, exc),
                  flush=True)
            rows.append(row)
            continue
        row["elapsed_s"] = round(time.time() - t0, 1)
        rows.append(row)
        print("[%2d] %s %-6s scale=%.4f chamfer=%.3f signed_med=%+.3f (%.0fs)"
              % (i, cid[:14], row["status"], row["uniform_scale"],
                 row["chamfer"], row["signed_overall_median"],
                 row["elapsed_s"]), flush=True)

    # ---- CSV
    csv_path = os.path.join(a.out, "fdi46_regression.csv")
    with open(csv_path, "w", newline="", encoding="ascii") as fh:
        w = csv.DictWriter(fh, fieldnames=CSV_COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if r.get(k) is None else r.get(k))
                        for k in CSV_COLUMNS})

    # ---- 聚合（系统性偏差从带符号分区域读）
    ok = [r for r in rows if r["status"] == "OK"]
    summary = {
        "n_cases": len(rows), "n_ok": len(ok),
        "config": config.as_dict(),
        "template": template.summary(),
        "chamfer_median": _med(ok, "chamfer"),
        "hausdorff_median": _med(ok, "hausdorff"),
        "margin_p95_median": _med(ok, "margin_error_p95"),
        "signed_overall_median_of_medians": _med(ok, "signed_overall_median"),
        "signed_by_region_median_of_medians": {
            reg: _med(ok, "signed_" + reg + "_median")
            for reg in ("cervical", "mesial", "distal", "buccal", "lingual",
                        "occlusal")},
        "failed": [r["case_uuid"] for r in rows if r["status"] != "OK"],
        "elapsed_total_s": round(time.time() - t_all, 1),
    }
    json.dump(summary, open(os.path.join(a.out, "batch_summary.json"), "w",
                            encoding="ascii"), indent=1)
    print("\n=== 批量汇总 ===")
    print(json.dumps({k: v for k, v in summary.items()
                      if k not in ("config", "template")}, indent=1))
    print("\n写出", csv_path)
    return 0


def _med(rows, key):
    vals = [float(r[key]) for r in rows
            if r.get(key) not in (None, "")]
    return round(float(np.median(vals)), 4) if vals else None


if __name__ == "__main__":
    raise SystemExit(main())
