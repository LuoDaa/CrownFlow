#!/usr/bin/env python3
"""H.2.2 —— Template Width Selection：**同 FDI** 候选模板的宽度/颈缘/解剖审计。

用户 2026-10-09 规格：这一轮只回答一个问题 ——
**是不是当前选错了"宽度基型"**。不做自动替换、不改 H.2.1/K0/J.2/Contact Gate。

评分只由三块组成（不把 Contact 结果塞进选择目标）：
  A. Neighbor-space fit  : mesial/distal penetration p95 / MD half-width / required shrink
  B. Margin / cervical   : margin p95 / max
  C. Protected anatomy   : buccal / lingual / occlusal 尺寸偏差（相对当前模板）

用法::

    CROWNFLOW_TOOTH_LIBRARY_ROOT=<牙库根> \
    python tools/run_h2_2_template_audit.py --case-dir <病例包> --out <目录> \
        [--libraries 炼金王,洋溢-长夜同在,超神2]
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.geometry.crown_outer import boundary as bd                     # noqa: E402
from app.geometry.crown_outer.contact import NeighborPatch              # noqa: E402
from app.geometry.crown_outer.laplacian import (DeformInputs,           # noqa: E402
                                                solve_laplacian)
from app.geometry.crown_outer.models import CrownOuterConfig            # noqa: E402
from app.geometry.crown_outer.patient_frame import build_patient_frame  # noqa: E402
from app.geometry.crown_outer.placement import (                        # noqa: E402
    patient_margin_perimeter,
    place_template,
)
from app.geometry.crown_outer.proximal_fit import (                     # noqa: E402
    ProximalFitParams,
    measure_proximal_prior,
)
from app.geometry.crown_outer.template import load_crown_template       # noqa: E402
from app.importers.tooth_library.library import discover_libraries      # noqa: E402
from app.services.crown_outer_pipeline import (                         # noqa: E402
    _orient_patches,
    patient_frame_from_meta,
)


def _patch(path):
    import trimesh
    m = trimesh.load(str(path), force="mesh")
    return NeighborPatch(np.asarray(m.vertices, float),
                         np.asarray(m.faces, np.int64))


def _run_h1(template, margin, axis, frame, patches):
    """统一 placement（与 H.1 一致，不做任何各向异性）+ margin 拟合 → 网格与指标。"""
    placement = place_template(template, build_patient_frame(margin, axis),
                               patient_margin_perimeter(margin))
    cfg = CrownOuterConfig(body_scale_decoupling=True)
    placed = placement.apply_body_decoupled(
        template.vertices, template.margin_loop_indices,
        template.crown_height_mm)
    faces = np.asarray(template.faces, np.int64)
    loop_pts = placed[template.margin_loop_indices]
    corr = bd.align_rings(loop_pts, margin, cfg.resample_n)
    targets = bd.targets_for_loop_vertices(loop_pts, corr)
    loop_idx = template.margin_loop_indices
    root_idx = np.flatnonzero(template.root_vertex_mask)
    crown_idx = np.flatnonzero(template.crown_vertex_mask)
    z_thr = np.percentile(placed[crown_idx, 2],
                          cfg.occlusal_anchor_percentile)
    occ_idx = crown_idx[placed[crown_idx, 2] >= z_thr]
    h1_v, info = solve_laplacian(DeformInputs(
        vertices_placed=placed, faces=faces,
        margin_vertex_indices=loop_idx, margin_targets=targets,
        root_vertex_indices=root_idx, occlusal_vertex_indices=occ_idx,
        config=cfg))
    out_mask = ~template.root_vertex_mask[faces].any(axis=1)
    import trimesh
    mesh = trimesh.Trimesh(vertices=np.asarray(h1_v, float),
                           faces=faces[out_mask], process=False)
    return mesh, info, np.asarray(placed, float)


def _dims(mesh, frame):
    v = np.asarray(mesh.vertices, float)
    md = (v - frame.origin) @ np.asarray(frame.x, float)
    bl = (v - frame.origin) @ np.asarray(frame.y, float)
    z = (v - frame.origin) @ np.asarray(frame.z, float)
    return {"md_half_width_mm": float((md.max() - md.min()) / 2.0),
            "bl_half_width_mm": float((bl.max() - bl.min()) / 2.0),
            "crown_height_mm": float(z.max() - z.min()),
            "tall_thin_ratio": float((bl.max() - bl.min())
                                     / max(md.max() - md.min(), 1e-9))}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--libraries", default=None,
                    help="逗号分隔的库名；缺省 = 全部库")
    ap.add_argument("--clearance-mm", type=float, default=0.05)
    a = ap.parse_args()

    import trimesh
    os.makedirs(a.out, exist_ok=True)
    meta = json.load(open(os.path.join(a.case_dir, "case.json"),
                          encoding="ascii"))
    fdi = int(meta["fdi"])
    frame = patient_frame_from_meta(meta)
    margin = bd.ensure_closed_ring_order(
        np.asarray(meta["margin_points"], float))
    axis = np.asarray(meta["insertion_axis"], float)
    patches = {"mesial": _patch(os.path.join(a.case_dir,
                                             "mesial_neighbor_patch.stl")),
               "distal": _patch(os.path.join(a.case_dir,
                                             "distal_neighbor_patch.stl"))}
    pf = ProximalFitParams(target_clearance_mm=a.clearance_mm)

    libs = discover_libraries()
    names = ([x.strip() for x in a.libraries.split(",")] if a.libraries
             else [l.name for l in libs])
    print("FDI %d | 候选库：%s" % (fdi, names))

    rows = []
    for lib in libs:
        if lib.name not in names:
            continue
        try:
            template = load_crown_template(lib, fdi)
        except Exception as exc:                               # noqa: BLE001
            print("  %-12s 取模板失败：%s" % (lib.name, str(exc)[:90]))
            continue
        mesh, info, placed = _run_h1(template, margin, axis, frame, patches)
        _orient_patches(patches, np.asarray(placed, float))
        audit = measure_proximal_prior(mesh, frame, patches["mesial"],
                                       patches["distal"], pf)
        dims = _dims(mesh, frame)
        rec = {
            "library": lib.name,
            "n_vertices": int(len(template.vertices)),
            "n_margin_loop": int(len(template.margin_loop_indices)),
            "margin_source": getattr(template, "margin_source", None),
            "margin_p95_mm": info.get("margin_constraint_residual_max_mm"),
            **dims,
            "mesial_pen_p95_mm": audit["mesial"]["penetration_mm"]["p95"],
            "mesial_pen_max_mm": audit["mesial"]["penetration_mm"]["max"],
            "distal_pen_p95_mm": audit["distal"]["penetration_mm"]["p95"],
            "distal_pen_max_mm": audit["distal"]["penetration_mm"]["max"],
            "mesial_shrink_p95_mm": audit["mesial"].get(
                "required_shrink_mm", {}).get("p95"),
            "distal_shrink_p95_mm": audit["distal"].get(
                "required_shrink_mm", {}).get("p95"),
        }
        rows.append(rec)
        mesh.export(os.path.join(a.out, "h1_%s.stl" % lib.name))
        print("  %-12s MD½=%.3f BL½=%.3f H=%.3f | pen p95 %.3f/%.3f | "
              "margin %.4f | loop %d"
              % (lib.name, rec["md_half_width_mm"], rec["bl_half_width_mm"],
                 rec["crown_height_mm"], rec["mesial_pen_p95_mm"],
                 rec["distal_pen_p95_mm"], rec["margin_p95_mm"] or 0.0,
                 rec["n_margin_loop"]))

    if not rows:
        print("没有可用候选（同 FDI 无候选模板）→ 按用户规格转入'参数化宽度重塑'后备路线")
        return 1

    base = next((r for r in rows if r["library"] ==
                 "炼金王"), rows[0])
    worst_base = max(base["mesial_pen_p95_mm"], base["distal_pen_p95_mm"])
    print("\n== H.2.2 Selection Audit（同 FDI；参考=当前库 %s）==" % base["library"])
    print("| template | MD½ mm | pen p95 M/D | margin p95 | BL½ dev | H dev | "
          "occl. dev |")
    print("|---|---:|---|---:|---:|---:|---:|")
    for r in rows:
        bl_dev = r["bl_half_width_mm"] - base["bl_half_width_mm"]
        h_dev = r["crown_height_mm"] - base["crown_height_mm"]
        print("| %s | %.3f | %.3f / %.3f | %.4f | %+.3f | %+.3f | %+.3f |"
              % (r["library"], r["md_half_width_mm"], r["mesial_pen_p95_mm"],
                 r["distal_pen_p95_mm"], r["margin_p95_mm"] or 0.0,
                 bl_dev, h_dev, h_dev))
    # 排序：先满足 margin 不退化（≤ 当前 + 0.01mm），再按最差侧穿透
    ok = [r for r in rows if (r["margin_p95_mm"] or 0.0)
          <= (base["margin_p95_mm"] or 0.0) + 0.01]
    best = min(ok or rows,
               key=lambda r: max(r["mesial_pen_p95_mm"], r["distal_pen_p95_mm"]))
    worst_best = max(best["mesial_pen_p95_mm"], best["distal_pen_p95_mm"])
    print("\n参考（当前）最差侧穿透 p95 = %.4fmm；最佳候选 %s = %.4fmm"
          % (worst_base, best["library"], worst_best))
    if worst_best < worst_base - 0.3:
        print("⇒ 有候选把系统性冲突**明显压小**（≥0.3mm）→ 进入 H.2.1 → J.2 → K0 → Contact")
        verdict = "CHALLENGER_FOUND"
    elif worst_best < worst_base - 0.1:
        print("⇒ 改善有限（0.1~0.3mm）→ 记为弱候选，需与参数化宽度重塑比较后再定")
        verdict = "WEAK"
    else:
        print("⇒ 所有同 FDI 模板都仍在 ~1mm 量级 → 换模板不是主解，"
              "按用户规格直接进入 B：同 FDI 参数化宽度重塑")
        verdict = "NO_TEMPLATE_SOLUTION"
    with open(os.path.join(a.out, "h2_2_selection_audit.json"), "w",
              encoding="utf-8") as fh:
        json.dump({"fdi": fdi, "rows": rows, "reference": base["library"],
                   "worst_side_p95_base_mm": worst_base,
                   "worst_side_p95_best_mm": worst_best,
                   "challenger": best["library"], "verdict": verdict},
                  fh, indent=1, ensure_ascii=False)
    print("\n写出 %s" % a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
