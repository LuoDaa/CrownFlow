#!/usr/bin/env python3
"""H.2.1 真实例收口报表：**三列网格各自独立重算**（不混网格、不混口径）。

用户 2026-10-09 §1–§4：
  * baseline = H.1（合法基线）；candidate = 本轮候选（可能被 Gate 拒绝）；
    accepted = 实际交付（首轮被拒时必须**逐位等于 H.1**）；
  * 每列的全部指标都从**该列自己的网格**重算；
  * 冻结域：audit_ids（全部冠表面顶点）/ editable_ids（未锁定的近远中候选），
    在 H.1 基线上生成；`audit − editable` 按排除原因分账（含穿透面积）；
  * 同一候选上同时记录：固定目标残差、**重查询**后的穿透/欠缺、
    最近面切换比例、冻结的 normal_sign；
  * 穿透与欠缺分开：`penetration = max(0, −signed)`、
    `deficit = max(0, clearance − signed)`。

用法::

    CROWNFLOW_TOOTH_LIBRARY_ROOT=<牙库根> \
    python tools/run_h2_1_case.py --case-dir <病例包> --out <目录> [--weight 300]
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
from app.geometry.crown_outer.proximal_constrained import (             # noqa: E402
    NeighborClearanceQuery,
    _vertex_area,
    build_active_constraints,
    edge_length_change_p95,
    exclude_reasons,
    proximal_candidates,
    solve_proximal_constrained,
)
from app.geometry.crown_outer.proximal_fit import (                     # noqa: E402
    ProximalFitParams,
    measure_proximal_prior,
)
from app.services.crown_outer_pipeline import (                         # noqa: E402
    _h1_template,
    _orient_patches,
    patient_frame_from_meta,
)


def _patch(path):
    import trimesh
    m = trimesh.load(str(path), force="mesh")
    return NeighborPatch(np.asarray(m.vertices, float),
                         np.asarray(m.faces, np.int64))


def _geometry(mesh) -> dict:
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
                g["metrics"].get("winding_inconsistent_edges") or 0)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--weight", type=float, default=300.0)
    ap.add_argument("--max-iterations", type=int, default=5)
    ap.add_argument("--protection-rings", type=int, default=4)
    ap.add_argument("--protection-decay", type=float, default=0.5)
    a = ap.parse_args()

    import trimesh
    os.makedirs(a.out, exist_ok=True)
    meta = json.load(open(os.path.join(a.case_dir, "case.json"),
                          encoding="ascii"))
    frame = patient_frame_from_meta(meta)
    margin = bd.ensure_closed_ring_order(
        np.asarray(meta["margin_points"], float))
    axis = np.asarray(meta["insertion_axis"], float)
    patches = {"mesial": _patch(os.path.join(a.case_dir,
                                             "mesial_neighbor_patch.stl")),
               "distal": _patch(os.path.join(a.case_dir,
                                             "distal_neighbor_patch.stl"))}
    pf = ProximalFitParams()
    template, lib = _h1_template({}, int(meta["fdi"]))
    faces = np.asarray(template.faces, np.int64)
    print("模板库 %s（FDI %d）| 顶点 %d" % (lib, meta["fdi"], len(template.vertices)))

    # ---- H.1：摆位 + margin 拟合（与 generate_crown_outer_v0 同一步骤）
    placement = place_template(template, build_patient_frame(margin, axis),
                               patient_margin_perimeter(margin))
    cfg = CrownOuterConfig(body_scale_decoupling=True)
    placed = placement.apply_body_decoupled(
        template.vertices, template.margin_loop_indices,
        template.crown_height_mm)
    _orient_patches(patches, np.asarray(placed, float))   # 与引擎同口径
    loop_pts = placed[template.margin_loop_indices]
    corr = bd.align_rings(loop_pts, margin, cfg.resample_n)
    targets = bd.targets_for_loop_vertices(loop_pts, corr)
    loop_idx = template.margin_loop_indices
    root_idx = np.flatnonzero(template.root_vertex_mask)
    crown_idx = np.flatnonzero(template.crown_vertex_mask)
    z_thr = np.percentile(placed[crown_idx, 2],
                          cfg.occlusal_anchor_percentile)
    occ_idx = crown_idx[placed[crown_idx, 2] >= z_thr]
    h1_v, h1_info = solve_laplacian(DeformInputs(
        vertices_placed=placed, faces=faces,
        margin_vertex_indices=loop_idx, margin_targets=targets,
        root_vertex_indices=root_idx, occlusal_vertex_indices=occ_idx,
        config=cfg))
    out_mask = ~template.root_vertex_mask[faces].any(axis=1)
    faces_out = faces[out_mask]

    def mk(vv):
        return trimesh.Trimesh(vertices=np.asarray(vv, float),
                               faces=faces_out, process=False)

    def col(vv):
        m = mk(vv)
        au = measure_proximal_prior(m, frame, patches["mesial"],
                                    patches["distal"], pf)
        lat = (np.asarray(vv, float) - frame.origin) @ np.asarray(frame.y, float)
        d = np.linalg.norm(np.asarray(vv, float) - np.asarray(h1_v, float),
                           axis=1)
        return {"mesh": m,
                "pen": {s: au[s]["penetration_mm"] for s in ("mesial", "distal")},
                "geometry": _geometry(m),
                "bl_drag_p95": float(np.quantile(d[np.abs(lat) > 0.5], .95)),
                "edge": edge_length_change_p95(h1_v, vv, faces)}

    h1 = mk(h1_v)
    c1 = col(h1_v)
    print("\n== H.1 ==")
    for s in ("mesial", "distal"):
        print("  %-7s pen p95=%.4f max=%.4f (n=%d)"
              % (s, c1["pen"][s]["p95"], c1["pen"][s]["max"], c1["pen"][s]["n"]))
    print("  margin resid max=%.5f  geometry=%s"
          % (h1_info["margin_constraint_residual_max_mm"],
             json.dumps(c1["geometry"], ensure_ascii=False)))

    # ---- 冻结域（H.1 基线上生成一次）
    qs = {s: NeighborClearanceQuery(patches[s]) for s in patches}
    frozen = {s: float(patches[s].normal_sign) for s in patches}
    idx0, tgt0, stats0, per_side0, _w0 = build_active_constraints(
        h1_v, faces, frame, patches, loop_idx, pf, qs=qs, root_idx=root_idx,
        frozen_sign=frozen)
    cand0, _z0 = proximal_candidates(h1_v, faces, frame, loop_idx, pf,
                                     root_idx=root_idx)
    audit_ids = np.arange(len(h1_v))
    editable_ids = np.flatnonzero(cand0["mesial"] | cand0["distal"])
    areas = _vertex_area(h1_v, faces)
    print("\n== 冻结域 ==")
    print("  audit_ids=%d（全部冠表面顶点）| editable_ids=%d（未锁定的近远中候选）"
          % (len(audit_ids), len(editable_ids)))
    for s in ("mesial", "distal"):
        st = stats0[s]
        print("  %-7s 候选=%d active=%d | penetration p95/max %.4f/%.4f "
              "(面积 %.2f mm²) | deficit p95/max %.4f/%.4f"
              % (s, st["n_candidate"], st["n_active"], st["penetration_p95_mm"],
                 st["penetration_max_mm"], st["penetration_area_mm2"],
                 st["deficit_p95_mm"], st["deficit_max_mm"]))
    reasons = exclude_reasons(h1_v, faces, frame, loop_idx, pf,
                              root_idx=root_idx)
    print("  排除原因分账（audit − editable；n / 穿透面积 mm² / 最大穿透）：")
    for k, mask in reasons.items():
        ids = np.flatnonzero(mask)
        if not len(ids):
            print("    %-16s n=0" % k)
            continue
        pen_max, area = 0.0, 0.0
        for s in ("mesial", "distal"):
            sc, _q, _n = qs[s].signed_clearance(h1_v[ids])
            pen = np.maximum(0.0, -sc)
            pen_max = max(pen_max, float(pen.max()) if len(pen) else 0.0)
            area += float(areas[ids][pen > 0].sum())
        print("    %-16s n=%-5d 穿透面积 %.3f  最大穿透 %.4f"
              % (k, len(ids), area, pen_max))

    # ---- H.2.1（单权重，不 sweep）
    cfg.weight_proximal_clearance = float(a.weight)
    out = solve_proximal_constrained(
        placed, faces, loop_idx, targets, frame, patches, root_idx=root_idx,
        occ_idx=occ_idx, config=cfg, params=pf,
        max_iterations=a.max_iterations, reference_v=h1_v,
        protection_rings=a.protection_rings,
        protection_decay=a.protection_decay)
    cand_v, acc_v = out["last_candidate_v"], out["accepted_v"]
    if cand_v is None:
        cand_v = acc_v
    c2, c3 = col(cand_v), col(acc_v)

    print("\n== 三列独立重算（W_prox=%.0f，未 sweep）==" % a.weight)
    print("| 指标 | H.1 | candidate | accepted |")
    print("|---|---:|---:|---:|")
    for s in ("mesial", "distal"):
        print("| %s pen p95 | %.4f | %.4f | %.4f |"
              % (s, c1["pen"][s]["p95"], c2["pen"][s]["p95"], c3["pen"][s]["p95"]))
        print("| %s pen max | %.4f | %.4f | %.4f |"
              % (s, c1["pen"][s]["max"], c2["pen"][s]["max"], c3["pen"][s]["max"]))
    for tag in ("self_intersections", "nonmanifold_edges",
                "winding_inconsistent_edges"):
        print("| %s | %d | %d | %d |"
              % (tag, c1["geometry"][tag], c2["geometry"][tag],
                 c3["geometry"][tag]))
    print("| buccal/lingual drag p95 | 0 | %.4f | %.4f |"
          % (c2["bl_drag_p95"], c3["bl_drag_p95"]))
    print("| edgeΔ p95 | 0 | %.4f | %.4f |" % (c2["edge"], c3["edge"]))
    print("| accepted == H.1 | — | — | **%s** |"
          % ("是（逐位相同）" if np.allclose(acc_v, h1_v) else "否"))

    print("\n== 每轮 α 回溯（目标/权重/冻结域/Gate 均未改）==")
    print("| it | active | α | deficit p95 before→after | deficit max before→after | "
          "new自交 | 累计翻面 | 锁定位移 | accepted |")
    print("|---|---:|---:|---|---|---:|---:|---|---|")
    for h in out["history"]:
        if h.get("stop"):
            print("| %d | — | — | %s | | | | | 停止 |" % (h["iteration"], h["stop"]))
            continue
        for t in h.get("alpha_trials", []):
            print("| %d | %d | %s | %.4f→%.4f | %.4f→%.4f | %d | %d | %.1e | %s |"
                  % (h["iteration"], h["n_active"], t["alpha"],
                     t["deficit_p95_before"], t["deficit_p95_after"],
                     t["deficit_max_before"], t["deficit_max_after"],
                     t["new_self_intersections"], t["cumulative_flips_severe"],
                     t["locked_displacement_mm"],
                     "✅" if t["accepted"] else ("✗ " + str(t.get("reject", ""))[:48])))
        for t in h.get("alpha_trials", []):
            if t.get("selfint_evidence"):
                print("    α=%s severe 自交对证据：%s"
                      % (t["alpha"], json.dumps(t["selfint_evidence"],
                                                ensure_ascii=False)))
        sd = h.get("set_drift") or {}
        if sd:
            print("    it%d 集合漂移：prev=%s cur=%s 新增=%s 删除=%s 相同=%s"
                  % (h["iteration"], sd.get("n_prev"), sd.get("n_cur"),
                     sd.get("added"), sd.get("removed"), sd.get("same")))
        for t in h.get("alpha_trials", []):
            rd = t.get("region_displacement") or {}
            if rd:
                print("    α=%-7s 分区位移 p95/max：%s | hard_regions_ok=%s "
                      "buccal/lingual≤%.2f=%s"
                      % (t["alpha"],
                         " ".join("%s %.3f/%.3f" % (k, v.get("p95_mm", 0),
                                                    v.get("max_mm", 0))
                                  for k, v in rd.items() if v.get("n")),
                         t.get("hard_regions_ok"),
                         pf.buccal_lingual_p95_allow_mm,
                         t.get("buccal_lingual_within_allow")))
                if t.get("per_side_deficit_delta"):
                    print("        逐点 deficit 变化（固定集）：%s"
                          % json.dumps(t["per_side_deficit_delta"],
                                       ensure_ascii=False))
        if h.get("blocked"):
            print("| %d | %d | — | | | | | | **阻断**：保留上一次 accepted |"
                  % (h["iteration"], h["n_active"]))

    # ---- 最小回归检查（用户 2026-10-09 要求）
    chk = []
    hist = [h for h in out["history"] if h.get("alpha_trials")]
    if hist:
        t0 = hist[0]["alpha_trials"]
        full_rejected = (t0 and not t0[0]["accepted"])
        smaller_ok = any(t["accepted"] and t["alpha"] < t0[0]["alpha"] for t in t0)
        chk.append(("全步被拒 → 小步可接受",
                    bool(full_rejected and smaller_ok) if full_rejected
                    else "n/a（全步未被拒）"))
    last = out["history"][-1] if out["history"] else {}
    if last.get("blocked") or last.get("stop"):
        chk.append(("所有步长被拒/停止 → 输出逐位保留上次 accepted",
                    bool(np.allclose(out["accepted_v"], h1_v))
                    if not any(h.get("alpha") for h in out["history"])
                    else "n/a（有步被接受）"))
    for name, res in chk:
        print("  [回归] %-44s %s" % (name, res))

    rec = (out["history"] or [{}])[0]
    print("\n候选诊断（it0）：fixed target resid p95=%.5f | normal_sign=%s | gate.ok=%s"
          % (rec.get("fixed_target_residual_mm", {}).get("p95", float("nan")),
             json.dumps(out["frozen_sign"]),
             (rec.get("gate") or {}).get("ok")))
    for s, dd in (rec.get("requery") or {}).items():
        print("  %-7s n=%-4d 穿透 p95 %.4f→%.4f | deficit p95 %.4f→%.4f | "
              "最近面切换 %.1f%%"
              % (s, dd["n_constrained"], dd["penetration_before_p95_mm"],
                 dd["penetration_after_p95_mm"], dd["deficit_before_p95_mm"],
                 dd["deficit_after_p95_mm"],
                 100.0 * dd["closest_face_changed_ratio"]))
    if (rec.get("gate") or {}).get("violations"):
        print("  gate violations: %s"
              % json.dumps(rec["gate"]["violations"], ensure_ascii=False))

    mk(h1_v).export(os.path.join(a.out, "anatomy_h1.stl"))
    mk(cand_v).export(os.path.join(a.out, "anatomy_h2_1_candidate.stl"))
    mk(acc_v).export(os.path.join(a.out, "anatomy_h2_1_accepted.stl"))
    with open(os.path.join(a.out, "h2_1_report.json"), "w",
              encoding="utf-8") as fh:
        json.dump({"h1": {k: v for k, v in c1.items() if k != "mesh"},
                   "candidate": {k: v for k, v in c2.items() if k != "mesh"},
                   "accepted": {k: v for k, v in c3.items() if k != "mesh"},
                   "frozen_domain": {"n_audit": int(len(audit_ids)),
                                     "n_editable": int(len(editable_ids)),
                                     "stats": stats0,
                                     "frozen_sign": out["frozen_sign"]},
                   "history": out["history"],
                   "config": {"weight_proximal_clearance": a.weight,
                              "clearance_mm": pf.target_clearance_mm}},
                  fh, indent=1, ensure_ascii=False)
    print("\n写出 %s" % a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
