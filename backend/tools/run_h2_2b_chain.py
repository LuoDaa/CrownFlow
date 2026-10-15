#!/usr/bin/env python3
"""H.2.2-B 全链：同 FDI 参数化宽度重塑（MD width field）→ H.2.1 → J.2 → K0 → Contact。

用户 2026-10-09：一次做完，每层都出账，最后看 Contact Gate。
顺序按用户纠正：**template → H.1(+宽度场) → H.2.1 → J.2 → K0 → Contact**
（J.2 在 K0 之前：minimum-thickness envelope 可能把 proximal 再顶出去）。

用法::

    CROWNFLOW_TOOTH_LIBRARY_ROOT=<牙库根> \
    CROWNFLOW_CROWN_CASES_DIR=<病例根> \
    python tools/run_h2_2b_chain.py --case-dir <病例包> --out <目录>
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
from app.geometry.crown_outer.md_width_field import (                   # noqa: E402
    MDWidthFieldParams,
    measure_md_width_field,
)
from app.geometry.crown_outer.models import CrownOuterConfig            # noqa: E402
from app.geometry.crown_outer.patient_frame import build_patient_frame  # noqa: E402
from app.geometry.crown_outer.placement import (                        # noqa: E402
    patient_margin_perimeter,
    place_template,
)
from app.geometry.crown_outer.proximal_constrained import (             # noqa: E402
    NeighborClearanceQuery,
    deficit_on_domain,
    frozen_editable_domain,
    solve_proximal_constrained,
)
from app.geometry.crown_outer.proximal_fit import (                     # noqa: E402
    ProximalFitParams,
    measure_proximal_prior,
)
from app.services.crown_outer_pipeline import (                         # noqa: E402
    DEFAULT_PARAMS,
    CrownBundle,
    _h1_template,
    _orient_patches,
    patient_frame_from_meta,
    stage_contact,
    stage_occlusion,
    stage_thickness_audit,
)


def _patch(path):
    import trimesh
    m = trimesh.load(str(path), force="mesh")
    return NeighborPatch(np.asarray(m.vertices, float),
                         np.asarray(m.faces, np.int64))


def _mk(vv, faces_out):
    import trimesh
    return trimesh.Trimesh(vertices=np.asarray(vv, float),
                           faces=faces_out, process=False)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--weight", type=float, default=300.0)
    ap.add_argument("--max-iterations", type=int, default=5)
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    meta = json.load(open(os.path.join(a.case_dir, "case.json"),
                          encoding="ascii"))
    # ---------- 前置 QC：邻牙 patch 作用域（用病例自己的 GT 冠作对照）
    # 用户纪律：生成前能发现的问题，不要跑完 40 分钟才发现。
    # 2026-10-09 实测：FDI46 的 patch 是半径 9mm 整段牙弓 patch，把目标牙自身/颈缘
    # 也包含进去 —— 临床已交付的 GT 冠在它上面读出的穿透 p95 ≈ 1.1mm（> 0.2 允许），
    # 因此基于该 patch 的一切下游数字都不可信。
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from audit_neighbor_patch_scope import audit as _patch_scope_audit
    patch_qc = _patch_scope_audit(a.case_dir)
    print("patch 作用域 QC：status=%s | GT 穿透 p95=%s（允许 %.2f）"
          % (patch_qc["status"],
             ("%.3f" % patch_qc["gt_control"]["worst_penetration_p95_mm"])
             if patch_qc.get("gt_control") else "—",
             patch_qc.get("gt_control", {}).get("allow_mm", 0.2)))
    if patch_qc["status"] == "PATCH_SCOPE_CONTAMINATED":
        for f in patch_qc["findings"]:
            print("  [%s] %s" % (f["code"], f["detail"]))
        blocked = {"case_uuid": os.path.basename(os.path.normpath(a.case_dir)),
                   "patch_qc": patch_qc, "stages": [],
                   "blocked": True, "blocked_at": "PATCH_SCOPE_QC",
                   "blocked_reason": "PATCH_SCOPE_CONTAMINATED"}
        with open(os.path.join(a.out, "h2_2b_chain.json"), "w",
                  encoding="utf-8") as fh:
            json.dump(blocked, fh, indent=1, ensure_ascii=False)
        print("\n→ 前置 QC 不通过：**PATCH_SCOPE_CONTAMINATED** —— 邻牙 patch 不能"
              "作为接触参照，下游链不运行（数字会全是污染伪值）。")
        print("写出 %s（含 patch_qc 证据）" % a.out)
        return 2
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
    print("模板 %s（FDI %d）" % (lib, meta["fdi"]))

    # ---------- H.1（摆位 + margin 拟合）
    placement = place_template(template, build_patient_frame(margin, axis),
                               patient_margin_perimeter(margin))
    cfg = CrownOuterConfig(body_scale_decoupling=True)
    placed = placement.apply_body_decoupled(
        template.vertices, template.margin_loop_indices,
        template.crown_height_mm)
    _orient_patches(patches, np.asarray(placed, float))
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
    faces_out = faces[~template.root_vertex_mask[faces].any(axis=1)]

    def report(tag, vv):
        m = _mk(vv, faces_out)
        au = measure_proximal_prior(m, frame, patches["mesial"],
                                    patches["distal"], pf)
        return {"tag": tag,
                "mesial_p95": au["mesial"]["penetration_mm"]["p95"],
                "distal_p95": au["distal"]["penetration_mm"]["p95"],
                "mesial_max": au["mesial"]["penetration_mm"]["max"],
                "distal_max": au["distal"]["penetration_mm"]["max"],
                "mesh": m}

    st = [report("H.1", h1_v)]
    print("H.1        pen p95 %.4f/%.4f" % (st[-1]["mesial_p95"],
                                            st[-1]["distal_p95"]))

    # ---------- 冻结域（H.1 基线）
    qs = {s: NeighborClearanceQuery(patches[s]) for s in patches}
    frozen = {s: float(patches[s].normal_sign) for s in patches}
    dom = frozen_editable_domain(h1_v, faces, frame, patches, loop_idx, pf,
                                 root_idx=root_idx, qs=qs, frozen_sign=frozen)
    d0 = deficit_on_domain(h1_v, dom, pf.target_clearance_mm)
    print("冻结域 editable=%d protection=%d | deficit p95 %.4f/%.4f"
          % (len(dom["editable_ids"]), len(dom["protection_ids"]),
             d0["mesial"]["deficit_p95"], d0["distal"]["deficit_p95"]))

    # ---------- B：MD 宽度场（只测；目标在同一次求解里施加）
    wf = measure_md_width_field(h1_v, faces, frame, loop_idx, dom, pf,
                                MDWidthFieldParams())
    print("MD 宽度场 s_md(t)：%s" % json.dumps(
        [(round(t, 2), round(s, 3)) for t, s in wf.s_of_t],
        ensure_ascii=False))
    for b_ in wf.band_table:
        print("  band t=%s n=%s 半宽=%s deficit=%s s_md=%s"
              % (b_.get("t"), b_.get("n"), b_.get("md_half_width_mm"),
                 b_.get("deficit_p95_mm"), b_.get("s_md")))

    # ---------- B：宽度重塑（独立成层，自己的 α 回溯 + 几何门）
    out_b = solve_proximal_constrained(
        placed, faces, loop_idx, targets, frame, patches, root_idx=root_idx,
        occ_idx=occ_idx, config=cfg, params=pf,
        max_iterations=2, reference_v=h1_v,
        width_field={"result": wf, "params": MDWidthFieldParams()},
        width_field_only=True)
    b_v = out_b["accepted_v"]
    st.append(report("B(width)", b_v))
    print("B(width)   pen p95 %.4f/%.4f (α=%s)"
          % (st[-1]["mesial_p95"], st[-1]["distal_p95"],
             [h.get("alpha") for h in out_b["history"]]))
    _mk(b_v, faces_out).export(os.path.join(a.out, "anatomy_b.stl"))

    # ---------- H.2.1（在 B 之后做局部包络收尾）
    out = solve_proximal_constrained(
        placed, faces, loop_idx, targets, frame, patches, root_idx=root_idx,
        occ_idx=occ_idx, config=cfg, params=pf,
        max_iterations=a.max_iterations, reference_v=b_v)
    st.append(report("B+H.2.1", out["accepted_v"]))
    print("B+H.2.1    pen p95 %.4f/%.4f (α=%s)"
          % (st[-1]["mesial_p95"], st[-1]["distal_p95"],
             [h.get("alpha") for h in out["history"]]))
    _mk(out["accepted_v"], faces_out).export(os.path.join(a.out,
                                                          "anatomy_b_h21.stl"))

    # ---------- J.2（冻结外表面：anatomy ∪ MinThickness envelope）
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import subprocess
    j2_out = os.path.join(a.out, "j2")
    os.makedirs(j2_out, exist_ok=True)
    r = subprocess.run(
        [sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                      "run_boolean_outer_case.py"),
         "--case-dir", a.case_dir,
         "--anatomy", os.path.join(a.out, "anatomy_b_h21.stl"),
         "--intaglio", os.path.join(a.case_dir, "_intaglio_for_chain.stl"),
         "--out", j2_out],
        capture_output=True, text=True)
    print("\n== J.2 ==\n%s" % (r.stdout[-2500:] if r.stdout else r.stderr[-1200:]))

    # ---------- K0 finishing on the J.2 output
    print("\n== K0 finishing ==")
    k0_out = os.path.join(a.out, "k0")
    os.makedirs(k0_out, exist_ok=True)
    anatomy_j2 = os.path.join(j2_out, "outer_j2b.stl")
    if os.path.exists(anatomy_j2):
        st_j2 = report("J.2", _load_v(anatomy_j2))
        print("J.2        pen p95 %.4f/%.4f" % (st_j2["mesial_p95"],
                                                st_j2["distal_p95"]))
        st.append(st_j2)
    r2 = subprocess.run(
        [sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                      "run_proximal_fit_case.py"),
         "--case-dir", a.case_dir, "--anatomy", anatomy_j2,
         "--out", k0_out, "--fit"],
        capture_output=True, text=True)
    print(r2.stdout[-1800:] if r2.stdout else r2.stderr[-1000:])
    k0_stl = os.path.join(k0_out, "anatomy_prior_fitted.stl")
    if os.path.exists(k0_stl) and os.path.getsize(k0_stl) > 1000:
        st_k0 = report("K0", _load_v(k0_stl))
        print("K0         pen p95 %.4f/%.4f" % (st_k0["mesial_p95"],
                                                st_k0["distal_p95"]))
        st.append(st_k0)

    print("\n== 逐层 proximal p95 对照（mesial/distal）==")
    print("| 层 | mesial p95 | distal p95 |")
    print("|---|---:|---:|")
    for r_ in st:
        print("| %s | %.4f | %.4f |" % (r_["tag"], r_["mesial_p95"],
                                        r_["distal_p95"]))

    # ---------- Contact（Phase K 硬化链：Contact v2 → I.1 → I.2 + 硬几何门）
    # 用户锁定顺序的最后一段：contact 消费 K0 之后的外表面（没有则 J.2 输出）。
    print("\n== Contact（Phase K 硬化，Gate 不变）==")
    from pathlib import Path
    outer_used = k0_stl if (os.path.exists(k0_stl)
                            and os.path.getsize(k0_stl) > 1000) else anatomy_j2
    contact_out = os.path.join(a.out, "contact")
    os.makedirs(contact_out, exist_ok=True)
    ant_path = os.path.join(a.case_dir, "antagonist_patch.stl")
    bundle = CrownBundle(
        case_uuid=os.path.basename(os.path.normpath(a.case_dir)),
        case_dir=Path(a.case_dir), margin_points=margin, insertion_axis=axis,
        fdi=int(meta["fdi"]),
        mesial_path=Path(a.case_dir) / "mesial_neighbor_patch.stl",
        distal_path=Path(a.case_dir) / "distal_neighbor_patch.stl",
        antagonist_path=(Path(ant_path) if os.path.exists(ant_path) else None),
        meta=meta)
    res_c = stage_contact(bundle, Path(contact_out), dict(DEFAULT_PARAMS),
                          prev={"outer_h1": {"artifact": outer_used}})
    g = res_c["gate"]
    print("Contact 输入 = %s" % os.path.basename(outer_used))
    print("Gate  status=%s | geometry_clean=%s | 自交=%s 非流形=%s 翻面=%s"
          % (g["status"], g["geometry_clean"], g["self_intersections"],
             g["nonmanifold_edges"], g["flipped_faces"]))
    print("      margin_disp=%.4fmm occlusal_locked_disp=%.4fmm | "
          "editable 残留 p95=%.4fmm (allow %.4f) | within=%s"
          % (g["margin_displacement_mm"],
             g["occlusal_locked_displacement_mm"],
             g["residual_editable_p95_mm"], g["residual_allow_mm"],
             g["within_allowance"]))
    if g["status"] != "PASS":
        print("→ 下游被 Gate 阻断（%s）：Contact 之后再无模块运行。" % g["status"])
    else:
        # ---------- Contact PASS → 继续 E2E（用户锁序：Occlusion → Thickness Audit）
        print("\n== Occlusion v1（只动咬合面；颈缘/邻接核心构造锁死）==")
        o = stage_occlusion(bundle, Path(contact_out), dict(DEFAULT_PARAMS),
                            prev={"contact": {"artifact": res_c["artifact"]}})
        print("Occlusion  status=%s | margin_disp=%.4fmm | proximal_core_disp=%.4fmm"
              " | 翻面=%d" % (o["status"], o["margin_max_displacement_mm"],
                              o["proximal_core_max_displacement_mm"],
                              o["n_flipped_faces"]))
        print("\n== Thickness Audit（只测不修）==")
        t = stage_thickness_audit(
            bundle, Path(contact_out), dict(DEFAULT_PARAMS),
            prev={"occlusion": {"artifact": o["artifact"]},
                  "intaglio": {"artifact": os.path.join(
                      a.case_dir, "_intaglio_for_chain.stl")},
                  "outer_h1": {"artifact": outer_used}})
        print("Thickness  status=%s | artifact=%s"
              % (t.get("status"), t.get("artifact")))
        print("→ Thickness Repair / Final Stitch **尚未实现**（路线图后续件）："
              "当前 E2E 可交付到 Thickness Audit 为止。")
        with open(os.path.join(a.out, "e2e_post_contact.json"), "w",
                  encoding="utf-8") as fh:
            json.dump({"occlusion": {k: v for k, v in o.items()
                                     if k not in ("history",)},
                       "thickness": {"status": t.get("status"),
                                     "artifact": t.get("artifact")}},
                      fh, indent=1, ensure_ascii=False)
    with open(os.path.join(a.out, "h2_2b_chain.json"), "w",
              encoding="utf-8") as fh:
        json.dump({"case_uuid": os.path.basename(os.path.normpath(a.case_dir)),
                   "stages": [{k: v for k, v in r_.items() if k != "mesh"}
                              for r_ in st],
                   "width_field": wf.as_dict(),
                   "domain": {"n_editable": len(dom["editable_ids"]),
                              "n_protection": len(dom["protection_ids"]),
                              "n_ambiguous": dom.get("n_ambiguous_side"),
                              "deficit_before": d0},
                   "contact": {"gate": g,
                               "outer_used": outer_used,
                               "artifact": res_c["artifact"],
                               "hardened_artifact": res_c["hardened_artifact"],
                               "hardening": res_c["hardening"]},
                   "patch_qc": patch_qc},
                  fh, indent=1, ensure_ascii=False)
    if g["status"] != "PASS":
        # 用户决策 ①：Contact Gate 不通过 → 自动出正式评审工件（不改设计参数、
        # 不放宽 Gate），供前端告知技师"为什么不能自动生成"。
        try:
            from emit_case_review import build_review
            review = build_review(a.out)
            rp = os.path.join(a.out, "case_review.json")
            with open(rp, "w", encoding="utf-8") as fh:
                json.dump(review, fh, indent=1, ensure_ascii=False)
            print("已出评审工件：%s（status=%s / reason=%s）"
                  % (rp, review["status"], review["reason_code"]))
        except Exception as exc:                       # noqa: BLE001
            print("评审工件生成失败（如实上报，不静默）：%r" % (exc,))
    print("\n写出 %s" % a.out)
    return 0


def _load_v(path):
    import trimesh
    return np.asarray(trimesh.load(path, force="mesh").vertices, float)


if __name__ == "__main__":
    raise SystemExit(main())
