"""H.2.2-B 单层探针：只跑宽度重塑层（width_field_only），完整打印 α 回溯证据。

用法：python tools/run_h2_2b_b_probe.py [case_dir]
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import trimesh

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.geometry.crown_outer import boundary as bd
from app.geometry.crown_outer.contact import NeighborPatch
from app.geometry.crown_outer.laplacian import DeformInputs, solve_laplacian
from app.geometry.crown_outer.md_width_field import (MDWidthFieldParams,
                                                     measure_md_width_field)
from app.geometry.crown_outer.models import CrownOuterConfig
from app.geometry.crown_outer.patient_frame import build_patient_frame
from app.geometry.crown_outer.placement import (patient_margin_perimeter,
                                                place_template)
from app.geometry.crown_outer.proximal_constrained import (
    NeighborClearanceQuery, frozen_editable_domain, solve_proximal_constrained)
from app.geometry.crown_outer.proximal_fit import ProximalFitParams
from app.services.crown_outer_pipeline import (_h1_template, _orient_patches,
                                               patient_frame_from_meta)

CASE = (sys.argv[1] if len(sys.argv) > 1
        else r"D:\code\牙冠自动生成\数据\crown_outer_cases\cf_b1b73bae221211b1")


def main():
    meta = json.load(open(os.path.join(CASE, "case.json"), encoding="ascii"))
    frame = patient_frame_from_meta(meta)
    margin = bd.ensure_closed_ring_order(np.asarray(meta["margin_points"], float))
    axis = np.asarray(meta["insertion_axis"], float)

    def pat(name):
        m = trimesh.load(os.path.join(CASE, name), force="mesh")
        return NeighborPatch(np.asarray(m.vertices, float),
                             np.asarray(m.faces, np.int64))

    patches = {"mesial": pat("mesial_neighbor_patch.stl"),
               "distal": pat("distal_neighbor_patch.stl")}
    pf = ProximalFitParams()
    tpl, _lib = _h1_template({}, int(meta["fdi"]))
    faces = np.asarray(tpl.faces, np.int64)
    pl = place_template(tpl, build_patient_frame(margin, axis),
                        patient_margin_perimeter(margin))
    cfg = CrownOuterConfig(body_scale_decoupling=True)
    placed = pl.apply_body_decoupled(tpl.vertices, tpl.margin_loop_indices,
                                     tpl.crown_height_mm)
    _orient_patches(patches, np.asarray(placed, float))
    loop_pts = placed[tpl.margin_loop_indices]
    corr = bd.align_rings(loop_pts, margin, cfg.resample_n)
    targets = bd.targets_for_loop_vertices(loop_pts, corr)
    root_idx = np.flatnonzero(tpl.root_vertex_mask)
    crown_idx = np.flatnonzero(tpl.crown_vertex_mask)
    z_thr = np.percentile(placed[crown_idx, 2], cfg.occlusal_anchor_percentile)
    occ_idx = crown_idx[placed[crown_idx, 2] >= z_thr]
    h1_v, _info = solve_laplacian(DeformInputs(
        vertices_placed=placed, faces=faces,
        margin_vertex_indices=tpl.margin_loop_indices, margin_targets=targets,
        root_vertex_indices=root_idx, occlusal_vertex_indices=occ_idx,
        config=cfg))
    qs = {s: NeighborClearanceQuery(patches[s]) for s in patches}
    frozen = {s: float(patches[s].normal_sign) for s in patches}
    dom = frozen_editable_domain(h1_v, faces, frame, patches,
                                 tpl.margin_loop_indices, pf, root_idx=root_idx,
                                 qs=qs, frozen_sign=frozen)
    wf = measure_md_width_field(h1_v, faces, frame, tpl.margin_loop_indices,
                                dom, pf, MDWidthFieldParams())
    print("s_md(t) =", [(round(float(a), 3), round(float(b), 3))
                        for a, b in wf.s_of_t])
    for rec in wf.band_table:
        print("  band", rec)
    out = solve_proximal_constrained(
        placed, faces, tpl.margin_loop_indices, targets, frame, patches,
        root_idx=root_idx, occ_idx=occ_idx, config=cfg, params=pf,
        max_iterations=1, reference_v=h1_v,
        width_field={"result": wf, "params": MDWidthFieldParams()},
        width_field_only=True)
    h = out["history"][0]
    print("B 层: width rows 覆盖顶点 =", int(h.get("n_active") or 0),
          "| stop =", h.get("stop"))
    for t in h.get("alpha_trials", []):
        rd = t.get("region_displacement") or {}
        regs = " ".join("%s %.4f" % (k, rd[k].get("max_mm", 0.0))
                        for k in ("margin", "cervical", "occlusal")
                        if rd.get(k, {}).get("n"))
        print("α=%-7s p95 %.4f→%.4f | max %.4f→%.4f | selfint(all/severe) %s/%s "
              "| cumFlips %s | lock %.1e | bl95 %.4f | regs[%s] | %s"
              % (t["alpha"], t["deficit_p95_before"], t["deficit_p95_after"],
                 t["deficit_max_before"], t["deficit_max_after"],
                 t.get("new_self_intersections_all"), t.get("new_self_intersections_severe"),
                 t.get("cumulative_flips_severe"), t.get("locked_displacement_mm", -1),
                 (rd.get("buccal_lingual") or {}).get("p95_mm", 0.0), regs,
                 "**ACCEPT**" if t.get("accepted") else str(t.get("reject"))))
    print("accepted == H.1 ?", bool(np.allclose(out["accepted_v"], h1_v)))
    print("冻结域 AMBIGUOUS_PATCH_EDGE（H.1 锁定）：", dom.get("n_ambiguous_side"),
          "| 各侧候选 n:", {s: int(len(dom["per_side"][s])) for s in dom["per_side"]})
    lc = out.get("last_candidate_v")
    if lc is not None and not np.allclose(lc, h1_v):
        from app.geometry.crown_outer.md_width_field import _height_param
        _t = _height_param(h1_v, frame, tpl.margin_loop_indices)
        _u = (h1_v - h1_v[np.asarray(tpl.margin_loop_indices, np.int64)].mean(axis=0)) \
            @ np.asarray(frame.x, float)
        print("— 候选步归因（trial = accepted + α·(candidate − accepted)）—")
        # 超细定点取证：最差回归顶点的几何事实（位置/位移/最近面/带符号间隙）
        mp = patches["mesial"]
        print("patch face counts: mesial=%d distal=%d"
              % (len(mp.faces), len(patches["distal"].faces)))
        for alpha in (1.0, 0.25, 0.0625):
            tr = h1_v + alpha * (lc - h1_v)
            regs = []
            for s in ("mesial", "distal"):
                sel = np.asarray(dom["per_side"][s], np.int64)
                db = np.maximum(0.0, pf.target_clearance_mm
                                - qs[s].signed_clearance_decision(h1_v[sel])[0])
                da = np.maximum(0.0, pf.target_clearance_mm
                                - qs[s].signed_clearance_decision(tr[sel])[0])
                dd = da - db
                for jj in np.argsort(dd)[::-1][:2]:
                    regs.append((float(dd[jj]), s, int(sel[jj])))
            regs.sort(reverse=True)
            for dlt, s, vid in regs[:2]:
                c0, f0, _s0, d0 = qs[s].closest(h1_v[vid][None])
                sc0, _q0, _n0, vd0 = qs[s].signed_clearance_valid(h1_v[vid][None])
                raw0, _qr0, _nr0 = qs[s].signed_clearance_raw(h1_v[vid][None])
                c1, f1, _s1, d1 = qs[s].closest(tr[vid][None])
                sc1, _q1, _n1, vd1 = qs[s].signed_clearance_valid(tr[vid][None])
                raw1, _qr1, _nr1 = qs[s].signed_clearance_raw(tr[vid][None])
                disp = tr[vid] - h1_v[vid]
                print("  α=%-6s %s id=%d Δ=%.5f t=%.3f u=%+.3f |disp|=%.6f "
                      "euclid %.5f→%.5f | raw %+.5f→%+.5f | robust %+.5f→%+.5f "
                      "(valid %s→%s)"
                      % (alpha, s, vid, dlt, _t[vid], _u[vid],
                         float(np.linalg.norm(disp)), float(d0[0]), float(d1[0]),
                         float(raw0[0]), float(raw1[0]),
                         float(sc0[0]), float(sc1[0]), bool(vd0[0]), bool(vd1[0])))
                print("      pos0=%s  pos1=%s  closest0=%s closest1=%s"
                      % (np.round(h1_v[vid], 3), np.round(tr[vid], 3),
                         np.round(c0[0], 3), np.round(c1[0], 3)))
        for alpha in (1.0, 0.25, 0.0625):
            tr = h1_v + alpha * (lc - h1_v)
            rows = []
            for s in ("mesial", "distal"):
                sel = np.asarray(dom["per_side"][s], np.int64)
                db = np.maximum(0.0, pf.target_clearance_mm
                                - qs[s].signed_clearance_decision(h1_v[sel])[0])
                da = np.maximum(0.0, pf.target_clearance_mm
                                - qs[s].signed_clearance_decision(tr[sel])[0])
                dd = da - db
                j = int(np.argmax(dd))
                vid = int(sel[j])
                _c0, f0, _s0, _d0 = qs[s].closest(h1_v[vid][None])
                _c1, f1, _s1, _d1 = qs[s].closest(tr[vid][None])
                rows.append((float(dd[j]), s, vid, float(_t[vid]), float(abs(_u[vid])),
                             float(db[j]), float(da[j]), int(f0[0]), int(f1[0])))
            rows.sort(reverse=True)
            for r in rows[:2]:
                print("  α=%-6s %s id=%d t=%.3f |u|=%.2f deficit %.4f→%.4f "
                      "closestFace %d→%d (Δ%+.5f)"
                      % (alpha, r[1], r[2], r[3], r[4], r[5], r[6], r[7], r[8], r[0]))

    acc = out["accepted_v"]
    for s in ("mesial", "distal"):
        sc, _q, _n = qs[s].signed_clearance_decision(acc[dom["per_side"][s]])
        pen = np.maximum(0.0, -sc)
        d = np.maximum(0.0, pf.target_clearance_mm - sc)
        print("%s: pen p95 %.4f max %.4f | deficit p95 %.4f max %.4f"
              % (s, float(np.quantile(pen, .95)), float(pen.max()),
                 float(np.quantile(d, .95)), float(d.max())))


if __name__ == "__main__":
    main()
