"""一次性取证：H.1 基线上的 |raw|/euclid 斜切分布 + 不同排除阈值下的 deficit 统计。

用法：python tools/run_h2_2b_obliquity_stats.py [case_dir]
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
from app.geometry.crown_outer.models import CrownOuterConfig
from app.geometry.crown_outer.patient_frame import build_patient_frame
from app.geometry.crown_outer.placement import (patient_margin_perimeter,
                                                place_template)
from app.geometry.crown_outer.proximal_constrained import (
    NeighborClearanceQuery, frozen_editable_domain)
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
    d_clear = float(pf.target_clearance_mm)
    for s in ("mesial", "distal"):
        ids = np.asarray(dom["per_side"][s], np.int64)
        q, fids, signed, dist = patches[s].closest(h1_v[ids])
        obl = np.abs(signed) / np.maximum(dist, 1e-12)
        deficit_raw = np.maximum(0.0, d_clear - signed)
        print("== %s（n=%d）|raw|/euclid 分位：p1=%.3f p5=%.3f p10=%.3f "
              "p25=%.3f p50=%.3f p90=%.3f"
              % (s, len(ids), *[float(np.quantile(obl, qq))
                                for qq in (.01, .05, .10, .25, .50, .90)]))
        for thr in (0.5, 0.8, 0.9, 0.95, 0.99):
            keep = obl >= thr
            if keep.sum() == 0:
                continue
            # 排除斜切点后：幅值用 euclid、符号沿用 raw
            sc = np.where(keep, signed, np.sign(signed) * dist)
            d = np.maximum(0.0, d_clear - sc[keep])
            print("   thr=%.2f 排除 %4d/%4d → deficit p95=%.4f max=%.4f "
                  "(排除者中 deficit_raw>0.02 的 %d 个)"
                  % (thr, int((~keep).sum()), len(ids),
                     float(np.quantile(d, .95)), float(d.max()),
                     int((deficit_raw[~keep] > 0.02).sum())))


if __name__ == "__main__":
    main()
