#!/usr/bin/env python3
"""I.2 归因诊断：Contact v2 输出上的残留穿透到底在哪里、为什么修不掉。

**只诊断，不改求解器。** 对真实病例的 Contact v2 输出网格，逐侧把每个穿透
顶点分入四类（沿用 v2 修正后的法向语义，**不退回 z≥p75**）：

    OCCLUSAL_LOCKED   咬合面（高 z 且 |n̂·轴| ≥ 0.6）—— Contact 不许动
    CERVICAL_LOCKED   颈环 + cervical lock 带 —— Contact 不许动
    PROXIMAL_EDITABLE 邻接壁（可修）
    OTHER             其余（patch 边界歧义等）

并回答"步长够不够"：每个穿透顶点的 required_correction（沿法向到 target_gap）
与现有 trust region（max_step_mm）的比值分布 —— 直接区分
"方向错了" vs "方向对但永远走不到"。

用法:
    python tools/diag_contact_residual.py
输出:
    数据/crown_outer_results/contact_v2_residual_attribution.json + 控制台摘要
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import trimesh

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from app.geometry.crown_outer.contact import (                      # noqa: E402
    ContactParameters, NeighborPatch, optimize_contacts)
from app.geometry.crown_outer.contact_refine import (               # noqa: E402
    RefineParams, occ_table_mask)
from app.geometry.crown_outer.generator import generate_crown_outer_v0  # noqa: E402
from app.geometry.crown_outer.models import (                       # noqa: E402
    CrownOuterConfig, PatientCrownContext)
from app.geometry.crown_outer.patient_frame import PatientFrame     # noqa: E402

CASE = r"D:\code\牙冠自动生成\数据\crown_outer_cases\cf_b1b73bae221211b1"
OUT_DIR = r"D:\code\牙冠自动生成\数据\crown_outer_results"


def classify(vertices, faces, frame, params, loop_idx):
    """→ (region 数组, 各类 id)。语义与 Contact v2 完全一致。"""
    n = len(vertices)
    region = np.full(n, "PROXIMAL_EDITABLE", dtype=object)
    occ_t = occ_table_mask(vertices, faces, frame, params)
    region[occ_t] = "OCCLUSAL_LOCKED"
    region[loop_idx] = "CERVICAL_LOCKED"
    # cervical lock 带：距颈环 < cervical_lock_mm（与 v1/v2 的锁带同义）
    from scipy.spatial import cKDTree
    d_loop = cKDTree(vertices[loop_idx]).query(vertices, k=1)[0]
    band = (d_loop < params.cervical_lock_mm) & (region == "PROXIMAL_EDITABLE")
    region[band] = "CERVICAL_LOCKED"
    return region


def side_diag(name, patch, vertices, faces, frame, params, loop_idx, region,
              trust):
    wids = np.arange(len(vertices))
    cp, fids, signed, dist = patch.closest(vertices[wids])
    inside = (signed < 0) & (dist <= params.inside_reach_mm)
    ids = np.flatnonzero(inside)
    pen = -signed[ids] - params.penetration_tolerance_mm
    pen = np.maximum(pen, 0.0)
    # required correction：沿法向把 phi 拉回 target_gap 的距离
    required = np.maximum(params.target_gap_mm - signed[ids], 0.0)
    regions = region[ids]
    by_region = {}
    for r in ("PROXIMAL_EDITABLE", "OCCLUSAL_LOCKED", "CERVICAL_LOCKED",
              "OTHER"):
        m = regions == r
        by_region[r] = {
            "count": int(m.sum()),
            "pen_sum_mm": round(float(pen[m].sum()), 4) if m.any() else 0.0,
            "pen_max_mm": round(float(pen[m].max()), 4) if m.any() else 0.0,
            "required_max_mm": round(float(required[m].max()), 4)
            if m.any() else 0.0,
            "required_over_trust_p50": round(float(np.median(
                required[m] / trust)), 2) if m.any() else None,
            "required_over_trust_p90": round(float(np.quantile(
                required[m] / trust, 0.9)), 2) if m.any() else None,
        }
    # 最深 20 个顶点明细
    order = np.argsort(-pen)[:20]
    deepest = [{
        "xyz": [round(float(x), 3) for x in vertices[ids[i]]],
        "penetration_mm": round(float(pen[i]), 4),
        "required_mm": round(float(required[i]), 4),
        "region": str(regions[i]),
        "patch_dist_mm": round(float(dist[ids[i]]), 3),
    } for i in order]
    q = (lambda a, p: round(float(np.quantile(a, p)), 4) if len(a) else 0.0)
    return {
        "side": name,
        "penetrating_vertex_count": int(len(ids)),
        "penetration_median_mm": q(pen, .5),
        "penetration_p95_mm": q(pen, .95),
        "penetration_max_mm": q(pen, 1.0),
        "required_max_mm": round(float(required.max()), 4) if len(required)
        else 0.0,
        "required_over_trust_max": round(float(required.max() / trust), 2)
        if len(required) else 0.0,
        "penetration_by_region": by_region,
        "deepest_20": deepest,
    }


def main():
    meta = json.load(open(os.path.join(CASE, "case.json"), encoding="ascii"))
    origin = np.mean(np.asarray(meta["margin_points"]), axis=0)
    z = np.asarray(meta["insertion_axis"], float)
    z /= np.linalg.norm(z)
    m = np.asarray(meta["axis_mesial"], float)
    m = m - (m @ z) * z
    x = m / np.linalg.norm(m)
    y = np.cross(z, x)
    frame = PatientFrame(origin=origin, z=z, x=x, y=y)

    libs = {l.name: l for l in __import__(
        "app.importers.tooth_library.library", fromlist=["discover_libraries"])
        .discover_libraries()}
    template = load_template(libs)
    ctx = PatientCrownContext(
        case_uuid=meta["case_uuid"], fdi=46,
        margin_points=np.asarray(meta["margin_points"]),
        insertion_axis=np.asarray(meta["insertion_axis"]),
        final_crown_gt_path=os.path.join(CASE, "final_crown.stl"))
    h1 = generate_crown_outer_v0(template, ctx.margin_points,
                                 ctx.insertion_axis,
                                 config=CrownOuterConfig(
                                     body_scale_decoupling=True))

    def patch_of(name):
        mm = trimesh.load(os.path.join(CASE, name), process=False)
        return NeighborPatch(np.asarray(mm.vertices), np.asarray(mm.faces))

    params = ContactParameters()
    res = optimize_contacts(h1.mesh, frame, patch_of("mesial_neighbor_patch.stl"),
                            patch_of("distal_neighbor_patch.stl"), params)
    v2 = np.asarray(res.mesh.vertices)
    faces = np.asarray(res.mesh.faces)
    loop_idx = _loop(res)
    region = classify(v2, faces, frame, params, loop_idx)
    # trust region 0.35mm 是 I.1（RefineParams）的步长上限；
    # Contact v1 主求解无 trust region（一步到位），其残留来自线性化。
    trust = RefineParams().max_step_mm
    out = {"case_uuid": meta["case_uuid"],
           "trust_region_mm": trust,
           "target_gap_mm": params.target_gap_mm,
           "occlusal_table_normal_dot": params.occ_table_normal_dot,
           "v2_status": res.status,
           "sides": {}}
    for name, patch in (("mesial", patch_of("mesial_neighbor_patch.stl")),
                        ("distal", patch_of("distal_neighbor_patch.stl"))):
        patch.orient_against_crown(v2, _vertex_normals(v2, faces))
        out["sides"][name] = side_diag(name, patch, v2, faces, frame, params,
                                       loop_idx, region, trust)
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, "contact_v2_residual_attribution.json")
    json.dump(out, open(path, "w"), indent=1, ensure_ascii=True)
    _report(out, path)


def load_template(libs):
    from app.geometry.crown_outer.template import load_crown_template
    return load_crown_template(libs["炼金王"], 46)


def _loop(res):
    """v2 输出网格的颈环（开放底边）。"""
    from app.geometry.crown_outer.contact import _boundary_loop
    return np.asarray(_boundary_loop(np.asarray(res.mesh.vertices),
                                     np.asarray(res.mesh.faces)), dtype=np.int64)


def _vertex_normals(v, f):
    n = np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])
    L = np.linalg.norm(n, axis=1)
    L[L == 0] = 1.0
    vn = np.zeros_like(v)
    for k in range(3):
        np.add.at(vn, f[:, k], n / L[:, None])
    Ln = np.linalg.norm(vn, axis=1)
    Ln[Ln == 0] = 1.0
    return vn / Ln[:, None]


def _report(out, path):
    print("=== I.2 残留穿透归因（%s，trust=%.2fmm）==="
          % (out["case_uuid"][:16], out["trust_region_mm"]))
    for side, d in out["sides"].items():
        print("\n[%s] 穿透顶点 %d，median=%.3f p95=%.3f max=%.3f mm"
              % (side, d["penetrating_vertex_count"], d["penetration_median_mm"],
                 d["penetration_p95_mm"], d["penetration_max_mm"]))
        print("%-18s %6s %10s %10s %12s %14s" %
              ("region", "count", "pen_sum", "pen_max", "req/trust p50",
               "req/trust p90"))
        for r, v in d["penetration_by_region"].items():
            print("%-18s %6d %10.3f %10.3f %12s %14s" %
                  (r, v["count"], v["pen_sum_mm"], v["pen_max_mm"],
                   v["required_over_trust_p50"], v["required_over_trust_p90"]))
        print("  最深 5 个：")
        for e in d["deepest_20"][:5]:
            print("    pen=%.3f req=%.3f (%.1fx trust) %s @%s"
                  % (e["penetration_mm"], e["required_mm"],
                     e["required_mm"] / out["trust_region_mm"],
                     e["region"], e["xyz"]))
    print("\nJSON →", path)


if __name__ == "__main__":
    main()
