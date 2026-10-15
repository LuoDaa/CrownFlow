"""Contact v1 真实链路验收（唯一双侧可用病例 cf_b1b73bae221211b1）。

    H.1 Crown Outer + 近中/远中邻牙 patch → optimize_contacts → v1

对照（NeighborGeometryAudit 反求的 REALIZED_GEOMETRY）：
    GT mesial min 0.079mm / ~2.6mm²；GT distal min 0.036mm / ~11.1mm²

重点不是 Chamfer 大幅下降，而是：margin 稳定、咬合面基本不动、
mesial/distal 带符号误差朝 GT 方向改善、预测接触质心/面积接近 GT。
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.geometry.crown_outer import (  # noqa: E402
    CrownOuterConfig,
    PatientCrownContext,
    generate_crown_outer_v0,
    load_crown_template,
    sample_surface_points,
)
from app.geometry.crown_outer.contact import (  # noqa: E402
    ContactParameters,
    NeighborPatch,
    optimize_contacts,
)
from app.geometry.crown_outer.gt_compare import (  # noqa: E402
    anatomical_regions,
    region_signed_stats,
    signed_surface_error,
)
from app.importers.tooth_library import discover_libraries  # noqa: E402

CASE = r"D:\code\牙冠自动生成\数据\crown_outer_cases\cf_b1b73bae221211b1"
BATCH_H1 = (r"D:\code\牙冠自动生成\数据\crown_outer_results"
            r"\fdi46_batch_h1")
OUT = r"D:\code\牙冠自动生成\数据\crown_outer_results\contact_v1_case"


def patient_frame_from_meta(meta):
    """与审计脚本同一约定的患者系：X = AxisMesial ⊥ Z（不用 PCA）。"""
    import trimesh.geometry as tg
    from app.geometry.crown_outer.patient_frame import PatientFrame
    margin = np.asarray(meta["margin_points"])
    origin = margin.mean(axis=0)
    z = np.asarray(meta["insertion_axis"], dtype=float)
    z = z / np.linalg.norm(z)
    m = np.asarray(meta["axis_mesial"], dtype=float)
    m = m - (m @ z) * z
    x = m / np.linalg.norm(m)
    y = np.cross(z, x)
    y = y / np.linalg.norm(y)
    return PatientFrame(origin=origin, z=z, x=x, y=y)


def main() -> int:
    import trimesh
    os.makedirs(OUT, exist_ok=True)
    meta = json.load(open(os.path.join(CASE, "case.json"), encoding="ascii"))
    frame = patient_frame_from_meta(meta)

    # ---- H.1 冠（与批量回归同一配置重新生成，保证口径一致）
    libs = {l.name: l for l in discover_libraries()}
    template = load_crown_template(libs["炼金王"], 46)
    ctx = PatientCrownContext(
        case_uuid=meta["case_uuid"], fdi=46,
        margin_points=np.asarray(meta["margin_points"]),
        insertion_axis=np.asarray(meta["insertion_axis"]),
        final_crown_gt_path=os.path.join(CASE, "final_crown.stl"))
    cfg = CrownOuterConfig(body_scale_decoupling=True)
    h1 = generate_crown_outer_v0(template, ctx.margin_points,
                                 ctx.insertion_axis, config=cfg)
    h1.mesh.export(os.path.join(OUT, "h1_crown.stl"))
    print("H.1 状态:", h1.status, "scale:", h1.uniform_scale)

    # ---- 邻牙 patch
    mesial = NeighborPatch(
        *(lambda m: (m.vertices, m.faces))(
            trimesh.load(os.path.join(CASE, "mesial_neighbor_patch.stl"),
                         force="mesh")))
    distal = NeighborPatch(
        *(lambda m: (m.vertices, m.faces))(
            trimesh.load(os.path.join(CASE, "distal_neighbor_patch.stl"),
                         force="mesh")))

    # ---- GT 反求基准（Final Crown ↔ patches，与审计同一口径）
    gt = trimesh.load(ctx.final_crown_gt_path, force="mesh")
    # 外表面 GT（acceptance-only）：整颗 Final Crown 含 Intaglio、法向朝内，
    # 拿它当参照会给出假的 +5mm 舌侧误差。之前脚本一直漏了这一步。
    from app.geometry.crown_outer import extract_outer_surface
    gt_faces = extract_outer_surface(gt, np.asarray(meta["insertion_axis"]),
                                     np.asarray(meta["margin_points"]).mean(0))
    gt_outer = gt.submesh([gt_faces], append=True, repair=False)
    gt_pts, _ = sample_surface_points(gt, n=30000)
    for side, patch in (("mesial", mesial), ("distal", distal)):
        cp, _f, signed, dist = patch.closest(gt_pts)
        gap = dist
        near = gap <= 0.15
        print("GT %-7s min=%.3fmm area~%.1fmm2 (audit: %s)"
              % (side, gap.min(), near.sum() * 0.02,
                 "0.079/2.6" if side == "mesial" else "0.036/11.1"))

    # ---- Contact v1
    params = ContactParameters(target_gap_mm=0.05, design_target_mm=0.0)
    res = optimize_contacts(h1.mesh, frame, mesial, distal, params)
    print("v1 状态:", res.status, res.notes)
    print(json.dumps(res.summary(), indent=1, ensure_ascii=False))
    res.mesh.export(os.path.join(OUT, "crown_outer_v1.stl"))

    # ---- H.1 vs v1 的分区域带符号误差（对 GT 冠）
    for name, mesh in (("h1", h1.mesh), ("v1", res.mesh)):
        pts, nrm = sample_surface_points(mesh, n=15000)
        gt_pts, gt_nrm = sample_surface_points(gt, n=15000)
        signed = signed_surface_error(pts, gt_pts, gt_nrm)
        regions = anatomical_regions(pts, np.asarray(meta["margin_points"]).mean(0),
                                     np.asarray(meta["insertion_axis"]),
                                     meta.get("axis_mesial"),
                                     meta.get("axis_buccal"))
        stats = region_signed_stats(pts, signed, regions)
        line = {k: v["signed_median_mm"] for k, v in stats.items()
                if v["signed_median_mm"] is not None}
        print("%s 区域带符号中位:" % name,
              json.dumps({k: round(v, 3) for k, v in line.items()}))

    json.dump(res.summary(), open(os.path.join(OUT, "contact_result.json"),
                                  "w"), indent=1)

    # ---- I.1 精修：V_ref 固定为 H.1，起点为 Contact v1 结果
    from app.geometry.crown_outer.contact_refine import (
        RefineParams, refine_contacts)
    rp = RefineParams(target_gap_mm=0.05)
    ref = refine_contacts(h1.mesh, frame, mesial, distal, rp,
                          start_from=res.mesh)
    print("")
    print("I.1 状态:", ref.status, ref.notes)
    print(json.dumps({k: v for k, v in ref.summary().items()
                      if k in ("status", "mesial", "distal",
                               "margin_max_displacement_mm",
                               "occlusal_p95_displacement_mm",
                               "n_flipped_faces")},
                     indent=1, ensure_ascii=False))
    print("I.1 逐轮：")
    for h in ref.history:
        print("  it=%d acc=%-5s pen=%.3f p95=%.3f gap=%.3f flip=%.4f occ=%.4f"
              % (h["iteration"], h.get("accepted", "base"),
                 h["penetration_max_mm"], h["penetration_p95_mm"],
                 h["contact_gap_error_p95_mm"], h["flipped_ratio"],
                 h["occlusal_disp_p95_mm"]))
    ref.mesh.export(os.path.join(OUT, "crown_outer_v1_1.stl"))
    for name, mesh in (("h1", h1.mesh), ("v1", res.mesh), ("v1_1", ref.mesh)):
        pts, nrm = sample_surface_points(mesh, n=15000)
        # 必须比**外表面** GT：整颗 Final Crown 含 Intaglio、法向朝内，
        # 拿它当参照会给出假的 +5mm 舌侧误差（实测踩过）。
        gt_pts, gt_nrm = sample_surface_points(gt_outer, n=15000)
        signed = signed_surface_error(pts, gt_pts, gt_nrm)
        regions = anatomical_regions(
            pts, np.asarray(meta["margin_points"]).mean(0),
            np.asarray(meta["insertion_axis"]), meta.get("axis_mesial"),
            meta.get("axis_buccal"))
        st = region_signed_stats(pts, signed, regions)
        print("%-5s 区域带符号:" % name, json.dumps(
            {k: round(v["signed_median_mm"], 3) for k, v in st.items()
             if v["signed_median_mm"] is not None}))
    json.dump(ref.summary() | {"history": ref.history},
              open(os.path.join(OUT, "contact_refine_result.json"), "w"),
              indent=1)
    print("写出", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
