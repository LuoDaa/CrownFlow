#!/usr/bin/env python3
"""邻牙 patch **作用域** QC —— 用病例自己的设计冠（GT control）验 patch 能不能当接触参照。

## 为什么需要（2026-10-09 实测，FDI46）

`grow_neighbor_patch.py` 按"离冠最近的牙弓三角形"做半径 9mm 区域生长 —— 长出来的
是**整段牙弓 patch**（实测 u 跨 −16..+15mm），把目标牙**自身位置/颈缘组织**也包含
进去。于是一个**临床已交付**的设计冠（`final_crown.stl`）在这些 patch 上也读出
穿透 p95 ≈ 1.0–1.13mm —— 说明读数不是"邻间隙不够"，而是 patch 作用域污染：

* 真正的近接大陆（|lat|≤1.5、距 MD 远端 1.2mm 内）：GT 冠与生成冠**零穿透**、
  间隙 1.0–1.8mm；
* 读出的"穿透"集中在**颊舌侧非近接区**，最近点有 ~37% 落在 GT 冠体**内部**；
* patch 的 MD 跨段覆盖目标牙自身区域。

⇒ **GT 冠读不出的测量，不能拿来判生成冠。** 本工具把这条做成**前置 QC**：
GT 冠（若病例包里有）在 patch 上的穿透 p95 超过 Contact Gate 的允许值 → patch
不可作为接触参照 → 下游链的结论一律不可信。

用法::

    python tools/audit_neighbor_patch_scope.py --case-dir <病例包> [--gt final_crown.stl]
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.geometry.crown_outer.contact import NeighborPatch             # noqa: E402
from app.geometry.crown_outer.thickness import winding_number          # noqa: E402
from app.services.crown_outer_pipeline import (                        # noqa: E402
    _orient_patches, patient_frame_from_meta)

#: 与 Contact Gate 同一允许值（残留 p95 ≤ 0.2mm）——GT 冠自己都过不了的
#: 测量，不能拿去判生成冠。
ALLOW_P95_MM = 0.2


def _load(path):
    import trimesh
    m = trimesh.load(str(path), force="mesh")
    return np.asarray(m.vertices, np.float64), np.asarray(m.faces, np.int64)


def audit(case_dir: str, gt_name: str = "final_crown.stl") -> dict:
    meta = json.load(open(os.path.join(case_dir, "case.json"),
                          encoding="ascii"))
    fr = patient_frame_from_meta(meta)
    x = np.asarray(fr.x, float)
    y = np.asarray(fr.y, float)
    out = {"case_uuid": os.path.basename(os.path.normpath(case_dir)),
           "gt_control": None, "patches": {}, "findings": [], "status": "OK"}
    gt_path = os.path.join(case_dir, gt_name)
    patches = {}
    for side, fn in (("mesial", "mesial_neighbor_patch.stl"),
                     ("distal", "distal_neighbor_patch.stl")):
        p = os.path.join(case_dir, fn)
        if not os.path.exists(p):
            continue
        pv, pf = _load(p)
        patches[side] = (NeighborPatch(pv, pf), pv, pf)
    if not patches:
        out["status"] = "INPUT_INVALID"
        out["findings"].append({"code": "NO_NEIGHBOR_PATCH",
                                "detail": "病例包没有邻牙 patch，无法审计"})
        return out
    if not os.path.exists(gt_path):
        out["status"] = "UNVERIFIABLE_NO_GT"
        out["findings"].append({
            "code": "NO_GT_CONTROL",
            "detail": "病例包没有 final_crown.stl 作对照 —— patch 作用域无法证伪，"
                      "结论按不可信处理"})
        return out
    gv, gf = _load(gt_path)
    pl = {s: v[0] for s, v in patches.items()}
    _orient_patches(pl, gv)
    # ① GT 冠在 patch 上的穿透（与 Contact Gate 同一口径/同一允许值）
    for side, (patch, pv, pf) in patches.items():
        sc, valid, cp, fids, _p = patch.clearance_detail(gv)
        if not valid.any():
            continue
        pen = np.maximum(-sc[valid], 0.0)
        u = (pv - fr.origin) @ x
        lat = (gv - fr.origin) @ y
        deep = np.flatnonzero(valid & (sc < -ALLOW_P95_MM))
        inside_frac = (float((winding_number(cp[deep], gv[gf]) > 0.5).mean())
                       if len(deep) else 0.0)
        out["patches"][side] = {
            "gt_penetration_p95_mm": float(np.quantile(pen, .95)),
            "gt_penetration_max_mm": float(pen.max()),
            "n_deep": int(len(deep)),
            "deep_closest_inside_gt_frac": inside_frac,
            "patch_u_span_mm": [float(u.min()), float(u.max())],
            "deep_lateral_abs_p50_mm": (float(np.median(np.abs(lat[deep])))
                                        if len(deep) else None)}
    # ② 真正的近接大陆上的 GT 间隙（若大陆干净，说明污染不在近接区）
    for side, sign in (("mesial", 1.0), ("distal", -1.0)):
        if side not in patches:
            continue
        patch = patches[side][0]
        u = (gv - fr.origin) @ x
        lat = (gv - fr.origin) @ y
        m0 = np.abs(lat) <= 1.5
        if not m0.any():
            continue
        thr = sign * u[m0].max() - 1.2
        sel = m0 & (sign * u >= thr)
        sc, valid, _cp, _f, _p = patch.clearance_detail(gv[sel])
        if valid.any():
            s = sc[valid]
            out["patches"][side]["gt_land_clearance_p50_mm"] = float(
                np.median(s))
            out["patches"][side]["gt_land_penetration_p95_mm"] = float(
                np.quantile(np.maximum(-s, 0.0), .95))
    # ③ 判定
    worst = max((v.get("gt_penetration_p95_mm") or 0.0)
                for v in out["patches"].values())
    out["gt_control"] = {"worst_penetration_p95_mm": worst,
                         "allow_mm": ALLOW_P95_MM}
    if worst > ALLOW_P95_MM:
        out["status"] = "PATCH_SCOPE_CONTAMINATED"
        out["findings"].append({
            "code": "GT_CROWN_FAILS_ON_PATCH",
            "detail": ("临床已交付的设计冠（GT）在 patch 上读出的穿透 p95 = "
                       "%.3fmm > %.2fmm —— patch 不能作为接触参照；"
                       "此前所有基于该 patch 的 K0/H.2/Contact 结论不可信"
                       % (worst, ALLOW_P95_MM))})
        lands = [v.get("gt_land_penetration_p95_mm") for v in
                 out["patches"].values()]
        if lands and all(l is not None and l <= 1e-9 for l in lands):
            out["findings"].append({
                "code": "CONTAMINATION_OUTSIDE_PROXIMAL_LAND",
                "detail": ("近接大陆上 GT 冠零穿透（间隙 1.0–1.8mm）——"
                           "污染位于颊舌侧非近接区（patch 含目标牙自身/颈缘）")})
    else:
        out["findings"].append({
            "code": "GT_CROWN_PASSES_ON_PATCH",
            "detail": "GT 冠在 patch 上通过（p95 %.3f ≤ %.2f）—— patch 可作参照"
                      % (worst, ALLOW_P95_MM)})
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case-dir", required=True)
    ap.add_argument("--gt", default="final_crown.stl")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    res = audit(a.case_dir, a.gt)
    out = a.out or os.path.join(a.case_dir, "neighbor_patch_scope_qc.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1, ensure_ascii=False)
    print("== 邻牙 patch 作用域 QC（GT control）==")
    print("status = %s" % res["status"])
    for side, v in res["patches"].items():
        print("  %s: GT 穿透 p95=%.3f max=%.3f | 深穿点 %d（最近点落在 GT 体内 %.0f%%）"
              " | patch u 跨 [%.1f, %.1f] | 大陆间隙 p50=%s"
              % (side, v["gt_penetration_p95_mm"], v["gt_penetration_max_mm"],
                 v["n_deep"], 100.0 * v["deep_closest_inside_gt_frac"],
                 v["patch_u_span_mm"][0], v["patch_u_span_mm"][1],
                 ("%.3f" % v["gt_land_clearance_p50_mm"])
                 if "gt_land_clearance_p50_mm" in v else "—"))
    for f in res["findings"]:
        print("  [%s] %s" % (f["code"], f["detail"]))
    print("写出 %s" % out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
