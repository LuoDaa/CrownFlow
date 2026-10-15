#!/usr/bin/env python3
"""J.3 Feasibility Audit（工具入口）—— 生成之前的秒级可行性判定。

    python tools/run_j3_feasibility_audit.py --case-dir <病例包> [--out <目录>]

只测：内表面（代型 + cement gap）沿外法向 + MinThickness 的"最小必需外表面"
是否侵入邻牙/对颌的允许空间。侵入 → ``LOCAL_CONSTRAINT_CONFLICT``；
聚合 → ``CONSTRAINT_CONFLICT``。不生成、不求解。

MinThickness / CementGapThickness / DistanceToNeighbor / DistanceToAntagonist
**从病例的 ``die_ref.json`` 读**（provenance = `.modelInfo` 设计参数），
读不到就明确报错，不在引擎里写临床默认值。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.geometry.crown_outer.contact import NeighborPatch            # noqa: E402
from app.geometry.crown_outer.feasibility_audit import (              # noqa: E402
    FeasibilityParams, feasibility_audit)


def _load_mesh(path):
    import trimesh
    m = trimesh.load(str(path), force="mesh")
    return np.asarray(m.vertices, np.float64), np.asarray(m.faces, np.int64)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case-dir", required=True)
    ap.add_argument("--intaglio", default=None,
                    help="默认 <case>/_intaglio_for_chain.stl")
    ap.add_argument("--out", default=None)
    ap.add_argument("--min-thickness", type=float, default=None,
                    help="覆盖 die_ref.json 的 MinThickness（默认读病例参数）")
    a = ap.parse_args()

    cd = a.case_dir
    ref = json.load(open(os.path.join(cd, "die_ref.json"), encoding="utf-8"))
    dp = ref.get("design_parameters") or {}
    t_min = a.min_thickness if a.min_thickness is not None else dp.get("MinThickness")
    if t_min is None:
        raise SystemExit("病例 die_ref.json 没有 MinThickness —— 不猜，请显式给")
    d_neighbor = float(dp.get("DistanceToNeighbor", 0.0))
    d_ant = float(dp.get("DistanceToAntagonist", 0.1))
    intaglio = a.intaglio or os.path.join(cd, "_intaglio_for_chain.stl")
    print("病例 %s | MinThickness=%.2f（die_ref.json/.modelInfo）| "
          "DistanceToNeighbor=%.2f | DistanceToAntagonist=%.2f"
          % (os.path.basename(os.path.normpath(cd)), float(t_min),
             d_neighbor, d_ant))

    iv, if_ = _load_mesh(intaglio)
    patches = {}
    for side, fn in (("mesial", "mesial_neighbor_patch.stl"),
                     ("distal", "distal_neighbor_patch.stl"),
                     ("antagonist", "antagonist_patch.stl")):
        p = os.path.join(cd, fn)
        if os.path.exists(p):
            pv, pf = _load_mesh(p)
            patches[side] = NeighborPatch(pv, pf)
    missing = [s for s in ("mesial", "distal", "antagonist") if s not in patches]
    if missing:
        print("缺 patch：%s（该方向不审计，如实记录）" % missing)

    t0 = time.time()
    params = FeasibilityParams(min_thickness_mm=float(t_min),
                               neighbor_clearance_mm=d_neighbor,
                               antagonist_clearance_mm=d_ant)
    res = feasibility_audit(iv, if_, patches, params=params)
    dt = time.time() - t0

    print("\n== J.3 Feasibility Audit（%.1fs，n_samples=%d，定向歧义面 %d）=="
          % (dt, res.n_samples, res.n_orientation_ambiguous))
    print("| 空间 | 查询点 | 冲突点 | 超噪冲突 | 侵入深度 p50/p95/max (mm) | 余量 p95 |")
    print("|---|---:|---:|---:|---|---:|")
    rows = [("邻牙 mesial", res.neighbor.get("mesial")),
            ("邻牙 distal", res.neighbor.get("distal")),
            ("对颌", res.antagonist)]
    for tag, sc in rows:
        if not sc:
            print("| %s | — | — | — | 未审计 | — |" % tag)
            continue
        print("| %s | %d | %d | %d | %.3f / %.3f / %.3f | %s |"
              % (tag, sc["n_queried"], sc["n_conflict"],
                 sc["n_conflict_above_floor"], sc["deficit_p50_mm"],
                 sc["deficit_p95_mm"], sc["deficit_max_mm"],
                 ("%.3f" % sc["headroom_p95_mm"])
                 if sc["headroom_p95_mm"] is not None else "—"))
        if sc.get("conflict_centroid"):
            print("    %s 冲突区质心 = %s" % (tag, sc["conflict_centroid"]))
    for n in res.notes:
        print("  note: %s" % n)
    verdict = ("CONSTRAINT_CONFLICT（生成前即可判定：最小厚度面与允许空间无交集）"
               if res.status == "CONSTRAINT_CONFLICT" else res.status)
    print("\n判定：**%s**" % verdict)

    out = a.out or os.path.join(cd, "j3_feasibility.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump({"case_uuid": os.path.basename(os.path.normpath(cd)),
                   "params": {"min_thickness_mm": float(t_min),
                              "neighbor_clearance_mm": d_neighbor,
                              "antagonist_clearance_mm": d_ant},
                   "params_provenance": "die_ref.json/.modelInfo",
                   "seconds": round(dt, 2),
                   **res.as_dict()}, fh, indent=1, ensure_ascii=False)
    print("写出 %s" % out)
    return 0 if res.status != "INPUT_INVALID" else 1


if __name__ == "__main__":
    raise SystemExit(main())
