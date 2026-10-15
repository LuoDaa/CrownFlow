#!/usr/bin/env python3
"""K0 真实例：先出账（只测不改），再决定要不要做 proximal fit。

用法::

    python tools/run_proximal_fit_case.py \
        --case-dir <...>/crown_outer_cases/cf_b1b73bae221211b1 \
        --anatomy  <H.1 冠 open shell 的 STL> \
        --out      <...>/proximal_fit_case \
        [--fit]            # 不给就只测
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.geometry.crown_outer.contact import NeighborPatch          # noqa: E402
from app.geometry.crown_outer.proximal_fit import (                 # noqa: E402
    ProximalFitParams,
    fit_proximal_prior,
    measure_proximal_prior,
)


def _patch(path):
    import trimesh
    m = trimesh.load(str(path), force="mesh")
    return NeighborPatch(np.asarray(m.vertices, float),
                         np.asarray(m.faces, np.int64))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case-dir", required=True)
    ap.add_argument("--anatomy", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fit", action="store_true")
    ap.add_argument("--clearance-mm", type=float, default=0.05)
    a = ap.parse_args()

    import trimesh
    from app.services.crown_outer_pipeline import patient_frame_from_meta

    os.makedirs(a.out, exist_ok=True)
    meta = json.load(open(os.path.join(a.case_dir, "case.json"),
                          encoding="ascii"))
    frame = patient_frame_from_meta(meta)
    anat = trimesh.load(a.anatomy, force="mesh")
    mes = _patch(os.path.join(a.case_dir, "mesial_neighbor_patch.stl"))
    dis = _patch(os.path.join(a.case_dir, "distal_neighbor_patch.stl"))
    p = ProximalFitParams(target_clearance_mm=a.clearance_mm)

    audit = measure_proximal_prior(anat, frame, mes, dis, p)
    print("== K0 账（只测不改）==")
    print(json.dumps(audit, indent=1, ensure_ascii=False))
    with open(os.path.join(a.out, "proximal_audit.json"), "w",
              encoding="utf-8") as fh:
        json.dump(audit, fh, indent=1, ensure_ascii=False)
    if not a.fit:
        print("\n（未给 --fit：只出账，不动几何）")
        return 0

    print("\n== K0 Proximal Fit ==")
    res = fit_proximal_prior(anat, frame, mes, dis, p)
    print("status:", res.status)
    for n in res.notes:
        print("note:", n)
    for r in res.fit["rounds"]:
        if r.get("stop"):
            print("round %s stop: %s" % (r["round"], r["stop"]))
        else:
            print("round %s %-6s α=%-7s pen %s→%s %s"
                  % (r["round"], r["side"], r["alpha"],
                     r.get("penetration_p95_before"),
                     r.get("penetration_p95_after"),
                     "ACCEPT" if r.get("accepted") else
                     ("reject: " + str(r.get("reject"))[:70])))
    print("moved=%s max_disp=%.4f" % (res.fit["n_moved_vertices"],
                                      res.fit["max_displacement_mm"]))
    print("off-axis/buccal-lingual:", json.dumps(
        res.fit["buccal_lingual_deviation_mm"], ensure_ascii=False))
    print("final gate:", json.dumps(res.fit["final_gate"]["ok"]),
          json.dumps(res.fit["final_gate"]["violations"], ensure_ascii=False))
    after = measure_proximal_prior(res.mesh, frame, mes, dis, p)
    print("\n== 修后账 ==")
    for side in ("mesial", "distal"):
        print("  %-7s pen p95 %s→%s  max %s→%s  (n=%s)"
              % (side,
                 round(audit[side]["penetration_mm"]["p95"], 4),
                 round(after[side]["penetration_mm"]["p95"], 4),
                 round(audit[side]["penetration_mm"]["max"], 4),
                 round(after[side]["penetration_mm"]["max"], 4),
                 after[side]["penetration_mm"]["n"]))
    if res.mesh is not None:
        res.mesh.export(os.path.join(a.out, "anatomy_prior_fitted.stl"))
    with open(os.path.join(a.out, "proximal_fit_report.json"), "w",
              encoding="utf-8") as fh:
        json.dump({"audit": audit, "fit": res.as_dict(),
                   "audit_after": after}, fh, indent=1, ensure_ascii=False)
    print("\n写出 %s" % a.out)
    return 0 if res.status == "OK" else 1


if __name__ == "__main__":
    raise SystemExit(main())
