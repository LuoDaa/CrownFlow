#!/usr/bin/env python3
"""Contact 之后的 E2E 续跑（Occlusion → Thickness Audit）。

用于：链工具跑到 Contact PASS 后，接着把 E2E 走完能走的部分。
（Thickness Repair / Final Stitch 尚未实现 —— 当前可交付到 Thickness Audit。）

    python tools/run_post_contact_e2e.py --run <链输出目录> --case-dir <病例包>
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.crown_outer_pipeline import (                        # noqa: E402
    DEFAULT_PARAMS, CrownBundle, stage_occlusion, stage_thickness_audit)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--case-dir", required=True)
    a = ap.parse_args()
    chain = json.load(open(os.path.join(a.run, "h2_2b_chain.json"),
                           encoding="utf-8"))
    gate = (chain.get("contact") or {}).get("gate") or {}
    if gate.get("status") != "PASS":
        print("Contact Gate 不是 PASS（%s）—— 按纪律不跑下游。"
              % gate.get("status"))
        return 1
    meta = json.load(open(os.path.join(a.case_dir, "case.json"),
                          encoding="ascii"))
    ant = os.path.join(a.case_dir, "antagonist_patch.stl")
    bundle = CrownBundle(
        case_uuid=os.path.basename(os.path.normpath(a.case_dir)),
        case_dir=Path(a.case_dir),
        margin_points=__import__("numpy").asarray(meta["margin_points"], float),
        insertion_axis=__import__("numpy").asarray(meta["insertion_axis"], float),
        fdi=int(meta["fdi"]),
        mesial_path=Path(a.case_dir) / "mesial_neighbor_patch.stl",
        distal_path=Path(a.case_dir) / "distal_neighbor_patch.stl",
        antagonist_path=(Path(ant) if os.path.exists(ant) else None),
        meta=meta)
    out = Path(a.run)
    contact_artifact = (chain.get("contact") or {}).get("artifact")
    print("输入（Contact 硬化输出）= %s" % contact_artifact)
    o = stage_occlusion(bundle, out, dict(DEFAULT_PARAMS),
                        prev={"contact": {"artifact": contact_artifact}})
    print("Occlusion  status=%s | margin_disp=%.4f | proximal_core=%.4f | 翻面=%d"
          % (o["status"], o["margin_max_displacement_mm"],
             o["proximal_core_max_displacement_mm"], o["n_flipped_faces"]))
    for n in (o.get("notes") or [])[:5]:
        print("   note: %s" % n)
    t = stage_thickness_audit(
        bundle, out, dict(DEFAULT_PARAMS),
        prev={"occlusion": {"artifact": o["artifact"]},
              "intaglio": {"artifact": os.path.join(
                  a.case_dir, "_intaglio_for_chain.stl")},
              "outer_h1": {"artifact": (chain.get("contact") or {}).get(
                  "outer_used")}})
    print("Thickness  status=%s | artifact=%s" % (t.get("status"), t.get("artifact")))
    with open(os.path.join(a.run, "e2e_post_contact.json"), "w",
              encoding="utf-8") as fh:
        json.dump({"occlusion": {k: v for k, v in o.items() if k != "history"},
                   "thickness": {"status": t.get("status"),
                                 "artifact": t.get("artifact")}},
                  fh, indent=1, ensure_ascii=False)
    print("写出 %s" % os.path.join(a.run, "e2e_post_contact.json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
