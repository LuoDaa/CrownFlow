#!/usr/bin/env python3
"""端到端验收：通过 HTTP API 跑真实例全链路（主链 + 分层热力图 + 产物下载）。

    CROWNFLOW_CROWN_CASES_DIR=<...>/crown_outer_cases \
    CROWNFLOW_TOOTH_LIBRARY_ROOT=<...>/牙齿数据库 \
    python3 tools/run_crown_api_e2e.py [case_uuid]

跑完把完整结果写到 ``e2e_result.json``（同目录），并打印阶段账 / 壁厚审计摘要 /
分层产物与可下载性。**这是"封装"这条线的验收口**：它走的是前端同一条 API。
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fastapi.testclient import TestClient                        # noqa: E402
from app.main import app                                          # noqa: E402

CASE = sys.argv[1] if len(sys.argv) > 1 else "cf_b1b73bae221211b1"
MAIN = ["outer_h1", "intaglio", "contact", "occlusion", "thickness_audit"]
ARTIFACTS = ("crown_h1.stl", "intaglio.stl", "crown_contact.stl",
             "crown_occluded.stl", "thickness_audit.json", "layer_thin.glb",
             "layer_contact.glb", "layer_occlusal.glb")


def main() -> int:
    c = TestClient(app)
    r = c.post("/api/crown/cases/register", json={"case_uuid": CASE})
    print("register:", r.status_code, r.json().get("case_id") or r.text[:120])
    if r.status_code != 200:
        return 1

    r = c.post("/api/crown/runs", json={"case_uuid": CASE, "stages": MAIN,
                                        "params": {}})
    if r.status_code != 200:
        print("start 失败:", r.status_code, r.text[:300])
        return 1
    rid = r.json()["run_id"]
    print("run_id:", rid)

    t0, seen = time.time(), 0
    d = {}
    while True:
        d = c.get(f"/api/crown/runs/{rid}").json()
        log = d.get("log") or []
        for e in log[seen:]:
            print("  [%s] %s" % (e["at"], e["msg"]), flush=True)
        seen = len(log)
        if d["status"] in ("done", "failed"):
            print("FINAL:", d["status"], d.get("error") or "")
            break
        if time.time() - t0 > 1800:
            print("TIMEOUT"); break
        time.sleep(5)

    res = d.get("results") or {}
    print("\n=== 阶段账 ===")
    for k, v in res.items():
        keys = {kk: vv for kk, vv in v.items()
                if kk in ("status", "elapsed_s", "iterations", "uniform_scale",
                          "margin_max_displacement_mm", "n_flipped_faces",
                          "proximal_core_max_displacement_mm")}
        print("  %-16s %s" % (k, json.dumps(keys, ensure_ascii=False)))

    ta = res.get("thickness_audit") or {}
    rep = ta.get("report") or {}
    s = rep.get("summary") or {}
    print("\n=== 壁厚审计 ===")
    print("  status:", rep.get("status"))
    for k in ("n_samples", "min_thickness_mm", "p05_thickness_mm",
              "median_thickness_mm", "thin_area_mm2", "n_outside_outer",
              "max_deficit_mm", "n_ambiguous_thickness"):
        if k in s:
            v = s[k]
            print("  %-22s %s" % (k, round(v, 4) if isinstance(v, float) else v))
    print("  negative_coverage:", json.dumps(s.get("negative_coverage") or {},
                                             ensure_ascii=False)[:220])
    print("  thin_by_kind:", json.dumps(s.get("thin_area_by_kind_mm2") or {},
                                        ensure_ascii=False))
    print("  input_qc:", json.dumps(
        {k: v for k, v in (rep.get("input_qc") or {}).items()
         if k in ("self_intersections", "flipped_ratio", "nonmanifold_edges",
                  "outside_by_winding", "outside_by_face_normal")},
        ensure_ascii=False))
    print("\n=== 分层 GLB ===", json.dumps(ta.get("layers") or {},
                                           ensure_ascii=False))

    print("\n=== 产物可下载性 ===")
    ok = True
    for name in ARTIFACTS:
        rr = c.get(f"/api/crown/runs/{rid}/artifact/{name}")
        size = len(rr.content) if rr.status_code == 200 else "-"
        print("  %-24s %s %s" % (name, rr.status_code, size))
        if rr.status_code == 200 and name.endswith(".glb"):
            head = rr.content[:4]
            ok &= head == b"glTF"
            print("      GLB 魔数:", head)
    print("GLB 头检查:", "OK" if ok else "FAILED")

    out = Path(__file__).with_name("e2e_result.json")
    out.write_text(json.dumps(
        {"run_id": rid, "status": d["status"], "results": res,
         "log": d.get("log")}, ensure_ascii=False, indent=1), encoding="utf-8")
    print("\n写出", out)
    return 0 if d["status"] == "done" else 1


if __name__ == "__main__":
    raise SystemExit(main())
