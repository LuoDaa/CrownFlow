#!/usr/bin/env python3
"""从 **train** 的 GT Margin 分布推导 v2 的跳变硬门（不是拍脑袋）。

做法：对每一行 train 病例，按与引擎**完全相同**的方式建立坐标系与扇区，
把 GT Margin 点分到扇区，统计**相邻扇区**的 Δh / Δr：

    p50 / p90 / p95 / p99 / max

然后用 ``p99 + 0.5·(p99 − p50)`` 作为硬门（对重尾稳健；用户规格"p99 + margin"）。
写到 `/tmp/automargin/m1_gates.json`，供 v2 配置引用。

**这不是 GT leakage**：这是在 train 上做**参数设计**（用户明确允许）；
val 只用于在给定参数下选择方案，test 全程封存。
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys

sys.path.insert(0, "/tmp/gt")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.geometry.auto_margin import candidate_band, features, prep_frame  # noqa: E402
from app.geometry.auto_margin.models import AutoMarginConfig               # noqa: E402
from build_auto_margin_gt import (read_stl, case_dirs, _hash_to_path)      # noqa: E402


def _q(v, p):
    v = sorted(v)
    return v[min(len(v) - 1, int(p * (len(v) - 1)))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--shard", default="1/1")
    a = ap.parse_args()
    cfg = AutoMarginConfig()
    ok = [r for r in csv.DictReader(
        open(os.path.join(a.out, "auto_margin_gt.csv"), encoding="ascii"))
        if r["qc_status"] == "OK"]
    man = json.load(open(os.path.join(a.out, "split_manifest.json")))
    sel = {(r["case_uuid"], r["fdi"]): r for r in csv.DictReader(
        open(os.path.join(a.out, "die_selection.csv"), encoding="ascii"))}
    rows = [r for r in ok if man["assign"].get(r["group_id"]) == "train"]
    i, n = (a.shard.split("/") + ["1"])[:2]
    rows = rows[int(i) - 1::int(n)]
    print("train 行数 %d（shard %s）" % (len(rows), a.shard), flush=True)

    dh_all, dr_all = [], []
    for r in rows:
        key = (r["case_uuid"], r["fdi"])
        s = sel.get(key)
        if not s:
            continue
        die = _hash_to_path(case_dirs(key[0])).get(s["die_asset_id"])
        if die is None:
            continue
        tris = read_stl(die)
        cents = [((t[0][0]+t[1][0]+t[2][0])/3.0, (t[0][1]+t[1][1]+t[2][1])/3.0,
                  (t[0][2]+t[1][2]+t[2][2])/3.0) for t in tris]
        cv = json.load(open(os.path.join(
            a.out, "labels",
            r["prep_mesh_ref"].replace("_labels.bin", "_curves.json"))))
        gt = [tuple(p) for p in (cv.get("projected") or cv["margin_gt"])]
        frame = prep_frame.build_frame(cents, tuple(cv["axis"]))
        step = 2.0 * math.pi / cfg.n_sectors
        per = {}
        for p in gt:
            h, rad, th = frame.coords(p)
            per.setdefault(int(th / step) % cfg.n_sectors, []).append((h, rad))
        secs = sorted(per)
        if len(secs) < 12:
            continue
        # 同扇区可能落多个 GT 点 → 取均值
        hh = {s0: sum(x[0] for x in per[s0]) / len(per[s0]) for s0 in secs}
        rr = {s0: sum(x[1] for x in per[s0]) / len(per[s0]) for s0 in secs}
        for k in range(len(secs)):
            a0, b0 = secs[k], secs[(k + 1) % len(secs)]
            dh_all.append(abs(hh[b0] - hh[a0]))
            dr_all.append(abs(rr[b0] - rr[a0]))
        print("  %s fdi=%s GT 扇区 %d" % (key[0][:14], key[1], len(secs)),
              flush=True)

    out = {"n_train_rows": len(rows), "n_dh_samples": len(dh_all),
           "dh_mm": {}, "dr_mm": {}}
    for name, vals in (("dh_mm", dh_all), ("dr_mm", dr_all)):
        if not vals:
            continue
        d = {"p50": round(_q(vals, .5), 4), "p90": round(_q(vals, .9), 4),
             "p95": round(_q(vals, .95), 4), "p99": round(_q(vals, .99), 4),
             "max": round(max(vals), 4)}
        # 硬门 = p99 + 0.5·(p99 − p50)（对重尾稳健）
        d["gate"] = round(d["p99"] + 0.5 * (d["p99"] - d["p50"]), 4)
        out[name] = d
    path = os.path.join(a.out, "m1_gates%s.json"
                        % ("" if a.shard == "1/1" else "_shard%s" % i))
    json.dump(out, open(path, "w"), indent=1)
    print(json.dumps(out, indent=1))
    print("写盘", path)


if __name__ == "__main__":
    main()
