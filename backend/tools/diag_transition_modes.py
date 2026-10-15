#!/usr/bin/env python3
"""② transition 三模式验收（用户 2026-10-01 规格）。

第一验收项：6 条 GT-infeasible catastrophic，GT 路径在三种模式下是否可行
    V2_FROZEN 0/6 → MESH_AWARE ?/6 → HYBRID ?/6（成功线 ≥5/6）

第三诊断：把每条拆成
    A GT 路径现在可行 + DP 选对      → transition 修复成功
    B GT 路径现在可行 + DP 仍选错    → 归到 ranking / Safety Gate
    C GT 路径仍不可行                → transition 表达仍不足

第二验收项（--phase good）：140 条 GOOD 回归，看有没有被放开到错误结构上。

用法:
    python3 diag_transition_modes.py --out /tmp/automargin --phase cat6
    python3 diag_transition_modes.py --out /tmp/automargin --phase good --shard 1/4
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys

sys.path.insert(0, "/tmp/gt")

from app.geometry.auto_margin import (candidate_band, engine, features,    # noqa: E402
                                      graph_search, prep_frame)
from app.geometry.auto_margin.graph_search import _sectors_spanned         # noqa: E402
from app.geometry.auto_margin.models import AutoMarginConfig, M1_V2_FROZEN  # noqa: E402
from app.geometry.auto_margin.projection import point_to_polyline_dist     # noqa: E402
from build_auto_margin_gt import case_dirs, content_hash, read_stl         # noqa: E402

MODES = ("V2_FROZEN", "MESH_AWARE", "HYBRID")


def load(out):
    man = json.load(open(os.path.join(out, "blind_generation_manifest.json"),
                         encoding="ascii"))
    evl = json.load(open(os.path.join(out, "blind_evaluation_manifest.json"),
                         encoding="ascii"))
    ev = {r["case_uuid"] + "|" + str(r["fdi"]): r
          for r in csv.DictReader(open(os.path.join(out, "blind_eval_modeldie.csv"),
                                       encoding="ascii"))}
    ent = {(e["case_uuid"], str(e["fdi"])): e
           for e in man["by_source"]["modeldie"]}
    gt = {(e["case_uuid"], str(e["fdi"])): e
          for e in evl["by_source"]["modeldie"]}
    return ent, gt, ev


def resolve_die(key, e):
    for d in case_dirs(key[0]):
        for dp, _dn, fns in os.walk(d):
            for f in fns:
                q = os.path.join(dp, f)
                if f.lower().endswith(".stl") and content_hash(q) == e["die_asset_id"]:
                    return q
    return None


def gt_path_feasible(stats, adj, frame, tris, cfg, gt_pts, surf, mode):
    """逐扇区取离 GT 最近的候选 → 用**该模式的 transition_cost** 检查可行环。"""
    band = candidate_band.candidate_facet_ids(stats, cfg)
    by_id = {s.fid: s for s in stats}
    pool = {}
    for fid in band:
        for sec in _sectors_spanned(by_id[fid], frame, tris[fid],
                                    cfg.n_sectors):
            pool.setdefault(sec, []).append(fid)
    secs = sorted(pool)
    chosen = {}
    for sec in secs:
        best_d, best_f = float("inf"), None
        for fid in pool[sec]:
            d = point_to_polyline_dist(by_id[fid].centroid, gt_pts, closed=True)
            if d < best_d:
                best_d, best_f = d, fid
        chosen[sec] = best_f
    rs = [by_id[f].r for f in chosen.values()]
    expected = 2.0 * math.pi * (sum(rs) / max(len(rs), 1)) / cfg.n_sectors
    bad = 0
    max_cost = 0.0
    for i in range(len(secs)):
        a = by_id[chosen[secs[i]]]
        b = by_id[chosen[secs[(i + 1) % len(secs)]]]
        c = graph_search.transition_cost(mode, a, b, cfg, surf, expected)
        if c is None:
            bad += 1
        else:
            max_cost = max(max_cost, c)
    return bad == 0, bad, max_cost


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--phase", default="cat6", choices=["cat6", "good"])
    ap.add_argument("--shard", default="1/1")
    ap.add_argument("--min-p95", type=float, default=1.0)
    a = ap.parse_args()
    ent, gt, ev = load(a.out)
    keys = []
    for k, r in ev.items():
        if r["status"] != "AUTO_ACCEPTED" or not r.get("p95_mm"):
            continue
        p95 = float(r["p95_mm"])
        if a.phase == "cat6" and p95 > a.min_p95:
            u, f = k.split("|")
            keys.append((u, f))
        if a.phase == "good" and p95 <= 0.5:
            u, f = k.split("|")
            keys.append((u, f))
    i, n = (a.shard.split("/") + ["1"])[:2]
    keys = sorted(keys)[int(i) - 1::int(n)]
    print("phase=%s 行数 %d（shard %s）" % (a.phase, len(keys), a.shard),
          flush=True)
    out_rows = []
    for key in keys:
        e, g = ent.get(key), gt.get(key)
        if not e or not g or not g.get("margin_gt"):
            continue
        die = resolve_die(key, e)
        if die is None:
            continue
        tris = read_stl(die)
        cents = [((t[0][0]+t[1][0]+t[2][0])/3.0, (t[0][1]+t[1][1]+t[2][1])/3.0,
                  (t[0][2]+t[1][2]+t[2][2])/3.0) for t in tris]
        frame = prep_frame.build_frame(cents, tuple(e["axis"]))
        stats, adj = features.face_stats(tris, frame)
        eadj = features.build_edge_adjacency(tris)
        surf = graph_search.SurfaceDistance(
            tris, eadj, {s.fid: s.centroid for s in stats})
        gt_pts = [tuple(p) for p in g["margin_gt"]]
        line = ["%s fdi=%s" % (key[0][:14], key[1])]
        feas = {}
        for mode in MODES:
            cfg = AutoMarginConfig(**dict(M1_V2_FROZEN, transition_mode=mode,
                                          fallback_mode="REVIEW"))
            cfg = __import__("dataclasses").replace(
                cfg, _face_scale_mm=features.face_scale_mm(stats))
            ok, bad, mc = gt_path_feasible(stats, adj, frame, tris, cfg,
                                           gt_pts, surf, mode)
            feas[mode] = ok
            res = engine.predict_margin(tris, tuple(e["axis"]), cfg,
                                       stats=stats, adj=adj, frame=frame)
            st = res.status
            # 选中路径到 GT 的距离（有曲线时）
            sel = None
            if res.margin is not None and len(res.margin.points) >= 3:
                d = [point_to_polyline_dist(p, gt_pts, closed=True)
                     for p in res.margin.points]
                d.sort()
                sel = d[min(len(d) - 1, int(.95 * (len(d) - 1)))]
            line.append("%s: GTpath=%s(%d段被挡) DP=%s p95=%s"
                        % (mode, "可行" if ok else "不可行", bad, st,
                           ("%.3f" % sel) if sel is not None else "-"))
            out_rows.append({"case_uuid": key[0], "fdi": key[1],
                             "phase": a.phase, "mode": mode,
                             "gt_path_feasible": ok, "n_blocked_segments": bad,
                             "dp_status": st,
                             "selected_p95_mm": (round(sel, 4)
                                                 if sel is not None else None),
                             "gt_p95_mm": ev[key[0] + "|" + key[1]]["p95_mm"]})
        print("  " + " | ".join(line), flush=True)
        # 分类（仅 cat6）
        if a.phase == "cat6" and feas["V2_FROZEN"] is False:
            fixed = feas["MESH_AWARE"] or feas["HYBRID"]
            dp_ok = any(r["selected_p95_mm"] is not None
                        and r["selected_p95_mm"] <= 0.5
                        for r in out_rows[-3:])
            tag = ("A transition修复+DP选对" if fixed and dp_ok else
                   "B transition已修但DP仍选错" if fixed else
                   "C GT路径仍不可行")
            print("       → 归类: %s" % tag, flush=True)
    p = os.path.join(a.out, "transition_modes_%s%s.csv"
                     % (a.phase, "" if a.shard == "1/1" else "_shard%s" % i))
    cols = sorted({k for r in out_rows for k in r})
    with open(p, "w", newline="", encoding="ascii") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in out_rows:
            w.writerow(r)
    print("写出", p)


if __name__ == "__main__":
    main()
