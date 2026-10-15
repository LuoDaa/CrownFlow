#!/usr/bin/env python3
"""诊断 15 条 catastrophic miss：正确候选是否存在于候选空间、能否组成可行环。

**不改算法**，只用冻结配置下的候选场与硬门做三件事：

1. 逐扇区找"离 GT Margin 最近"的候选（池子 = 全部候选，不是 Top-K），
   记录距离与其**分数排名** → within 0.10/0.20/0.50 召回、是否每个扇区都有；
2. 把这些 GT-nearest 候选串成"GT 路径"，检查**冻结硬门**（|Δh|/|Δr| + 离散度）
   能否逐段通过 → 用 freezing 的 gate 回答"正确环在不在可行空间里"；
3. 对照：同一行的 DP 实际选中路径在这些扇区里排第几。

结论分流：
  A 正确候选存在且可组成可行环 → 纯 Safety Gate（阈值/置信）即可解决安全性
  B 正确候选不存在/不可行     → 需要改 candidate generator 或 learned ranker
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys

sys.path.insert(0, "/tmp/gt")

from app.geometry.auto_margin import candidate_band, features, prep_frame   # noqa: E402
from app.geometry.auto_margin.graph_search import _sectors_spanned          # noqa: E402
from app.geometry.auto_margin.models import AutoMarginConfig, frozen_config_id  # noqa: E402
from app.geometry.auto_margin.projection import point_to_polyline_dist      # noqa: E402
from build_auto_margin_gt import case_dirs, content_hash, read_stl          # noqa: E402

TAUS = (0.10, 0.20, 0.50)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--eval-csv", required=True)
    ap.add_argument("--min-p95", type=float, default=1.0)
    a = ap.parse_args()

    rows = [r for r in csv.DictReader(open(a.eval_csv, encoding="ascii"))
            if r["status"] == "AUTO_ACCEPTED" and r.get("p95_mm")
            and float(r["p95_mm"]) > a.min_p95]
    print("catastrophic 行 %d（AUTO_ACCEPTED 且 p95 > %.1fmm）"
          % (len(rows), a.min_p95))
    man = json.load(open(os.path.join(a.out, "blind_generation_manifest.json"),
                         encoding="ascii"))
    entries = {(e["case_uuid"], str(e["fdi"])): e
               for e in man["by_source"]["modeldie"]}
    evl = json.load(open(os.path.join(a.out, "blind_evaluation_manifest.json"),
                         encoding="ascii"))
    gt_of = {(e["case_uuid"], str(e["fdi"])): e
             for e in evl["by_source"]["modeldie"]}
    cfg = AutoMarginConfig(**{k: v for k, v in
                              dict(candidate_mode="faces", top_k=24,
                                   score="sharp", method="dp",
                                   max_dh_mm=0.301, max_dr_mm=0.219,
                                   gate_disc_factor=1.0, lambda_h=1.0,
                                   lambda_r=1.0, lambda_p=0.6,
                                   huber_delta_mm=0.35, n_sectors=120,
                                   resample_n=240).items()})
    print("frozen config_id:", frozen_config_id(cfg))
    print()
    summ = {"n": 0, "recall_all_020": 0, "recall_all_050": 0,
            "gate_feasible": 0, "n_gate_blocked": 0, "ranks": [],
            "max_dh": [], "max_dr": []}
    for r in sorted(rows, key=lambda x: x["case_uuid"]):
        key = (r["case_uuid"], str(r["fdi"]))
        e = entries.get(key)
        gt = gt_of.get(key, {}).get("margin_gt")
        if not e or not gt:
            continue
        die = None
        for d in case_dirs(key[0]):
            for dp, _dn, fns in os.walk(d):
                for f in fns:
                    q = os.path.join(dp, f)
                    if f.lower().endswith(".stl") and content_hash(q) == e["die_asset_id"]:
                        die = q
                        break
                if die:
                    break
            if die:
                break
        if die is None:
            continue
        tris = read_stl(die)
        cents = [((t[0][0]+t[1][0]+t[2][0])/3.0, (t[0][1]+t[1][1]+t[2][1])/3.0,
                  (t[0][2]+t[1][2]+t[2][2])/3.0) for t in tris]
        frame = prep_frame.build_frame(cents, tuple(e["axis"]))
        stats, adj = features.face_stats(tris, frame)
        band = candidate_band.candidate_facet_ids(stats, cfg)
        by_id = {s.fid: s for s in stats}
        # 逐扇区：全池里离 GT 最近的候选 + 它的分数排名
        pool = {}
        for fid in band:
            for sec in _sectors_spanned(by_id[fid], frame, tris[fid],
                                        cfg.n_sectors):
                pool.setdefault(sec, []).append(fid)
        secs = sorted(pool)
        gt_pts = [tuple(p) for p in gt]
        best_of = {}
        ranks = []
        for sec in secs:
            # 排名必须用**冻结 score**（M1_V2_FROZEN 里是 geometry_score 的
            # "sharp" = FaceStat.sharp_deg 本身；不是 sharp_axis）
            order = sorted(pool[sec], key=lambda f: -by_id[f].sharp_deg)
            best_d, best_fid = float("inf"), None
            for fid in order:
                dd = point_to_polyline_dist(by_id[fid].centroid, gt_pts,
                                            closed=True)
                if dd < best_d:
                    best_d, best_fid = dd, fid
            ranks.append(order.index(best_fid) + 1)
            best_of[sec] = (best_fid, best_d)
        # 召回（扇区级）
        hit = {t: sum(1 for _s, (_f, d) in best_of.items() if d <= t)
               for t in TAUS}
        n_sec = len(best_of)
        # GT 路径能否过冻结硬门
        disc = getattr(cfg, "_face_scale_mm", 0.0)
        disc = features.face_scale_mm(stats) * cfg.gate_disc_factor
        max_dh = cfg.max_dh_mm + disc
        max_dr = cfg.max_dr_mm + disc
        dh_max = dr_max = 0.0
        feasible = True
        prev = None
        for sec in secs:
            fid, _d = best_of[sec]
            st = by_id[fid]
            if prev is not None:
                dh = abs(st.h - prev.h)
                dr = abs(st.r - prev.r)
                dh_max = max(dh_max, dh)
                dr_max = max(dr_max, dr)
                if dh > max_dh or dr > max_dr:
                    feasible = False
            prev = st
        summ["n"] += 1
        summ["ranks"].append(sorted(ranks)[len(ranks)//2])
        summ["max_dh"].append(dh_max)
        summ["max_dr"].append(dr_max)
        if hit[0.20] == n_sec:
            summ["recall_all_020"] += 1
        if hit[0.50] == n_sec:
            summ["recall_all_050"] += 1
        if feasible:
            summ["gate_feasible"] += 1
        else:
            summ["n_gate_blocked"] += 1
        print("%s fdi=%-3s GT候选召回(扇区级) 0.10=%.2f 0.20=%.2f 0.50=%.2f | "
              "GT候选排名中位=%d | 硬门 maxΔh=%.3f/%.3f maxΔr=%.3f/%.3f → %s"
              % (key[0][:14], key[1], hit[0.10]/n_sec, hit[0.20]/n_sec,
                 hit[0.50]/n_sec, sorted(ranks)[len(ranks)//2], dh_max, max_dh,
                 dr_max, max_dr, "可行环 ✅" if feasible else "被硬门挡 ❌"))
    print()
    print("=== 汇总（%d 行）===" % summ["n"])
    if summ["n"]:
        print("  全扇区召回 @0.20mm: %d/%d" % (summ["recall_all_020"], summ["n"]))
        print("  全扇区召回 @0.50mm: %d/%d" % (summ["recall_all_050"], summ["n"]))
        print("  GT 候选能组成合法环（过冻结硬门）: %d/%d"
              % (summ["gate_feasible"], summ["n"]))
        print("  被硬门挡: %d" % summ["n_gate_blocked"])
        rs = sorted(summ["ranks"])
        print("  GT 候选的分数排名中位: p10=%d p50=%d p90=%d max=%d"
              % (rs[int(.1*len(rs))], rs[len(rs)//2], rs[int(.9*len(rs))], rs[-1]))


if __name__ == "__main__":
    main()
