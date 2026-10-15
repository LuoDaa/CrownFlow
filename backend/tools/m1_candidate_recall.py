#!/usr/bin/env python3
"""M1 v3 候选召回诊断 —— **v3 的第一指标，先于最终 p95 看**。

对每一行、每个扇区，问同一个问题：

    "该扇区给出的候选里，有没有一个点落在 GT Margin 的 τ 邻域内？"
    τ ∈ {0.10, 0.20, 0.50} mm

三级统计：扇区级（分母 = 有候选的扇区数）、行级（该行扇区召回）、病例级（该病例
的均值）。同时对照两种候选生成器（**同一网格、同一特征、同一扇区划分**）：

    faces   = v2（每扇区按分数 Top-24 的**面**）
    ridges  = v3（每扇区 Top-6 的 ridge **簇**）

用户规格：若 v3 明显提高 recall 且 DP 成功率同步提高 → 方向正确；若 GT 附近
仍经常不在候选里，说明是**纯几何特征看不到 Margin**，那就该进 M2。
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import math
import os
import sys
import time

sys.path.insert(0, "/tmp/gt")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.geometry.auto_margin import (candidate_band, features, prep_frame,   # noqa: E402
                                      ridge_cluster)
from app.geometry.auto_margin.models import AutoMarginConfig                  # noqa: E402
from app.geometry.auto_margin.projection import point_to_polyline_dist        # noqa: E402
from build_auto_margin_gt import read_stl, case_dirs, _hash_to_path           # noqa: E402

TAUS = (0.10, 0.20, 0.50)
COLS = ("case_uuid", "fdi", "split", "mode", "k", "n_sectors",
        "recall_0p10", "recall_0p20", "recall_0p50",
        "sectors_0p10", "sectors_0p20", "sectors_0p50", "elapsed_s")


def recall_row(sector_cands, point_of, gt):
    """→ (n_sectors, {tau: 命中扇区数})"""
    n, hit = 0, {t: 0 for t in TAUS}
    for _sec, items in sector_cands.items():
        if not items:
            continue
        n += 1
        best = min(point_to_polyline_dist(point_of(cid), gt, closed=True)
                   for _sc, cid in items)
        for t in TAUS:
            if best <= t:
                hit[t] += 1
    return n, hit


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", default="train,val")
    ap.add_argument("--shard", default="1/1")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--allow-seen", action="store_true",
                    help="允许跑 test（它已升级为 hard validation set，可重复看）")
    a = ap.parse_args()
    splits = [x.strip() for x in a.split.split(",") if x.strip()]
    if "test" in splits and not a.allow_seen:
        raise SystemExit("test 现在是 hard validation set —— 需要 --allow-seen 显式确认")

    ok = [r for r in csv.DictReader(
        open(os.path.join(a.out, "auto_margin_gt.csv"), encoding="ascii"))
        if r["qc_status"] == "OK"]
    man = json.load(open(os.path.join(a.out, "split_manifest.json")))
    sel = {(r["case_uuid"], r["fdi"]): r for r in csv.DictReader(
        open(os.path.join(a.out, "die_selection.csv"), encoding="ascii"))}
    rows = [r for r in ok if man["assign"].get(r["group_id"]) in splits]
    i, n = (a.shard.split("/") + ["1"])[:2]
    rows = rows[int(i) - 1::int(n)]
    if a.limit:
        rows = rows[:a.limit]
    cfg_faces = AutoMarginConfig(candidate_mode="faces", top_k=24)
    cfg_ridge = AutoMarginConfig(candidate_mode="ridges")
    print("召回诊断 %d 行（splits=%s shard=%s）" % (len(rows), splits, a.shard),
          flush=True)
    # 注意：**不要**与 run_auto_margin_m1.py --recall 的输出同名
    # （踩过：两个进程交叉写同一个文件，CSV 表头互相覆盖）
    path = os.path.join(a.out, "m1_cand_recall%s.csv"
                        % ("" if a.shard == "1/1" else "_shard%s" % i))
    t_all = time.time()
    done = 0
    with open(path, "w", newline="", encoding="ascii") as fh:
        w = csv.DictWriter(fh, fieldnames=list(COLS))
        w.writeheader()
        for r in rows:
            key = (r["case_uuid"], r["fdi"])
            s = sel.get(key)
            if not s or not s.get("die_asset_id"):
                continue
            die = _hash_to_path(case_dirs(key[0])).get(s["die_asset_id"])
            if die is None:
                continue
            tris = read_stl(die)
            cv = json.load(open(os.path.join(
                a.out, "labels",
                r["prep_mesh_ref"].replace("_labels.bin", "_curves.json"))))
            gt = [tuple(p) for p in (cv.get("projected") or cv["margin_gt"])]
            cents = [((t[0][0]+t[1][0]+t[2][0])/3.0,
                      (t[0][1]+t[1][1]+t[2][1])/3.0,
                      (t[0][2]+t[1][2]+t[2][2])/3.0) for t in tris]
            frame = prep_frame.build_frame(cents, tuple(cv["axis"]))
            stats, adj = features.face_stats(tris, frame)
            cand_ids = candidate_band.candidate_facet_ids(stats, cfg_ridge)
            by_id = {s0.fid: s0 for s0 in stats}
            for mode in ("faces", "ridges"):
                t0 = time.time()
                if mode == "faces":
                    sc = candidate_band.sector_candidates(
                        stats, adj, frame, cfg_faces, cand_ids, tris)
                    k = cfg_faces.top_k
                    point_of = lambda cid: by_id[cid].centroid
                else:
                    cl, sc = ridge_cluster.build_clusters(
                        stats, adj, frame, cfg_ridge, cand_ids, tris)
                    k = cfg_ridge.ridge_top_k
                    cl_by = {c.cid: c for c in cl}
                    point_of = lambda cid: cl_by[cid].centroid
                ns, hit = recall_row(sc, point_of, gt)
                rec = {c2: "" for c2 in COLS}
                rec.update(case_uuid=r["case_uuid"], fdi=r["fdi"],
                           split=man["assign"].get(r["group_id"]), mode=mode,
                           k=k, n_sectors=ns,
                           elapsed_s=round(time.time()-t0, 1))
                for t in TAUS:
                    key2 = "0p%02d" % int(round(t * 100))
                    rec["recall_" + key2] = (round(hit[t] / ns, 4)
                                             if ns else None)
                    rec["sectors_" + key2] = hit[t]
                w.writerow(rec)
                fh.flush()
            done += 1
            if done % 5 == 0:
                print("  %d/%d（%.0fs）" % (done, len(rows),
                                           time.time() - t_all), flush=True)
    print("完成 %d 行 → %s（%.0fs）" % (done, path, time.time() - t_all))


if __name__ == "__main__":
    main()
