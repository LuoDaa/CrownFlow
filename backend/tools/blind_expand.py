#!/usr/bin/env python3
"""Blind 数据扩容 —— **只做输入合法性 QC，绝不看 GT 误差/标签/可视化**。

背景：AutoMarginGTSet v1 的 44 病例已全部曝光（train/val/hard-val），final blind
需要全新病例。v1 的 122 行全部来自"只有 modelInfo"的病例；几何选代型对**双快照
病例**（同时有 constructionInfo + modelInfo）用的是 cis[0] = constructionInfo 的
margin（**扫描系**），投到模型系的代型上必然失败（BEST_STILL_OFF p50=7.9mm）。

本脚本对每个**未使用**病例的**每份快照分别**做 margin→候选代型投影（modeldie 或
modelbase 模式），从而救回双快照病例 —— 这是输入 QC（用户 blind 流程明确允许
"Margin projection QC"），不是 GT 误差评估。

模式：
  --source modeldie   候选 = *modeldie*.stl（主路线）
  --source modelbase  候选 = *modelbase*.stl（探针：模型基座可能含预备体表面）

输出 blind_expand_<source>.csv（PHI-free：case_uuid 哈希 + 计数 + 误差）。
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time

sys.path.insert(0, "/tmp/gt")

from build_auto_margin_gt import (TriIndex, case_dirs, content_hash,  # noqa: E402
                                  read_stl, INDEX_CSV)
from app.importers.exocad.construction_info import parse_construction_case  # noqa: E402

COLS = ("case_uuid", "fdi", "best_method", "best_asset_id", "n_snapshots", "n_candidates",
        "best_p50_mm", "best_p95_mm", "best_max_mm", "best_n_faces",
        "decision", "skipped_big", "elapsed_s")


def snapshots(dirs):
    """病例里全部设计快照，各自独立解析（双快照病例是本轮的救回目标）。"""
    out = []
    for d in dirs:
        for dp, _dn, fns in os.walk(d):
            for f in fns:
                lf = f.lower()
                if lf.endswith(".constructioninfo"):
                    out.append((os.path.join(dp, f), "CONSTRUCTION_INFO"))
                elif lf.endswith(".modelinfo"):
                    out.append((os.path.join(dp, f), "MODEL_INFO"))
    return sorted(out)


def stl_candidates(dirs, needle, max_files=8):
    out = {}
    for d in dirs:
        for dp, _dn, fns in os.walk(d):
            for f in fns:
                lf = f.lower()
                if lf.endswith(".stl") and needle in lf:
                    p = os.path.join(dp, f)
                    try:
                        h = content_hash(p)
                    except OSError:
                        continue
                    out.setdefault(h, p)
                    if len(out) >= max_files:
                        return out
    return out


def project_stats(idx, pts):
    ds = []
    for p in pts:
        d, _f = idx.nearest(p)
        if d is None:
            return None
        ds.append(d)
    ds.sort()
    n = len(ds)
    return ds[n // 2], ds[min(n - 1, int(.95 * (n - 1)))], ds[-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--source", default="modeldie",
                    choices=["modeldie", "modelbase"])
    ap.add_argument("--shard", default="1/1")
    ap.add_argument("--max-tris", type=int, default=300_000)
    a = ap.parse_args()

    used = set()
    p = os.path.join(a.out, "auto_margin_gt.csv")
    if os.path.isfile(p):
        for r in csv.DictReader(open(p, encoding="ascii")):
            used.add(r["group_id"])
    rows = [r for r in csv.DictReader(open(INDEX_CSV, encoding="ascii"))
            if r["has_margin"] == "True"]
    by_case = {}
    for r in rows:
        if r["case_uuid"] not in used:
            by_case.setdefault(r["case_uuid"], []).append(r)
    cases = sorted(by_case)
    i, n = (a.shard.split("/") + ["1"])[:2]
    mine = cases[int(i) - 1::int(n)]
    print("未使用病例 %d（全库 %d）；本分片 %d；source=%s"
          % (len(cases), len(used) and len({r["case_uuid"] for r in rows}),
             len(mine), a.source), flush=True)

    needle = "modeldie" if a.source == "modeldie" else "modelbase"
    path = os.path.join(a.out, "blind_expand_%s%s.csv"
                        % (a.source, "" if a.shard == "1/1"
                           else "_shard%s" % i))
    # 断点续跑：本文件里已出现过的病例不再重算
    already = set()
    if os.path.isfile(path):
        for r in csv.DictReader(open(path, encoding="ascii")):
            already.add(r["case_uuid"])
        mine = [u for u in mine if u not in already]
        print("  续跑：跳过已写 %d 病例，剩 %d" % (len(already), len(mine)),
              flush=True)
    mode = "a" if already else "w"
    t_all = time.time()
    done = 0
    with open(path, mode, newline="", encoding="ascii") as fh:
        w = csv.DictWriter(fh, fieldnames=list(COLS))
        if mode == "w":
            w.writeheader()
        for uuid in mine:
            t0 = time.time()
            dirs = case_dirs(uuid)
            if not dirs:
                continue
            cands = stl_candidates(dirs, needle)
            snaps = snapshots(dirs)
            if not cands or not snaps:
                continue
            # 每个候选只建一次索引（供该病例全部 (fdi, snapshot) 复用）
            built = {}
            skipped_big = 0
            for h, q in cands.items():
                try:
                    tris = read_stl(q, max_tris=a.max_tris)
                except Exception:                 # noqa: BLE001
                    skipped_big += 1
                    continue
                try:
                    built[h] = (TriIndex(tris), len(tris))
                except Exception:                 # noqa: BLE001
                    continue
            if not built:
                continue
            best = {}
            for snap_path, method in snaps:
                try:
                    case = parse_construction_case(snap_path)
                except Exception:                 # noqa: BLE001
                    continue
                for r in by_case[uuid]:
                    try:
                        fdi = int(r["fdi"])
                    except ValueError:
                        continue
                    tooth = case.tooth(fdi)
                    if tooth is None or not tooth.margin_points \
                            or not tooth.insertion_axis:
                        continue
                    margin = [(float(p[0]), float(p[1]), float(p[2]))
                              for p in tooth.margin_points]
                    sub = (margin if len(margin) <= 48
                           else margin[::max(1, len(margin) // 48)])
                    for h, (idx, nf) in built.items():
                        st = project_stats(idx, sub)
                        if st is None:
                            continue
                        p50s, p95s, _mx = st
                        if p50s > 1.0:            # 早停：粗筛就远 → 不做全量
                            continue
                        full = project_stats(idx, margin)
                        if full is None:
                            continue
                        p50, p95, mx = full
                        key = (uuid, r["fdi"])
                        cur = best.get(key)
                        if cur is None or p95 < cur[2]:
                            best[key] = (method, len(built), p50, p95, mx, nf,
                                         h)
            for (uuid2, fdi), (m, nc, p50, p95, mx, nf, bh) in sorted(
                    best.items()):
                rec = {k: "" for k in COLS}
                rec.update(case_uuid=uuid2, fdi=fdi, best_method=m,
                           best_asset_id=bh,
                           n_snapshots=len(snaps), n_candidates=nc,
                           best_p50_mm=round(p50, 4), best_p95_mm=round(p95, 4),
                           best_max_mm=round(mx, 4), best_n_faces=nf,
                           decision=("MARGIN_ON_DIE" if p95 <= 0.20
                                     else "BEST_STILL_OFF"),
                           skipped_big=skipped_big,
                           elapsed_s=round(time.time() - t0, 1))
                w.writerow(rec)
                fh.flush()
            done += 1
            if done % 10 == 0:
                print("  %d/%d（%.0fs）" % (done, len(mine),
                                           time.time() - t_all), flush=True)
    print("完成 %d 病例 → %s（%.0fs）" % (done, path, time.time() - t_all))


if __name__ == "__main__":
    main()
