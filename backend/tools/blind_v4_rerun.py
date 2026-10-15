#!/usr/bin/env python3
"""①′：在 **M1_V4_HARD_DEV_SET（原 Blind v1，198 行）** 上用冻结的 v4 transition
（HYBRID）完整重算一遍 —— 产出迁移矩阵、三信号新分布、Safety-Coverage curve。

**只做分析，不改任何阈值**；本数据集已开过 GT（已降级为 Hard Dev），
所以这里允许读 evaluation manifest，但结论只用于确定 Gate 的"形态"。

用法:
    python3 blind_v4_rerun.py --out /tmp/automargin --source modeldie \
        --transition HYBRID --shard 1/4
"""
from __future__ import annotations

import argparse
import dataclasses
import csv
import json
import math
import os
import sys
import time

sys.path.insert(0, "/tmp/gt")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.geometry.auto_margin import engine, features, prep_frame      # noqa: E402
from app.geometry.auto_margin.graph_search import _sectors_spanned     # noqa: E402
from app.geometry.auto_margin.models import (AutoMarginConfig,         # noqa: E402
                                             M1_V2_PRODUCTION_CANDIDATE,
                                             frozen_config_id)
from app.geometry.auto_margin.projection import point_to_polyline_dist  # noqa: E402
from build_auto_margin_gt import case_dirs, content_hash, read_stl     # noqa: E402

COLS = ("case_uuid", "fdi", "transition", "config_id", "status", "reason",
        "p95_mm", "median_mm", "hausdorff_mm",
        "energy_gap", "second_energy", "second_separation_mm",
        "ambiguous_second_cycle", "chosen_rank_mean", "chosen_rank_max",
        "max_transition_mm", "filled_sector_frac", "n_self_intersections",
        "elapsed_s")


def resolve_die(uuid, want):
    for d in case_dirs(uuid):
        for dp, _dn, fns in os.walk(d):
            for f in fns:
                q = os.path.join(dp, f)
                if f.lower().endswith(".stl") and content_hash(q) == want:
                    return q
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--source", default="modeldie")
    ap.add_argument("--transition", default="HYBRID",
                    choices=["V2_FROZEN", "MESH_AWARE", "HYBRID"])
    ap.add_argument("--shard", default="1/1")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()

    gen = json.load(open(os.path.join(a.out, "blind_generation_manifest.json"),
                         encoding="ascii"))
    evl = json.load(open(os.path.join(a.out, "blind_evaluation_manifest.json"),
                         encoding="ascii"))
    ent = {(e["case_uuid"], str(e["fdi"])): e
           for e in gen["by_source"][a.source]}
    gt = {(e["case_uuid"], str(e["fdi"])): e.get("margin_gt")
          for e in evl["by_source"][a.source]}
    keys = sorted(ent)
    i, n = (a.shard.split("/") + ["1"])[:2]
    keys = keys[int(i) - 1::int(n)]
    cfg = AutoMarginConfig(**dict(M1_V2_PRODUCTION_CANDIDATE,
                                  transition_mode=a.transition))
    print("frozen config: %s" % frozen_config_id(cfg), flush=True)
    print("行数 %d（shard %s）" % (len(keys), a.shard), flush=True)

    path = os.path.join(a.out, "blind_v4rerun_%s%s%s.csv"
                        % (a.transition.lower(),
                           ("_" + a.tag) if a.tag else "",
                           "" if a.shard == "1/1" else "_shard%s" % i))
    t_all = time.time()
    done = 0
    with open(path, "w", newline="", encoding="ascii") as fh:
        w = csv.DictWriter(fh, fieldnames=list(COLS))
        w.writeheader()
        for key in keys:
            e = ent[key]
            gtp = gt.get(key)
            t0 = time.time()
            rec = {k: "" for k in COLS}
            rec.update(case_uuid=key[0], fdi=key[1], transition=a.transition,
                       config_id=frozen_config_id(cfg))
            die = resolve_die(key[0], e["die_asset_id"])
            if die is None or not gtp:
                rec.update(status="FAILED", reason="input_unresolvable")
                w.writerow(rec)
                fh.flush()
                done += 1
                continue
            tris = read_stl(die)
            cents = [((t[0][0]+t[1][0]+t[2][0])/3.0,
                      (t[0][1]+t[1][1]+t[2][1])/3.0,
                      (t[0][2]+t[1][2]+t[2][2])/3.0) for t in tris]
            frame = prep_frame.build_frame(cents, tuple(e["axis"]))
            stats, adj = features.face_stats(tris, frame)
            res = engine.predict_margin(tris, tuple(e["axis"]), cfg,
                                        stats=stats, adj=adj, frame=frame)
            rec.update(status=res.status, reason=res.reason,
                       energy_gap=res.energy_gap,
                       second_energy=res.second_energy,
                       chosen_rank_mean=res.candidate_rank_stats.get("mean"),
                       chosen_rank_max=res.candidate_rank_stats.get("max"),
                       max_transition_mm=res.quality.get("max_transition_mm"),
                       filled_sector_frac=res.quality.get("filled_sector_frac"),
                       n_self_intersections=res.quality.get(
                           "n_self_intersections"),
                       elapsed_s=round(time.time() - t0, 1))
            # second_separation（结构性不同的次优环；engine 已算进 quality）
            rec["second_separation_mm"] = res.quality.get("second_separation_mm")
            rec["ambiguous_second_cycle"] = res.quality.get(
                "ambiguous_second_cycle")
            if res.margin is not None and len(res.margin.points) >= 3:
                d = [point_to_polyline_dist(p, [tuple(x) for x in gtp],
                                            closed=True)
                     for p in res.margin.points]
                d.sort()
                rec["p95_mm"] = round(d[min(len(d) - 1, int(.95 * (len(d) - 1)))],
                                      4)
                rec["median_mm"] = round(d[len(d) // 2], 4)
                rec["hausdorff_mm"] = round(max(d), 4)
            w.writerow(rec)
            fh.flush()
            done += 1
            if done % 5 == 0:
                print("  %d/%d（%.0fs）" % (done, len(keys),
                                           time.time() - t_all), flush=True)
    print("完成 %d 行 → %s（%.0fs）" % (done, path, time.time() - t_all))


if __name__ == "__main__":
    main()
