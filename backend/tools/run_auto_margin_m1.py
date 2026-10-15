#!/usr/bin/env python3
"""M1 Geometry Auto Margin —— v1(argmax) vs v2(cyclic DP) 对照评测。

    GT 只在 evaluate 里出现；生成阶段只看（代型网格 + 就位轴）。

用法（服务器）:
    python3 run_auto_margin_m1.py --out /tmp/automargin --split train,val \
        --scores sharp,normal_turn --method argmax,dp --shard 1/4 --recall

**test 分片默认封存**：不传 --allow-test 直接拒绝（用户规格：M1 参数冻结后再开）。
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time

sys.path.insert(0, "/tmp/gt")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.geometry.auto_margin import engine, features, prep_frame         # noqa: E402
from app.geometry.auto_margin.models import (AutoMarginConfig,           # noqa: E402
                                             frozen_config_id)
from app.geometry.auto_margin.projection import point_to_polyline_dist    # noqa: E402
from build_auto_margin_gt import read_stl, case_dirs, _hash_to_path       # noqa: E402

COLS = ("case_uuid", "fdi", "split", "config_id", "score", "method", "status", "n_points",
        "closed", "n_self_intersections", "filled_sector_frac",
        "median_mm", "p95_mm", "max_mm", "chamfer_mm", "hausdorff_mm",
        "frac_within_0p10", "frac_within_0p20",
        "gt_perimeter_mm", "pred_perimeter_mm", "perimeter_ratio",
        "proj_disp_p95_mm",
        # --- 生产语义 / 置信度 / provenance（2026-10-01）
        "energy_gap", "chosen_rank_mean", "chosen_rank_max",
        "max_transition_mm", "score_method", "transition_method",
        "elapsed_s", "notes")

RECALL_COLS = ("case_uuid", "fdi", "split", "score", "n_sectors",
               "gt_rank_p50", "gt_rank_p90", "gt_rank_p99",
               "frac_gt_in_top1", "frac_gt_in_top5",
               "n_bad_sectors", "frac_gt_in_top5_on_bad_sectors")


class GtGrid:
    """GT 折线点的均匀网格（用于**同一扇区内候选排序**的近似 GT 距离）。

    用"到 GT 点的最近距离"而非精确折线距离：它只用来排序（同扇区所有候选同一
    把尺子，结论不受影响），最终指标仍用精确的 ``point_to_polyline_dist``。
    """

    def __init__(self, pts, cell=1.0):
        self.pts = pts
        self.cell = cell
        self.bins = {}
        for i, p in enumerate(pts):
            self.bins.setdefault(self._k(p), []).append(i)

    def _k(self, p):
        return (int(p[0] // self.cell), int(p[1] // self.cell),
                int(p[2] // self.cell))

    def dist(self, p):
        k0 = self._k(p)
        best = float("inf")
        for ring in range(7):
            for dx in range(-ring, ring + 1):
                for dy in range(-ring, ring + 1):
                    for dz in range(-ring, ring + 1):
                        if max(abs(dx), abs(dy), abs(dz)) != ring:
                            continue
                        for i in self.bins.get((k0[0]+dx, k0[1]+dy,
                                                k0[2]+dz), ()):
                            d = math.dist(p, self.pts[i])
                            if d < best:
                                best = d
            if best < float("inf") and ring * self.cell > best:
                break
        return best


def _quant(v, p):
    if not v:
        return None
    v = sorted(v)
    return v[min(len(v) - 1, int(p * (len(v) - 1)))]


def metrics(pred_pts, gt_poly):
    d_p2g = [point_to_polyline_dist(p, gt_poly, closed=True) for p in pred_pts]
    d_g2p = [point_to_polyline_dist(q, pred_pts, closed=True) for q in gt_poly]
    return {
        "median_mm": round(_quant(d_p2g, .5), 4),
        "p95_mm": round(_quant(d_p2g, .95), 4),
        "max_mm": round(max(d_p2g), 4),
        "chamfer_mm": round(0.5 * (sum(d_p2g) / len(d_p2g)
                                   + sum(d_g2p) / len(d_g2p)), 4),
        "hausdorff_mm": round(max(max(d_p2g), max(d_g2p)), 4),
        "frac_within_0p10": round(sum(1 for d in d_p2g if d <= 0.10)
                                 / len(d_p2g), 4),
        "frac_within_0p20": round(sum(1 for d in d_p2g if d <= 0.20)
                                 / len(d_p2g), 4),
    }


def recall_stats(candidates, v1_chosen, gt, centroid_of):
    """候选召回诊断（用户规格的核心问题）。

    逐扇区：在**全部候选**（分数降序，比 Top-K 更大）里找离 GT 最近的那个，
    记它的分数排名。再单看"v1 选中的点离 GT > 0.5mm"的**大误差扇区**：
    这些扇区里 GT 是否仍在 Top-5？—— 是，则 DP 路线方向正确；否，则是
    特征/候选生成的问题，靠优化修不了。
    """
    grid = GtGrid(gt)
    ranks, bad_ranks = [], []
    for sec, items in candidates["full"].items():
        if not items:
            continue
        best_rank, best_d = None, float("inf")
        for idx, (_sc, fid) in enumerate(items):
            d = grid.dist(centroid_of(fid))
            if d < best_d:
                best_d, best_rank = d, idx + 1
        ranks.append(best_rank)
        cfid = v1_chosen.get(sec)
        if cfid is not None and grid.dist(centroid_of(cfid)) > 0.5:
            bad_ranks.append(best_rank)
    return {
        "n_sectors": len(ranks),
        "gt_rank_p50": _quant(ranks, .5),
        "gt_rank_p90": _quant(ranks, .9),
        "gt_rank_p99": _quant(ranks, .99),
        "frac_gt_in_top1": (round(sum(1 for r in ranks if r <= 1) / len(ranks), 4)
                            if ranks else None),
        "frac_gt_in_top5": (round(sum(1 for r in ranks if r <= 5) / len(ranks), 4)
                            if ranks else None),
        "n_bad_sectors": len(bad_ranks),
        "frac_gt_in_top5_on_bad_sectors": (
            round(sum(1 for r in bad_ranks if r <= 5) / len(bad_ranks), 4)
            if bad_ranks else None),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", default="train,val")
    ap.add_argument("--scores", default="sharp,normal_turn")
    ap.add_argument("--method", default="argmax,dp")
    ap.add_argument("--shard", default="1/1")
    ap.add_argument("--top-k", type=int, default=24)
    ap.add_argument("--transition-mode", default="V2_FROZEN",
                    choices=["V2_FROZEN", "MESH_AWARE", "HYBRID"],
                    help="transition 模式（①′ 起用 HYBRID 作为 v4 冻结 transition）")
    ap.add_argument("--candidate-mode", default="faces",
                    help="faces=v2（Top-K 面）/ ridges=v3（Top ridge 簇）")
    ap.add_argument("--ridge-top-k", type=int, default=6)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--recall", action="store_true")
    ap.add_argument("--tag", default="",
                    help="输出文件名后缀（并行跑多组参数时必须给，否则互相覆盖）")
    ap.add_argument("--allow-test", action="store_true")
    a = ap.parse_args()

    splits = [x.strip() for x in a.split.split(",") if x.strip()]
    if "test" in splits and not a.allow_test:
        raise SystemExit("test 分片已封存 —— 冻结 M1 参数后才可用 --allow-test 打开")

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
    scores = [x.strip() for x in a.scores.split(",") if x.strip()]
    methods = [x.strip() for x in a.method.split(",") if x.strip()]
    print("评测 %d 行 × score %s × method %s（shard %s）"
          % (len(rows), scores, methods, a.shard), flush=True)

    suffix = (("_%s" % a.tag) if a.tag else "") + (
        "" if a.shard == "1/1" else "_shard%s" % i)
    path = os.path.join(a.out, "m1_baseline%s.csv" % suffix)
    rpath = os.path.join(a.out, "m1_recall%s.csv" % suffix)
    t_all = time.time()
    done = 0
    with open(path, "w", newline="", encoding="ascii") as fh, \
            open(rpath, "w", newline="", encoding="ascii") as fr:
        w = csv.DictWriter(fh, fieldnames=list(COLS))
        wr = csv.DictWriter(fr, fieldnames=list(RECALL_COLS))
        w.writeheader()
        wr.writeheader()
        for r in rows:
            key = (r["case_uuid"], r["fdi"])
            s = sel.get(key)
            if not s or not s.get("die_asset_id"):
                continue
            die = _hash_to_path(case_dirs(key[0])).get(s["die_asset_id"])
            if die is None:
                continue
            try:
                tris = read_stl(die)
            except Exception as exc:              # noqa: BLE001
                print("  读网格失败 %s: %s" % (key[0][:14], exc), flush=True)
                continue
            cv = json.load(open(os.path.join(
                a.out, "labels",
                r["prep_mesh_ref"].replace("_labels.bin", "_curves.json"))))
            gt = [tuple(p) for p in (cv.get("projected") or cv["margin_gt"])]
            axis = tuple(cv["axis"])
            gt_perim = sum(math.dist(gt[j], gt[(j + 1) % len(gt)])
                           for j in range(len(gt)))
            # 特征只算一次（同一网格多组配置复用；结果不变）
            cents = [((t[0][0]+t[1][0]+t[2][0])/3.0,
                      (t[0][1]+t[1][1]+t[2][1])/3.0,
                      (t[0][2]+t[1][2]+t[2][2])/3.0) for t in tris]
            frame = prep_frame.build_frame(cents, axis)
            stats, adj = features.face_stats(tris, frame)
            cen = {s0.fid: s0.centroid for s0 in stats}
            for score in scores:
                v1_chosen = {}
                pool = None
                for method in methods:
                    t0 = time.time()
                    cfg = AutoMarginConfig(score=score, method=method,
                                           top_k=a.top_k,
                                           candidate_mode=a.candidate_mode,
                                           ridge_top_k=a.ridge_top_k,
                                           transition_mode=a.transition_mode)
                    res = engine.predict_margin(tris, axis, cfg, stats=stats,
                                                adj=adj, frame=frame,
                                                return_candidates=a.recall)
                    if res.candidates:
                        pool = res.candidates
                    if method == "argmax" and res.candidates:
                        v1_chosen = res.candidates["chosen"]
                    rec = {k: "" for k in COLS}
                    rec.update(case_uuid=r["case_uuid"], fdi=r["fdi"],
                               split=man["assign"].get(r["group_id"]),
                               config_id=frozen_config_id(cfg),
                               score=score, method=method, status=res.status,
                               elapsed_s=round(time.time()-t0, 1),
                               notes=res.reason[:60])
                    rec.update(energy_gap=res.energy_gap,
                               chosen_rank_mean=res.candidate_rank_stats.get(
                                   "mean"),
                               chosen_rank_max=res.candidate_rank_stats.get(
                                   "max"),
                               max_transition_mm=res.quality.get(
                                   "max_transition_mm"),
                               score_method=res.score_method,
                               transition_method=res.transition_method)
                    if res.margin is not None and len(res.margin.points) >= 3:
                        rec.update(metrics(res.margin.points, gt))
                        rec.update(
                            n_points=len(res.margin.points),
                            closed=res.margin.closed,
                            n_self_intersections=res.quality.get(
                                "n_self_intersections"),
                            filled_sector_frac=res.quality.get(
                                "filled_sector_frac"),
                            proj_disp_p95_mm=res.quality.get(
                                "proj_disp_p95_mm"),
                            gt_perimeter_mm=round(gt_perim, 3),
                            pred_perimeter_mm=res.margin.length_mm(),
                            perimeter_ratio=round(
                                res.margin.length_mm() / max(gt_perim, 1e-9), 4))
                    w.writerow(rec)
                    fh.flush()
                if a.recall and pool:
                    rs = recall_stats(pool, v1_chosen, gt,
                                      lambda fid: cen.get(fid, (0.0, 0.0, 0.0)))
                    wr.writerow(dict(case_uuid=r["case_uuid"], fdi=r["fdi"],
                                     split=man["assign"].get(r["group_id"]),
                                     score=score, **rs))
                    fr.flush()
            done += 1
            if done % 5 == 0:
                print("  %d/%d（%.0fs）" % (done, len(rows),
                                           time.time() - t_all), flush=True)
    print("完成 %d 行 → %s（%.0fs）" % (done, path, time.time() - t_all))


if __name__ == "__main__":
    main()
