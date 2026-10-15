#!/usr/bin/env python3
"""Blind **评估阶段**（独立进程）：打开 GT，出四层报告。

用户 2026-10-01 §8 规格：

    第一层 Coverage      AUTO_ACCEPT / REVIEW / FAILED %
    第二层 Accuracy given AUTO_ACCEPT   median / p95 / Hausdorff / p95≤0.20 / ≤0.50
    第三层 Case-weighted  病例内取 median 再聚合（避免多牙病例权重过大）
    第四层 Failure audit  全部评完之后才看：REVIEW 原因、**AUTO_ACCEPT 里有没有
                          p95 > 1mm 的 catastrophic miss**、分布

**生成阶段不碰这个文件**：它只读 sealed predictions + evaluation manifest，
并且先校验 seal 的 sha256（预测文件被改过就直接拒绝）。
"""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import math
import os
import sys

sys.path.insert(0, "/tmp/gt")

from app.geometry.auto_margin.projection import point_to_polyline_dist  # noqa: E402


def _q(v, p):
    v = sorted(x for x in v if x is not None)
    return v[min(len(v) - 1, int(p * (len(v) - 1)))] if v else None


def metrics(pred, gt):
    if len(pred) < 3 or len(gt) < 3:
        return None
    d_p2g = [point_to_polyline_dist(p, gt, closed=True) for p in pred]
    d_g2p = [point_to_polyline_dist(q, pred, closed=True) for q in gt]
    return {
        "median_mm": round(_q(d_p2g, .5), 4),
        "p95_mm": round(_q(d_p2g, .95), 4),
        "max_mm": round(max(d_p2g), 4),
        "chamfer_mm": round(0.5 * (sum(d_p2g) / len(d_p2g)
                                   + sum(d_g2p) / len(d_g2p)), 4),
        "hausdorff_mm": round(max(max(d_p2g), max(d_g2p)), 4),
        "frac_within_0p20": round(sum(1 for d in d_p2g if d <= 0.20)
                                  / len(d_p2g), 4),
        "frac_within_0p50": round(sum(1 for d in d_p2g if d <= 0.50)
                                  / len(d_p2g), 4),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--source", default="modeldie")
    ap.add_argument("--pred", required=True, help="sealed predictions jsonl")
    ap.add_argument("--gen-manifest",
                    default="/tmp/automargin/blind_generation_manifest.json",
                    help="用于校验 seal 里的输入清单哈希")
    ap.add_argument("--eval-manifest",
                    default="/tmp/automargin/blind_evaluation_manifest.json")
    a = ap.parse_args()

    # ---- seal 校验（预测文件必须与生成阶段封存的一致）
    # 优先用**增强 seal**（锁预测 + 运行条件，用户 §2）；否则退回 runner 自带 seal
    raw = open(a.pred, "rb").read()
    got = hashlib.sha256(raw).hexdigest()
    strong = os.path.join(a.out, "blind_seal_%s.json" % a.source)
    if os.path.isfile(strong):
        seal = json.load(open(strong, encoding="ascii"))
        if seal.get("prediction_file_sha256") != got:
            raise SystemExit("预测文件哈希与 seal 不符（被改过？）—— 拒绝评估")
        gm = hashlib.sha256(open(a.gen_manifest, "rb").read()).hexdigest()
        if seal.get("generation_manifest_sha256") != gm:
            raise SystemExit("输入清单哈希与 seal 不符 —— 拒绝评估")
        n = len([l for l in raw.decode("ascii").splitlines() if l.strip()])
        if seal.get("n_written") != n or seal.get("n_expected") != n:
            raise SystemExit("seal 条数不符：seal(%s/%s) vs 文件 %d"
                             % (seal.get("n_expected"), seal.get("n_written"), n))
        print("增强 seal 校验通过：commit=%s runner=%s… n=%d/%d"
              % (seal.get("git_commit"), (seal.get("runner_file_sha256") or "")[:12],
                 seal.get("n_expected"), n))
        print("  config_id=%s" % seal.get("config_id"))
    else:
        seal = None
        for f in os.listdir(a.out):
            if f.startswith("blind_prediction_seal_%s_" % a.source):
                cand = json.load(open(os.path.join(a.out, f), encoding="ascii"))
                if cand.get("file") == os.path.basename(a.pred):
                    seal = cand
                    break
        if seal is None:
            raise SystemExit("找不到与 %s 匹配的 seal —— 拒绝评估" % a.pred)
        if got != seal["sha256"]:
            raise SystemExit("预测文件哈希与 seal 不符（被改过？）—— 拒绝评估")
        print("seal 校验通过：%s（config_id=%s）"
              % (seal["file"], seal["config_id"]))

    preds = [json.loads(l) for l in raw.decode("ascii").splitlines() if l.strip()]
    evl = json.load(open(a.eval_manifest, encoding="ascii"))
    gt_of = {(e["case_uuid"], str(e["fdi"])): e
             for e in (evl.get("by_source") or {}).get(a.source, [])}
    print("预测 %d 条；评估侧 GT %d 条" % (len(preds), len(gt_of)))

    rows = []
    for p in preds:
        e = gt_of.get((p["case_uuid"], str(p["fdi"])))
        rec = {"case_uuid": p["case_uuid"], "fdi": p["fdi"],
               "status": p["status"], "reason": p.get("reason", "")}
        if e and e.get("margin_gt") and p.get("points"):
            m = metrics([tuple(x) for x in p["points"]],
                        [tuple(x) for x in e["margin_gt"]])
            if m:
                rec.update(m)
        rows.append(rec)

    # ---- 第一层：Coverage
    n = len(rows)
    cnt = collections.Counter(r["status"] for r in rows)
    print()
    print("=== 第一层 Coverage（%d 条）===" % n)
    for k in ("AUTO_ACCEPTED", "REVIEW_REQUIRED", "FAILED"):
        print("   %-16s %4d  %5.1f%%" % (k, cnt.get(k, 0),
                                         100.0 * cnt.get(k, 0) / max(n, 1)))

    def layer(sub, tag):
        ok = [r for r in sub if r.get("p95_mm") is not None]
        if not ok:
            print("   %-14s 无可评条目" % tag)
            return
        print("   %-14s n=%3d  median=%.3f p95=%.3f hausdorff=%.3f  "
              "≤0.20=%.2f ≤0.50=%.2f"
              % (tag, len(ok), _q([r["median_mm"] for r in ok], .5),
                 _q([r["p95_mm"] for r in ok], .5),
                 _q([r["hausdorff_mm"] for r in ok], .5),
                 sum(r["frac_within_0p20"] for r in ok) / len(ok),
                 sum(r["frac_within_0p50"] for r in ok) / len(ok)))

    acc = [r for r in rows if r["status"] == "AUTO_ACCEPTED"]
    print()
    print("=== 第二层 Accuracy given AUTO_ACCEPT（行权）===")
    layer(acc, "row-weighted")
    ok = [r for r in acc if r.get("p95_mm") is not None]
    b20 = sum(1 for r in ok if r["p95_mm"] <= 0.20)
    b50 = sum(1 for r in ok if r["p95_mm"] <= 0.50)
    print("   p95≤0.20 的行: %d/%d (%.0f%%)   p95≤0.50 的行: %d/%d (%.0f%%)"
          % (b20, len(ok), 100.0 * b20 / max(len(ok), 1), b50, len(ok),
             100.0 * b50 / max(len(ok), 1)))

    # ---- 第三层：Case-weighted
    print()
    print("=== 第三层 Case-weighted（病例内取 median 再聚合）===")
    by = collections.defaultdict(list)
    for r in acc:
        if r.get("p95_mm") is not None:
            by[r["case_uuid"]].append(r)
    cps = [_q([r["p95_mm"] for r in v], .5) for v in by.values()]
    cms = [_q([r["median_mm"] for r in v], .5) for v in by.values()]
    n_cases = len({r["case_uuid"] for r in rows})
    if cps:
        print("   有 AUTO_ACCEPT 的病例 %d/%d；病例 median(median)=%.3f "
              "median(p95)=%.3f p90(p95)=%.3f"
              % (len(by), n_cases, _q(cms, .5), _q(cps, .5), _q(cps, .9)))
        print("   病例 p95≤0.20: %d/%d (%.0f%%)   p95≤0.50: %d/%d (%.0f%%)"
              % (sum(1 for x in cps if x <= 0.20), len(cps),
                 100.0 * sum(1 for x in cps if x <= 0.20) / len(cps),
                 sum(1 for x in cps if x <= 0.50), len(cps),
                 100.0 * sum(1 for x in cps if x <= 0.50) / len(cps)))

    # ---- 第四层：Failure audit
    print()
    print("=== 第四层 Failure audit ===")
    print("   REVIEW 原因:", dict(collections.Counter(
        r["reason"] for r in rows if r["status"] == "REVIEW_REQUIRED")))
    print("   FAILED 原因:", dict(collections.Counter(
        r["reason"] for r in rows if r["status"] == "FAILED")))
    cat = [r for r in ok if r["p95_mm"] > 1.0]
    print("   **AUTO_ACCEPT 且 p95 > 1mm 的 catastrophic miss: %d 条**%s"
          % (len(cat), "" if not cat else " ← 必须逐条查"))
    for r in cat[:5]:
        print("      %s fdi=%s p95=%.3f hausdorff=%.3f"
              % (r["case_uuid"][:16], r["fdi"], r["p95_mm"], r["hausdorff_mm"]))
    print("   AUTO_ACCEPT 中 p95>0.5mm: %d 条" % (len(ok) - b50))
    out = os.path.join(a.out, "blind_eval_%s.csv" % a.source)
    cols = sorted({k for r in rows for k in r})
    with open(out, "w", newline="", encoding="ascii") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print("逐条结果 →", out)


if __name__ == "__main__":
    main()
