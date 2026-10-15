#!/usr/bin/env python3
"""① Gate 形态分析（**只定形态，不冻结阈值**）。

在 `M1_V4_HARD_DEV_SET`（原 Blind v1，198 条）上：

1. 三项 GT-free 信号在 GOOD / MID / CATASTROPHIC 三组的完整分位分布
   —— 重点看**重叠**（只看组中位会骗人）；
2. 四种**单调、可解释**规则族的 安全-覆盖 Pareto 表：
      A  low_gap AND high_separation
      B  low_gap AND high_rank
      C  (low_gap AND high_sep) OR (high_rank AND high_sep)
      D  三项归一化风险的简单加权和
3. 优化顺序按用户规格：catastrophic accepted 最少 → good false-review 最少 →
   coverage 第三。

**明确不做**：不选"最优阈值"、不冻结任何数值。transition 改了以后
（energy_gap / chosen_mean / second_separation 都会变），这些数值要全部重算。
"""
from __future__ import annotations

import csv
import glob
import json
import os

OUT = "/tmp/automargin"


def fn(d, k):
    try:
        return float(d.get(k))
    except (TypeError, ValueError):
        return None


def q(v, p):
    v = sorted(x for x in v if x is not None)
    return v[min(len(v) - 1, int(p * (len(v) - 1)))] if v else None


# ---- 载入：预测里的 GT-free 置信度 + 评估里的 p95（后者仅用于分组，不参与规则）
ev = {r["case_uuid"] + "|" + str(r["fdi"]): r
      for r in csv.DictReader(open(os.path.join(OUT, "blind_eval_modeldie.csv"),
                                   encoding="ascii"))}
rows = []
for f in glob.glob(os.path.join(OUT, "blind_predictions_*.jsonl")):
    for line in open(f):
        p = json.loads(line)
        e = ev.get(p["case_uuid"] + "|" + str(p["fdi"]))
        if not e or e.get("p95_mm") in (None, ""):
            continue
        c = dict(p.get("confidence") or {})
        c["_gap"] = p.get("energy_gap")
        c["_rank"] = (p.get("candidate_rank_stats") or {}).get("mean")
        c["_sep"] = c.get("second_separation_mm")
        c["_p95"] = float(e["p95_mm"])
        c["_status"] = p["status"]
        rows.append(c)
acc = [r for r in rows if r["_status"] == "AUTO_ACCEPTED"]
good = [r for r in acc if r["_p95"] <= 0.5]
mid = [r for r in acc if 0.5 < r["_p95"] <= 1.0]
cat = [r for r in acc if r["_p95"] > 1.0]
print("AUTO_ACCEPT %d 条：GOOD %d / MID %d / CATASTROPHIC %d"
      % (len(acc), len(good), len(mid), len(cat)))
print()

print("=== 一、三信号完整分布（看重叠，不只看中位）===")
print("%-24s %-14s %7s %7s %7s %7s %7s %7s %7s %7s"
      % ("signal", "group", "n", "min", "p10", "p25", "p50", "p75", "p90", "max"))
for name, key in (("energy_gap", "_gap"),
                  ("chosen_mean", "_rank"),
                  ("second_separation_mm", "_sep")):
    for tag, grp in (("GOOD(p95<=0.5)", good), ("MID(0.5-1.0)", mid),
                     ("CATASTROPHIC(>1)", cat)):
        v = [r[key] for r in grp if r[key] is not None]
        n_missing = len(grp) - len(v)
        fmt = "%.3f" if name != "energy_gap" else "%.4f"
        if not v:
            print("%-24s %-14s %7d   （全部缺失）" % (name, tag, len(grp)))
            continue
        print("%-24s %-14s %7d %7s %7s %7s %7s %7s %7s %7s%s"
              % (name, tag, len(grp), fmt % min(v), fmt % q(v, .1),
                 fmt % q(v, .25), fmt % q(v, .5), fmt % q(v, .75),
                 fmt % q(v, .9), fmt % max(v),
                 "（缺 %d）" % n_missing if n_missing else ""))
    print()

print("=== 二、规则族 安全-覆盖（阈值扫分位；未冻结）===")
print("%-42s %8s %8s %8s %8s"
      % ("rule（阈值取分位）", "AUTO", "good保", "cat挡", "cat漏"))
print("-" * 82)
base = len(acc)


def evaluate(rule, tag):
    keep = [r for r in acc if not rule(r)]
    g = sum(1 for r in keep if r["_p95"] <= 0.5)
    cmiss = sum(1 for r in keep if r["_p95"] > 1.0)
    print("%-42s %7.0f%% %8d %8d %8d"
          % (tag, 100.0 * len(keep) / base, g, len(cat) - cmiss, cmiss))
    return {"tag": tag, "auto": len(keep), "good": g, "cat_miss": cmiss}


pareto = []
for qq in (5, 10, 20, 30, 50):
    tg = q([r["_gap"] for r in good if r["_gap"] is not None], qq / 100.0)
    tr = q([r["_rank"] for r in good if r["_rank"] is not None], qq / 100.0)
    ts = q([r["_sep"] for r in good if r["_sep"] is not None], qq / 100.0)
    # A: gap 低 且 sep 大
    pareto.append(evaluate(
        lambda r, tg=tg, ts=ts: (r["_gap"] is None or r["_gap"] <= tg)
        and (r["_sep"] is not None and r["_sep"] >= ts),
        "A low_gap & high_sep  (q%d: %.3f/%.2f)" % (qq, tg or 0, ts or 0)))
    # B: gap 低 且 rank 大
    pareto.append(evaluate(
        lambda r, tg=tg, tr=tr: (r["_gap"] is None or r["_gap"] <= tg)
        and (r["_rank"] is not None and r["_rank"] >= tr),
        "B low_gap & high_rank (q%d: %.3f/%.2f)" % (qq, tg or 0, tr or 0)))
    # C: (gap 低 且 sep 大) 或 (rank 大 且 sep 大)
    pareto.append(evaluate(
        lambda r, tg=tg, tr=tr, ts=ts: (
            ((r["_gap"] is None or r["_gap"] <= tg) and r["_sep"] is not None
             and r["_sep"] >= ts)
            or ((r["_rank"] is not None and r["_rank"] >= tr)
                and r["_sep"] is not None and r["_sep"] >= ts)),
        "C A∪B 变体             (q%d)" % qq))
print()
print("=== 三、Pareto 前沿（cat_miss 最小 → good 保留最多）===")
seen = []
for p in sorted(pareto, key=lambda x: (x["cat_miss"], -x["good"])):
    if any(s[0] <= p["cat_miss"] and s[1] >= p["good"] for s in seen):
        continue
    seen.append((p["cat_miss"], p["good"], p["auto"], p["tag"]))
for cm, g, a, tag in seen[:6]:
    print("   cat漏=%-2d  good保留=%-3d  AUTO=%5.1f%%   %s"
          % (cm, g, 100.0 * a / base, tag))
print()
print("=== 四、MID 组（0.5<p95≤1.0）在三规则下的处理 ===")
for tag, rule in (("A", lambda r: (r["_gap"] is None or r["_gap"] <= 0.25)
                   and (r["_sep"] or 0) >= 4.0),
                  ("C", lambda r: ((r["_gap"] is None or r["_gap"] <= 0.25)
                                   and (r["_sep"] or 0) >= 4.0)
                   or ((r["_rank"] or 0) >= 2.5 and (r["_sep"] or 0) >= 4.0))):
    rej = sum(1 for r in mid if rule(r))
    print("   规则 %s：MID %d 条中会被 REJECT %d 条（%.0f%%）"
          % (tag, len(mid), rej, 100.0 * rej / max(len(mid), 1)))
