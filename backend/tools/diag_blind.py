"""诊断：GT-free 置信度字段能否区分 catastrophic miss 与好结果。
（只做分析，不动任何阈值 —— 阈值要改必须换新 blind set。）"""
import csv, json, glob, collections

ev = {r["case_uuid"] + "|" + str(r["fdi"]): r
      for r in csv.DictReader(open("/tmp/automargin/blind_eval_modeldie.csv",
                                   encoding="ascii"))}
preds = []
for f in glob.glob("/tmp/automargin/blind_predictions_*.jsonl"):
    preds += [json.loads(l) for l in open(f)]


def fn(d, k):
    v = d.get(k)
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


groups = {"catastrophic(p95>1mm)": [], "good(p95<=0.5)": [],
          "mid(0.5<p95<=1)": []}
for p in preds:
    e = ev.get(p["case_uuid"] + "|" + str(p["fdi"]))
    if not e or e.get("p95_mm") in (None, ""):
        continue
    q = float(e["p95_mm"])
    c = p.get("confidence") or {}
    c["_chosen_max"] = (p.get("candidate_rank_stats") or {}).get("max")
    c["_chosen_mean"] = (p.get("candidate_rank_stats") or {}).get("mean")
    c["_gap"] = p.get("energy_gap")
    if q > 1.0:
        groups["catastrophic(p95>1mm)"].append(c)
    elif q <= 0.5:
        groups["good(p95<=0.5)"].append(c)
    else:
        groups["mid(0.5<p95<=1)"].append(c)


def q(v, p):
    v = sorted(x for x in v if x is not None)
    return v[min(len(v) - 1, int(p * (len(v) - 1)))] if v else None


print("=== GT-free 置信度 vs 实际结果（AUTO_ACCEPT 行）===")
print("%-22s %6s %10s %10s %10s %10s" %
      ("group", "n", "chosen_max", "chosen_mean", "energy_gap", "sep_mm"))
for g, rows in groups.items():
    if not rows:
        continue
    print("%-22s %6d %10s %10s %10s %10s" %
          (g, len(rows),
           "%.0f" % (q([r.get("_chosen_max") for r in rows], .5) or 0),
           "%.2f" % (q([r.get("_chosen_mean") for r in rows], .5) or 0),
           "%.2e" % (q([r.get("_gap") for r in rows], .5) or 0),
           "%.2f" % (q([r.get("second_separation_mm") for r in rows], .5) or 0)))
print()
print("=== catastrophic 组逐条（看是否有共同特征）===")
for r in groups["catastrophic(p95>1mm)"][:12]:
    print("   chosen_max=%-3s chosen_mean=%-6s gap=%-9s sep=%-7s "
          "transition=%-6s closed=%s 自交=%s"
          % (r.get("_chosen_max"), round(r.get("_chosen_mean") or 0, 2),
             ("%.1e" % r["_gap"]) if r.get("_gap") is not None else "-",
             round(r.get("second_separation_mm") or 0, 2),
             round(r.get("max_transition_mm") or 0, 2),
             r.get("closed"), r.get("n_self_intersections")))
print()
c = collections.Counter()
for r in groups["catastrophic(p95>1mm)"]:
    c["chosen_max>=20" if (r.get("_chosen_max") or 0) >= 20 else "chosen_max<20"] += 1
for r in groups["good(p95<=0.5)"]:
    c["GOOD chosen_max>=20" if (r.get("_chosen_max") or 0) >= 20
      else "GOOD chosen_max<20"] += 1
print("chosen_max 分布对比:", dict(c))
