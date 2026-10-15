"""①′ 最终版：在 Hard Dev（Blind v1 198 行）上的迁移矩阵 + 三信号 + Safety-Coverage。"""
import csv, glob, collections

V2 = {r["case_uuid"] + "|" + str(r["fdi"]): r
      for r in csv.DictReader(open("/tmp/automargin/blind_eval_modeldie.csv",
                                   encoding="ascii"))}
V4 = {}
for f in sorted(glob.glob("/tmp/automargin/blind_v4rerun_hybrid_shard*.csv")):
    for r in csv.DictReader(open(f, encoding="ascii")):
        V4[r["case_uuid"] + "|" + str(r["fdi"])] = r


def fn(r, k):
    try:
        return float(r[k])
    except (TypeError, ValueError):
        return None


def bucket(r):
    if r["status"] != "AUTO_ACCEPTED":
        return "REVIEW"
    p = fn(r, "p95_mm")
    if p is None:
        return "NO_METRIC"
    return "GOOD" if p <= 0.5 else ("MID" if p <= 1.0 else "CATA")


def q(v, p):
    v = sorted(x for x in v if x is not None)
    return v[min(len(v) - 1, int(p * (len(v) - 1)))] if v else None


print("=== ①′-1 迁移矩阵（Blind v1 的 V2 → v4/HYBRID），n=%d ===" % len(V4))
mat = collections.Counter()
for k, r4 in V4.items():
    r2 = V2.get(k)
    if not r2:
        continue
    mat[(bucket(r2), bucket(r4))] += 1
print("%-20s → %-20s %5s" % ("V2(原 blind)", "v4(HYBRID)", "n"))
for (a, b), n in sorted(mat.items(), key=lambda x: -x[1]):
    warn = "   ⚠️ 必须为 0" if (a, b) == ("GOOD", "CATA") else ""
    print("%-20s → %-20s %5d%s" % (a, b, n, warn))
print()
g2c = mat[("GOOD", "CATA")]
print("**GOOD → CATA = %d**" % g2c)
st4 = collections.Counter(bucket(r) for r in V4.values())
print("v4 总体：", dict(st4))
acc4 = [r for r in V4.values() if bucket(r) != "REVIEW"]
p4 = [fn(r, "p95_mm") for r in acc4 if fn(r, "p95_mm") is not None]
print("accepted-only：median=%.3f p95=%.3f p95≤0.5 %d/%d p95≤0.2 %d/%d"
      % (q(p4, .5), q(p4, .95), sum(1 for x in p4 if x <= 0.5), len(p4),
         sum(1 for x in p4 if x <= 0.2), len(p4)))
print("REVIEW 原因：", dict(collections.Counter(
    r["reason"] for r in V4.values() if bucket(r) == "REVIEW")))
print()
print("=== ①′-2 三信号在新分组下的分布（含 second_separation）===")
for key, fmt in (("energy_gap", "%.4f"), ("chosen_rank_mean", "%.2f"),
                 ("second_separation_mm", "%.3f")):
    for g in ("GOOD", "MID", "CATA"):
        v = [fn(r, key) for r in V4.values() if bucket(r) == g]
        n_all = sum(1 for r in V4.values() if bucket(r) == g)
        if not any(x is not None for x in v):
            print("%-22s %-6s n=%3d  （缺）" % (key, g, n_all))
            continue
        print("%-22s %-6s n=%3d  min=%s p25=%s p50=%s p75=%s p90=%s max=%s"
              % (key, g, n_all, fmt % q(v, 0), fmt % q(v, .25), fmt % q(v, .5),
                 fmt % q(v, .75), fmt % q(v, .9), fmt % q(v, 1.0)))
    print()
print("=== ①′-3 Safety-Coverage curve（规则族；阈值取 GOOD 分位，**未冻结**）===")
good4 = [r for r in V4.values() if bucket(r) == "GOOD"]
cat4 = [r for r in V4.values() if bucket(r) == "CATA"]
mid4 = [r for r in V4.values() if bucket(r) == "MID"]
base = len(V4)
print("%-44s %7s %7s %7s %7s %7s"
      % ("rule", "AUTO%", "good保", "MID拒", "cat挡", "cat漏"))
print("-" * 84)


def ev(rule, tag):
    keep = [r for r in acc4 if not rule(r)]
    g = sum(1 for r in keep if fn(r, "p95_mm") <= 0.5)
    m = sum(1 for r in keep if 0.5 < (fn(r, "p95_mm") or 0) <= 1.0)
    cm = sum(1 for r in keep if (fn(r, "p95_mm") or 0) > 1.0)
    print("%-44s %7.1f %7d %7d %7d %7d"
          % (tag, 100.0 * len(keep) / base, g, len(mid4) - m,
             (len(cat4) + len(mid4)) - cm - (len(mid4) - m), cm))
    return (cm, -g)


results = []
results.append(ev(lambda r: False, "none（v4 现状）"))
for tq in (5, 10, 20):
    tg = q([fn(r, "energy_gap") for r in good4], tq / 100.0)
    results.append(ev(lambda r, tg=tg: fn(r, "energy_gap") is None
                      or fn(r, "energy_gap") <= tg,
                      "gap only   (τ=GOOD q%d=%.3f)" % (tq, tg or 0)))
for tq in (5, 10):
    ts = q([fn(r, "second_separation_mm") for r in good4], 1 - tq / 100.0)
    results.append(ev(lambda r, ts=ts: fn(r, "second_separation_mm") is not None
                      and fn(r, "second_separation_mm") >= ts,
                      "sep only   (τ=GOOD q%d=%.3f)" % (100 - tq, ts or 0)))
for tq in (5, 10, 20):
    tg = q([fn(r, "energy_gap") for r in good4], tq / 100.0)
    ts = q([fn(r, "second_separation_mm") for r in good4], 1 - tq / 100.0)
    results.append(ev(lambda r, tg=tg, ts=ts: (
        fn(r, "energy_gap") is None or fn(r, "energy_gap") <= tg)
        and (fn(r, "second_separation_mm") is not None
             and fn(r, "second_separation_mm") >= ts),
        "gap AND sep (τ=%.3f/%.3f)" % (tg or 0, ts or 0)))
print()
print("Pareto（cat漏 最小 → good 保留最多）：")
best = sorted(results, key=lambda x: (x[0], x[1]))[:4]
for cm, ng in best:
    print("   cat漏=%d good保留=%d" % (cm, -ng))
