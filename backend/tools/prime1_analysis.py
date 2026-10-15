"""①′ 三份输出：迁移矩阵 + 三信号重算 + Safety-Coverage curve（HYBRID transition）。"""
import csv, glob, collections

V2, V4 = {}, {}
# V2 基线必须用**显式冻结配置 + fail-loud 语义**的那次运行（prod tag）；
# v2frozen 那次发生在状态改名（OK→AUTO_ACCEPTED）之前，status 列是旧值，
# 会被 bucket() 全判成 REVIEW（实测踩到）。
for f in sorted(glob.glob("/tmp/automargin/m1_baseline_prod_shard*.csv")):
    for r in csv.DictReader(open(f, encoding="ascii")):
        if r["method"] == "dp":
            V2[(r["case_uuid"], r["fdi"])] = r
for f in sorted(glob.glob("/tmp/automargin/m1_baseline_v4hy_shard*.csv")):
    for r in csv.DictReader(open(f, encoding="ascii")):
        if r["method"] == "dp":
            V4[(r["case_uuid"], r["fdi"])] = r


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


print("=== ①′-1 迁移矩阵（V2 → v4/HYBRID），n=%d ===" % len(V4))
mat = collections.Counter()
for k in V4:
    mat[(bucket(V2[k]), bucket(V4[k]))] += 1
print("%-18s → %-18s %5s" % ("V2", "v4", "n"))
for (a, b), n in sorted(mat.items(), key=lambda x: -x[1]):
    flag = "  ⚠️" if (a, b) == ("GOOD", "CATA") or (a, b) == ("REVIEW", "CATA") else ""
    print("%-18s → %-18s %5d%s" % (a, b, n, flag))
g2c = mat[("GOOD", "CATA")]
print()
print("**GOOD → CATA = %d**（transition 修复不合格线 = 0）" % g2c)
st = collections.Counter(bucket(r) for r in V4.values())
print("v4 总体：", dict(st))
p4 = [fn(r, "p95_mm") for r in V4.values() if bucket(r) != "REVIEW"]
print("accepted-only：median=%.3f p95=%.3f p95≤0.5 %d/%d"
      % (q(p4, .5), q(p4, .95), sum(1 for x in p4 if x <= 0.5), len(p4)))
cat4 = [k for k, r in V4.items() if bucket(r) == "CATA"]
print("v4 catastrophic 行 %d：" % len(cat4))
for k in cat4:
    print("   %s fdi=%s p95=%.3f（V2 时 %.3f）"
          % (k[0][:14], k[1], fn(V4[k], "p95_mm"), fn(V2[k], "p95_mm")))
print()
print("=== ①′-2 三信号在新分组下的分布 ===")
print("%-22s %-16s %7s %7s %7s %7s %7s %7s %7s"
      % ("signal", "group", "n", "min", "p25", "p50", "p75", "p90", "max"))
for key, name in (("energy_gap", "energy_gap"),
                  ("chosen_rank_mean", "chosen_mean"),
                  ("max_transition_mm", None)):
    pass
groups = collections.defaultdict(list)
for r in V4.values():
    groups[bucket(r)].append(r)
for key, name in (("energy_gap", "energy_gap"), ("chosen_rank_mean", "chosen_mean")):
    for g in ("GOOD", "MID", "CATA"):
        v = [fn(r, key) for r in groups[g]]
        if not any(x is not None for x in v):
            print("%-22s %-16s %7d   （缺）" % (name, g, len(v)))
            continue
        fmt = "%.4f" if key == "energy_gap" else "%.2f"
        print("%-22s %-16s %7d %7s %7s %7s %7s %7s %7s"
              % (name, g, len(v), fmt % q(v, 0), fmt % q(v, .25), fmt % q(v, .5),
                 fmt % q(v, .75), fmt % q(v, .9), fmt % q(v, 1.0)))
# second_separation 需要从 DP 的 structurally-distinct second 得到 —— runner 没落盘
# 它；用近似的 (E2-E1 已有 energy_gap) + 迁移信息替代，并单独说明
print()
print("（注：runner 未落盘 second_separation；①′ 的分离度信号用 energy_gap 与")
print("  chosen_mean 判读，second_separation 需在 Gate 校准阶段单独补跑。）")
print()
print("=== ①′-3 Safety-Coverage curve（规则族 A：low_gap AND high_sep）===")
print("（second_separation 未落盘 → 本轮先用 gap 单信号 + rank 组合的保守下界）")
print("%-34s %7s %8s %8s %8s" % ("rule", "AUTO", "good保", "cat挡", "cat漏"))
print("-" * 74)
acc4 = [r for r in V4.values() if bucket(r) != "REVIEW"]
good4 = [r for r in acc4 if fn(r, "p95_mm") <= 0.5]
cat4r = [r for r in acc4 if fn(r, "p95_mm") > 1.0]
print("%-34s %7s %8s %8s %8s" % ("none（现状）", "%.0f%%" % (100.0*len(acc4)/198),
                                 len(good4), 0, len(cat4r)))
for tq in (10, 20, 30, 50):
    tg = q([fn(r, "energy_gap") for r in good4], tq / 100.0)
    tr = q([fn(r, "chosen_rank_mean") for r in good4], tq / 100.0)
    keep = [r for r in acc4
            if not ((fn(r, "energy_gap") is None or fn(r, "energy_gap") <= tg))]
    g = sum(1 for r in keep if fn(r, "p95_mm") <= 0.5)
    cm = sum(1 for r in keep if fn(r, "p95_mm") > 1.0)
    print("%-34s %7.0f%% %8d %8d %8d"
          % ("gap only  (τ=GOOD q%d=%.3f)" % (tq, tg or 0),
             100.0 * len(keep) / 198, g, len(cat4r) - cm, cm))
for tq in (50, 90):
    tg = q([fn(r, "energy_gap") for r in good4], 0.30)
    tr = q([fn(r, "chosen_rank_mean") for r in good4], tq / 100.0)
    keep = [r for r in acc4 if not (
        (fn(r, "energy_gap") is None or fn(r, "energy_gap") <= tg)
        and (fn(r, "chosen_rank_mean") is not None
             and fn(r, "chosen_rank_mean") >= tr))]
    g = sum(1 for r in keep if fn(r, "p95_mm") <= 0.5)
    cm = sum(1 for r in keep if fn(r, "p95_mm") > 1.0)
    print("%-34s %7.0f%% %8d %8d %8d"
          % ("gap AND high_rank (q%d=%.2f)" % (tq, tr or 0),
             100.0 * len(keep) / 198, g, len(cat4r) - cm, cm))
