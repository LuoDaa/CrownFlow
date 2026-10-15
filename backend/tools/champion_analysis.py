"""①′ 第一步：对齐 6 条的账 + GOOD 全量回归 + 选 Transition Champion。"""
import csv, collections, glob

cat = list(csv.DictReader(open("/tmp/automargin/transition_modes_cat6.csv",
                               encoding="ascii")))
good = []
for f in sorted(glob.glob("/tmp/automargin/transition_modes_good_shard*.csv")):
    good += list(csv.DictReader(open(f, encoding="ascii")))


def p95(r):
    v = r.get("selected_p95_mm")
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


# ---- 1) 6 条 GT-infeasible 逐条账
ev = {r["case_uuid"] + "|" + str(r["fdi"]): r
      for r in csv.DictReader(open("/tmp/automargin/blind_eval_modeldie.csv",
                                   encoding="ascii"))}
by = collections.defaultdict(dict)
for r in cat:
    by[(r["case_uuid"], r["fdi"])][r["mode"]] = r
infeas6 = [(k, m) for k, m in by.items()
           if m.get("V2_FROZEN", {}).get("gt_path_feasible") == "False"]
print("=== 1) V2 下 GT 路径不可行的 %d 条：逐条账 ===" % len(infeas6))
print("%-16s %-4s %-12s %-12s %-12s %-12s" %
      ("case", "fdi", "V2", "MESH", "HYBRID", "MESH p95"))
for (u, f), m in sorted(infeas6):
    print("%-16s %-4s %-12s %-12s %-12s %-12s" %
          (u[:14], f,
           "挡%s" % m["V2_FROZEN"]["n_blocked_segments"],
           "可行" if m["MESH_AWARE"]["gt_path_feasible"] == "True" else "不可行",
           "可行" if m["HYBRID"]["gt_path_feasible"] == "True" else "不可行",
           ("%.3f" % p95(m["MESH_AWARE"])) if p95(m["MESH_AWARE"]) else "-"))
print()

# ---- 2) GOOD 全量回归：四项检查
gb = collections.defaultdict(dict)
for r in good:
    gb[(r["case_uuid"], r["fdi"])][r["mode"]] = r
print("=== 2) GOOD 全量回归（%d 行）===" % len(gb))
for mode in ("V2_FROZEN", "MESH_AWARE", "HYBRID"):
    ok = [m[mode] for m in gb.values() if mode in m]
    acc = sum(1 for r in ok if r["dp_status"] == "AUTO_ACCEPTED")
    p = [p95(r) for r in ok if p95(r) is not None]
    p.sort()
    q = lambda t: p[min(len(p) - 1, int(t * (len(p) - 1)))] if p else 0
    b50 = sum(1 for x in p if x <= 0.5)
    b10 = sum(1 for x in p if x > 1.0)
    print("  %-11s AUTO_ACCEPT %3d/%3d  p95中位=%.3f  p90=%.3f  "
          "p95≤0.5 %3d/%3d  p95>1(新cata) %d"
          % (mode, acc, len(ok), q(.5), q(.9), b50, len(p), b10))
# 逐行退化对比：V2 好 → 新模式变差/变cata
deg = []
for k, m in sorted(gb.items()):
    v2, me, hy = p95(m.get("V2_FROZEN", {})), p95(m.get("MESH_AWARE", {})), \
        p95(m.get("HYBRID", {}))
    if v2 is not None and v2 <= 0.5:
        for nm, nv in (("MESH_AWARE", me), ("HYBRID", hy)):
            if nv is not None and nv > 1.0:
                deg.append((k, nm, v2, nv))
            elif nv is not None and nv > v2 * 2.0:
                deg.append((k, nm + "(2x)", v2, nv))
print("  V2 好(≤0.5) → 新模式 >1mm 或 ≥2x 退化: %d 条" % len(deg))
for k, nm, a, b in deg[:6]:
    print("     %s fdi=%s %s: %.3f → %.3f" % (k[0][:14], k[1], nm, a, b))
print()

# ---- 3) REVIEW 变化（cat6 里 DP 状态）
print("=== 3) catastrophic 15 条的 DP 状态（有无新增 REVIEW）===")
for mode in ("V2_FROZEN", "MESH_AWARE", "HYBRID"):
    st = collections.Counter(m[mode]["dp_status"] for m in by.values()
                             if mode in m)
    print("  %-11s %s" % (mode, dict(st)))
