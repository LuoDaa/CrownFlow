"""② 验收汇总：分类计数 + 救回幅度。"""
import csv, collections
rows = list(csv.DictReader(open("/tmp/automargin/transition_modes_cat6.csv",
                                encoding="ascii")))
by = collections.defaultdict(dict)
for r in rows:
    by[(r["case_uuid"], r["fdi"])][r["mode"]] = r
print("catastrophic 行 %d" % len(by))
cls = collections.Counter()
rescue = []
for k, m in sorted(by.items()):
    v2 = m.get("V2_FROZEN", {})
    fixed = (m.get("MESH_AWARE", {}).get("gt_path_feasible") == "True"
             or m.get("HYBRID", {}).get("gt_path_feasible") == "True")
    v2f = v2.get("gt_path_feasible") == "True"

    def p95(mode):
        v = m.get(mode, {}).get("selected_p95_mm")
        return float(v) if v else None
    best = min([x for x in (p95("MESH_AWARE"), p95("HYBRID")) if x is not None],
               default=None)
    if not v2f and fixed:
        good = best is not None and best <= 0.5
        cls["A transition修复+DP选对" if good else "B transition已修但DP仍选错"] += 1
        if good:
            rescue.append((k[0][:14], k[1], p95("V2_FROZEN"), best))
    elif v2f:
        cls["B' 本来就可行的选错（与transition无关）"] += 1
    else:
        cls["C GT路径仍不可行"] += 1
print()
for k, v in cls.most_common():
    print("  %-40s %d" % (k, v))
print()
print("=== 被 transition 救回的行（p95: V2 → MESH/HYBRID）===")
for c, f, a, b in rescue:
    print("   %s fdi=%-3s %.3f → %.3f mm" % (c, f, a or 0, b or 0))
