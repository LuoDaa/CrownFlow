"""扩容资格漏斗（只看 eligibility，不看任何 accuracy）。**分来源统计，不混总数**。"""
import csv, glob, collections

for src in ("modeldie", "modelbase"):
    rows = []
    for f in sorted(glob.glob("/tmp/automargin/blind_expand_%s_shard*.csv" % src)):
        rows += list(csv.DictReader(open(f, encoding="ascii")))
    if not rows:
        print("=== %s: 还没有结果 ===" % src.upper())
        continue
    cases_all = {r["case_uuid"] for r in rows}
    ok = [r for r in rows if r["decision"] == "MARGIN_ON_DIE"]
    ok_cases = {r["case_uuid"] for r in ok}
    print("=== %s ===" % src.upper())
    print("  有候选的病例            : %d（行 %d）" % (len(cases_all), len(rows)))
    print("  几何选择成功(MARGIN_ON_DIE): %d 行 / %d 病例"
          % (len(ok), len(ok_cases)))
    print("  BEST_STILL_OFF          : %d 行 / %d 病例"
          % (len(rows) - len(ok), len(cases_all - ok_cases)))
    bym = collections.Counter(r["best_method"] for r in ok)
    print("  命中来源                : %s" % dict(bym))
    p95 = sorted(float(r["best_p95_mm"]) for r in ok)
    if p95:
        print("  选中代型的 proj_p95      : p50=%.4f p90=%.4f max=%.4f"
              % (p95[len(p95)//2], p95[int(.9*len(p95))], p95[-1]))
    print("  每病例牙位数            : %s"
          % dict(collections.Counter(
              collections.Counter(r["case_uuid"] for r in ok).values())))
    print()
