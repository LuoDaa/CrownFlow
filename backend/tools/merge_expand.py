"""合并扩容分片 → blind_expand_<source>.csv（按 case_uuid|fdi 去重，保留最优）。"""
import csv, glob, sys

src = sys.argv[1] if len(sys.argv) > 1 else "modeldie"
rows = []
for f in sorted(glob.glob("/tmp/automargin/blind_expand_%s_shard*.csv" % src)):
    rows += list(csv.DictReader(open(f, encoding="ascii")))
print("分片行 %d" % len(rows))
best = {}
for r in rows:
    k = (r["case_uuid"], r["fdi"])
    cur = best.get(k)
    if cur is None:
        best[k] = r
        continue
    try:
        better = float(r["best_p95_mm"]) < float(cur["best_p95_mm"])
    except (TypeError, ValueError):
        better = False
    if better:
        best[k] = r
print("去重后 %d 行 / %d 病例"
      % (len(best), len({k[0] for k in best})))
cols = list(rows[0].keys())
out = "/tmp/automargin/blind_expand_%s.csv" % src
with open(out, "w", newline="", encoding="ascii") as fh:
    w = csv.DictWriter(fh, fieldnames=cols)
    w.writeheader()
    for k in sorted(best):
        w.writerow(best[k])
print("写出", out)
