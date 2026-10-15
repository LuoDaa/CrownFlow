#!/usr/bin/env python3
"""Blind 数据集的三层 manifest（用户 2026-10-01 收紧版）。

    ① blind_selection_record.json      选择证据（含 GT 派生量；生成永不读取）
    ② blind_generation_manifest.json   **喂给 Auto Margin 的唯一输入**
                                       只有部署时真实能拿到的字段
    ③ blind_evaluation_manifest.json   只有 evaluator 打开（GT Margin）

生成侧禁止任何由 GT Margin 派生的字段（投影 p95/max、side method、形态结论、
甚至 margin 来源）—— 详见 ``app.geometry.auto_margin.blind_guard``。

**Final Blind v1 默认只用 ``modeldie``**（验证充分的路线）。``modelbase`` 仍是
探针，留到 AutoMarginGTSet v2 extension 单独验证语义后再进 blind。

用法:
    python3 blind_manifest.py --out /tmp/automargin --sources modeldie --n 50
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, "/tmp/gt")

from app.geometry.auto_margin.blind_guard import (      # noqa: E402
    assert_evaluation_manifest_ok, assert_generation_manifest_clean,
    split_selection_evidence)
from app.importers.exocad.construction_info import parse_construction_case  # noqa: E402
from build_auto_margin_gt import case_dirs                                 # noqa: E402


def case_sort_key(uuid: str) -> str:
    """确定性选样键：按 case_uuid 哈希排序，**绝不看病例形态**。"""
    return hashlib.sha256(("blind|" + uuid).encode()).hexdigest()


def _snapshot_paths(uuid):
    out = []
    for d in case_dirs(uuid):
        for dp, _dn, fns in os.walk(d):
            for f in fns:
                lf = f.lower()
                if lf.endswith(".constructioninfo") or lf.endswith(".modelinfo"):
                    out.append(os.path.join(dp, f))
    return sorted(out)


def input_of(uuid, fdi):
    """只取**部署时真实可得**的输入：就位轴（来自设计快照的输入字段）。"""
    for path in _snapshot_paths(uuid):
        try:
            case = parse_construction_case(path)
        except Exception:                     # noqa: BLE001
            continue
        tooth = case.tooth(int(fdi))
        if tooth is None or not tooth.insertion_axis:
            continue
        a = [float(x) for x in tooth.insertion_axis]
        n = sum(v * v for v in a) ** 0.5 or 1.0
        return {"axis": [v / n for v in a]}
    return None


def margin_points_of(uuid, fdi, method):
    """**只给评估侧用**：现场解析该牙位 GT Margin 点列。"""
    want = ".modelinfo" if (method or "").upper() == "MODEL_INFO" else \
        ".constructioninfo"
    for d in case_dirs(uuid):
        for dp, _dn, fns in os.walk(d):
            for f in fns:
                if not f.lower().endswith(want):
                    continue
                try:
                    tooth = parse_construction_case(
                        os.path.join(dp, f)).tooth(int(fdi))
                except Exception:                 # noqa: BLE001
                    continue
                if tooth is not None and tooth.margin_points:
                    return [[round(float(x), 5) for x in p]
                            for p in tooth.margin_points]
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--sources", default="modeldie",
                    help="Final Blind v1 默认只用 modeldie；modelbase 留作 v2 探针")
    ap.add_argument("--n", type=int, default=0,
                    help="确定性选取的病例数上限（0 = 全部通过 input QC 的病例）")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    rows = []
    for src in [s.strip() for s in a.sources.split(",") if s.strip()]:
        p = os.path.join(a.out, "blind_expand_%s.csv" % src)
        if not os.path.isfile(p):
            print("缺 %s（扩容还没跑完？）" % p)
            continue
        for r in csv.DictReader(open(p, encoding="ascii")):
            if r.get("decision") != "MARGIN_ON_DIE":
                continue
            r["source"] = src
            rows.append(r)
    print("输入 QC 通过行 %d（sources=%s）" % (len(rows), a.sources))
    if not rows:
        raise SystemExit("没有通过 input QC 的行 —— 等扩容跑完再建 manifest")

    cases = sorted({r["case_uuid"] for r in rows}, key=case_sort_key)
    if a.n:
        cases = cases[:a.n]
    keep = set(cases)
    print("确定性选取病例 %d（按 sha256('blind|'+case_uuid) 升序）" % len(cases))

    sel, gen, evl = [], {}, {}
    for r in sorted(rows, key=lambda x: (x["case_uuid"], x["fdi"])):
        if r["case_uuid"] not in keep:
            continue
        inp = input_of(r["case_uuid"], r["fdi"])
        if inp is None:
            continue
        row = dict(r)
        row["axis"] = inp["axis"]
        g, e = split_selection_evidence(row)
        sel.append({**{k: v for k, v in e.items() if k != "margin_gt"},
                    "source": r["source"]})
        gen.setdefault(r["source"], []).append(g)
        e["margin_gt"] = margin_points_of(r["case_uuid"], r["fdi"],
                                          r.get("best_method"))
        evl.setdefault(r["source"], []).append(e)

    for src, gl in gen.items():
        for entry in gl:
            assert_generation_manifest_clean(entry)
    obj_sel = {"name": "blind_selection_record", "version": "v1",
               "warning": "含 GT 派生信息（投影 QC / 形态结论）；"
                          "**生成进程永不读取本文件**",
               "selection_rule": ("全部通过输入 QC 的行按 "
                                  "sha256('blind|'+case_uuid) 升序取前 N"
                                  "（无人工挑选）"),
               "records": sel}
    obj_gen = {"name": "blind_generation_manifest", "version": "v1",
               "warning": "喂给 Auto Margin 的唯一输入；"
                          "**不含任何 GT 派生字段**",
               "input_schema_version": "blind_gen_v1",
               "created_utc": datetime.now(timezone.utc).strftime(
                   "%Y-%m-%dT%H:%M:%SZ"),
               "by_source": gen}
    obj_evl = {"name": "blind_evaluation_manifest", "version": "v1",
               "warning": "只有 evaluator 打开；生成阶段禁止访问",
               "by_source": evl}
    assert_generation_manifest_clean({"case_uuid": "x", "fdi": "46",
                                      "source": "modeldie",
                                      "die_asset_id": "h", "die_sha256": "h",
                                      "axis": [0, 0, 1], "mesh_n_faces": 1,
                                      "input_schema_version": "blind_gen_v1"})
    assert_evaluation_manifest_ok(obj_evl)
    print("自检通过：生成侧无 GT 派生字段")
    for src, gl in gen.items():
        print("  %-10s 生成侧条目 %d（病例 %d）"
              % (src, len(gl), len({x["case_uuid"] for x in gl})))
    if a.dry_run:
        print("dry-run：未写盘")
        return
    for name, obj in (("blind_selection_record.json", obj_sel),
                      ("blind_generation_manifest.json", obj_gen),
                      ("blind_evaluation_manifest.json", obj_evl)):
        json.dump(obj, open(os.path.join(a.out, name), "w"),
                  ensure_ascii=True, indent=1)
        print("写出 %s" % os.path.join(a.out, name))


if __name__ == "__main__":
    main()
