#!/usr/bin/env python3
"""Blind **生成阶段**：只读 generation manifest，绝不接触 GT。

用户 2026-10-01 §1 / §新 Blind 流程：生成过程**永远不能 import / open
evaluation manifest**。本文件因此：

* 只读 ``blind_generation_manifest.json``（输入：代型哈希 + 轴 + 输入 QC）
* 用**冻结的生产候选配置**（M1_V2_PRODUCTION_CANDIDATE）预测
* 把预测落盘 + 写 seal（哈希），**不计算任何误差指标**
* 文件里**不出现** evaluation manifest 的文件名（有一条测试盯着这件事）

用法:
    python3 run_blind_generation.py --out /tmp/automargin \
        --gen-manifest /tmp/automargin/blind_generation_manifest.json \
        --source modeldie
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone

sys.path.insert(0, "/tmp/gt")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.geometry.auto_margin import engine                              # noqa: E402
from app.geometry.auto_margin.models import (AutoMarginConfig,           # noqa: E402
                                             M1_V2_PRODUCTION_CANDIDATE,
                                             frozen_config_id)
from build_auto_margin_gt import (case_dirs, content_hash,          # noqa: E402
                                  read_stl)

#: 生成阶段允许出现的输入键 —— **从 blind_guard 取，避免两处实现分叉**
from app.geometry.auto_margin.blind_guard import (    # noqa: E402
    GEN_KEYS, assert_generation_manifest_clean)

ALLOWED_INPUT_KEYS = set(GEN_KEYS)


# 注意：**必须复用选择阶段同一个 content_hash**（build_auto_margin_gt）。
# 我曾在生成侧本地重写了一份（"sha:"+32 位），格式不一致 → 198 条全部
# die_asset_not_found。同一件事两份实现是这个项目的经典陷阱。


def _resolve_die(uuid, want_hash):
    for d in case_dirs(uuid):
        for dp, _dn, fns in os.walk(d):
            for f in fns:
                if f.lower().endswith(".stl"):
                    q = os.path.join(dp, f)
                    try:
                        if content_hash(q) == want_hash:
                            return q
                    except OSError:
                        continue
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--gen-manifest", required=True)
    ap.add_argument("--source", default="modeldie")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    man = json.load(open(a.gen_manifest, encoding="ascii"))
    entries = (man.get("by_source") or {}).get(a.source) or []
    if not entries:
        raise SystemExit("generation manifest 里没有 source=%s 的条目" % a.source)
    bad = sorted({k for e in entries for k in e if k not in ALLOWED_INPUT_KEYS})
    if bad:
        raise SystemExit("generation manifest 含非输入字段：%s" % bad)
    for e in entries:                     # 结构自检（GT 派生字段一律拒绝）
        assert_generation_manifest_clean(e)
    cfg = AutoMarginConfig(**M1_V2_PRODUCTION_CANDIDATE)
    cid = frozen_config_id(cfg)
    print("冻结生产配置 %s" % cid)
    print("生成侧条目 %d（source=%s）" % (len(entries), a.source), flush=True)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    pred_path = os.path.join(a.out, "blind_predictions_%s_%s.jsonl"
                             % (a.source, stamp))
    t0 = time.time()
    n = 0
    with open(pred_path, "w", encoding="ascii") as fh:
        for e in entries:
            if a.limit and n >= a.limit:
                break
            uuid, fdi = e["case_uuid"], int(e["fdi"])
            die = _resolve_die(uuid, e["die_asset_id"])
            if die is None:
                fh.write(json.dumps({"case_uuid": uuid, "fdi": fdi,
                                     "status": "FAILED",
                                     "reason": "die_asset_not_found"}) + "\n")
                n += 1
                continue
            tris = read_stl(die)
            res = engine.predict_margin(tris, tuple(e["axis"]), cfg)
            rec = {"case_uuid": uuid, "fdi": fdi, "source": a.source,
                   "config_id": res.config_id or cid,
                   "status": res.status, "reason": res.reason,
                   "n_points": (len(res.margin.points)
                                if res.margin is not None else 0),
                   "points": ([list(p) for p in res.margin.points]
                              if res.margin is not None else []),
                   "confidence": res.confidence,
                   "loop_energy": res.loop_energy,
                   "second_energy": res.second_energy,
                   "energy_gap": res.energy_gap,
                   "candidate_rank_stats": res.candidate_rank_stats,
                   "elapsed_s": res.elapsed_s}
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            n += 1
            if n % 5 == 0:
                print("  %d/%d（%.0fs）" % (n, len(entries), time.time() - t0),
                      flush=True)
    raw = open(pred_path, "rb").read()
    seal = {"file": os.path.basename(pred_path),
            "sha256": hashlib.sha256(raw).hexdigest(),
            "n_predictions": n, "config_id": cid,
            "generated_utc": datetime.now(timezone.utc).strftime(
                "%Y-%m-%dT%H:%M:%SZ"),
            "note": "生成阶段完成；评估由独立进程打开 GT 后进行"}
    sp = os.path.join(a.out, "blind_prediction_seal_%s_%s.json"
                      % (a.source, stamp))
    json.dump(seal, open(sp, "w"), indent=1, ensure_ascii=True)
    print("写出 %s（%d 条，%.0fs）" % (pred_path, n, time.time() - t0))
    print("seal %s" % sp)


if __name__ == "__main__":
    main()
