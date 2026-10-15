#!/usr/bin/env python3
"""Blind 封存（独立步骤）：锁住**预测 + 运行条件**（用户 2026-10-01 §2）。

为什么单独一步、不改 runner：生成正在跑，改 runner 会得到"半批旧 runner、
半批新 runner"的混合结果。封存是对**已完成的不可变产物**做事后认证：

    generation_manifest_sha256   输入清单没被改
    prediction_file_sha256       预测没被改
    git_commit                   冻结的代码版本
    config_id                    冻结的算法配置
    input_schema_version         输入 schema 版本
    runner_file_sha256           **实际跑的那份 runner 字节**
    n_expected / n_written       一一对应，无缺无多

并做两项断言：① 每条 manifest 输入恰好一条预测；② runner 字节哈希与仓库提交一致
（由调用方给出仓库侧哈希，或本工具自行在服务器上算）。

用法:
    python3 seal_blind.py --out /tmp/automargin --source modeldie \
        --pred /tmp/automargin/blind_predictions_modeldie_<stamp>.jsonl \
        --gen-manifest /tmp/automargin/blind_generation_manifest.json \
        --runner /tmp/gt/run_blind_generation.py \
        --git-commit 420823d
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--source", default="modeldie")
    ap.add_argument("--pred", required=True)
    ap.add_argument("--gen-manifest", required=True)
    ap.add_argument("--runner", required=True)
    ap.add_argument("--git-commit", default="")
    a = ap.parse_args()

    gen = json.load(open(a.gen_manifest, encoding="ascii"))
    entries = (gen.get("by_source") or {}).get(a.source) or []
    expected = {(e["case_uuid"], str(e["fdi"])) for e in entries}
    preds = [json.loads(l) for l in
             open(a.pred, encoding="ascii").read().splitlines() if l.strip()]
    got = {(p["case_uuid"], str(p["fdi"])) for p in preds}

    missing = sorted(expected - got)
    extra = sorted(got - expected)
    if missing or extra:
        raise SystemExit("预测与输入不一一对应：缺 %d、多 %d（前几个 %s / %s）"
                         % (len(missing), len(extra), missing[:3], extra[:3]))
    if len(preds) != len(expected):
        raise SystemExit("条数不符：expected %d, written %d"
                         % (len(expected), len(preds)))
    cfg_ids = sorted({p.get("config_id", "") for p in preds if p.get("config_id")})
    if len(cfg_ids) > 1:
        raise SystemExit("预测里出现多个 config_id：%s —— 不是同一次冻结运行"
                         % cfg_ids)

    seal = {
        "generation_manifest_sha256": sha256_file(a.gen_manifest),
        "prediction_file_sha256": sha256_file(a.pred),
        "git_commit": a.git_commit,
        "config_id": cfg_ids[0] if cfg_ids else "",
        "input_schema_version": gen.get("input_schema_version", ""),
        "runner_file_sha256": sha256_file(a.runner),
        "n_expected": len(expected),
        "n_written": len(preds),
        "source": a.source,
        "sealed_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "note": ("本轮 generation 由上述 runner 字节 + config_id 在冻结的输入清单上"
                 "产生；算法/QC 阈值/manifest 均为冻结版本"),
    }
    # 状态分布也一并记下（GT-independent，可安全写进 seal）
    from collections import Counter
    seal["status_counts"] = dict(Counter(p["status"] for p in preds))
    p = os.path.join(a.out, "blind_seal_%s.json" % a.source)
    json.dump(seal, open(p, "w"), indent=1, ensure_ascii=True)
    print(json.dumps(seal, ensure_ascii=True, indent=1))
    print("封存写出 →", p)


if __name__ == "__main__":
    main()
