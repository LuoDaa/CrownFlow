#!/usr/bin/env python3
"""把一个真实病例的 CrownOuter 输入「摘」出来，落到 ``crown_outer_cases/``。

纯 stdlib（在数据机 / 服务器上跑，那边没有 numpy）。产出**无 PHI**：
文件名一律用 case_uuid + FDI，原始文件名只以内容哈希形式出现。

    CROWNFLOW_CASE_ROOT=/data/luoda/dentistry/crownflow_data/crown_outer_cases \
    python3 tools/stage_crown_outer_case.py \
        --index /tmp/gt_freeze_v1/frozen_tooth_index.csv \
        --case cf_d310209b58909a45 --fdi 46

每个病例目录内容：

    case.json          margin 点列 / Center / Axis / 各种哈希与计数
    final_crown.stl    导出冠的副本（改名为中性名字）

``final_crown.stl`` 的选取必须与索引器一致，否则评测对象就错了。
这里用**内容哈希**对账：索引里记了 ``final_crown_asset_id``，摘出来的文件
必须哈希相同，否则报错退出。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys

sys.path.insert(0, os.environ.get("CROWNFOW_APP_PATH", "/tmp/gt"))

from app.groundtruth.assets import content_hash  # noqa: E402
from app.importers.exocad.construction_info import parse_construction_case  # noqa: E402

DEFAULT_ROOTS = ("/data1/welo/dentistry/exo/all_data",
                 "/data1/welo/dentistry/exo/EXO")


def find_case_dir(case_uuid: str, roots) -> str:
    want = case_uuid[3:] if case_uuid.startswith("cf_") else case_uuid
    for root in roots:
        for dp, dn, _fn in os.walk(root):
            for d in dn:
                if hashlib.sha256(d.encode()).hexdigest()[:16] == want:
                    return os.path.join(dp, d)
    raise SystemExit("找不到病例目录（uuid=%s）" % case_uuid)


def collect(case_dir: str) -> dict:
    stls, cis = [], []
    for dp, _dn, fns in os.walk(case_dir):
        for f in fns:
            p = os.path.join(dp, f)
            if f.lower().endswith(".stl"):
                stls.append(p)
            elif f.endswith(".constructionInfo") or f.endswith(".modelInfo"):
                cis.append(p)
    cis.sort(key=lambda p: (not p.endswith(".constructionInfo"),))
    return {"stls": sorted(stls), "cis": cis}


def pick_crown(case, fdi, stls) -> str | None:
    """按索引器同一规则选导出冠：先查导出清单，再退回文件名里的 _cad + FDI。"""
    by_fdi = {}
    for cf in case.construction_files:
        for n in (cf.tooth_numbers or []):
            by_fdi.setdefault(int(n), []).append(cf)
    for cf in by_fdi.get(fdi, []):
        base = os.path.basename(cf.filename or "")
        for p in stls:
            if os.path.basename(p) == base:
                return p
    for p in stls:
        low = os.path.basename(p).lower()
        if "_cad" not in low:
            continue
        if str(fdi) in os.path.basename(p):
            return p
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", required=True, help="frozen_tooth_index.csv")
    ap.add_argument("--case", required=True, help="case_uuid")
    ap.add_argument("--fdi", type=int, required=True)
    ap.add_argument("--out", default=os.environ.get("CROWNFLOW_CASE_ROOT"))
    ap.add_argument("--root", action="append", default=None)
    a = ap.parse_args()
    if not a.out:
        raise SystemExit("未指定输出目录：--out 或 CROWNFLOW_CASE_ROOT")

    import csv
    row = None
    with open(a.index, newline="", encoding="ascii") as fh:
        for d in csv.DictReader(fh):
            if d["case_uuid"] == a.case and int(d["fdi"]) == a.fdi:
                row = d
                break
    if row is None:
        raise SystemExit("索引里没有 (%s, %d)" % (a.case, a.fdi))
    want_hash = row["final_crown_asset_id"]

    case_dir = find_case_dir(a.case, a.root or DEFAULT_ROOTS)
    print("病例目录:", case_dir)
    files = collect(case_dir)
    print("STL %d 个，设计文件 %d 个" % (len(files["stls"]), len(files["cis"])))
    if not files["cis"]:
        raise SystemExit("没有 constructionInfo/modelInfo")

    case = parse_construction_case(files["cis"][0])
    tooth = case.tooth(a.fdi)
    if tooth is None or not tooth.margin_points:
        raise SystemExit("FDI%d 没有 Margin" % a.fdi)
    if not tooth.insertion_axis:
        raise SystemExit("FDI%d 没有 InsertionAxis" % a.fdi)

    crown = pick_crown(case, a.fdi, files["stls"])
    if not crown:
        raise SystemExit("没找到 FDI%d 的导出冠 STL" % a.fdi)
    got_hash = content_hash(crown)
    print("候选冠:", os.path.basename(crown)[:24], "…")
    print("  内容哈希 %s（索引 %s）" % (got_hash, want_hash))
    if want_hash and got_hash and got_hash != want_hash:
        raise SystemExit("冠 STL 与索引记录不一致 —— 选错文件，拒绝写出")

    out_dir = os.path.join(a.out, a.case)
    os.makedirs(out_dir, exist_ok=True)
    dst_stl = os.path.join(out_dir, "final_crown.stl")
    shutil.copyfile(crown, dst_stl)

    meta = {
        "case_uuid": a.case,
        "fdi": a.fdi,
        "source_design_snapshot": ("CONSTRUCTION_INFO"
                                   if files["cis"][0].endswith(".constructionInfo")
                                   else "MODEL_INFO"),
        "design_hash": content_hash(files["cis"][0]),
        "final_crown_hash": got_hash,
        "final_crown_stl_bytes": os.path.getsize(dst_stl),
        "n_margin_points": len(tooth.margin_points),
        "margin_points": [list(map(float, p)) for p in tooth.margin_points],
        "center": [float(v) for v in tooth.center] if tooth.center else None,
        "insertion_axis": [float(v) for v in tooth.insertion_axis],
        "axis_occlusal_of_tooth_model": (
            [float(v) for v in tooth.axis_occlusal_of_tooth_model]
            if tooth.axis_occlusal_of_tooth_model else None),
        "axis_mesial": ([float(v) for v in tooth.axis_mesial]
                        if tooth.axis_mesial else None),
        # 颊侧轴向：分区域统计（颊/舌/近中/远中）需要患者自己的解剖方向，
        # 不猜。缺失时下游按"近中/远中由 PCA 轴 ±、颊舌标记为 unknown"降级。
        "axis_buccal": ([float(v) for v in tooth.axis_buccal]
                        if tooth.axis_buccal else None),
        "reconstruction_type": tooth.reconstruction_type,
        "tooth_library_name_present": bool(tooth.tooth_library_name),
    }
    with open(os.path.join(out_dir, "case.json"), "w", encoding="ascii") as fh:
        json.dump(meta, fh, indent=1)

    print("\n写出:", out_dir)
    print("  case.json        %d 个 margin 点" % len(tooth.margin_points))
    print("  final_crown.stl  %.2f MB" % (meta["final_crown_stl_bytes"] / 1e6))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
