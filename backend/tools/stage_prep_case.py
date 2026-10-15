#!/usr/bin/env python3
"""摘取「代型（prep）指向」——Thickness Engine 的内表面来源。纯 stdlib，服务器跑。

## 为什么需要它

CrownOuter 链路目前只生成**外表面**（开放壳，颈环即边界）。厚度审计的内表面
必须来自**代型**：真实 CAD 里 intaglio = 代型表面 + cement gap。而代型不在
crown_outer_cases 暂存物里，且**索引里的 ``modeldie_asset_id`` 对本病例是
修复体壳本身**（AutoMargin 审计已知的 94.9% 根因），不能信。

## 本脚本做什么

不改任何源数据（该账号只读）。只从 ``.modelInfo`` 读出：

1. 该 FDI 的 **代型导出记录**：``<ConstructionFile>`` 里 ``PartType=Model*Die``
   且 ``Filename`` 含 ``-<fdi>-`` 的那条 → 拿它的 SHA256（内容对账用，不写原文件名）；
2. 该 FDI 的 **Margin 点列**（MODEL 坐标系）——与代型同帧由几何证据证明
   （投影 p95 = 0.0000mm）；
3. 该 FDI 的 **设计参数**（MinThickness / CementGapThickness / …）——厚度阈值
   的 provenance 来源，避免"拍脑袋定 T_min"；
4. 顺带记下 ``MatrixToScanDataFiles`` 是否存在（坐标系证据，不参与求解）。

写出 ``die_ref.json``（PHI-free：只有坐标、哈希、计数、参数键值；**不写**
原始文件名 / 内网路径 / PatientId）。

用法::

    python3 stage_prep_case.py cf_b1b73bae221211b1 --fdi 46
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import xml.etree.ElementTree as ET

DEFAULT_ROOTS = ("/data1/welo/dentistry/exo/all_data",
                 "/data1/welo/dentistry/exo/EXO")
CASE_ROOT = "/data/luoda/dentistry/crownflow_data/crown_outer_cases"

#: 该 FDI 的厚度规则相关参数（写进 die_ref.json 的白名单）。
THICKNESS_PARAM_KEYS = (
    "MinThickness", "MinThicknessNearPreparationMargin",
    "MinThicknessFullDistanceToPreparationMargin", "MinThicknessOcclusalOffset",
    "CementGapThickness", "CementGapTopThickness", "CementGapStart",
    "CementGapEnd", "CementGapBorderSmoothing",
    "CrownAdditionalSpacingX", "CrownAdditionalSpacingY",
    "CrownAdditionalSpacingZ", "CrownAdditionalSpacingZOnTop",
    "DistanceToAntagonist", "DistanceToNeighbor",
    "MillingDiameter", "MillingMachineAxes", "CrownBlockoutAngle",
    "CrownBorderHorizontalSize", "CrownBorderAngledSize", "CrownBorderAngle",
)


def sha256_of(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def find_case_dirs(case_uuid: str, roots) -> list[str]:
    want = case_uuid[3:] if case_uuid.startswith("cf_") else case_uuid
    out = []
    for root in roots:
        if not os.path.isdir(root):
            continue
        for dp, dn, _fn in os.walk(root):
            for d in dn:
                if hashlib.sha256(d.encode()).hexdigest()[:16] == want:
                    out.append(os.path.join(dp, d))
    return out


def modelinfo_path(dirs: list[str]) -> str | None:
    for d in dirs:
        for f in sorted(os.listdir(d)):
            if f.endswith(".modelInfo"):
                return os.path.join(d, f)
    return None


def _vec(e) -> list[float]:
    return [float(e.find("x").text), float(e.find("y").text),
            float(e.find("z").text)]


def _matrix(e) -> list[list[float]]:
    return [[float(e.find("_%d%d" % (i, j)).text) for j in range(4)]
            for i in range(4)]


def extract(modelinfo: str, fdi: int) -> dict:
    root = ET.parse(modelinfo).getroot()
    teeth = [t for t in root.find("Teeth")
             if t.findtext("Number") == str(fdi)]
    if not teeth:
        raise SystemExit("modelInfo 里没有 FDI %d" % fdi)
    tooth = teeth[0]

    margin = [_vec(v) for v in tooth.find("Margin")]
    params = {}
    ps = tooth.find("ParametersUsed")
    if ps is not None:
        for p in ps:
            params[p.findtext("Type")] = p.findtext("Value")

    # 代型导出记录：PartType 含 Die 且 Filename 是本 FDI 的
    die_rec = None
    cfl = root.find("ConstructionFileList")
    if cfl is not None:
        for cf in cfl:
            fn = cf.findtext("Filename") or ""
            pt = cf.findtext("PartType") or ""
            if "Die" in pt and ("-%d-" % fdi) in fn:
                die_rec = {"part_type": pt,
                           "sha256": (cf.findtext("FileHashSHA256") or "").lower(),
                           "height_mm": cf.findtext("Height")}
    if die_rec is None:
        raise SystemExit("modelInfo 里找不到 FDI %d 的代型导出记录" % fdi)

    m2s = root.find("MatrixToScanDataFiles")
    return {
        "fdi": fdi,
        "modelinfo_margin_points": margin,
        "modelinfo_n_margin_points": len(margin),
        "die": die_rec,
        "design_parameters": {k: params.get(k)
                              for k in THICKNESS_PARAM_KEYS if k in params},
        "design_parameter_count_total": len(params),
        "has_matrix_to_scan_data_files": m2s is not None,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("case_uuid")
    ap.add_argument("--fdi", type=int, required=True)
    ap.add_argument("--roots", nargs="*", default=list(DEFAULT_ROOTS))
    ap.add_argument("--out-root", default=CASE_ROOT)
    ap.add_argument("--print-die", action="store_true",
                    help="定位代型 STL 并打印其路径（供下载）")
    a = ap.parse_args()

    dirs = find_case_dirs(a.case_uuid, a.roots)
    if not dirs:
        raise SystemExit("找不到病例目录: %s" % a.case_uuid)
    mi = modelinfo_path(dirs)
    if mi is None:
        raise SystemExit("找不到 .modelInfo")
    print("modelInfo:", mi)
    ref = extract(mi, a.fdi)
    ref["case_uuid"] = a.case_uuid
    ref["modelinfo_asset_sha256"] = sha256_of(mi)

    out_dir = os.path.join(a.out_root, a.case_uuid)
    os.makedirs(out_dir, exist_ok=True)
    dst = os.path.join(out_dir, "die_ref.json")
    with open(dst, "w", encoding="ascii") as fh:
        json.dump(ref, fh, indent=1)
    print("写出", dst)
    print("  die:", ref["die"]["part_type"], ref["die"]["sha256"][:16],
          "height=%s" % ref["die"]["height_mm"])
    print("  margin:", ref["modelinfo_n_margin_points"], "点")
    print("  设计参数(厚度相关):",
          json.dumps(ref["design_parameters"], ensure_ascii=False))

    if a.print_die:
        want = ref["die"]["sha256"]
        for d in dirs:
            for dp, _dn, fns in os.walk(d):
                for f in fns:
                    q = os.path.join(dp, f)
                    if f.lower().endswith(".stl") and sha256_of(q) == want:
                        print("DIE_PATH:", q)
                        return 0
        raise SystemExit("按哈希找不到代型 STL")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
