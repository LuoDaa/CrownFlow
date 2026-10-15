"""Bite Reference Audit（C2 第一步，2026-10-13）：**先证明坐标语义，再谈算法**。

用户冻结的顺序：Prep Conditioning ✅ → **Bite Reference Audit** → Bite Registration → …
本批**不写任何配准算法**，只回答四件事：

1. 每个病例的三份扫描各自扮演什么角色（`ScanFiles/ScanFile` 的 ``part_type``：``UnsegmentedJaw``
   预备体颌 / ``Antagonist`` 对颌 / ``BiteRegister`` **咬合对齐扫描** / 其他）；
2. 这些扫描**几何是否可读**（实测：15 例全部是 ``Schema=CE`` 的 3Shape CE 加密载荷 ⇒ 无密钥不可读）；
3. 目录里**还有哪些可读几何资产**，以及它们的角色（用 ``construction_files`` 的
   ``tooth_numbers`` 做**元数据映射**，不靠文件名猜）；
4. 用既有 P1 契约对"逐牙位代型（ModelDie）"这类资产跑一遍输入契约 —— 因为 P1 报告里
   "合格单位 100% 无支撑资产"的结论只查了病例包里的 ``prep.stl/die.stl``，**没有查这些 STL**。

## 术语纪律（用户 2026-10-13 修正，必须遵守）

- ``biteregister.DCM`` = **bite / alignment scan**（第三份扫描几何），**不是**"exocad 算出的注册真值"；
- 真正的 vendor reference 是 **exocad 已经注册好的上下颌最终相对位置**；
- 因此报告里只用 ``EXOCAD_REGISTERED_POSE``（candidate / unavailable），**绝不写 GROUND_TRUTH**，
  也绝不把 bite scan 本身当 pose。

本模块只读：不修改任何病例文件，不落库。
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path

#: ``ScanFiles/ScanFile/@PartType`` 的已知取值 → 人话
PART_TYPE_ROLES = {
    "unsegmentedjaw": "PREP_JAW_SCAN",
    "antagonist": "ANTAGONIST_SCAN",
    "biteregister": "BITE_ALIGNMENT_SCAN",
    "waxupscan": "WAXUP_SCAN",
    "situscan": "SITU_SCAN",
}

#: 可读几何资产的角色（``construction_files`` 命中优先，其次按标准命名兜底）
ASSET_CAD_RESTORATION = "CAD_RESTORATION"
ASSET_MODEL_DIE = "MODEL_DIE"
ASSET_MODEL_BASE = "MODEL_BASE"
ASSET_UNKNOWN = "UNKNOWN_ASSET"

#: C2（配准）在该病例上的可行性
C2_NO_READABLE_SCANS = "NO_READABLE_JAW_OR_BITE_GEOMETRY"
C2_READABLE = "READABLE_SCANS_PRESENT"
C2_PARTIAL = "PARTIAL_READABLE_GEOMETRY"

#: 代型名里的牙位：``吴宝刚-24-modeldie.stl`` / ``xxx-36-modeldie.stl``
_DIE_FDI_RE = re.compile(r"(?<!\d)([1-4][1-8])(?!\d)")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _scan_is_encrypted(path: Path, *, head_bytes: int = 4000) -> tuple[bool, str]:
    """只读文件头判断 DCM 载荷是否 CE 加密（不加载网格）。"""
    try:
        head = path.open("rb").read(head_bytes).decode("utf-8", "replace")
    except OSError as exc:
        return True, "读取失败：%s" % exc
    schema = (re.search(r"<Schema>(.*?)</Schema>", head) or [None, ""])[1].strip()
    if schema.upper() == "CE":
        return True, "HPS Schema=CE（3Shape CE 加密载荷；无合法密钥不可读）"
    return False, "HPS Schema=%s" % (schema or "?")


def _matrix_class(matrix) -> str:
    if not matrix:
        return "ABSENT"
    import numpy as np
    array = np.asarray(matrix, float)
    if array.size != 16:
        return "MALFORMED"
    return "IDENTITY" if np.allclose(array.reshape(4, 4), np.eye(4), atol=1e-9) else "CUSTOM"


def audit_case(case_dir: Path | str, *, run_die_contract: bool = True) -> dict:
    """审计单个病例目录（只读）。"""
    case = Path(case_dir)
    result = {"case": case.name, "scans": [], "construction_files": [], "readable_assets": [],
              "geometry_availability": {}, "c2_feasibility": None, "die_contracts": [],
              "notes": []}

    infos = sorted(case.glob("*.constructionInfo")) or sorted(case.glob("*.modelInfo"))
    parsed = None
    if not infos:
        result["notes"].append("无 constructionInfo/modelInfo —— 设计元数据只在二进制快照里")
    else:
        if len(infos) > 1:
            result["notes"].append("目录里有 %d 份 constructionInfo（可能含多个订单/设计），"
                                   "本次审计用 %s" % (len(infos), infos[0].name))
        result["metadata_file"] = infos[0].name
        try:
            from ..importers.exocad.construction_info import parse_construction_case
            parsed = parse_construction_case(infos[0])
        except Exception as exc:                                  # noqa: BLE001
            result["notes"].append("元数据解析失败：%s" % type(exc).__name__)

    # 1) 扫描角色 + 可读性（角色来自元数据；可读性来自文件头）
    if parsed is not None:
        for scan in parsed.scan_files:
            entry = {"part_type": scan.part_type,
                     "role": PART_TYPE_ROLES.get((scan.part_type or "").strip().lower(),
                                                 "OTHER"),
                     "file_name": scan.file_name,
                     "has_file_hash": bool(scan.file_hash),
                     "transformation_matrix": _matrix_class(scan.transformation_matrix),
                     "readable": None, "readable_reason": None}
            path = case / scan.file_name if scan.file_name else None
            if path is None or not path.is_file():
                entry.update(readable=False, readable_reason="文件不在目录里")
            elif path.suffix.lower() == ".dcm":
                encrypted, reason = _scan_is_encrypted(path)
                entry.update(readable=(not encrypted), readable_reason=reason)
            else:
                entry.update(readable=True, readable_reason="非 DCM")
            result["scans"].append(entry)

    # 2) 可读几何资产目录（CAD 文件用元数据映射牙位/材料，不靠文件名猜）
    cad_by_name = {}
    if parsed is not None:
        for item in getattr(parsed, "construction_files", []) or []:
            name = getattr(item, "filename", None)
            if not name:
                continue
            cad_by_name[Path(name).name] = {
                "tooth_numbers": list(getattr(item, "tooth_numbers", []) or []),
                "reconstruction_units": getattr(item, "reconstruction_units", None),
                "material": getattr(item, "material", None),
                "material_name": getattr(item, "material_name", None)}
            result["construction_files"].append({"file_name": name, **cad_by_name[Path(name).name]})

    for path in sorted(case.iterdir()):
        if not path.is_file():
            continue
        low = path.suffix.lower()
        if low == ".dcm":
            encrypted, _reason = _scan_is_encrypted(path)
            result["readable_assets"].append({"file_name": path.name, "suffix": "dcm",
                                              "role": "ENCRYPTED_SCAN" if encrypted else "SCAN",
                                              "readable": not encrypted,
                                              "size_mb": round(path.stat().st_size / 1e6, 2)})
            continue
        if low != ".stl":
            continue
        entry = {"file_name": path.name, "suffix": "stl", "readable": True,
                 "size_mb": round(path.stat().st_size / 1e6, 2)}
        meta = cad_by_name.get(path.name)
        if meta:
            entry.update(role=ASSET_CAD_RESTORATION, meta=meta)
        else:
            stem = path.stem
            if re.search(r"modeldie", stem, re.I):
                entry["role"] = ASSET_MODEL_DIE
                fdi = _DIE_FDI_RE.findall(stem)
                entry["fdi"] = int(fdi[-1]) if fdi else None
            elif re.search(r"modelbase", stem, re.I):
                entry["role"] = ASSET_MODEL_BASE
            elif re.search(r"_cad|slm_cad", stem, re.I):
                entry["role"] = ASSET_CAD_RESTORATION
            else:
                entry["role"] = ASSET_UNKNOWN
        result["readable_assets"].append(entry)

    # 3) 几何可得性 + C2 可行性
    roles = {s["role"]: s for s in result["scans"]}
    prep_scan = roles.get("PREP_JAW_SCAN") or {}
    ant_scan = roles.get("ANTAGONIST_SCAN") or {}
    bite_scan = roles.get("BITE_ALIGNMENT_SCAN") or {}
    jaw_pair_readable = bool(prep_scan.get("readable") and ant_scan.get("readable"))
    bite_readable = bool(bite_scan.get("readable"))
    assets = result["readable_assets"]
    result["geometry_availability"] = {
        "prep_jaw_scan_readable": bool(prep_scan.get("readable")),
        "antagonist_scan_readable": bool(ant_scan.get("readable")),
        "bite_alignment_scan_readable": bite_readable,
        "jaw_pair_readable": jaw_pair_readable,
        "second_bite_scan_present": any(
            p.name.lower().endswith("biteregister2.dcm") for p in case.iterdir() if p.is_file()),
        "model_die_count": sum(1 for a in assets if a["role"] == ASSET_MODEL_DIE),
        "model_die_fdis": sorted({a.get("fdi") for a in assets if a["role"] == ASSET_MODEL_DIE
                                  and a.get("fdi")}),
        "model_base_count": sum(1 for a in assets if a["role"] == ASSET_MODEL_BASE),
        "cad_restoration_count": sum(1 for a in assets if a["role"] == ASSET_CAD_RESTORATION),
        "unknown_asset_count": sum(1 for a in assets if a["role"] == ASSET_UNKNOWN),
    }
    if jaw_pair_readable and bite_readable:
        result["c2_feasibility"] = C2_READABLE
    elif jaw_pair_readable or bite_readable:
        result["c2_feasibility"] = C2_PARTIAL
    else:
        result["c2_feasibility"] = C2_NO_READABLE_SCANS
    # vendor reference 命名纪律：没有可读几何就谈不上"已注册位姿"能否被验证
    result["vendor_reference_pose"] = {
        "reference_type": "EXOCAD_REGISTERED_POSE",
        "status": ("UNAVAILABLE_NO_READABLE_GEOMETRY" if not jaw_pair_readable
                   else "VERIFIABLE_BUT_NOT_YET_VERIFIED"),
        "note": "biteregister 是 bite/alignment scan，**不是** pose；只有已注册的上下颌相对位姿才是 reference",
    }

    # 4) 逐牙位代型 → 跑既有 P1 契约（本批唯一"动手"的地方，且只读）
    if run_die_contract and parsed is not None:
        result["die_contracts"] = _assess_die_assets(case, parsed, result["readable_assets"])
    return result


def _assess_die_assets(case: Path, parsed, assets: list[dict]) -> list[dict]:
    """把 ``*-modeldie.stl`` 喂给 P1 契约（需要该 FDI 的 margin + 插入轴）。"""
    from ..geometry.io import load_mesh
    from .design_dependencies import file_digest
    from .prep_conditioning import (Indication, assess_prep_surface)
    import numpy as np

    teeth = {int(getattr(t, "tooth_fdi", 0) or 0): t for t in getattr(parsed, "teeth", []) or []}
    out = []
    for asset in assets:
        if asset.get("role") != ASSET_MODEL_DIE or not asset.get("fdi"):
            continue
        tooth = teeth.get(int(asset["fdi"]))
        entry = {"file_name": asset["file_name"], "fdi": asset["fdi"],
                 "asset_role": "PRINTED_MODEL_DIE_NOT_INTRAORAL_PREP"}
        if tooth is None or not getattr(tooth, "margin_points", None) \
                or not getattr(tooth, "insertion_axis", None):
            entry["contract"] = {"classification": {
                "readiness": "REVIEW_REQUIRED", "rejection_code": "MARGIN_OR_AXIS_MISSING",
                "rejection_class": "REVIEW_REQUIRED",
                "rejection_reason": "该牙位在 constructionInfo 里没有 margin/插入轴"}}
            out.append(entry)
            continue
        path = case / asset["file_name"]
        try:
            contract = assess_prep_surface(
                load_mesh(path), np.asarray(tooth.margin_points, float),
                np.asarray(tooth.insertion_axis, float),
                case_id=case.name, target_fdi=int(asset["fdi"]),
                source_asset_sha256=file_digest(path),
                margin_point_count=len(tooth.margin_points),
                indication=Indication.NATURAL_TOOTH_SINGLE_ANATOMICAL_CROWN.value,
                screw_channel_present=None,
                provenance={"source": "exocad_model_die_export"})
            entry["contract"] = contract.as_report()
        except Exception as exc:                                  # noqa: BLE001
            entry["contract"] = {"classification": {
                "readiness": "REVIEW_REQUIRED", "rejection_code": "ASSET_LOAD_FAILED",
                "rejection_reason": "%s: %s" % (type(exc).__name__, exc)}}
        out.append(entry)
    return out


def audit_corpus(cases_root: Path | str, *, run_die_contract: bool = True) -> dict:
    root = Path(cases_root)
    cases = [audit_case(p, run_die_contract=run_die_contract)
             for p in sorted(p for p in root.iterdir() if p.is_dir())]
    report = {"generated_at": _now(), "cases_root": str(root), "total_cases": len(cases),
              "c2_feasibility": {}, "geometry_availability_totals": {}, "cases": cases}
    for entry in cases:
        key = entry["c2_feasibility"] or "UNKNOWN"
        report["c2_feasibility"][key] = report["c2_feasibility"].get(key, 0) + 1
    totals = {}
    for entry in cases:
        for key, value in (entry.get("geometry_availability") or {}).items():
            if isinstance(value, bool):
                totals[key] = totals.get(key, 0) + int(value)
            elif isinstance(value, int):
                totals[key] = totals.get(key, 0) + value
    report["geometry_availability_totals"] = totals
    die_results = [c for entry in cases for c in (entry.get("die_contracts") or [])]
    report["die_contract_summary"] = {
        "attempted": len(die_results),
        "by_readiness": {}} 
    for item in die_results:
        readiness = ((item.get("contract") or {}).get("classification") or {}).get("readiness", "UNKNOWN")
        report["die_contract_summary"]["by_readiness"][readiness] = \
            report["die_contract_summary"]["by_readiness"].get(readiness, 0) + 1
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Bite Reference Audit（只读；不跑配准算法）")
    parser.add_argument("--cases", required=True, help="exocad 完整病例目录")
    parser.add_argument("--out", default="bite_reference_audit.json")
    parser.add_argument("--no-die-contract", action="store_true",
                        help="跳过逐牙位代型的 P1 契约评估")
    args = parser.parse_args()
    report = audit_corpus(args.cases, run_die_contract=not args.no_die_contract)
    Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("total_cases", "c2_feasibility",
                                             "geometry_availability_totals",
                                             "die_contract_summary")},
                     ensure_ascii=False, indent=2))
    print("written: %s" % args.out)
    return 0


if __name__ == "__main__":                                       # pragma: no cover
    raise SystemExit(main())
