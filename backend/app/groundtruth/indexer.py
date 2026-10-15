"""GroundTruthIndexer：把文件堆变成一张「病例 × FDI」真值索引。

**纯 stdlib。** 数据在服务器上，而那台机器没有 numpy/trimesh；索引器必须能就地跑，
否则"把 566 GB 变成可复现的索引"就得先把数据搬走。有测试断言本模块与
``assets.py`` 不 import numpy/trimesh。

## 三种来源目录，同一病例会被拆开

实测 ``exo/EXO/<id>/`` 与 ``exo/all_data/<id>/`` 是**同一病例的两半**：
前者放模型与截图，后者放设计导出（``constructionInfo`` + ``*_cad.stl``）。
所以索引器按**病例 id（目录名）聚合后再索引**，跨 root 合并文件清单 ——
否则同一病例会被算成两行，或者只看到一半的文件。

## 承重假设在这里落地

``target_center_surface_distance_mm`` / ``target_center_surface_coverage`` /
``arch_is_sectioned_or_unsectioned`` 三项来自同一个几何判定：

    目标牙 <Center> 到整颌候选表面的距离 + 该点四周有没有面

未分割整颌里预备体还在 → 中心处有实体；代型被抽走的 ``modelbase`` →
中心处是空洞。**只有 unsectioned 才能当「整颌 → 找目标牙」的正样本。**

## 用法

    python -m app.groundtruth.indexer --root /data1/welo/dentistry/exo \\
        --out /data1/welo/gt_index

在只读数据目录之外写输出（源数据目录对本人只读）。
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

from .assets import (
    CENTER_SURFACE_NEAR_MM,
    CENTER_SURFACE_VOID_MM,
    StlProbe,
    assert_phi_free,
    content_hash,
    fdis_in_name,
    probe_stl,
    read_vertex_grid,
    safe_asset_path,
    safe_case_id,
    safe_text_token,
    watertight_and_euler,
)
from .models import (
    COLUMNS,
    GT_SETS,
    ToothGroundTruth,
    add_note,
    build_all_sets,
    case_group_key,
    is_margin_valid,
)

#: ``.constructionInfo`` 后缀（注意大小写两种都出现过）。
_CI_SUFFIXES = (".constructionInfo", ".modelInfo")

#: 几何选代型结果：{(case_uuid, fdi): die_asset_id}。由 build_auto_margin_gt
#: --stage select 产出（PHI-free），CLI 用 --die-selection 注入。空则只用文件名证据。
_DIE_SELECTION: dict = {}


def load_die_selection(path):
    """读 die_selection*.csv，收集 selection == MARGIN_ON_DIE 的行。"""
    import csv as _csv
    n = 0
    with open(path, encoding="ascii") as fh:
        for r in _csv.DictReader(fh):
            if r.get("selection") != "MARGIN_ON_DIE":
                continue
            h = r.get("die_asset_id") or ""
            if h:
                _DIE_SELECTION[(r["case_uuid"], int(r["fdi"]))] = h
                n += 1
    return n
_STL = ".stl"


@dataclass
class CaseFiles:
    """一个病例 id 名下聚合到的全部相关文件。"""

    case_id: str
    construction_infos: list[str] = field(default_factory=list)
    dental_projects: list[str] = field(default_factory=list)
    stls: list[str] = field(default_factory=list)

    @property
    def has_construction_info(self) -> bool:
        return bool(self.construction_infos)


def discover_cases(roots: list[str]) -> dict[str, CaseFiles]:
    """扫描 roots，按**目录名**聚合病例。跨 root 合并同一 id。"""
    cases: dict[str, CaseFiles] = {}
    for root in roots:
        for dirpath, _dirnames, filenames in os.walk(root):
            if not filenames:
                continue
            cid = os.path.basename(dirpath)
            if not cid:
                continue
            bucket = cases.setdefault(cid, CaseFiles(case_id=cid))
            for fn in filenames:
                full = os.path.join(dirpath, fn)
                low = fn.lower()
                if fn.endswith(_CI_SUFFIXES):
                    # .modelInfo 与 .constructionInfo 同 schema；优先 constructionInfo
                    if fn.endswith(".constructionInfo"):
                        bucket.construction_infos.insert(0, full)
                    else:
                        bucket.construction_infos.append(full)
                elif fn.endswith(".dentalProject"):
                    bucket.dental_projects.append(full)
                elif low.endswith(_STL):
                    bucket.stls.append(full)
    return cases


# ---------------------------------------------------------------- 单项判据

def _axis_valid(axis) -> bool:
    if not axis or len(axis) != 3:
        return False
    if not all(isinstance(x, float) and math.isfinite(x) for x in axis):
        return False
    n = math.sqrt(sum(x * x for x in axis))
    return abs(n - 1.0) <= 0.05


def _margin_metrics(points) -> tuple[float, float]:
    """``(闭合周长, 最大段长)``。点数 < 2 返回 (0, 0)。"""
    if len(points) < 2:
        return 0.0, 0.0
    total = 0.0
    worst = 0.0
    n = len(points)
    for i in range(n):
        a, b = points[i], points[(i + 1) % n]      # 闭合
        d = math.dist(a, b)
        total += d
        if d > worst:
            worst = d
    return total, worst


def _p95_distance_to_surface(points, grid, *, sample: int = 25) -> float | None:
    """颈缘点到冠表面的 95 分位距离。用来判"颈缘是否真属于这颗冠"。"""
    if grid is None or not points:
        return None
    step = max(1, len(points) // sample)
    ds = []
    for p in points[::step]:
        d, _found = grid.nearest_distance(tuple(p), search_radius_mm=4.0)
        ds.append(d)
    if not ds:
        return None
    ds.sort()
    return ds[min(len(ds) - 1, int(round(0.95 * (len(ds) - 1))))]


def classify_arch_sectioned(surface_distance_mm: float | None,
                            coverage: float | None) -> str:
    """``unsectioned`` / ``sectioned`` / ``unknown``。

    判据是**两个独立信号取一致**，不一致就老实说 unknown：
      * 中心到表面距离：有实体则近（几个 mm 内）
      * 中心四周方向覆盖率：有实体则各方向都能在 6 mm 内碰到面
    只看距离不稳（中心可能落在洞边因而很近），只看覆盖率也不稳（大平面会偏大）。
    """
    if surface_distance_mm is None:
        return "unknown"
    d = surface_distance_mm
    c = coverage if coverage is not None else 0.0
    if d <= CENTER_SURFACE_NEAR_MM and c >= 0.50:
        return "unsectioned"
    if d >= CENTER_SURFACE_VOID_MM or c <= 0.25:
        return "sectioned"
    return "unknown"


# ---------------------------------------------------------------- 主流程

def build_index(cases: dict[str, CaseFiles], *,
                max_arch_probes: int = 3,
                crown_topology: bool = True,
                do_hash: bool = True,
                progress=None) -> list[ToothGroundTruth]:
    """对每个病例产出若干行（一行一个牙位）。"""
    rows: list[ToothGroundTruth] = []
    for n, (cid, files) in enumerate(sorted(cases.items()), 1):
        if not files.has_construction_info:
            continue
        if progress and n % 50 == 0:
            progress(f"  ...已处理 {n}/{len(cases)} 个病例，当前行数 {len(rows)}")
        rows.extend(_index_case(cid, files, max_arch_probes=max_arch_probes,
                                crown_topology=crown_topology, do_hash=do_hash))
    return rows


def _index_case(cid: str, files: CaseFiles, *, max_arch_probes: int,
                crown_topology: bool, do_hash: bool) -> list[ToothGroundTruth]:
    # 延迟 import：让"本模块不依赖重库"这件事只由这一处的链路决定
    from ..importers.exocad.construction_info import parse_construction_case

    ci_path = files.construction_infos[0]
    # 设计快照来源：constructionInfo 优先（canonical）；只有 modelInfo 时
    # 仍然解析（记录 metadata/研究候选），但 provenance 会如实标注，
    # GT 集合按政策把它排除在几何真值之外。
    is_model_info = ci_path.lower().endswith(".modelinfo")
    try:
        case = parse_construction_case(ci_path)
    except Exception as exc:
        return [ToothGroundTruth(
            case_uuid=_case_uuid(cid), source_case_id=safe_case_id(cid),
            source_order_id=None, fdi=0, reconstruction_type="PARSE_FAILED",
            has_construction_info=True,
            notes=f"constructionInfo 解析失败: {type(exc).__name__}")]

    case_uuid = _case_uuid(cid)
    safe_id = safe_case_id(cid)
    if is_model_info:
        source_kind, frame_kind = "MODEL_INFO", "MODEL_INFO_NATIVE"
    else:
        source_kind, frame_kind = "CONSTRUCTION_INFO", "EXPORT_CANONICAL"
    # 分组键：**不用 PatientId**（那是 PHI）。病例 id 本身就把同一订单的多个牙位
    # 归在一起，这正是训练/验证拆分需要的粒度。
    #
    # 但 `<ProjectName>` **不能直接当它用** —— 实测有病例把项目命名为患者姓名
    # （例如 `杨柳`），直接写出就是 PHI 泄漏。这里做稳定脱敏：
    # 纯 ASCII 的订单号原样保留，含非 ASCII 的换成确定性 token ——
    # **同名映射到同 token**，所以 group by 依然成立，指标不会被拆散。
    order_id = safe_text_token(case.project_name or cid, prefix="order")

    # --- 资产探测：只探体积最大的若干个，避免每个病例都读全量
    probes: dict[str, StlProbe] = {}
    for p in files.stls:
        try:
            probes[p] = probe_stl(p)
        except Exception:
            continue

    archs = [p for p, pr in probes.items() if pr.ok and pr.scale_class == "arch"]
    singles = [p for p, pr in probes.items() if pr.ok and pr.scale_class == "single_tooth"]
    archs.sort(key=lambda p: -probes[p].bytes)
    archs = archs[:max_arch_probes]

    # --- 导出清单：以 constructionInfo 的 ToothNumbers 为准（权威）
    export_by_fdi: dict[int, list] = {}
    for cf in case.construction_files:
        if not cf.looks_like_cad_export:
            continue
        for fdi in cf.tooth_numbers:
            export_by_fdi.setdefault(fdi, []).append(cf)

    def find_stl_by_name(filename: str) -> str | None:
        base = os.path.basename(filename or "")
        if not base:
            return None
        for p in files.stls:
            if os.path.basename(p) == base:
                return p
        return None

    out: list[ToothGroundTruth] = []
    for tooth in case.teeth:
        if tooth.tooth_fdi is None:
            continue
        row = ToothGroundTruth(
            case_uuid=case_uuid, source_case_id=safe_id, source_order_id=order_id,
            fdi=tooth.tooth_fdi,
            reconstruction_type=tooth.reconstruction_type.value,
            has_construction_info=True,
            has_parameters=bool(tooth.parameters),
            parameter_count=len(tooth.parameters),
            design_source_kind=source_kind,
            design_source_asset_id=_asset_id(ci_path, do_hash),
            coordinate_frame_kind=frame_kind,
        )

        # --- 颈缘线
        pts = tooth.margin_points
        row.has_margin = len(pts) > 0
        row.margin_point_count = len(pts)
        if pts:
            per, worst = _margin_metrics(pts)
            row.margin_perimeter_mm = per
            row.margin_max_segment_mm = worst
            diag = case.diagnostics.get(f"fdi_{tooth.tooth_fdi}_margin", {})
            if diag.get("n_bad"):
                add_note(row, f"margin_bad_points={diag['n_bad']}")
        row.margin_valid = is_margin_valid(row)

        # --- 就位方向 / 牙架 / 库牙 / 矩阵
        row.has_axis = tooth.insertion_axis is not None
        row.axis_valid = _axis_valid(tooth.insertion_axis)
        row.has_center = tooth.center is not None
        if tooth.center is not None and len(tooth.center) == 3:
            row.center = (tooth.center[0], tooth.center[1], tooth.center[2])
        row.has_tooth_frame = all(a is not None for a in (
            tooth.axis_occlusal_of_tooth_model, tooth.axis_mesial, tooth.axis_buccal))
        row.has_tooth_library = bool(tooth.tooth_library_name)
        row.tooth_library_name = safe_text_token(
            tooth.tooth_library_name, prefix="lib")
        row.tooth_model_matrix_valid = bool(
            tooth.tooth_model_matrix and len(tooth.tooth_model_matrix) == 16
            and all(isinstance(x, float) and math.isfinite(x)
                    for x in tooth.tooth_model_matrix))

        # --- 最终冠：先按导出清单匹配，匹配不到退到文件名
        crown_path = None
        fdi_count = 0
        matched = export_by_fdi.get(tooth.tooth_fdi)
        if matched:
            for cf in matched:
                # ToothNumbers 权威：桥件的多个 FDI 都在里面
                fdi_count = max(fdi_count, len(cf.tooth_numbers))
                cand = find_stl_by_name(cf.filename)
                if cand:
                    crown_path = cand
                    break
        if crown_path is None:
            for p in files.stls:
                low = os.path.basename(p).lower()
                if "_cad" not in low:
                    continue
                named = fdis_in_name(p)
                if tooth.tooth_fdi in named:
                    crown_path = p
                    # 这里必须报**文件名里的全部 FDI**，而不是最后一个 ——
                    # 旧实现只取最后一个，把 `-12-11-21-22-bridge_slm_cad` 当成单冠。
                    fdi_count = max(fdi_count, len(named))
                    break
        if crown_path is not None:
            row.has_final_crown = True
            row.final_crown_fdi_count = max(1, fdi_count)
            row.final_crown_single_unit = (row.final_crown_fdi_count == 1)
            row.final_crown_asset_id = _asset_id(crown_path, do_hash)
            if crown_topology:
                wt, euler = watertight_and_euler(crown_path)
                row.final_crown_watertight = wt
                row.final_crown_euler = euler
            # 颈缘 ↔ 冠 一致性
            grid = read_vertex_grid(crown_path)
            p95 = _p95_distance_to_surface(pts, grid) if pts else None
            row.margin_crown_p95_mm = p95
            row.margin_crown_consistent = bool(p95 is not None and p95 <= 0.5)

        # --- 代型（support）
        # 用户决策 2026-09-30 ③：**区分"有证据的代型"与"最近的 STL"**。
        # 证据来源两种，几何证据优先（它自带验证：Margin 投影误差）：
        #   1) MARGIN_PROJECTION_VERIFIED —— 由 build_auto_margin_gt --stage select
        #      产出（该牙位 Margin 投影到候选代型，p95 <= 0.20mm 视为命中）
        #   2) FILENAME_AND_FDI —— 文件名含 modeldie 且含该牙位 FDI
        # 都没有时**只写 nearest_asset_***，has_modeldie 保持 False。
        # 旧实现把回退候选当 modeldie，实测 2417/2546 = 94.9% 的"代型"其实是
        # 修复体 CAD 壳（docs/AUTO_MARGIN_M0_ASSET_AUDIT.md），必须修。
        die_path = None
        selection = None
        sel_hash = _DIE_SELECTION.get((row.case_uuid, tooth.tooth_fdi))
        if sel_hash:
            for p in singles:
                if _asset_id(p, do_hash) == sel_hash:
                    die_path = p
                    selection = "MARGIN_PROJECTION_VERIFIED"
                    break
        if die_path is None:
            for p in singles:
                if "modeldie" in os.path.basename(p).lower() and \
                        tooth.tooth_fdi in fdis_in_name(p):
                    die_path = p
                    selection = "FILENAME_AND_FDI"
                    break
        if die_path is not None:
            row.has_modeldie = True
            row.modeldie_asset_id = _asset_id(die_path, do_hash)
            row.asset_selection_method = selection
            if row.center is not None:
                d = math.dist(probes[die_path].center, row.center)
                row.modeldie_center_distance_mm = d
                # 同坐标系的判据：代型中心与该牙设计中心的距离只有几个 mm
                row.modeldie_same_frame = d <= 10.0
        elif row.center is not None and singles:
            # 回退候选：**只写 nearest_asset_***，绝不写 modeldie_*
            best = None
            best_d = 1e9
            for p in singles:
                d = math.dist(probes[p].center, row.center)
                if d < best_d:
                    best_d, best = d, p
            if best is not None and best_d <= 10.0:
                row.nearest_asset_id = _asset_id(best, do_hash)
                row.nearest_asset_center_distance_mm = best_d
                row.asset_selection_method = "NEAREST_TO_CENTER_FALLBACK"

        # --- 整颌候选 + 承重假设
        if archs and row.center is not None:
            nearest = None
            nearest_d = 1e9
            for p in archs:
                d = math.dist(probes[p].center, row.center)
                if d < nearest_d:
                    nearest_d, nearest = d, p
            row.has_full_arch_candidate = True
            row.full_arch_asset_id = _asset_id(nearest, do_hash)
            ext = probes[nearest].extents
            row.full_arch_bbox_mm = ";".join(f"{e:.1f}" for e in ext)
            grid = read_vertex_grid(nearest)
            d = None
            cov = None
            if grid is not None:
                # 搜索半径取得比"空洞"判据大，于是"找不到"就等价于"确实很远"
                d, found = grid.nearest_distance(
                    row.center, search_radius_mm=CENTER_SURFACE_VOID_MM * 2.5)
                if not found:
                    add_note(row, "surface_distance_saturated")
                cov = grid.coverage(row.center, 6.0)
            row.target_center_surface_distance_mm = d
            row.target_center_surface_coverage = cov
            row.target_center_has_surface = bool(
                row.target_center_surface_distance_mm is not None
                and row.target_center_surface_distance_mm <= CENTER_SURFACE_NEAR_MM)
            row.arch_is_sectioned_or_unsectioned = classify_arch_sectioned(
                row.target_center_surface_distance_mm, cov)
            # 对颌：≥2 个弓候选时，取离目标牙最远的那个（启发式，写进 notes）
            if len(archs) >= 2:
                far = max(archs, key=lambda p: math.dist(probes[p].center, row.center))
                if far != nearest:
                    row.has_antagonist = True
                    row.antagonist_asset_id = _asset_id(far, do_hash)
                    add_note(row, "antagonist=heuristic_farthest_arch")

        if source_kind == "CONSTRUCTION_INFO":
            row.geometry_validation_status = "CANONICAL"
        elif row.has_final_crown:
            # modelInfo 行碰巧带冠导出时，用 margin→crown 一致性直接验证
            row.geometry_validation_status = (
                "VALIDATED" if row.margin_crown_consistent else "INCONSISTENT")
        else:
            row.geometry_validation_status = "UNVALIDATABLE_NO_CROWN"
        out.append(row)
    return out


def _case_uuid(cid: str) -> str:
    """与 ``app/importers/exocad/case_loader`` 同一约定：``cf_`` + sha256 前 16 位。"""
    return "cf_" + hashlib.sha256(cid.encode("utf-8")).hexdigest()[:16]


def _asset_id(path: str, do_hash: bool) -> str:
    """资产 id = 内容哈希。只存它，就不必把含姓名的原文件名写进索引。"""
    if do_hash:
        h = content_hash(path)
        if h:
            return h
    return "sha:" + hashlib.sha256(os.path.basename(path).encode("utf-8")).hexdigest()[:16]


# ---------------------------------------------------------------- 输出

def write_outputs(rows: list[ToothGroundTruth], out_dir: str | Path) -> dict:
    """写 ``tooth_index.csv`` + ``gt_sets.json`` + ``assets_manifest.csv``。

    写之前先过 PHI 闸：任何字段出现非 ASCII 就**拒绝写出**（不是打警告）。
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    flat = [r.as_row() for r in rows]
    assert_phi_free(flat)

    index_path = out / "tooth_index.csv"
    with index_path.open("w", newline="", encoding="ascii") as f:
        w = csv.DictWriter(f, fieldnames=list(COLUMNS))
        w.writeheader()
        w.writerows(flat)

    sets = build_all_sets(rows)
    summary = {
        "n_rows": len(rows),
        "n_cases": len({r.case_uuid for r in rows}),
        "n_case_groups": len({case_group_key(r) for r in rows}),
        "set_counts": {name: len(members) for name, members in sets.items()},
        "set_case_counts": {
            name: len({case_group_key(m) for m in members})
            for name, members in sets.items()},
        "reconstruction_type_counts": _counts(r.reconstruction_type for r in rows),
        "sectioned_counts": _counts(
            r.arch_is_sectioned_or_unsectioned for r in rows),
        "fdi_counts": {str(k): v for k, v in sorted(
            _counts(r.fdi for r in rows).items())},
    }
    (out / "gt_sets.json").write_text(
        json.dumps(summary, ensure_ascii=True, indent=2), encoding="ascii")

    # 集合成员另存，方便直接拿来训练/评测（同样只含 case_uuid 与 FDI）
    with (out / "gt_set_members.csv").open("w", newline="", encoding="ascii") as f:
        w = csv.writer(f)
        w.writerow(["set_name", "case_uuid", "source_order_id", "fdi",
                    "reconstruction_type"])
        for name, members in sets.items():
            for m in members:
                w.writerow([name, m.case_uuid, m.source_order_id, m.fdi,
                            m.reconstruction_type])
    return summary


def _counts(iterable) -> dict:
    out: dict = {}
    for x in iterable:
        out[x] = out.get(x, 0) + 1
    return out


# ---------------------------------------------------------------- CLI

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="WelooAI GroundTruthIndexer")
    ap.add_argument("--root", action="append", required=True,
                    help="病例根目录，可重复（如 exo/EXO 与 exo/all_data）")
    ap.add_argument("--out", required=True, help="输出目录（不要写在只读数据目录里）")
    ap.add_argument("--limit", type=int, default=0, help="只处理前 N 个病例（调试）")
    ap.add_argument("--no-hash", action="store_true", help="跳过内容哈希（更快，但不可追溯）")
    ap.add_argument("--no-topology", action="store_true",
                    help="跳过水密/欧拉数检查（更快，但 final_crown_watertight 会是 None）")
    ap.add_argument("--max-arch-probes", type=int, default=3)
    ap.add_argument("--die-selection", action="append", default=None,
                    help="几何选代型结果 CSV（可重复给多个分片）。给了就用它作为"
                         "代型的几何证据（优于文件名证据）")
    a = ap.parse_args(argv)

    if a.die_selection:
        total = 0
        for path in a.die_selection:
            total += load_die_selection(path)
        print(f"  几何选代型证据载入 {total} 行", flush=True)

    print(f"扫描 {len(a.root)} 个根目录 ...", flush=True)
    cases = discover_cases(a.root)
    with_ci = {k: v for k, v in cases.items() if v.has_construction_info}
    print(f"  病例 id 总数 {len(cases)}，其中有 constructionInfo 的 {len(with_ci)}", flush=True)
    if a.limit:
        with_ci = dict(sorted(with_ci.items())[:a.limit])
        print(f"  --limit 生效，只处理 {len(with_ci)} 个", flush=True)

    rows = build_index(with_ci, max_arch_probes=a.max_arch_probes,
                       crown_topology=not a.no_topology,
                       do_hash=not a.no_hash,
                       progress=lambda m: print(m, flush=True))
    summary = write_outputs(rows, a.out)
    print(json.dumps(summary, ensure_ascii=True, indent=2))
    print(f"\n写出到 {a.out}/tooth_index.csv")
    return 0


if __name__ == "__main__":       # pragma: no cover
    sys.exit(main())
