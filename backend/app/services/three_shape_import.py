"""3Shape 订单包导入的服务层。

分成两步，**顺序不能颠倒**：

    preview  —— 解析 + 跑验收闸门，**零副作用**。不建病例、不写网格、不落库。
    commit   —— 只有在闸门全过时，才把几何写进病例目录并建库。

这样操作员可以先看到"这个包被读成了什么、缺什么"，再决定要不要建病例。
也正因为如此，**"元数据能解析"与"可以进设计管线"是两件事** ——
闸门里任何一条 FAIL 都会让 commit 拒绝，而不是建一个进不了设计的空病例。

## 面板（panel）

``preview`` 返回一个已经排好序的 ``panel``，供前端直接渲染。
**通过与否的判定只有一处** —— 由 ``validate_package`` 决定，面板只是它的呈现层，
不复制一份判定逻辑，避免界面显示 ✅ 而闸门实际是 FAIL。

## 关于加密的措辞（不可改）

`测试_胡昭` 实测：扫描格式是受支持的 HPS，只是顶点载荷用 3Shape CE 加密，
当前没有合法密钥。面向操作员必须说**"已加密 / 无合法密钥"**，
**不能**说成"文件损坏 / 不支持格式" —— 后者会让操作员去重新导出，而问题不在那。
"""

from __future__ import annotations

import json
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path

from ..config import CASE_DIR
from ..db import db
from ..importers.three_shape.order import PHI_PROPERTY_NAMES
from .phi_policy import (PhiPolicy, PhiPolicyError, drop_phi_keys, file_sha256,
                         redact_phi_fields, resolve_policy, sanitize_logical_path,
                         sanitize_name, workspace_rules)
from ..geometry.io import load_mesh, mesh_info, save_mesh
from ..importers.three_shape import (
    import_package,
    mesh_physical_path,
    open_package_root,
    validate_package,
)
from ..importers.three_shape.models import (
    ScanAsset,
    ScanDecodeStatus,
    ThreeShapePackage,
)
from ..importers.three_shape.validator import CheckResult, ValidationReport

#: 记录文件名。留在病例目录里，作为"这个病例从哪来"的可追溯凭据。
RECORD_FILENAME = "three_shape_import.json"
SOURCE_PACKAGE_FILENAME = "source_package.zip"
#: 解析器版本（写进 provenance；解析行为变了必须改这里）。
IMPORTER_VERSION = "three_shape/1.0"


class ImportRejected(Exception):
    """元数据可解析，但不满足进入设计管线的最小集。"""

    def __init__(self, message: str, report: ValidationReport):
        super().__init__(message)
        self.report = report


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ------------------------------------------------------------------ 面板

def _check_map(report: ValidationReport) -> dict[str, object]:
    return {c.name: c for c in report.checks}


def _row(label: str, check, detail: str | None = None) -> dict:
    """一行面板。``check`` 为 None 时由调用方给 detail 与状态。"""
    return {
        "label": label,
        "status": check.result.value if check is not None else "INFO",
        "detail": detail if detail is not None else (check.detail if check else ""),
        "remedy": check.remedy if check is not None else None,
    }


def _scan_format_row(pkg: ThreeShapePackage) -> dict:
    """扫描**容器格式**是否受支持 —— 与几何是否可用是两件事，分两行。

    实测三态是 ``SUPPORTED_BUT_ENCRYPTED``：格式受支持，顶点加密。
    这一行必须是 PASS 并把加密说清楚，否则操作员会以为格式不被支持。
    """
    statuses = {s.decode_status for s in pkg.scans}
    if not statuses:
        return _row("扫描格式", None, "包内没有识别到扫描")
    if ScanDecodeStatus.UNSUPPORTED in statuses:
        return _row("扫描格式", None,
                    "存在不受支持的扫描容器（受支持：HPS 11.1 明文与 CE 压缩）")
    if statuses == {ScanDecodeStatus.DECODED}:
        detail = "HPS 11.1 明文（CC 压缩），顶点可直接解码"
    else:
        detail = ("HPS 11.1 / CE（CC 压缩 + 顶点载荷加密）—— 格式受支持；"
                  "顶点坐标使用加密载荷，当前无合法密钥。这不是文件损坏。")
    return _row("扫描格式", None, detail) | {"status": CheckResult.PASS.value}


def _scan_row(label: str, pkg: ThreeShapePackage, kind: str, jaw: str | None) -> dict:
    a = pkg.scan_of_type(kind)
    if a is None:
        return _row(label, None, "包内没有这个扫描")
    bits = [f"颌位 {jaw}" if jaw else "颌位未判定", a.resolved_path]
    if a.geometry_ready:
        src = getattr(a.preferred_geometry_source, "value", None)
        bits.append(f"几何来源 {src}")
    else:
        bits.append(f"几何 {a.geometry_state}")
    return _row(label, None, " · ".join(bits)) | {
        "status": (CheckResult.PASS.value if a.geometry_ready
                   else CheckResult.FAIL.value)}


def build_panel(pkg: ThreeShapePackage, report: ValidationReport) -> list[dict]:
    """把闸门结果排成操作员看得懂的面板。**判定来自 report，不在这里重算。**"""
    c = _check_map(report)
    p = pkg.prescription
    teeth = pkg.teeth[0] if pkg.teeth else None

    fdi_detail = None
    if p.target_fdi is not None:
        univ = teeth.tooth_number_universal if teeth else None
        fdi_detail = (f"FDI {p.target_fdi}" + (f"（Universal {univ}）" if univ else ""))

    material = pkg.material_record or {}
    mat_detail = None
    if p.material_id:
        # 两个名字都给：订单上的 CacheMaterialName 是技师下的单，
        # 材料库里的 Name 是实际引用的记录 —— 实测它们并不相同
        # （Zirkon vs 'Crown Zircon'），只显示一个会让人以为对不上。
        mat_detail = (f"{p.material_name or '?'} · {p.material_id}"
                      + (f" -> {material.get('name')!r}" if material.get("name") else ""))

    cad = pkg.cad_parameters or []
    unresolved = [x for x in cad if x.get("state") == "UNRESOLVED"]
    if cad:
        cad_detail = (f"找到 {len(cad)} 项"
                      + (f"，其中 {len(unresolved)} 项语义未确认（保留记录、不参与自动设计）"
                         if unresolved else ""))
    else:
        cad_detail = "未找到与修复体类型匹配的参数集"

    return [
        _row("目标牙位", c.get("target_fdi"), fdi_detail),
        _row("修复体", c.get("restoration_type"),
             f"{p.restoration_type}（{p.tooth_element_type_id}）"
             if p.restoration_type else None),
        _row("材料", c.get("material_reference"), mat_detail),
        _scan_row("预备体扫描", pkg, "preparation", p.preparation_jaw),
        _scan_row("对颌扫描", pkg, "antagonist", p.antagonist_jaw),
        _row("颌位分离", c.get("jaw_separation")),
        _scan_format_row(pkg),
        _row("扫描几何", c.get("scan_geometry")),
        _row("CAD 参数", c.get("cad_parameters"), cad_detail),
    ]


def _next_steps(pkg: ThreeShapePackage, report: ValidationReport) -> list[str]:
    steps: list[str] = []
    if not report.ready:
        for chk in report.failed:
            if chk.remedy:
                steps.append(chk.remedy)
    else:
        prep = pkg.scan_of_type("preparation")
        if prep is not None:
            src = getattr(prep.preferred_geometry_source, "value", None)
            steps.append(
                f"几何来自 {src}；在病例里对预备体颌做一次 ROI 取冠，生成 support 后再建设计。")
        steps.append("导入只铺几何与处方，不自动套参数 —— 参数仍需在设计中显式确认。")
    return steps


# ------------------------------------------------------------------ preview

def preview(path: Path | str) -> dict:
    """解析 + 验收，**零副作用**。"""
    pkg = import_package(path)
    report = validate_package(pkg)
    return {
        "source_name": pkg.source_name,
        "import_status": ("CASE_READY" if report.ready else "CASE_NOT_READY"),
        # 预填名称给前端 —— 不含 PHI，患者名要操作员自己决定用不用。
        "suggested_case_name": _default_case_name(pkg),
        "panel": build_panel(pkg, report),
        "validation": report.as_dict(),
        "prescription": pkg.prescription.as_dict(),
        "scans": [s.as_dict() for s in pkg.scans],
        "material_record": pkg.material_record,
        "cad_parameters": pkg.cad_parameters,
        # 与患者身份有关的字段单独一组，供前端决定是否用于命名 —— 不混进正常字段里。
        "phi_fields": pkg.phi_fields,
        "order_meta": pkg.order_meta,
        "diagnostics": pkg.diagnostics,
        "next_steps": _next_steps(pkg, report),
        "can_commit": report.ready,
    }


# ------------------------------------------------------------------ commit

#: 扫描落在哪个资产 kind，**只由颌位决定**（判不出就不写）。
def _asset_kind_for(jaw: str | None) -> str | None:
    return jaw if jaw in ("upper", "lower") else None


def _default_case_name(pkg: ThreeShapePackage) -> str:
    """默认病例名**不含 PHI**。患者名只在操作员显式选择时才用。"""
    p = pkg.prescription
    parts = ["3Shape", p.material_name or "?", p.restoration_type or "?"]
    if p.target_fdi is not None:
        parts.append(f"FDI{p.target_fdi}")
    return " ".join(parts)


def commit(path: Path | str, *, case_name: str | None = None,
           policy: PhiPolicy | str | None = None,
           keep_source_package: bool | None = None) -> dict:
    """把包落成病例：建病例 + 写几何资产 + 记导入凭据。

    **闸门不过就拒绝建病例。** 不给"先建着、以后再补几何"的口子 ——
    那种病例在库里和真正可设计的病例长得一模一样，迟早有人拿它去跑设计。

    **PHI 策略（G0）**：默认 ``PSEUDONYMIZED`` —— 工作区里不放原始包、PHI 字段只留
    "存在性 + 分类"、名字逐段脱敏、技术 provenance（来源 sha256 / 解析器版本）保留。
    要保留原始包必须显式 ``policy=IDENTIFIED``；在 PSEUDONYMIZED 下显式要求保留会被
    **拒绝**并提示改档（不静默照做）。
    """
    resolved_policy = resolve_policy(policy)
    rules = workspace_rules(resolved_policy)
    if keep_source_package is True and not rules.copy_source_package:
        raise PhiPolicyError(
            "策略 %s 下不允许把原始订单包留在工作区（包里含订单 XML 原文）。"
            "如确需保留原始包复核来源，请显式以 IDENTIFIED 导入。" % resolved_policy.value)
    copy_source = rules.copy_source_package if keep_source_package is None else bool(keep_source_package)
    if copy_source and not rules.copy_source_package:
        raise PhiPolicyError("策略 %s 不允许保留原始包" % resolved_policy.value)

    source = Path(path)
    is_zip = source.suffix.lower() == ".zip"
    source_sha = file_sha256(source) if source.is_file() else None

    # 用 open_package_root 而不是 import_package：ZIP 的解包目录必须在
    # 拷完网格之后才允许消失，否则 ScanAsset.absolute_path 全是死路径。
    with open_package_root(source) as root:
        pkg = import_package(root)          # root 已是目录，不会再开临时目录
        report = validate_package(pkg)
        if not report.ready:
            raise ImportRejected(
                "导入未通过验收闸门：" + "；".join(
                    f"{c.name}: {c.detail}" for c in report.failed),
                report)

        # 几何要落到哪个资产 kind，由颌位决定 —— 判不出颌位就不写，
        # 不能默认塞进 lower（那会让对颌图出现在下颌上，比缺文件更糟）。
        placements: list[tuple[str, ScanAsset]] = []
        for kind, jaw in (("preparation", pkg.prescription.preparation_jaw),
                          ("antagonist", pkg.prescription.antagonist_jaw)):
            asset = pkg.scan_of_type(kind)
            if asset is None:
                continue
            target_kind = _asset_kind_for(jaw)
            if target_kind is None:
                raise ImportRejected(
                    f"{kind} 扫描的颌位无法判定（{asset.resolved_path}），"
                    "无法确定它属于上颌还是下颌；请确认包内 Scans/<Jaw>/ 目录结构。",
                    report)
            placements.append((target_kind, asset))

        kinds = [k for k, _ in placements]
        if len(set(kinds)) != len(kinds):
            raise ImportRejected(
                f"预备体与对颌被判为同一颌（{kinds}），无法同时作为上下颌扫描。", report)

        case_id = uuid.uuid4().hex
        case_dir = CASE_DIR / case_id
        # 名字立刻脱敏：默认病例名本来就不含 PHI；操作员显式给的名字按策略处理，
        # **原值不落库、不落文件、不进日志**（redactions 只在本进程内用于替换）。
        raw_name = (case_name or "").strip() or _default_case_name(pkg)
        redactions = [v for v in (pkg.phi_fields or {}).values() if isinstance(v, str) and v.strip()]
        redactions.append(source.name)
        name = sanitize_name(raw_name, policy=resolved_policy, redactions=redactions, fallback="case")
        case_dir.mkdir(parents=True, exist_ok=True)
        written: list[dict] = []
        try:
            # 病例行必须先插：assets.case_id 有外键约束，
            # 反过来写会在插入第一个资产时就 IntegrityError。
            with db() as c:
                c.execute("INSERT INTO cases(id,name,created_at) VALUES(?,?,?)",
                          (case_id, name, _now()))

            for target_kind, asset in placements:
                logical = asset.geometry_source_path
                physical = mesh_physical_path(pkg, root, logical)
                if physical is None:
                    raise ImportRejected(
                        f"几何文件在包内找不到：{logical}", report)
                # 走 load_mesh/save_mesh 而不是裸拷贝：与 /assets 上传同一条规范化路径，
                # 并且能在写库前就发现坏网格。
                #
                # **必须显式传 file_type**：物理文件名是 GUID、没有扩展名，
                # 靠扩展名分派会直接 NotImplementedError。格式取自逻辑名。
                logical_suffix = Path(logical).suffix.lower().lstrip(".")
                stored_logical = sanitize_logical_path(logical, policy=resolved_policy,
                                                       redactions=redactions)
                mesh = load_mesh(physical, file_type=logical_suffix or None)
                dest = case_dir / f"{target_kind}_{uuid.uuid4().hex}.stl"
                save_mesh(mesh, dest)
                info = mesh_info(mesh)
                with db() as c:
                    c.execute(
                        """INSERT INTO assets(id,case_id,kind,path,original_name,
                                              mesh_info_json,created_at)
                           VALUES(?,?,?,?,?,?,?)""",
                        (uuid.uuid4().hex, case_id, target_kind, str(dest),
                         stored_logical, json.dumps(info), _now()))
                written.append({
                    "kind": target_kind,
                    "source": getattr(asset.preferred_geometry_source, "value", None),
                    "original_geometry_source": getattr(
                        asset.original_geometry_source, "value", None),
                    # 逻辑路径按策略脱敏后落库/落报告：包内目录名可能就是患者名。
                    "logical_path": stored_logical,
                    "source_sha256": file_sha256(Path(physical)),
                    "mesh": info,
                })

            record = {
                "imported_at": _now(),
                "importer": "three_shape",
                "importer_version": IMPORTER_VERSION,
                # 来源可追（技术 provenance）：字节哈希 + 脱敏后的来源名。
                "source_sha256": source_sha,
                "source_name": sanitize_name(pkg.source_name, policy=resolved_policy,
                                             redactions=redactions, fallback="source"),
                "case_name": name,
                "prescription": pkg.prescription.as_dict(),
                "validation": report.as_dict(),
                "panel": build_panel(pkg, report),
                "scans": [s.as_dict() for s in pkg.scans],
                "material_record": pkg.material_record,
                "cad_parameters": pkg.cad_parameters,
                # PHI 只留"存在性 + 分类"（+ 可选 HMAC 摘要）；原值只在 IDENTIFIED 档出现。
                "phi_report": redact_phi_fields(pkg.phi_fields, policy=resolved_policy),
                "phi_policy": resolved_policy.value,
                "phi_rules": rules.as_report(),
                # order_meta 是整份订单属性，PHI 混在里面 —— 按字段名白名单剔除。
                "order_meta": drop_phi_keys(pkg.order_meta, phi_names=PHI_PROPERTY_NAMES),
                "diagnostics": pkg.diagnostics,
                "assets": written,
            }
            order_source = {"prescription": record["prescription"],
                            "source_name": record["source_name"], "importer": "three_shape",
                            "importer_version": IMPORTER_VERSION,
                            "phi_policy": resolved_policy.value,
                            "source_sha256": source_sha,
                            "imported_at": record["imported_at"], "configuration_status": "UNCONFIRMED",
                            "material_policy": "SOURCE_LABEL_ONLY_NOT_VALIDATED_PROCESS"}
            with db() as c:
                c.execute("UPDATE cases SET order_json=? WHERE id=?",
                          (json.dumps(order_source, ensure_ascii=False), case_id))
            (case_dir / RECORD_FILENAME).write_text(
                json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
            if is_zip and copy_source:
                shutil.copyfile(source, case_dir / SOURCE_PACKAGE_FILENAME)
        except Exception:
            # 失败不留半成品病例 —— 库里出现一个没有几何的病例最危险，
            # 它和真正可设计的病例长得一模一样，迟早有人拿它去跑设计。
            # assets 有 ON DELETE CASCADE，删病例行即可连带清掉。
            with db() as c:
                c.execute("DELETE FROM cases WHERE id=?", (case_id,))
            shutil.rmtree(case_dir, ignore_errors=True)
            raise

    return {
        "case_id": case_id,
        "case_name": name,
        "phi_policy": resolved_policy.value,
        "source_package_retained": bool(is_zip and copy_source),
        "assets": written,
        "prescription": pkg.prescription.as_dict(),
        "panel": build_panel(pkg, report),
        "validation": report.as_dict(),
        # 明确告诉前端：导入**没有**自动延伸出设计。
        "design_created": False,
        "next_steps": [
            "病例已建立。support 仍需对预备体颌做 ROI 取冠后生成。",
            "参数不随导入自动套用 —— 建设计时需显式确认。",
        ],
        "record_path": str(case_dir / RECORD_FILENAME),
    }


def repreview(case_id: str) -> dict:
    """重读已落库病例的导入凭据（不回读源包）。"""
    path = CASE_DIR / case_id / RECORD_FILENAME
    if not path.is_file():
        return {"has_record": False, "case_id": case_id}
    return {"has_record": True, "case_id": case_id,
            "record": json.loads(path.read_text(encoding="utf-8"))}
