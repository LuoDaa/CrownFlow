"""Import Validation Gate —— 只有全部通过才允许进入自动设计。

对应流程里的"导入完成后不直接一键出 Final Crown"。每一条检查都给出
**可执行的失败原因**，而不是一个笼统的"导入失败"。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from .models import ScanDecodeStatus, ThreeShapePackage
from .materials import MappingStatus

#: 支持自动设计的修复体类型（本轮只确证了单冠）。
SUPPORTED_RESTORATIONS: frozenset[str] = frozenset({"CROWN"})


class CheckResult(str, Enum):
    PASS = "PASS"
    WARN = "WARN"
    FAIL = "FAIL"


@dataclass
class Check:
    name: str
    result: CheckResult
    detail: str
    #: 失败的修复建议（给操作员看的下一步动作）。
    remedy: str | None = None

    def as_dict(self) -> dict:
        return {"name": self.name, "result": self.result.value,
                "detail": self.detail, "remedy": self.remedy}


@dataclass
class ValidationReport:
    checks: list[Check] = field(default_factory=list)

    @property
    def failed(self) -> list[Check]:
        return [c for c in self.checks if c.result is CheckResult.FAIL]

    @property
    def ready(self) -> bool:
        return not self.failed

    def as_dict(self) -> dict:
        return {
            "ready": self.ready,
            "n_pass": sum(1 for c in self.checks if c.result is CheckResult.PASS),
            "n_warn": sum(1 for c in self.checks if c.result is CheckResult.WARN),
            "n_fail": len(self.failed),
            "checks": [c.as_dict() for c in self.checks],
            "remedies": [c.remedy for c in self.failed if c.remedy],
        }


def validate_package(pkg: ThreeShapePackage) -> ValidationReport:
    """跑完全部检查。任一项 FAIL → 不允许自动设计。"""
    r = ValidationReport()

    def add(name, result, detail, remedy=None):
        r.checks.append(Check(name, result, detail, remedy))

    # 1. 目标牙位
    fdi = pkg.prescription.target_fdi
    if fdi is None:
        univ = next((t.tooth_number_universal for t in pkg.teeth), None)
        add("target_fdi", CheckResult.FAIL,
            f"订单未给出可识别的牙位（ToothNumber={univ}）"
            if univ is not None else "订单里没有 ToothElement",
            "在 3Shape 订单里确认牙位后重新导出")
    else:
        add("target_fdi", CheckResult.PASS, f"FDI {fdi}")

    # 2. 修复体类型
    rt = pkg.prescription.restoration_type
    if rt is None:
        add("restoration_type", CheckResult.FAIL,
            f"无法从 toothElementTypeID={pkg.prescription.tooth_element_type_id!r} 确证修复体类型",
            "确认订单的修复体类型；当前只支持已确证的 CROWN")
    elif rt not in SUPPORTED_RESTORATIONS:
        add("restoration_type", CheckResult.FAIL,
            f"{rt} 暂不支持自动设计（已确证：{sorted(SUPPORTED_RESTORATIONS)}）")
    else:
        add("restoration_type", CheckResult.PASS, rt)

    # 3/4. 两种扫描都要在
    for kind, label in (("preparation", "预备体"), ("antagonist", "对颌")):
        asset = pkg.scan_of_type(kind)
        if asset is None:
            add(f"{kind}_scan", CheckResult.FAIL,
                f"包内没有{label}扫描", "确认 3Shape 订单包含该扫描后重新导出")
        else:
            add(f"{kind}_scan", CheckResult.PASS, asset.resolved_path)

    # 5. 扫描几何可用性 —— **元数据就绪不等于几何就绪**
    #
    # 判定必须按角色看"实际可用的几何"，而不是只看 HPS 原件的三态：
    # 原件是 CE 加密但同目录有 STL companion 时，几何是可用的。
    by_role = {}
    for kind in ("preparation", "antagonist"):
        a = pkg.scan_of_type(kind)
        if a is not None:
            by_role[kind] = a

    missing_geometry = [k for k, a in by_role.items() if not a.geometry_ready]
    encrypted = [a for a in by_role.values()
                 if a.decode_status is ScanDecodeStatus.SUPPORTED_BUT_ENCRYPTED]

    if not by_role:
        add("scan_geometry", CheckResult.FAIL, "没有可识别的预备体/对颌扫描")
    elif not missing_geometry:
        sources = {k: getattr(a.preferred_geometry_source, "value", None)
                   for k, a in by_role.items()}
        add("scan_geometry", CheckResult.PASS, f"几何可用：{sources}")
    else:
        detail_bits = []
        for k in missing_geometry:
            a = by_role[k]
            counts = (f"{a.plaintext_meta.get('vertex_count')} 顶点 / "
                      f"{a.plaintext_meta.get('facet_count')} 面")
            # 面向用户的措辞：必须说"加密"，**不能**说成"文件损坏/不支持格式" ——
            # 已实测证明格式是受支持的（CE ∈ hpsdecode.SUPPORTED_SCHEMAS），
            # 缺的只是合法密钥。这两者混为一谈会误导操作员去重新导出。
            if a.decode_status is ScanDecodeStatus.SUPPORTED_BUT_ENCRYPTED:
                why = "扫描格式受支持，但顶点坐标使用 3Shape CE 加密载荷，当前无合法密钥"
            elif a.decode_status is ScanDecodeStatus.UNSUPPORTED:
                why = "扫描格式不受支持"
            else:
                why = "扫描可解码但未被采用"
            detail_bits.append(f"{k}（{a.resolved_path}）：{why}；明文可见 {counts}")
        remedy = None
        if encrypted:
            # 纯文本，会原样出现在界面与报告里 —— 不写 markdown 的 ** 强调，
            # 那些星号在网页上是照原样显示的。
            remedy = ("在 3Shape 里把缺几何的扫描导出为 STL，与原扫描同目录同主名放进包里 "
                      "（如 Scans/Lower/PreparationScan.stl），或提供合法的 HPS_ENCRYPTION_KEY。"
                      "注意：不通过提取/绕过厂商密钥来获取几何。")
        add("scan_geometry", CheckResult.FAIL,
            "以下扫描缺可用几何：" + "；".join(detail_bits), remedy)

    # 6. 材料 profile 引用有效（订单里的 MaterialID 必须在材料库里存在）
    if pkg.prescription.material_id and pkg.material_record is None:
        add("material_reference", CheckResult.FAIL,
            f"MaterialID={pkg.prescription.material_id} 在 Materials.xml 里找不到",
            "确认包内包含订单引用材料所在的材料库")
    elif pkg.prescription.material_id:
        add("material_reference", CheckResult.PASS,
            f"{pkg.prescription.material_id} -> {pkg.material_record.get('name')!r}")
    else:
        add("material_reference", CheckResult.WARN, "订单未给出 MaterialID")

    # 7. CAD 参数：UNRESOLVED 的不得进入自动设计
    unresolved = [p for p in pkg.cad_parameters
                  if p.get("state") == MappingStatus.UNRESOLVED.value]
    if unresolved:
        names = ", ".join(sorted({f"{p['source_field']}" for p in unresolved}))
        add("cad_parameters", CheckResult.WARN,
            f"{len(unresolved)} 个 3Shape 参数语义未确认（{names}）："
            "保留记录但不参与自动设计，需厂商文档或技师确认",
            None)
    else:
        add("cad_parameters", CheckResult.PASS, "无可疑参数")

    # 8. 扫描颌位必须分离（预备体与对颌不能在同一颌）
    prep = pkg.scan_of_type("preparation")
    ant = pkg.scan_of_type("antagonist")
    if prep and ant and pkg.prescription.preparation_jaw and pkg.prescription.antagonist_jaw:
        if pkg.prescription.preparation_jaw == pkg.prescription.antagonist_jaw:
            add("jaw_separation", CheckResult.FAIL,
                f"预备体与对颌都被判为 {pkg.prescription.preparation_jaw} 颌",
                "检查包内扫描目录结构")
        else:
            add("jaw_separation", CheckResult.PASS,
                f"prep={pkg.prescription.preparation_jaw}, "
                f"antagonist={pkg.prescription.antagonist_jaw}")
    else:
        add("jaw_separation", CheckResult.WARN, "颌位未从路径完整判定")

    return r
