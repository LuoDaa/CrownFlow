"""3Shape 订单包导入器的单元测试 —— **合成 fixture，可提交 Git**。

真实「胡昭」病例包含 PHI（患者姓名出现在订单号、Patient_LastName、目录名），
按与 exocad 数据一致的规则**不进 Git**；它的验收测试在
``tests/acceptance/three_shape/test_real_package.py``（本地 fixture，缺失即 skip）。

这里用人工构造的最小包，只证明 parser 行为。重点锁住本轮踩到的三个真实 bug：

  1. 包内 GUID 物理文件名**没有扩展名** → 类型判断必须用 AliasFilenames 的逻辑文件名
  2. 订单字段位于 **descendants** 而非直接 children → 只遍历 children 会全部找不到
  3. **MaterialID 不能被 ItemID 短路** → 修复体类型与材料引用是两回事

以及一条规则性断言：**companion mesh 绝不位置绑定**。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests._three_shape_fixture import (
    GUID,
    MATERIAL_ID,
    PATIENT,
    TYPE_ID,
    build_package,
    zip_package,
)

from app.importers.three_shape import import_package, validate_package
from app.importers.three_shape.geometry_source import BindingEvidence, GeometrySource
from app.importers.three_shape.models import ScanDecodeStatus, ScanRole

# ============================================================ bug 1: 扩展名

def test_type_detection_uses_logical_name_not_physical(tmp_path):
    """包内 GUID 物理文件名没有扩展名，类型判断必须用 AliasFilenames 的逻辑名。

    若在物理路径上取 ``suffix``，永远是空串 —— 订单 XML 判不出、
    扫描一个都筛不到（实测就是这个原因导致处方全空、scans=0）。
    """
    root = build_package(tmp_path / "pkg")

    # 前提：物理文件确实没有扩展名
    assert (root / GUID["order"]).suffix == ""
    assert (root / GUID["prep_dcm"]).suffix == ""

    pkg = import_package(root)
    assert pkg.diagnostics["order_xml"] is not None, "订单 XML 未被识别"
    assert len(pkg.scans) == 2, f"应筛出 2 个扫描，实际 {len(pkg.scans)}"


# ============================================================ bug 2: descendants

def test_order_fields_found_through_nested_lists(tmp_path):
    """订单字段位于孙节点（TDM_List_X / TDM_Item_X），必须用 descendants。

    只遍历直接 children 会把牙位/材料/扫描条目全部漏掉。
    """
    pkg = import_package(build_package(tmp_path / "pkg"))

    assert pkg.prescription.target_fdi == 46, "Universal 30 应转为 FDI 46"
    assert pkg.prescription.restoration_type == "CROWN"
    assert pkg.prescription.material_name == "Zirkon"
    assert pkg.prescription.material_id == MATERIAL_ID
    assert pkg.prescription.process_status == "psScanned"
    assert len(pkg.teeth) == 1


# ============================================================ bug 3: ItemID 短路

def test_material_id_not_short_circuited_by_item_id(tmp_path):
    """``TDM_Item_ToothElementType_Crown`` 的 ItemID 是修复体类型 ID、MaterialID 才是材料。

    实现若写 ``ItemID or MaterialID``，会短路到错误的那个，材料记录查不到。
    """
    pkg = import_package(build_package(tmp_path / "pkg"))

    assert pkg.material_record is not None, "材料记录未找到（疑似 ItemID 短路）"
    assert pkg.material_record["material_id"] == MATERIAL_ID
    assert pkg.material_record["name"] == "Crown Zircon"
    # 同时确认 ItemID 与 MaterialID 是**不同**的值 —— 否则这条测试是空的
    assert pkg.material_record["properties"]["ItemID"] == TYPE_ID
    assert pkg.material_record["properties"]["ItemID"] != MATERIAL_ID


# ==================================== 参数：材质 + 修复体子类型两级收窄

def test_cad_parameters_narrowed_by_material_and_subtype(tmp_path):
    """参数必须按 [材质 + 修复体子类型] 两级收窄。

    实测同一材质下有 Crown / Coping / Pontic / Inlay / Veneer / Tabletop 多套参数集，
    只按材质过滤会混用不同修复体类型的参数。
    """
    pkg = import_package(build_package(tmp_path / "pkg"))
    by_field = {p["source_field"]: p for p in pkg.cad_parameters}

    assert "CementSpace.CementGap" in by_field
    # 只应留下 Crown 那条的 0.04，而不是 Coping 的 0.09 或别的材质的 0.11
    assert by_field["CementSpace.CementGap"]["value"] == 0.04, (
        f"取到了错误的参数集：{[p['value'] for p in pkg.cad_parameters if 'CementGap' in p['source_field']]}")
    assert by_field["Overlay.WallThickness"]["value"] == 0.4


def test_conflicting_fields_are_not_mapped(tmp_path):
    """语义有冲突的字段必须 UNRESOLVED，不得按名字硬映射。

    实测冲突：CementGap（3Shape 拆成两个）、WallThickness（设计目标 vs exocad 最小值）、
    DistToPrepLine（与 exocad UntouchableZone 差一个量级）、RemoveUndercut（布尔 vs 模式+角度）。
    """
    pkg = import_package(build_package(tmp_path / "pkg"))
    state = {p["source_field"]: p["state"] for p in pkg.cad_parameters}

    for field in ("CementSpace.CementGap", "Overlay.WallThickness",
                  "CementSpace.DistToPrepLine", "CementSpace.RemoveUndercut"):
        assert state.get(field) == "UNRESOLVED", f"{field} 被错误地当作已确认映射"
    # DistToPrepLine 明确不映射到任何规范名（差一个量级）
    d2p = next(p for p in pkg.cad_parameters if p["source_field"] == "CementSpace.DistToPrepLine")
    assert d2p["canonical_name"] is None
    # 而语义清楚的邻接/咬合项可以是 PROPOSED
    assert state.get("CrownOverlay.Cut2AntagonistEnabled") in (None, "PROPOSED")


# ============================================ companion mesh：绝不位置绑定

def test_companion_bound_by_same_stem(tmp_path):
    """同目录同主名的 STL 应被绑定，且两份 provenance 都保留。"""
    root = build_package(tmp_path / "pkg", include_companions=True)
    pkg = import_package(root)

    prep = pkg.scan_of_type("preparation")
    assert prep is not None
    assert prep.companion is not None
    assert prep.companion.binding is BindingEvidence.SIBLING_SAME_STEM
    assert prep.geometry_ready is True
    assert prep.geometry_state == "GEOMETRY_READY"
    # 双 provenance：原件是 CE 加密，实际用 companion
    assert prep.original_geometry_source is GeometrySource.HPS_CE_ENCRYPTED
    assert prep.preferred_geometry_source is GeometrySource.COMPANION_STL
    assert pkg.geometry_available_for_design is True


def test_companion_not_bound_positionally(tmp_path):
    """**名字不符合约定时不得按顺序硬塞。**

    包里有两个 STL 但名字与原扫描无关 —— 必须拒绝绑定，而不是
    "第一个 STL = Prep、第二个 = Antagonist"。后者以后一定出事故。
    """
    root = build_package(tmp_path / "pkg", include_companions=True,
                         companion_names={"prep": f"{PATIENT}/mesh_a.stl",
                                          "ant": f"{PATIENT}/mesh_b.stl"})
    pkg = import_package(root)

    for scan in pkg.scans:
        assert scan.companion is not None
        assert scan.companion.binding is BindingEvidence.NONE, (
            "无名 STL 被位置绑定了")
        assert scan.geometry_ready is False
    assert pkg.geometry_available_for_design is False


# ================================================ 元数据就绪 != 几何就绪

def test_metadata_ready_but_geometry_not_ready(tmp_path):
    """元数据齐全、几何不可用时，闸门必须是 NOT_READY 且点名 scan_geometry。

    **不能因为 importer 能解析 metadata 就让病例进入设计管线。**
    """
    pkg = import_package(build_package(tmp_path / "pkg"))
    report = validate_package(pkg).as_dict()

    assert report["ready"] is False
    failed = {c["name"] for c in report["checks"] if c["result"] == "FAIL"}
    assert failed == {"scan_geometry"}, f"预期只卡在几何上，实际 {failed}"

    # 元数据侧应全部通过
    passed = {c["name"] for c in report["checks"] if c["result"] == "PASS"}
    assert {"target_fdi", "restoration_type", "preparation_scan",
            "antagonist_scan", "material_reference", "jaw_separation"} <= passed

    # 且补救建议必须明确、且不暗示"格式损坏"
    remedy = next(c["remedy"] for c in report["checks"] if c["name"] == "scan_geometry")
    assert "STL" in remedy
    detail = next(c["detail"] for c in report["checks"] if c["name"] == "scan_geometry")
    assert "加密" in detail and "损坏" not in detail and "不支持" not in detail


def test_gate_ready_when_companions_present(tmp_path):
    """补上 companion mesh 后，闸门应转为就绪。"""
    root = build_package(tmp_path / "pkg", include_companions=True)
    report = validate_package(import_package(root)).as_dict()
    assert report["ready"] is True, [c for c in report["checks"] if c["result"] == "FAIL"]


# ================================================ 三态判定与 PHI

def test_scan_classified_as_supported_but_encrypted(tmp_path):
    """CE schema + 无密钥 → SUPPORTED_BUT_ENCRYPTED（不是 UNSUPPORTED）。"""
    pkg = import_package(build_package(tmp_path / "pkg"))
    for scan in pkg.scans:
        assert scan.decode_status is ScanDecodeStatus.SUPPORTED_BUT_ENCRYPTED
        assert scan.plaintext_meta["schema"] == "CE"
        assert scan.plaintext_meta["vertex_count"] == 64
        assert scan.plaintext_meta["vertex_payload_entropy"] > 7.5, "未能识别为加密载荷"


def test_roles_and_jaws_from_path(tmp_path):
    """角色与颌位从映射路径判定，不猜 ScanType 的语义。"""
    pkg = import_package(build_package(tmp_path / "pkg"))
    prep = pkg.scan_of_type("preparation")
    ant = pkg.scan_of_type("antagonist")

    assert prep is not None and prep.role is ScanRole.PREPARATION
    assert ant is not None and ant.role is ScanRole.ANTAGONIST
    assert pkg.prescription.preparation_jaw == "lower"
    assert pkg.prescription.antagonist_jaw == "upper"


def test_phi_collected_for_deidentification(tmp_path):
    """PHI 字段必须被收集出来，供脱敏阶段使用。"""
    pkg = import_package(build_package(tmp_path / "pkg"))
    assert "Patient_LastName" in pkg.phi_fields
    assert pkg.phi_fields["Patient_LastName"] == PATIENT
    assert "OperatorName" in pkg.phi_fields


# ================================================ ZIP 形态与失败模式

def test_import_from_zip(tmp_path):
    """操作员上传的是 ZIP，必须直接支持。"""
    zpath = zip_package(build_package(tmp_path / "src"), tmp_path / "case.zip")

    pkg = import_package(zpath)
    assert pkg.prescription.target_fdi == 46
    assert len(pkg.scans) == 2


def test_missing_alias_file_fails_loudly(tmp_path):
    """没有 AliasFilenames.xml 时必须明确报错，不得靠文件名猜。"""
    root = tmp_path / "not_a_package"
    root.mkdir()
    (root / "random.dcm").write_bytes(b"x")
    with pytest.raises(ValueError, match="AliasFilenames"):
        import_package(root)


def test_stl_only_package(tmp_path):
    from tests._three_shape_fixture import _write_alias_file
    from app.importers.three_shape import import_package, validate_package
    root=build_package(tmp_path/'stl_only',include_companions=True)
    package=import_package(root)
    mapping={logical:guid for guid,logical in package.alias_map.items() if not logical.lower().endswith('.dcm')}
    _write_alias_file(root,mapping)
    pkg=import_package(root)
    assert len(pkg.scans)==2
    assert pkg.prescription.preparation_jaw=='lower'
    assert pkg.prescription.antagonist_jaw=='upper'
    assert all(s.mesh_info for s in pkg.scans)
