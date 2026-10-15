"""3Shape 订单包的**合成** fixture 构造器（可提交 Git，不含任何 PHI）。

被两处复用：
  * ``tests/unit/importers/test_three_shape_package.py``  —— 解析器行为
  * ``tests/unit/api/test_three_shape_import_api.py``     —— HTTP 层

真实「胡昭」包含 PHI，不进 Git；它的验收测试用
``CROWNFLOW_THREESHAPE_FIXTURE_DIR`` 在本地受控数据区查找（缺失即 skip）。
"""

from __future__ import annotations

import zipfile
from pathlib import Path

import numpy as np
import trimesh

# ------------------------------------------------------------------ fixture 构造

_ALIAS_TEMPLATE = '''<?xml version="1.0" encoding="utf-8"?>
<AliasFilenameSettings version="1.0">
  <Object name="MainObject" type="TAliasFilenameDefinition">
    <Object name="FilenameMappings" type="TAliasFilenameList">
      <List name="Items">
{mappings}
      </List>
    </Object>
  </Object>
</AliasFilenameSettings>
'''

_ITEM = '''        <Object type="TAliasFilenameItem">
          <Property name="OriginalName" value="{original}"/>
          <Property name="AliasName" value="{alias}"/>
        </Object>
'''


def _write_alias_file(root: Path, mapping: dict[str, str]) -> None:
    """mapping: {逻辑路径: guid}"""
    body = "\n".join(_ITEM.format(original=orig, alias=g) for orig, g in mapping.items())
    (root / "AliasFilenames.xml").write_text(
        _ALIAS_TEMPLATE.format(mappings=body), encoding="utf-8")


def _order_xml(patient: str, *, tooth_universal: int, material_id: str,
               type_id: str, cache_material: str, extra_properties=(),
               patient_name: str | None = None) -> str:
    """构造订单 XML。

    **关键：条目对象挂在 TDM_List_X / TDM_Item_X 两层之下** —— 复现实测层级，
    这样"只遍历 children"的实现会失败（bug 2）。

    ``patient_name`` / ``extra_properties`` 是 G0 加的**PHI 注入点**：
    默认不注入（既有测试完全不受影响），只在 PHI 负向测试里填可用作搜索标记的假名。
    """
    address = (extra_properties or {}).get("DeliveryAddress1", "")
    return f'''<?xml version="1.0" encoding="utf-8"?>
<DentalContainer version="2025-1">
  <Object name="Container" type="TDM_Container">
    <Object name="OrderList" type="TDM_List_Order">
      <List name="Items">
        <Object type="TDM_Item_Order">
          <Property name="IntOrderID" value="{patient}"/>
          <Property name="Patient_LastName" value="{patient_name or patient}"/>
          <Property name="OperatorName" value="Tech01"/>
          <Property name="DeliveryAddress1" value="{address}"/>
          <Property name="CacheMaterialName" value="{cache_material}"/>
        </Object>
      </List>
    </Object>
    <Object name="ModelElementList" type="TDM_List_ModelElement">
      <List name="Items">
        <Object type="TDM_Item_ModelElement">
          <Property name="ModelElementID" value="ME1"/>
          <Property name="MaterialID" value="{material_id}"/>
          <Property name="ProcessStatusID" value="psScanned"/>
          <Property name="CacheMaterialName" value="{cache_material}"/>
        </Object>
      </List>
    </Object>
    <Object name="ToothElementList" type="TDM_List_ToothElement">
      <List name="Items">
        <Object type="TDM_Item_ToothElement">
          <Property name="ToothElementID" value="TE1"/>
          <Property name="toothElementTypeID" value="{type_id}"/>
          <Property name="ToothNumber" value="{tooth_universal}"/>
          <Property name="CacheToothTypeClass" value="teCrown"/>
        </Object>
      </List>
    </Object>
    <Object name="ScanList" type="TDM_List_Scan">
      <List name="Items">
        <Object type="TDM_Item_Scan">
          <Property name="ScanID" value="S1"/>
          <Property name="ScanType" value="stPreperation"/>
        </Object>
        <Object type="TDM_Item_Scan">
          <Property name="ScanID" value="S2"/>
          <Property name="ScanType" value="stAntagonistModel"/>
        </Object>
      </List>
    </Object>
  </Object>
</DentalContainer>
'''


def _materials_xml(material_id: str, type_id: str) -> str:
    """构造材料库。

    **关键：TDM_Item_ToothElementType_Crown 的 ItemID 装的是修复体类型 ID、
    MaterialID 才是材料 ID** —— 复现实测结构，锁住 bug 3。
    """
    return f'''<?xml version="1.0" encoding="utf-8"?>
<DentalContainer version="2025-1">
  <Object name="Container" type="TDM_Container">
    <Object name="ToothElementTypeList" type="TDM_List_ToothElementType">
      <List name="Items">
        <Object type="TDM_Item_ToothElementType_Crown">
          <Property name="ItemID" value="{type_id}"/>
          <Property name="MaterialID" value="{material_id}"/>
          <Property name="Name" value="Crown Zircon"/>
        </Object>
        <Object type="TDM_Item_ToothElementType_Crown">
          <Property name="ItemID" value="Configuration_11_IDCrownTi99"/>
          <Property name="MaterialID" value="Configuration_11_Material9"/>
          <Property name="Name" value="Crown Titanium"/>
        </Object>
      </List>
    </Object>
    <Object name="CementSpaceList" type="TDM_List_CementSpace">
      <List name="Items">
        <Object type="TDM_Item_CementSpace">
          <Property name="MaterialID" value="{material_id}"/>
          <Property name="Name" value="Zircon Crown"/>
          <Property name="RemoveUndercut" value="True"/>
          <Property name="CementGap" value="0.04"/>
          <Property name="ExtraCementGap" value="0.07"/>
          <Property name="DistToPrepLine" value="0.8"/>
        </Object>
        <Object type="TDM_Item_CementSpace">
          <Property name="MaterialID" value="{material_id}"/>
          <Property name="Name" value="Zircon Coping (tangent)"/>
          <Property name="CementGap" value="0.09"/>
        </Object>
        <Object type="TDM_Item_CementSpace">
          <Property name="MaterialID" value="Configuration_11_Material9"/>
          <Property name="Name" value="Crown Titanium"/>
          <Property name="CementGap" value="0.11"/>
        </Object>
      </List>
    </Object>
    <Object name="OverlayList" type="TDM_List_Overlay">
      <List name="Items">
        <Object type="TDM_Item_Overlay">
          <Property name="MaterialID" value="{material_id}"/>
          <Property name="Name" value="Zircon Crown"/>
          <Property name="WallThickness" value="0.4"/>
        </Object>
      </List>
    </Object>
  </Object>
</DentalContainer>
'''


def _hps_ce_bytes(*, vertex_count: int = 64, facet_count: int = 120) -> bytes:
    """构造一个 CE schema 的 HPS 外壳：顶点载荷高熵（模拟加密）。

    高熵是刻意的 —— ``hps_scan.plaintext_metadata`` 用熵判断是否加密，
    用零填充或可预测内容会让"加密"判定失效，测试就白写了。
    """
    rng = np.random.default_rng(7)
    verts = rng.integers(0, 256, size=vertex_count * 12, dtype=np.uint8)
    import base64
    v64 = base64.b64encode(verts.tobytes()).decode()
    f64 = base64.b64encode(rng.integers(0, 256, size=256, dtype=np.uint8).tobytes()).decode()
    return f'''<?xml version="1.0" encoding="utf-8"?>
<HPS version="1.1">
  <Packed_geometry>
    <Schema>CE</Schema>
    <Binary_data>
      <CE version="1.0">
        <Facets facet_count="{facet_count}" base64_encoded_bytes="256" color="8421504">{f64}</Facets>
        <Vertices vertex_count="{vertex_count}" base64_encoded_bytes="{vertex_count * 12}" check_value="1">{v64}</Vertices>
      </CE>
    </Binary_data>
  </Packed_geometry>
  <Properties>
    <Property name="EKID" value="1"/>
    <Property name="ScanSource" value="ssImportThirdPartySTL"/>
  </Properties>
</HPS>
'''.encode("utf-8")


def _write_stl(path: Path, *, translate=(0.0, 0.0, 0.0)) -> None:
    """写 STL 到指定路径。

    必须显式给 ``file_type``：包内物理文件名是 GUID、**没有扩展名**，
    trimesh 靠扩展名选导出器，``m.export(path)`` 会报
    "exporter not available"。（与 bug 1 同源：GUID 名不带扩展名。）
    """
    m = trimesh.creation.box(extents=[4.0, 4.0, 4.0])
    m.apply_translation(translate)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(m.export(file_type="stl"))


GUID = {
    "alias": "{00000000-0000-0000-0000-0000000000A1}",
    "materials": "{00000000-0000-0000-0000-0000000000A2}",
    "order": "{00000000-0000-0000-0000-0000000000A3}",
    "prep_dcm": "{00000000-0000-0000-0000-0000000000A4}",
    "ant_dcm": "{00000000-0000-0000-0000-0000000000A5}",
    "prep_stl": "{00000000-0000-0000-0000-0000000000A6}",
    "ant_stl": "{00000000-0000-0000-0000-0000000000A7}",
}
PATIENT = "TestPatient"
MATERIAL_ID = "Configuration_11_Material1"
TYPE_ID = "Configuration_11_IDCrownZircon10"


def build_package(root: Path, *, include_companions: bool = False,
                  companion_names: dict[str, str] | None = None,
                  patient_name: str | None = None,
                  extra_properties: dict[str, str] | None = None) -> Path:
    """构造一个最小但结构真实（含那两个坑）的 3Shape 包。

    ``companion_names`` 可覆盖 companion 的逻辑路径，用于测试**不位置绑定**：
    传入不符合约定的名字时，导入器必须拒绝绑定而不是按顺序硬塞。
    """
    root.mkdir(parents=True, exist_ok=True)
    logical = {
        f"{PATIENT}/{PATIENT}.xml": GUID["order"],
        f"{PATIENT}/Materials.xml": GUID["materials"],
        f"{PATIENT}/Scans/Lower/PreparationScan.dcm": GUID["prep_dcm"],
        f"{PATIENT}/Scans/Upper/AntagonistScan.dcm": GUID["ant_dcm"],
    }
    if include_companions:
        names = companion_names or {}
        logical[names.get("prep", f"{PATIENT}/Scans/Lower/PreparationScan.stl")] = GUID["prep_stl"]
        logical[names.get("ant", f"{PATIENT}/Scans/Upper/AntagonistScan.stl")] = GUID["ant_stl"]

    _write_alias_file(root, logical)

    # 物理文件名 = GUID 本身，**没有扩展名**（复现实测，锁 bug 1）
    (root / GUID["order"]).write_text(
        _order_xml(PATIENT, tooth_universal=30, material_id=MATERIAL_ID,
                   type_id=TYPE_ID, cache_material="Zirkon",
                   patient_name=patient_name,
                   extra_properties=extra_properties or {}), encoding="utf-8")
    (root / GUID["materials"]).write_text(
        _materials_xml(MATERIAL_ID, TYPE_ID), encoding="utf-8")
    (root / GUID["prep_dcm"]).write_bytes(_hps_ce_bytes())
    (root / GUID["ant_dcm"]).write_bytes(_hps_ce_bytes())

    if include_companions:
        # 物理文件名是 GUID（无扩展名）；逻辑路径由上面的 alias 映射给出。
        _write_stl(root / GUID["prep_stl"])
        _write_stl(root / GUID["ant_stl"], translate=(0.0, 20.0, 0.0))
    return root


def zip_package(root: Path, dest: Path) -> Path:
    """把构造好的包打成 ZIP —— **这才是操作员实际上传的形态**。

    打出来的 ZIP 里，物理文件名同样是 GUID 且没有扩展名，
    与真实包一致（见 bug 1）。
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(dest, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(root.rglob("*")):
            if p.is_file():
                z.write(p, p.relative_to(root).as_posix())
    return dest
