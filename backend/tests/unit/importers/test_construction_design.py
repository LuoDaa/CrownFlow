"""``.constructionInfo`` 设计几何解析的单元测试 —— **合成 fixture，可提交 Git**。

真实病例含 PHI（文件名里有患者姓名、``PatientId``/``ProjectDir``），不进 Git；
真实数据上的验收在 ``tests/acceptance/exocad_real/``（走环境变量，缺失即 skip）。

这里只证明 parser 行为。fixture 刻意复现真实 schema 里的几个坑：

  1. 根标签是 ``<ConstuctionInfo>`` —— exocad 自己拼错了（少了 r），不能"纠正"它
  2. ``<StepSize xsi:nil="true" />`` 自闭合标签：正则实现会在此错位配对
  3. ``<MinValue>NaN</MinValue>``：直接 ``float()`` 会炸
  4. ``<Axis>`` 的子标签名与 ``<Margin><Vec3>`` 不同，必须两种都能读
  5. ``EstimatedDirectionForCAM`` 是**加工方向**，与 ``<Axis>``（设计就位方向）
     是两回事，必须分开存

fixture 里的坐标是**编造的**，与任何真实病例无关。
"""

from __future__ import annotations

import xml.etree.ElementTree as ET

import pytest

from app.importers.exocad.construction_info import (
    parse_construction_case,
    parse_construction_info,
)

#: 编造的三维点，只用于验证解析通路。
_MARGIN_XYZ = [(17.83, 10.04, -12.49), (17.84, 10.09, -12.48), (17.85, 10.16, -12.46),
               (17.90, 10.30, -12.40), (18.00, 10.50, -12.30)]


def _margin_xml(points) -> str:
    """拼 ``<Vec3>`` 序列。元素是 (x,y,z) 三元组，或一段原始 XML
    （用来刻意造出"缺分量"的坏点）。"""
    parts = []
    for item in points:
        if isinstance(item, str):
            parts.append(item)
        else:
            x, y, z = item
            parts.append(f"<Vec3><x>{x}</x><y>{y}</y><z>{z}</z></Vec3>")
    return "".join(parts)


def _mat16(prefix: str = "") -> str:
    return "".join(
        f"<{prefix}_{r}{c}>{r * 4 + c}</{prefix}_{r}{c}>"
        for r in range(4) for c in range(4))


def build_construction_info(*, bad_margin: bool = False,
                            with_geometry: bool = True,
                            with_scan_files: bool = True,
                            with_exports: bool = True) -> str:
    """构造一份结构与真实文件同构的 ``.constructionInfo``。"""
    terms = list(_MARGIN_XYZ)
    if bad_margin:
        # 故意少一个 z —— 必须被计数为坏点，而不是静默当成 0
        terms.append("<Vec3><x>1.0</x><y>2.0</y></Vec3>")

    geometry = ""
    if with_geometry:
        geometry = f"""
          <Margin>{_margin_xml(terms)}</Margin>
          <MarginBottomType>Normal</MarginBottomType>
          <Axis><_a>2.8e-10</_a><_b>2.2e-09</_b><_c>0.99999999</_c></Axis>
          <AxisOcclusalOfToothModel>
            <_a>-0.035</_a><_b>0.231</_b><_c>0.972</_c>
          </AxisOcclusalOfToothModel>
          <AxisMesial><_a>0.979</_a><_b>0.202</_b><_c>-0.013</_c></AxisMesial>
          <AxisBuccal><_a>-0.200</_a><_b>0.952</_b><_c>-0.234</_c></AxisBuccal>
          <Center><x>13.90</x><y>10.82</y><z>-12.35</z></Center>
          <ToothLibraryName>002</ToothLibraryName>
          <ToothLibraryToothNumber>17</ToothLibraryToothNumber>
          <ToothModelMatrix>{_mat16()}</ToothModelMatrix>
          <ToothScanFileName>case-UpperJaw.ply</ToothScanFileName>
"""

    scans = ""
    if with_scan_files:
        scans = f"""
      <ScanFiles>
        <ScanFile>
          <FileName>case-UpperJaw.ply</FileName>
          <FileHash>abc</FileHash>
          <PartType>UnsegmentedJaw</PartType>
          <PartName>UpperJaw</PartName>
          <ToothNumbers><int>17</int></ToothNumbers>
          <TransformationMatrix>{_mat16()}</TransformationMatrix>
        </ScanFile>
        <ScanFile>
          <FileName>case-LowerJaw.ply</FileName>
          <PartType>Antagonist</PartType>
          <ToothNumbers><int>47</int></ToothNumbers>
        </ScanFile>
      </ScanFiles>
"""

    exports = ""
    if with_exports:
        exports = f"""
      <ConstructionFileList>
        <ConstructionFile>
          <Filename>CASE-17-crown_cad.stl</Filename>
          <MaterialName>           氧化锆</MaterialName>
          <Height>0</Height>
          <ToothNumbers><int>17</int></ToothNumbers>
          <EstimatedDirectionForCAM><_a>0</_a><_b>0</_b><_c>1</_c></EstimatedDirectionForCAM>
          <IsScrewRetained>false</IsScrewRetained>
          <ZRotationMatrix>{_mat16()}</ZRotationMatrix>
        </ConstructionFile>
        <ConstructionFile>
          <Filename>CASE-12-11-21-22-bridge_slm_cad.stl</Filename>
          <MaterialName>氧化锆</MaterialName>
          <ToothNumbers><int>12</int><int>11</int><int>21</int><int>22</int></ToothNumbers>
          <EstimatedDirectionForCAM><_a>0.1</_a><_b>0.2</_b><_c>0.97</_c></EstimatedDirectionForCAM>
          <IsScrewRetained>true</IsScrewRetained>
        </ConstructionFile>
      </ConstructionFileList>
"""

    return f'''<?xml version="1.0" encoding="utf-8"?>
<ConstuctionInfo xmlns:xsd="http://www.w3.org/2001/XMLSchema"
                 xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
  <Teeth>
    <Tooth>
      <Number>17</Number>
      <ReconstructionType>AnatomicCrown</ReconstructionType>
      <ParametersUsed>
        <Parameter>
          <Value>0</Value><MinValue>0</MinValue><MaxValue>0</MaxValue>
          <StepSize xsi:nil="true" />
          <Type>LoadToothModelForceUsingPonticMode</Type>
        </Parameter>
        <Parameter>
          <Value>0.025</Value><MinValue>NaN</MinValue><MaxValue>0.5</MaxValue>
          <StepSize xsi:nil="true" />
          <Type>CementGapThickness</Type>
        </Parameter>
      </ParametersUsed>
      <ImplantType />
      <FreeformScanPerformed>true</FreeformScanPerformed>{geometry}
    </Tooth>
    <Tooth>
      <Number>47</Number>
      <ReconstructionType>Antagonist</ReconstructionType>
      <ParametersUsed />
      <Center><x>1.0</x><y>2.0</y><z>3.0</z></Center>
    </Tooth>
  </Teeth>
  <MatrixToScanDataFiles>{_mat16()}</MatrixToScanDataFiles>
  {scans}
  {exports}
  <PatientId>848</PatientId>
  <PracticeId>7</PracticeId>
  <TechnicianId />
  <ProjectName>CASE</ProjectName>
  <ProjectDir>\\\\192.168.1.1\\share\\CASE</ProjectDir>
  <DesignGUID>2b60e9fc-4e66-408c-b11a-a6dc6c91ad07</DesignGUID>
  <ConstructionFileCount>2</ConstructionFileCount>
</ConstuctionInfo>
'''


@pytest.fixture()
def info_path(tmp_path):
    def _write(**kw):
        p = tmp_path / "CASE.constructionInfo"
        p.write_text(build_construction_info(**kw), encoding="utf-8")
        return p
    return _write


# ============================================================ 向后兼容

def test_old_api_still_returns_tooth_records(info_path):
    """``parse_construction_info`` 的契约不变：返回 list[ExocadToothRecord]。"""
    recs = parse_construction_info(info_path())
    assert isinstance(recs, list) and len(recs) == 2
    assert recs[0].tooth_fdi == 17
    assert recs[0].reconstruction_type.value == "AnatomicCrown"
    assert recs[1].tooth_fdi == 47


def test_self_closing_tag_does_not_shift_parameters(info_path):
    """``<StepSize xsi:nil="true" />`` 不得让后续 Value/Type 错位配对。"""
    rec = parse_construction_info(info_path())[0]
    assert "CementGapThickness" in rec.parameters
    cell = rec.parameters["CementGapThickness"]
    assert cell.value == pytest.approx(0.025)
    # 错位的典型症状：值被写成前一个参数的值
    assert rec.parameters["LoadToothModelForceUsingPonticMode"].value == pytest.approx(0.0)


def test_nan_bounds_do_not_crash_and_do_not_become_numbers(info_path):
    """``<MinValue>NaN</MinValue>`` 必须原样留在 provenance，不能被变成数值。"""
    cell = parse_construction_info(info_path())[0].parameters["CementGapThickness"]
    assert cell.provenance["min"] == "NaN"
    assert cell.value == pytest.approx(0.025)      # 取的是 Value，不是 MinValue


# ============================================================ 颈缘线

def test_margin_parsed_as_3d_points(info_path):
    rec = parse_construction_info(info_path())[0]
    assert len(rec.margin_points) == len(_MARGIN_XYZ)
    assert rec.margin_points[0] == pytest.approx(list(_MARGIN_XYZ[0]))
    assert rec.margin_bottom_type == "Normal"
    assert rec.has_design_geometry is True


def test_bad_margin_point_counted_not_silently_dropped(info_path):
    """缺分量的 Vec3 必须被计数，而不是静默丢弃或补 0。"""
    case = parse_construction_case(info_path(bad_margin=True))
    rec = case.tooth(17)
    assert len(rec.margin_points) == len(_MARGIN_XYZ)
    assert "fdi_17_margin" in case.diagnostics
    diag = case.diagnostics["fdi_17_margin"]
    assert diag["n_total"] == len(_MARGIN_XYZ) + 1
    assert diag["n_bad"] == 1
    assert diag["present"] is True


# ============================================================ 牙架与方向

def test_insertion_axis_and_tooth_frame_parsed(info_path):
    rec = parse_construction_info(info_path())[0]
    assert rec.insertion_axis == pytest.approx([2.8e-10, 2.2e-09, 0.99999999])
    assert rec.axis_occlusal_of_tooth_model == pytest.approx([-0.035, 0.231, 0.972])
    assert rec.axis_mesial == pytest.approx([0.979, 0.202, -0.013])
    assert rec.axis_buccal == pytest.approx([-0.200, 0.952, -0.234])


def test_axis_children_read_by_document_order_when_tags_unnamed(info_path):
    """``<Axis>`` 的子标签名未知，必须按文档序读 —— 不能要求它们叫 x/y/z。"""
    rec = parse_construction_info(info_path())[0]
    assert len(rec.insertion_axis) == 3


def test_axis_and_cam_direction_are_separate_fields(info_path):
    """设计就位方向与加工方向必须分开存，不得互相覆盖。"""
    case = parse_construction_case(info_path())
    rec = case.tooth(17)
    assert rec.insertion_axis == pytest.approx([2.8e-10, 2.2e-09, 0.99999999])
    cam = case.construction_files[0].estimated_direction_for_cam
    assert cam == pytest.approx([0.0, 0.0, 1.0])
    assert cam != rec.insertion_axis


def test_center_and_library_parsed(info_path):
    rec = parse_construction_info(info_path())[0]
    assert rec.center == pytest.approx([13.90, 10.82, -12.35])
    assert rec.tooth_library_name == "002"
    assert rec.tooth_library_tooth_number == "17"
    assert len(rec.tooth_model_matrix) == 16
    assert rec.tooth_model_matrix[0] == pytest.approx(0.0)
    assert rec.tooth_model_matrix[5] == pytest.approx(5.0)


# ============================================================ 病例级

def test_scan_files_and_part_types(info_path):
    """``PartType`` 是预备体颌/对颌的真值来源。"""
    case = parse_construction_case(info_path())
    assert len(case.scan_files) == 2
    assert case.unsegmented_jaw_scan.part_type == "UnsegmentedJaw"
    assert case.unsegmented_jaw_scan.tooth_numbers == [17]
    assert case.antagonist_scan.part_type == "Antagonist"
    assert case.antagonist_scan.tooth_numbers == [47]
    assert len(case.unsegmented_jaw_scan.transformation_matrix) == 16


def test_construction_file_list_and_cad_export_detection(info_path):
    case = parse_construction_case(info_path())
    assert len(case.construction_files) == 2
    single = case.construction_files[0]
    assert single.filename == "CASE-17-crown_cad.stl"
    assert single.tooth_numbers == [17]
    assert single.looks_like_cad_export is True
    assert single.is_screw_retained is False

    bridge = case.construction_files[1]
    assert bridge.tooth_numbers == [12, 11, 21, 22]      # 桥件覆盖多个牙位
    assert bridge.looks_like_cad_export is True
    assert bridge.is_screw_retained is True


def test_matrix_to_scan_data_files(info_path):
    case = parse_construction_case(info_path())
    assert case.matrix_to_scan_data_files is not None
    assert len(case.matrix_to_scan_data_files) == 16


def test_phi_fields_collected_separately(info_path):
    """身份字段必须单独收集 —— 文件名里的患者姓名、PatientId、内网路径都算。"""
    case = parse_construction_case(info_path())
    phi = case.phi_fields
    assert phi["PatientId"] == "848"
    assert phi["PracticeId"] == "7"
    assert "192.168.1.1" in phi["ProjectDir"]
    # 空值不进 PHI 字典
    assert "TechnicianId" not in phi
    # 正常字段里不该混进这些
    rec = case.tooth(17)
    assert "848" not in str(rec.center)


def test_restoration_and_design_geometry_helpers(info_path):
    case = parse_construction_case(info_path())
    assert [t.tooth_fdi for t in case.restoration_teeth] == [17]
    assert [t.tooth_fdi for t in case.teeth_with_design_geometry] == [17]


# ============================================================ 缺字段不能抛

def test_missing_design_geometry_degrades_to_empty_not_error(info_path):
    """没有设计几何的文件必须正常返回空，而不是抛异常。"""
    case = parse_construction_case(info_path(with_geometry=False,
                                             with_scan_files=False,
                                             with_exports=False))
    rec = case.tooth(17)
    assert rec.margin_points == []
    assert rec.insertion_axis is None
    assert rec.center is None
    assert rec.has_design_geometry is False
    assert case.scan_files == []
    assert case.construction_files == []
    assert case.teeth_with_design_geometry == []


def test_antagonist_tooth_without_geometry_is_not_a_restoration(info_path):
    case = parse_construction_case(info_path())
    ant = case.tooth(47)
    assert ant.has_design_geometry is False
    assert ant.center == pytest.approx([1.0, 2.0, 3.0])


def test_root_tag_is_read_as_written_including_exocad_typo(tmp_path):
    """根标签就是 ``ConstuctionInfo``（exocad 拼错）。纠正它会让真实文件读不出来。"""
    p = tmp_path / "typo.constructionInfo"
    p.write_text(build_construction_info(), encoding="utf-8")
    root = ET.parse(p).getroot()
    assert root.tag == "ConstuctionInfo"
    assert root.tag != "ConstructionInfo"
    assert parse_construction_case(p).teeth              # 仍然能正常解析
