"""Bite Reference Audit 与"同坐标系门"的测试（只读；不做配准）。

两组：
1. 合成病例目录 —— 钉住角色映射、加密判定、资产目录、**margin 跨坐标系必须拒绝**；
2. 真实语料 —— 报表能跑完、自洽、字段齐全（缺失即 skip）。
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from app.services import bite_reference_audit as AUD

_CE_DCM = b"""<?xml version="1.0" encoding="utf-8"?>
<HPS version="1.1"><Packed_geometry><Schema>CE</Schema><Binary_data><CE version="1.0">
<Vertices vertex_count="8" base64_encoded_bytes="96">AAAA</Vertices></CE></Binary_data>
</Packed_geometry></HPS>"""

_PLAIN_DCM = b"""<?xml version="1.0" encoding="utf-8"?>
<HPS version="1.1"><Packed_geometry><Schema>HPS</Schema></Packed_geometry></HPS>"""

_ORDER_XML = """<?xml version="1.0" encoding="utf-8"?>
<ConstuctionInfo version="1.0">
  <ScanFiles>
    <ScanFile><FileName>{case}-upperjaw.DCM</FileName><PartType>UnsegmentedJaw</PartType>
      <TransformationMatrix>{T}</TransformationMatrix><FileHash>{H}</FileHash>
      <LoadedBy>DentalProjectFileReader</LoadedBy></ScanFile>
    <ScanFile><FileName>{case}-lowerjaw.DCM</FileName><PartType>Antagonist</PartType>
      <TransformationMatrix>{T}</TransformationMatrix><FileHash>{H}</FileHash></ScanFile>
    <ScanFile><FileName>{case}-biteregister.DCM</FileName><PartType>BiteRegister</PartType>
      <TransformationMatrix>{T}</TransformationMatrix><FileHash>{H}</FileHash></ScanFile>
  </ScanFiles>
  <ConstructionFileList>
    <ConstructionFile><Filename>{case}-cad.stl</Filename><ToothNumbers>46</ToothNumbers>
      <Material>ZI</Material><MaterialName>氧化锆</MaterialName>
      <ZRotationMatrix>{T}</ZRotationMatrix></ConstructionFile>
  </ConstructionFileList>
  <Teeth>
    <Tooth><ToothFDI>24</ToothFDI><ToothLibraryName>合成</ToothLibraryName></Tooth>
  </Teeth>
</ConstuctionInfo>"""

# exocad 用 ``<_00>``…``<_33>`` 子标签存矩阵（解析器按标签定位，行主序、平移在最后一行）
_MATRIX = "".join("<_%d%d>%d</_%d%d>" % (r, c, 1 if r == c else 0, r, c)
                  for r in range(4) for c in range(4))


def _make_case(root: Path, name: str, *, encrypted: bool = True, with_die: bool = False,
               with_cad: bool = True) -> Path:
    case = root / name
    case.mkdir(parents=True, exist_ok=True)
    (case / f"{name}.constructionInfo").write_text(
        _ORDER_XML.format(case=name, T=_MATRIX, H="abc123"), encoding="utf-8")
    payload = _CE_DCM if encrypted else _PLAIN_DCM
    for suffix in ("upperjaw", "lowerjaw", "biteregister"):
        (case / f"{name}-{suffix}.DCM").write_bytes(payload)
    if with_cad:
        (case / f"{name}-cad.stl").write_bytes(b"solid x\nendsolid x\n")
    if with_die:
        import trimesh
        die = trimesh.creation.icosphere(subdivisions=2, radius=3.0)
        die.export(case / f"{name}-24-modeldie.stl")
    return case


def test_scan_roles_and_encryption_verdict(tmp_path):
    case = _make_case(tmp_path, "caseA")
    result = AUD.audit_case(case, run_die_contract=False)
    roles = {s["role"]: s for s in result["scans"]}
    assert set(roles) == {"PREP_JAW_SCAN", "ANTAGONIST_SCAN", "BITE_ALIGNMENT_SCAN"}
    assert all(s["readable"] is False for s in result["scans"]), "CE 加密载荷必须判为不可读"
    assert "CE" in roles["BITE_ALIGNMENT_SCAN"]["readable_reason"]
    assert roles["PREP_JAW_SCAN"]["transformation_matrix"] == "IDENTITY"
    assert result["c2_feasibility"] == AUD.C2_NO_READABLE_SCANS
    assert result["vendor_reference_pose"]["status"] == "UNAVAILABLE_NO_READABLE_GEOMETRY"
    assert result["vendor_reference_pose"]["reference_type"] == "EXOCAD_REGISTERED_POSE"
    # 术语纪律：绝不把 bite scan 或 exocad 位姿写成 GROUND_TRUTH
    assert "GROUND_TRUTH" not in json.dumps(result, ensure_ascii=False)


def test_plain_scans_are_readable_and_c2_becomes_feasible(tmp_path):
    case = _make_case(tmp_path, "caseB", encrypted=False)
    result = AUD.audit_case(case, run_die_contract=False)
    assert result["geometry_availability"]["jaw_pair_readable"] is True
    assert result["geometry_availability"]["bite_alignment_scan_readable"] is True
    assert result["c2_feasibility"] == AUD.C2_READABLE
    assert result["vendor_reference_pose"]["status"] == "VERIFIABLE_BUT_NOT_YET_VERIFIED"


def test_asset_catalog_uses_metadata_for_cad_files(tmp_path):
    case = _make_case(tmp_path, "caseC")
    result = AUD.audit_case(case, run_die_contract=False)
    cad = [a for a in result["readable_assets"] if a["role"] == AUD.ASSET_CAD_RESTORATION]
    assert cad and cad[0]["meta"]["tooth_numbers"] == [46] and cad[0]["meta"]["material"] == "ZI"
    assert result["geometry_availability"]["cad_restoration_count"] == 1


def test_model_die_asset_gets_fdi_and_is_contract_checked(tmp_path):
    """逐牙位代型：FDI 从标准命名取出，并**必须**过同坐标系门（跨坐标系不许 READY）。"""
    case = _make_case(tmp_path, "caseD", with_die=True)
    result = AUD.audit_case(case)
    availability = result["geometry_availability"]
    assert availability["model_die_count"] == 1 and availability["model_die_fdis"] == [24]
    assert len(result["die_contracts"]) == 1
    classification = result["die_contracts"][0]["contract"]["classification"]
    # 合成病例的 constructionInfo 没有该牙位 margin ⇒ 如实报"缺 margin/轴"，不猜
    assert classification["rejection_code"] in ("MARGIN_OR_AXIS_MISSING",
                                                "MARGIN_NOT_ON_SUPPORT_SURFACE")


def test_corpus_report_shape_on_synthetic_corpus(tmp_path):
    _make_case(tmp_path, "caseA")
    _make_case(tmp_path, "caseB", encrypted=False)
    report = AUD.audit_corpus(tmp_path, run_die_contract=False)
    json.dumps(report, ensure_ascii=False)
    for key in ("generated_at", "cases_root", "total_cases", "c2_feasibility",
                "geometry_availability_totals", "cases", "die_contract_summary"):
        assert key in report, key
    assert report["total_cases"] == 2
    assert report["c2_feasibility"].get(AUD.C2_NO_READABLE_SCANS) == 1
    assert report["c2_feasibility"].get(AUD.C2_READABLE) == 1


# ---------------------------------------------------------------- 真实语料

_DEFAULT_CASES = Path(r"D:\code\牙冠自动生成\数据\15个完整病例")


def _resolve_corpus():
    value = os.environ.get("CROWNFLOW_EXOCAD_CASES_DIR")
    path = Path(value) if value else _DEFAULT_CASES
    if not path.is_dir():
        pytest.skip("exocad 病例目录不可用（设置 CROWNFLOW_EXOCAD_CASES_DIR）")
    return path


def test_real_corpus_audit_is_self_consistent(tmp_path):
    root = _resolve_corpus()
    report = AUD.audit_corpus(root, run_die_contract=False)
    out = tmp_path / "bite_reference_audit.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    assert report["total_cases"] == sum(report["c2_feasibility"].values())
    for entry in report["cases"]:
        assert entry["c2_feasibility"] in (AUD.C2_NO_READABLE_SCANS, AUD.C2_READABLE, AUD.C2_PARTIAL)
        for scan in entry["scans"]:
            assert scan["role"] in set(AUD.PART_TYPE_ROLES.values()) | {"OTHER"}
            assert scan["readable"] is not None, "每份扫描都必须给可读性结论"
        # 术语纪律：报告里不得出现把 exocad 位姿称作真值的字样
        assert "GROUND_TRUTH" not in json.dumps(entry, ensure_ascii=False)


def test_real_die_assets_never_pass_the_frame_gate_silently():
    """真实代型（模型/打印系）与设计系 margin 不同坐标系 ⇒ 不许 READY，且必须说明原因。"""
    from app.services import prep_conditioning as PC

    root = _resolve_corpus()
    report = AUD.audit_corpus(root, run_die_contract=True)
    for entry in report["cases"]:
        for item in entry.get("die_contracts") or []:
            classification = item["contract"]["classification"]
            if classification["readiness"] == PC.Readiness.PREP_SURFACE_READY.value:
                # 允许 READY，但必须是在同一坐标系里（距离在门内）
                assert (item["contract"]["diagnostics"].get("coordinate_frame_sane") is True), \
                    "跨坐标系的代型不许报 READY"
            else:
                assert classification["rejection_code"], item
                assert classification["rejection_reason"], "拒绝必须带理由"
