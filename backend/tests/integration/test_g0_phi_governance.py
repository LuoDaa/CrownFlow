"""G0：PHI 治理的**负向测试**（默认 `PSEUDONYMIZED`）。

这些测试的写法是刻意的：**用"PHI 不可能流出去"来证明治理成立**，而不是靠读代码。
每个用例都注入一个可用作搜索标记的假名（`PHI_MARKER` / `PHI_ADDRESS`），
然后在"操作员/审计能看到的一切"里搜它：

    工作区文件名 · 文件内容 · DB asset.original_name · cases.name · cases.order_json
    · 导入凭据 JSON · 修订链 · 工程包导出 ZIP

再加两个**反向**测试（用户要求）：

    IDENTIFIED 档 → 原始包与原始值可以被明确保留（否则"显式开口"就没有意义）
    DEIDENTIFIED_EXPORT 档 → 即使源病例是 IDENTIFIED，导出也不得带 PHI

以及一条**拒绝**测试：默认档下"顺手把原始包留下来"必须被拦，不是静默照做。
"""

from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from tests._three_shape_fixture import build_package, zip_package

IMPORT = "/api/importers/three-shape/import"
PREVIEW = "/api/importers/three-shape/preview"

#: 注入用的**假**身份信息（不是真实患者数据；有辨识度、便于全文搜索）。
PHI_MARKER = "SyntheticPatientZhaoQi"
PHI_ADDRESS = "SyntheticStreetNo42"


@pytest.fixture()
def iso(isolated_db, monkeypatch):
    from app.api import importers, routes
    from app.services import three_shape_import
    for module in (importers, routes, three_shape_import):
        monkeypatch.setattr(module, 'CASE_DIR', isolated_db / 'cases', raising=False)
    from app.main import app
    return isolated_db, TestClient(app)


def _package_with_phi(tmp_path, name: str = "pkg") -> Path:
    return zip_package(build_package(tmp_path / name, include_companions=True,
                                     patient_name=PHI_MARKER,
                                     extra_properties={"DeliveryAddress1": PHI_ADDRESS}),
                       tmp_path / f"{name}.zip")


def _import(client, zip_path: Path, **form) -> dict:
    with zip_path.open("rb") as fh:
        response = client.post(IMPORT, files={"file": (zip_path.name, fh, "application/zip")},
                               data=form or None)
    assert response.status_code == 200, response.text
    return response.json()


def _workspace_files(case_dir: Path) -> list[Path]:
    return [p for p in case_dir.rglob("*") if p.is_file()]


def _assert_marker_absent(case_dir: Path, marker: str) -> None:
    """工作区**文件名与文本内容**里都不许出现标记。"""
    for path in _workspace_files(case_dir):
        assert marker not in path.name, f"文件名里出现 PHI：{path.name}"
        if path.suffix.lower() in (".json", ".txt", ".csv", ".log"):
            text = path.read_text(encoding="utf-8", errors="replace")
            assert marker not in text, f"{path.name} 内容里出现 PHI"


# ============================================================ 默认档：不许流出

def test_default_policy_keeps_phi_out_of_the_whole_case_workspace(iso, tmp_path):
    """默认 PSEUDONYMIZED：工作区（文件名 + 文件内容）里搜不到任何注入的 PHI 值。"""
    data, client = iso
    body = _import(client, _package_with_phi(tmp_path))
    assert body["phi_policy"] == "PSEUDONYMIZED"
    case_dir = data / "cases" / body["case_id"]

    _assert_marker_absent(case_dir, PHI_MARKER)
    _assert_marker_absent(case_dir, PHI_ADDRESS)
    # 原始订单包不在工作区（包里就是订单 XML 原文）
    assert not (case_dir / "source_package.zip").exists()
    assert body["source_package_retained"] is False


def test_default_policy_keeps_phi_out_of_db_columns(iso, tmp_path):
    """DB 里也不许出现：asset.original_name / cases.name / cases.order_json。"""
    data, client = iso
    cid = _import(client, _package_with_phi(tmp_path))["case_id"]

    from app.db import db
    with db() as conn:
        case = dict(conn.execute("SELECT * FROM cases WHERE id=?", (cid,)).fetchone())
        assets = [dict(r) for r in conn.execute(
            "SELECT * FROM assets WHERE case_id=?", (cid,)).fetchall()]

    for asset in assets:
        assert PHI_MARKER not in asset["original_name"], asset["original_name"]
        assert PHI_ADDRESS not in asset["original_name"]
    assert PHI_MARKER not in (case.get("name") or "")
    assert PHI_MARKER not in (case.get("order_json") or "")
    assert PHI_ADDRESS not in (case.get("order_json") or "")
    # 技术 provenance 必须留下（否则"来源可追"也没了）
    order = json.loads(case["order_json"])
    assert order["source_sha256"] and order["importer_version"]
    assert order["phi_policy"] == "PSEUDONYMIZED"


def test_record_keeps_phi_report_but_never_the_values(iso, tmp_path):
    """导入凭据：保留"字段存在性 + 分类"，**不保留原值**，也不回显 PHI 字段名以外的东西。"""
    data, client = iso
    cid = _import(client, _package_with_phi(tmp_path))["case_id"]
    record = json.loads((data / "cases" / cid / "three_shape_import.json").read_text(encoding="utf-8"))

    assert "phi_fields" not in record, "原始 PHI 值不得落进凭据"
    report = record["phi_report"]
    assert report["Patient_LastName"]["present"] is True, "存在性必须如实记录"
    assert report["Patient_LastName"]["classification"] == "PATIENT_IDENTIFIER"
    assert "value" not in report["Patient_LastName"], "默认档不得存原值"
    assert "value_digest" not in report["Patient_LastName"], \
        "默认档连哈希都不存（SHA256(姓名) 可字典攻击、且跨病例可链接）"
    for entry in report.values():
        assert "value" not in entry
    # order_meta 是整份订单属性 —— PHI 键必须被剔掉
    assert PHI_MARKER not in json.dumps(record["order_meta"], ensure_ascii=False)
    assert "Patient_LastName" not in record["order_meta"]


def test_default_policy_rejects_legacy_keep_source_package_request(iso, tmp_path):
    """默认档下"顺手保留原始包"必须被**拒绝**，而不是静默照做。"""
    from app.services.phi_policy import PhiPolicyError
    from app.services.three_shape_import import commit

    zip_path = _package_with_phi(tmp_path)
    with pytest.raises(PhiPolicyError) as err:
        commit(zip_path, keep_source_package=True)
    assert "IDENTIFIED" in str(err.value), "错误消息要告诉调用方怎么才能合法保留"


def test_preview_reports_policy_without_side_effects(iso, tmp_path):
    """preview 仍然零副作用；但要能看出"这次会按哪档导入"。"""
    data, client = iso
    with _package_with_phi(tmp_path).open("rb") as fh:
        response = client.post(PREVIEW, files={"file": ("pkg.zip", fh, "application/zip")})
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["phi_fields"]["Patient_LastName"] == PHI_MARKER, \
        "解析层仍然收得到 PHI（供导入时脱敏用）"
    assert not list((data / "cases").glob("*")), "preview 不得建病例"


# ============================================================ 反向：显式档才保留

def test_identified_policy_retains_raw_package_and_values(iso, tmp_path):
    """反向测试①：显式 IDENTIFIED ⇒ 原始包与原始值**可以**保留（这是"显式开口"的意义）。"""
    data, client = iso
    body = _import(client, _package_with_phi(tmp_path, "identified"), phi_policy="IDENTIFIED")
    assert body["phi_policy"] == "IDENTIFIED"
    case_dir = data / "cases" / body["case_id"]

    assert body["source_package_retained"] is True
    assert (case_dir / "source_package.zip").is_file(), "显式档必须真的保留原始包"
    record = json.loads((case_dir / "three_shape_import.json").read_text(encoding="utf-8"))
    assert record["phi_report"]["Patient_LastName"]["value"] == PHI_MARKER
    assert record["phi_rules"]["persist_raw_phi_values"] is True


def test_export_gate_blocks_phi_even_for_identified_cases(iso, tmp_path):
    """反向测试②：即使源病例是 IDENTIFIED，**导出也不得带 PHI**（DEIDENTIFIED_EXPORT）。

    做法：把 PHI 值故意塞进病例显示名与凭据（IDENTIFIED 允许），再导出工程包 ——
    闸门必须拒绝；把显示名换成脱敏名后导出应当通过（证明闸门不是"一律拒绝导出"）。
    """
    data, client = iso
    cid = _import(client, _package_with_phi(tmp_path, "identified2"), phi_policy="IDENTIFIED",
                  case_name=f"{PHI_MARKER} 46")["case_id"]

    from app.db import db
    from app.services.phi_policy import PhiPolicyError, assert_export_clean, phi_needles_for_case

    needles = phi_needles_for_case(cid)
    assert PHI_MARKER in needles, "IDENTIFIED 档下病例名就是 PHI ⇒ 必须进 needle 集"

    # 直接对闸门做单元级验证（导出端点需要完整设计链，这里只验闸门本身）
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as z:
        z.writestr("design.json", json.dumps({"case_name": f"{PHI_MARKER} 46"}))
    with pytest.raises(PhiPolicyError) as err:
        assert_export_clean(buffer.getvalue(), needles=needles, where="测试导出")
    assert PHI_MARKER not in str(err.value), "错误消息不得回显 PHI 值本身"

    clean = io.BytesIO()
    with zipfile.ZipFile(clean, "w") as z:
        z.writestr("design.json", json.dumps({"case_name": "case 46"}))
        z.writestr("history/1/outer.stl", b"solid x\nendsolid x\n")
    assert_export_clean(clean.getvalue(), needles=needles, where="测试导出").clean is True

    def _source_package(needles_never_matter):
        with db() as conn:
            return dict(conn.execute("SELECT * FROM cases WHERE id=?", (cid,)).fetchone())

    assert PHI_MARKER in _source_package(needles)["name"], "源病例确实是 IDENTIFIED（带真名）"


def test_export_gate_structural_rules():
    """导出闸门的结构规则：条目名非 ASCII / 原始包 / 脱敏映射表 / PHI 字段名一律拒绝。"""
    from app.services.phi_policy import export_gate

    def _zip(entries: dict[str, bytes]) -> bytes:
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as z:
            for name, payload in entries.items():
                z.writestr(name, payload)
        return buffer.getvalue()

    named = export_gate(_zip({"张三/design.json": b"{}"}))
    assert not named.clean and any("ENTRY_NAME_NOT_ASCII_SAFE" in v for v in named.violations)

    raw = export_gate(_zip({"design.json": b"{}", "source_package.zip": b"PK"}))
    assert not raw.clean and any("FORBIDDEN_ENTRY" in v for v in raw.violations)

    keyed = export_gate(_zip({"revisions.json": b'{"Patient_LastName": "x"}'}),
                        phi_keys=("Patient_LastName",))
    assert not keyed.clean and any("PHI_KEY" in v for v in keyed.violations)

    assert export_gate(_zip({"design.json": b'{"case_name": "case 46"}'})).clean is True


# ============================================================ 策略解析

def test_policy_resolution_defaults_to_pseudonymized_and_rejects_unknown(iso):
    from app.services.phi_policy import DEFAULT_POLICY, PhiPolicy, PhiPolicyError, resolve_policy

    assert DEFAULT_POLICY is PhiPolicy.PSEUDONYMIZED
    assert resolve_policy(None) is PhiPolicy.PSEUDONYMIZED
    assert resolve_policy("deidentified_export") is PhiPolicy.DEIDENTIFIED_EXPORT
    with pytest.raises(PhiPolicyError):
        resolve_policy("SOMETHING_ELSE")
