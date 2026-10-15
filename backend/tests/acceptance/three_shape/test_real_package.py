"""真实 3Shape 订单包的验收测试 —— **fixture 不进 Git**。

`数据/测试_胡昭` 含 PHI（患者姓名同时出现在订单号、``Patient_LastName``、
包根目录名、以及 ``AliasFilenames.xml`` 的逻辑路径里），所以按与 exocad 数据
一致的规则**不提交**。本文件通过环境变量在本地受控数据区找它，找不到就 skip。

    CROWNFLOW_THREESHAPE_FIXTURE_DIR=D:/code/牙冠自动生成/数据/测试_胡昭

单元层的行为锁定在 ``tests/unit/importers/test_three_shape_package.py``（合成 fixture，
可提交）。这里只做真实病例的事实断言。
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from app.importers.three_shape import import_package, validate_package
from app.importers.three_shape.geometry_source import BindingEvidence
from app.importers.three_shape.models import ScanDecodeStatus, ScanRole

_ENV = "CROWNFLOW_THREESHAPE_FIXTURE_DIR"
_DEFAULT = Path(r"D:\code\牙冠自动生成\数据\测试_胡昭")


def _fixture_dir() -> Path | None:
    v = os.environ.get(_ENV)
    if v:
        p = Path(v)
        return p if p.exists() else None
    return _DEFAULT if _DEFAULT.exists() else None


@pytest.fixture(scope="module")
def package():
    d = _fixture_dir()
    if d is None:
        pytest.skip(
            f"真实 3Shape 订单包不可用。请设置 {_ENV} 指向本地受控数据区。"
            f"患者数据不进 Git，因此这里 skip 而不是失败 —— 但真实病例断言在"
            f"该数据可用前无法生效，这一点是已知的。默认查找位置 {_DEFAULT} 不存在。")
    return import_package(d)


# --------------------------------------------------------------- 处方

def test_prescription_from_real_package(package):
    """真实包应带出完整处方，用户不需要再手填牙位与材料。"""
    rx = package.prescription
    assert rx.target_fdi == 46, "Universal 30 应转为 FDI 46"
    assert rx.restoration_type == "CROWN"
    assert rx.material_name == "Zirkon"
    assert rx.material_id == "Configuration_11_Material1"
    assert rx.tooth_element_type_id == "Configuration_11_IDCrownZircon10"
    assert rx.process_status == "psScanned"
    assert rx.source_system == "3shape"


def test_material_reference_resolves(package):
    """订单引用的 MaterialID 必须能在材料库里解析出记录。

    这条锁住「ItemID 短路」那个 bug：``TDM_Item_ToothElementType_Crown``
    的 ItemID 是修复体类型 ID，材料 ID 在 MaterialID 里。
    """
    assert package.material_record is not None
    assert package.material_record["material_id"] == "Configuration_11_Material1"
    assert package.material_record["name"] == "Crown Zircon"
    # 载体对象的 ItemID 是**短形**修复体类型 id（实测 IDCrownZircon10），
    # 而订单里的 toothElementTypeID 是全形（Configuration_11_IDCrownZircon10）。
    # 关键是它与 MaterialID 不同 —— 这正是「ItemID 短路」bug 的成因。
    carrier_item_id = package.material_record["properties"]["ItemID"]
    assert carrier_item_id == "IDCrownZircon10"
    assert carrier_item_id != package.material_record["material_id"]


# --------------------------------------------------------------- 扫描

def test_preparation_scan(package):
    prep = package.scan_of_type("preparation")
    assert prep is not None
    assert prep.role is ScanRole.PREPARATION
    assert prep.resolved_path.endswith("Scans/Lower/PreparationScan.dcm")
    assert prep.decode_status is ScanDecodeStatus.SUPPORTED_BUT_ENCRYPTED
    assert prep.geometry_state == "SUPPORTED_BUT_ENCRYPTED"
    assert prep.geometry_ready is False
    assert prep.plaintext_meta["vertex_count"] == 131304
    assert prep.plaintext_meta["facet_count"] == 260445


def test_antagonist_scan(package):
    ant = package.scan_of_type("antagonist")
    assert ant is not None
    assert ant.role is ScanRole.ANTAGONIST
    assert ant.resolved_path.endswith("Scans/Upper/AntagonistScan.dcm")
    assert ant.decode_status is ScanDecodeStatus.SUPPORTED_BUT_ENCRYPTED
    assert ant.geometry_state == "SUPPORTED_BUT_ENCRYPTED"
    assert ant.geometry_ready is False
    assert ant.plaintext_meta["vertex_count"] == 286638
    assert ant.plaintext_meta["facet_count"] == 570241


def test_jaws_separated(package):
    assert package.prescription.preparation_jaw == "lower"
    assert package.prescription.antagonist_jaw == "upper"


def test_encryption_evidence_is_recorded(package):
    """加密判定必须有可复核的证据，而不是一个布尔值。

    实测：顶点载荷熵 7.9999 bit/byte（明文连通性只有 2.29），
    且按 float32 解释不像坐标。
    """
    for scan in package.scans:
        meta = scan.plaintext_meta
        assert meta["schema"] == "CE"
        assert meta["encryption_key_id"] == "1", "EKID 应记录使用的密钥 ID"
        assert meta["vertex_payload_entropy"] > 7.5, "顶点载荷未呈加密特征"
        assert meta["vertex_looks_like_coordinates"] is False
        assert meta["facet_payload_entropy"] < 4.0, "连通性应为明文（低熵）"


def test_no_companion_bound_in_this_package(package):
    """该包内没有 companion mesh，因此不得凭空绑定。

    同时确认绑定逻辑没有退化成位置式：包里只有两个 .dcm、没有 STL。
    """
    for scan in package.scans:
        assert scan.companion is not None
        assert scan.companion.binding is BindingEvidence.NONE
        assert scan.companion.mesh_path is None


# --------------------------------------------------------------- 闸门

def test_case_not_ready_and_blocked_only_on_geometry(package):
    """**元数据就绪不等于几何就绪**：病例必须停在 NOT_READY。

    不能因为 importer 能解析 metadata 就让病例进入设计管线。
    """
    report = validate_package(package).as_dict()

    assert report["ready"] is False
    failed = {c["name"] for c in report["checks"] if c["result"] == "FAIL"}
    assert failed == {"scan_geometry"}, f"预期只卡在几何上，实际 {failed}"

    passed = {c["name"] for c in report["checks"] if c["result"] == "PASS"}
    assert {"target_fdi", "restoration_type", "preparation_scan",
            "antagonist_scan", "material_reference", "jaw_separation"} <= passed


def test_geometry_not_available_for_design(package):
    assert package.geometry_available is False
    assert package.geometry_available_for_design is False


def test_failure_message_does_not_say_corrupt(package):
    """面向用户的措辞必须说"加密"，**不能**说成"文件损坏/不支持格式"。

    已实测证明格式受支持（CE 属于 hpsdecode.SUPPORTED_SCHEMAS），缺的只是合法密钥。
    混为一谈会让操作员误以为要重新导出整个包，而不是去导出 STL。
    """
    report = validate_package(package).as_dict()
    check = next(c for c in report["checks"] if c["name"] == "scan_geometry")

    assert "加密" in check["detail"]
    assert "损坏" not in check["detail"]
    assert "不支持" not in check["detail"]
    assert check["remedy"] and "STL" in check["remedy"]


# --------------------------------------------------------------- 参数映射纪律

def test_conflicting_parameters_stay_unresolved(package):
    """语义有冲突的字段必须 UNRESOLVED，且不得映射到任何规范名。

    实测四组冲突（详见 materials.py 的表格）。这条断言防止有人后来
    "看名字差不多"把它们接进解析链。
    """
    state = {p["source_field"]: p["state"] for p in package.cad_parameters}
    assert state, "该包应解析出 CAD 参数"

    for field in ("CementSpace.CementGap", "Overlay.WallThickness",
                  "CementSpace.RemoveUndercut"):
        assert state.get(field) == "UNRESOLVED", f"{field} 被错误地当成已确认映射"

    d2p = next((p for p in package.cad_parameters
                if p["source_field"] == "CementSpace.DistToPrepLine"), None)
    if d2p is not None:
        assert d2p["state"] == "UNRESOLVED"
        assert d2p["canonical_name"] is None, "与 exocad 差一个量级的量不得映射"


def test_parameters_narrowed_to_crown_subtype(package):
    """参数按 [材质 + 修复体子类型] 收窄，不应混入 Coping/Inlay/Veneer 的参数集。

    实测 Material1 下有 10 个 CementSpace，只按材质过滤会混用。
    """
    gaps = [p["value"] for p in package.cad_parameters
            if p["source_field"] == "CementSpace.CementGap"]
    assert len(gaps) <= 2, f"CementGap 取到 {len(gaps)} 条，疑似未按子类型收窄：{gaps}"


# --------------------------------------------------------------- PHI

def test_phi_collected_for_deidentification(package):
    """PHI 必须被收集出来供脱敏阶段使用，而不是留在包内无人处理。"""
    assert package.phi_fields, "未收集到任何 PHI 字段"
    assert package.diagnostics["package_root_name"], "未识别包根名（实测含患者姓名）"


def test_package_root_name_is_phi(package):
    """包根目录名本身就是 PHI —— 脱敏时必须一并重命名。

    这与 exocad 侧的六层 PHI 来源是同类问题，多了一层：3Shape 包根目录名。
    """
    root_name = package.diagnostics["package_root_name"]
    assert root_name == "胡昭"


# ==================================================================
# HTTP 层：真实包走一遍操作员实际会走的路径
#
# 操作员上传的是 ZIP，所以这里把 fixture 目录现打成 ZIP 再上传 ——
# 顺带验证"打出来的 ZIP 与真实包同构"（物理文件名是 GUID、没有扩展名）。
# ==================================================================

def _fixture_zip(tmp_path: Path) -> Path | None:
    d = _fixture_dir()
    if d is None:
        return None
    import zipfile
    dest = tmp_path / "real_package.zip"
    with zipfile.ZipFile(dest, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(d.rglob("*")):
            if p.is_file():
                z.write(p, p.relative_to(d).as_posix())
    return dest


@pytest.fixture()
def api_client(tmp_path, monkeypatch):
    """把库与病例目录指到临时路径 —— 绝不允许碰工作区里的真实病例库。"""
    from fastapi.testclient import TestClient
    from app import config, db
    from app.api import routes
    from app.services import three_shape_import

    data_dir = tmp_path / "data"
    cases = data_dir / "cases"
    cases.mkdir(parents=True)
    (data_dir / "templates").mkdir(parents=True)
    for mod in (config, routes, three_shape_import):
        monkeypatch.setattr(mod, "CASE_DIR", cases, raising=False)
    monkeypatch.setattr(config, "DATA_DIR", data_dir)
    monkeypatch.setattr(config, "TEMPLATE_DIR", data_dir / "templates")
    monkeypatch.setattr(config, "DB_PATH", data_dir / "t.sqlite3")
    monkeypatch.setattr(routes, "TEMPLATE_DIR", data_dir / "templates")
    db.init_db()

    from app.main import app
    with TestClient(app) as c:
        yield c, cases


def _post_zip(client, url, zp: Path, **form):
    with zp.open("rb") as f:
        return client.post(url, files={"file": (zp.name, f, "application/zip")},
                           data=form or None)


def test_preview_real_package_panel(tmp_path, api_client):
    """真实包的面板：处方全绿，唯独几何卡在加密。"""
    client, cases = api_client
    zp = _fixture_zip(tmp_path)
    if zp is None:
        pytest.skip("真实 3Shape 订单包不可用（见文件头说明）")

    r = _post_zip(client, "/api/importers/three-shape/preview", zp)
    assert r.status_code == 200, r.text
    body = r.json()
    rows = {x["label"]: x for x in body["panel"]}

    assert body["can_commit"] is False
    assert rows["目标牙位"]["status"] == "PASS" and "46" in rows["目标牙位"]["detail"]
    assert rows["修复体"]["status"] == "PASS" and "CROWN" in rows["修复体"]["detail"]
    assert rows["材料"]["status"] == "PASS" and "Zirkon" in rows["材料"]["detail"]
    assert rows["预备体扫描"]["status"] == "FAIL" and "lower" in rows["预备体扫描"]["detail"]
    assert rows["对颌扫描"]["status"] == "FAIL" and "upper" in rows["对颌扫描"]["detail"]
    assert rows["颌位分离"]["status"] == "PASS"
    # 格式受支持（PASS）与几何不可用（FAIL）必须分开呈现
    assert rows["扫描格式"]["status"] == "PASS"
    assert "加密" in rows["扫描格式"]["detail"]
    assert rows["扫描几何"]["status"] == "FAIL"
    assert "CE 加密" in rows["扫描几何"]["detail"]
    assert rows["CAD 参数"]["status"] in ("PASS", "WARN")

    # 明文可见的规模必须报出来（证明文件读到了、只有顶点不可解）
    assert "131304" in rows["预备体扫描"]["detail"] or "131304" in rows["扫描几何"]["detail"]

    assert body["prescription"]["target_fdi"] == 46
    assert [s["decode_status"] for s in body["scans"]] == [
        "SUPPORTED_BUT_ENCRYPTED", "SUPPORTED_BUT_ENCRYPTED"]
    assert all(s["companion_binding"] in (None, "NONE") for s in body["scans"])


def test_preview_real_package_writes_nothing(tmp_path, api_client):
    """真实包 preview 之后，病例库与病例目录都必须原样为空。"""
    client, cases = api_client
    zp = _fixture_zip(tmp_path)
    if zp is None:
        pytest.skip("真实 3Shape 订单包不可用（见文件头说明）")

    assert _post_zip(client, "/api/importers/three-shape/preview", zp).status_code == 200
    assert client.get("/api/cases").json() == []
    assert list(cases.iterdir()) == []


def test_import_real_package_refused(tmp_path, api_client):
    """真实包在当前没有合法密钥、也没有 STL 的情况下，必须拒绝建病例。

    这就是本轮的核心结论：元数据能解析，但**几何不可用**，病例不得进入设计管线。
    拒绝必须是 409 + 失败明细，而不是建一个进不了设计的空病例。
    """
    client, cases = api_client
    zp = _fixture_zip(tmp_path)
    if zp is None:
        pytest.skip("真实 3Shape 订单包不可用（见文件头说明）")

    r = _post_zip(client, "/api/importers/three-shape/import", zp)
    assert r.status_code == 409, r.text
    detail = r.json()["detail"]
    assert detail["import_status"] == "CASE_NOT_READY"
    failed = {c["name"] for c in detail["validation"]["checks"] if c["result"] == "FAIL"}
    assert failed == {"scan_geometry"}, failed
    remedies = " ".join(detail["validation"]["remedies"])
    assert "STL" in remedies
    assert "不通过提取/绕过厂商密钥" in remedies
    assert client.get("/api/cases").json() == []
    assert list(cases.iterdir()) == []


# ==================================================================
# 补救路径：操作员按建议导出 STL 之后，病例必须真的变成可导入
#
# 这一步很关键 —— 说明里写了"导出 STL 放进包里"，如果做完还是 CCC，
# 那条建议就是空话。这里在真实包的副本上补两个 STL（不碰原 fixture），
# 验证：几何就绪 -> 闸门全过 -> 建出病例，且两处 provenance 都在。
# ==================================================================

def _copy_and_add_stl_companions(tmp_path: Path, scan_logicals: dict[str, str]) -> Path | None:
    """把真实包拷一份，往 AliasFilenames.xml 追加 STL 条目并写出对应文件。

    与 3Shape 重新导出的效果一致：新文件拿到新 GUID，并在别名表里登记。
    网格内容是**合成盒体**（仅用于证明绑定与闸门逻辑，不代表真实扫描几何）。
    """
    import shutil, uuid as _uuid, trimesh
    d = _fixture_dir()
    if d is None:
        return None
    root = tmp_path / "pkg_with_stl"
    shutil.copytree(d, root)

    alias = root / "AliasFilenames.xml"
    text = alias.read_text(encoding="utf-8-sig")
    items = []
    guids = {}
    for logical in scan_logicals:
        guid = "{" + str(_uuid.uuid4()).upper() + "}"
        guids[logical] = guid
        items.append(
            "        <Object type=\"TAliasFilenameItem\">\n"
            f"          <Property name=\"OriginalName\" value=\"{logical}\"/>\n"
            f"          <Property name=\"AliasName\" value=\"{guid}\"/>\n"
            "        </Object>\n")
        m = trimesh.creation.box(extents=[6.0, 6.0, 6.0])
        (root / guid).write_bytes(m.export(file_type="stl"))
    alias.write_text(text.replace("      </List>", "".join(items) + "      </List>", 1),
                     encoding="utf-8")
    return root


def test_operator_stl_export_makes_case_importable(tmp_path, api_client):
    """按补救建议补上同主名 STL 后，病例必须从"拒绝"变成"可建"。"""
    import zipfile
    client, cases = api_client
    root = _copy_and_add_stl_companions(tmp_path, {
        "胡昭/Scans/Lower/PreparationScan.stl": "",
        "胡昭/Scans/Upper/AntagonistScan.stl": "",
    })
    if root is None:
        pytest.skip("真实 3Shape 订单包不可用（见文件头说明）")

    zp = tmp_path / "with_stl.zip"
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(root.rglob("*")):
            if p.is_file():
                z.write(p, p.relative_to(root).as_posix())

    r = _post_zip(client, "/api/importers/three-shape/preview", zp)
    assert r.status_code == 200, r.text
    body = r.json()
    rows = {x["label"]: x for x in body["panel"]}
    assert body["can_commit"] is True, rows["扫描几何"]
    assert rows["扫描几何"]["status"] == "PASS"
    for s in body["scans"]:
        assert s["geometry_state"] == "GEOMETRY_READY"
        assert s["original_geometry_source"] == "HPS_CE_ENCRYPTED", "原件来源不能丢"
        assert s["preferred_geometry_source"] == "COMPANION_STL"
        assert s["companion_binding"] == "SIBLING_SAME_STEM"
    # 原件仍标注加密 —— 采纳 STL 不等于把原件说成明文
    assert rows["扫描格式"]["status"] == "PASS"
    assert "加密" in rows["扫描格式"]["detail"]

    r = _post_zip(client, "/api/importers/three-shape/import", zp)
    assert r.status_code == 200, r.text
    created = r.json()
    assert created["design_created"] is False
    assert {a["kind"] for a in created["assets"]} == {"lower", "upper"}
    for a in created["assets"]:
        assert a["source"] == "COMPANION_STL"
    assert (cases / created["case_id"] / "three_shape_import.json").is_file()
