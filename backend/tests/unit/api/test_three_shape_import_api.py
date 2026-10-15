"""3Shape 导入入口的 HTTP 层测试（合成 fixture，可提交 Git）。

锁住的规则：

  1. **preview 零副作用** —— 不建病例、不写网格、不落库。
  2. **闸门不过就 409，不建病例** —— "元数据能解析"不等于"能进设计管线"。
  3. **加密 ≠ 不支持** —— ``扫描格式`` 行 PASS、"扫描几何"行 FAIL，
     措辞必须说"加密/无合法密钥"，**不得**出现"损坏 / 不支持格式"。
  4. **companion 存在时几何就绪**，且两处 provenance 都保留。
  5. 建病例必须**顺带写导入凭据**，且 ``design_created is False``。
"""

from __future__ import annotations

import json
from pathlib import Path

from tests._three_shape_fixture import MATERIAL_ID, PATIENT, build_package, zip_package

PREVIEW = "/api/importers/three-shape/preview"
IMPORT = "/api/importers/three-shape/import"

#: 操作员看到的面板行（顺序即呈现顺序）。
PANEL_LABELS = [
    "目标牙位", "修复体", "材料", "预备体扫描", "对颌扫描",
    "颌位分离", "扫描格式", "扫描几何", "CAD 参数",
]


def _upload(client, url, zip_path: Path, **form):
    with zip_path.open("rb") as f:
        return client.post(url, files={"file": (zip_path.name, f, "application/zip")},
                           data=form or None)


def _row(payload: dict, label: str) -> dict:
    return next(r for r in payload["panel"] if r["label"] == label)


# ==================================================== preview：零副作用

def test_preview_has_no_side_effects(client, iso, tmp_path):
    """preview 不得建病例、不得写网格、不得落库。"""
    zp = zip_package(build_package(tmp_path / "enc"), tmp_path / "enc.zip")
    r = _upload(client, PREVIEW, zp)
    assert r.status_code == 200, r.text
    body = r.json()

    assert body["can_commit"] is False
    assert body["import_status"] == "CASE_NOT_READY"
    # 病例目录里一个文件都不该多出来
    assert list((iso / "cases").iterdir()) == []
    assert client.get("/api/cases").json() == []


def test_preview_panel_has_operator_facing_rows(client, tmp_path):
    zp = zip_package(build_package(tmp_path / "enc"), tmp_path / "enc.zip")
    body = _upload(client, PREVIEW, zp).json()
    assert [r["label"] for r in body["panel"]] == PANEL_LABELS
    # 面板的判定必须与闸门同源，不能界面 ✅ 而闸门 FAIL
    gate = {c["name"]: c["result"] for c in body["validation"]["checks"]}
    assert _row(body, "扫描几何")["status"] == gate["scan_geometry"]


def test_preview_carries_order_facts(client, tmp_path):
    """处方从订单 XML 读出来，操作员不需要手填牙位与材料。"""
    zp = zip_package(build_package(tmp_path / "enc"), tmp_path / "enc.zip")
    body = _upload(client, PREVIEW, zp).json()

    assert body["prescription"]["target_fdi"] == 46
    assert body["prescription"]["restoration_type"] == "CROWN"
    assert body["prescription"]["material_name"] == "Zirkon"
    assert body["prescription"]["preparation_jaw"] == "lower"
    assert body["prescription"]["antagonist_jaw"] == "upper"
    assert body["material_record"]["material_id"] == MATERIAL_ID
    assert _row(body, "目标牙位")["status"] == "PASS"
    assert _row(body, "修复体")["status"] == "PASS"
    assert _row(body, "颌位分离")["status"] == "PASS"


# ============================================ 加密 ≠ 损坏 / 不支持

def test_encrypted_scan_reported_as_encrypted_not_unsupported(client, tmp_path):
    """实测这个包是 CE 加密、格式受支持。面板必须区分这两件事。

    ``扫描格式`` 走 PASS（HPS/CE 是受支持的容器），``扫描几何`` 走 FAIL
    （拿不到顶点）。把所有加密扫描一律说成"格式不支持"，操作员会去重新导出，
    而问题不在导出格式上。
    """
    zp = zip_package(build_package(tmp_path / "enc"), tmp_path / "enc.zip")
    body = _upload(client, PREVIEW, zp).json()

    fmt = _row(body, "扫描格式")
    geom = _row(body, "扫描几何")
    assert fmt["status"] == "PASS", fmt
    assert "加密" in fmt["detail"] and "不是文件损坏" in fmt["detail"]

    assert geom["status"] == "FAIL", geom
    assert "CE 加密" in geom["detail"]
    assert "无合法密钥" in geom["detail"]

    # 整份响应里不允许把加密**断言成**损坏/格式不受支持。
    # 注意 "这不是文件损坏" 是我们要的否定表述，不能一刀切禁掉"文件损坏"四个字，
    # 所以这里禁的是**肯定式**的说法。
    blob = json.dumps(body, ensure_ascii=False)
    for bad in ("文件已损坏", "扫描文件损坏", "文件损坏，", "格式不受支持",
                "不支持该格式", "无法识别的格式"):
        assert bad not in blob, f"响应里出现了误导性措辞：{bad}"
    assert "这不是文件损坏" in blob, "必须明确否掉'损坏'的解读"
    assert "格式受支持" in blob
    assert "SUPPORTED_BUT_ENCRYPTED" in blob   # 三态原样暴露，不吞掉

    # 补救建议必须指向 STL 导出，而不是"重新导出格式"
    remedy = geom["remedy"]
    assert remedy and "STL" in remedy
    assert "不通过提取/绕过厂商密钥" in remedy


def test_panel_text_has_no_markdown_emphasis(client, tmp_path):
    """面板是**纯文本**渲染的，不能把 markdown 的 ** 强调写进这些字符串。

    实测踩过一次：validator 的 remedy 里写了 ``**与原扫描同目录同主名**``，
    浏览器上原样显示成星号。
    """
    zp = zip_package(build_package(tmp_path / "enc"), tmp_path / "enc.zip")
    body = _upload(client, PREVIEW, zp).json()
    blob = json.dumps(body, ensure_ascii=False)
    assert "**" not in blob, "用户可见文案里出现了 markdown 强调标记"
    assert "``" not in blob


def test_commit_refused_when_geometry_unavailable(client, iso, tmp_path):
    """闸门不过 -> 409，且**不建病例**。"""
    zp = zip_package(build_package(tmp_path / "enc"), tmp_path / "enc.zip")
    r = _upload(client, IMPORT, zp)
    assert r.status_code == 409, r.text
    detail = r.json()["detail"]
    assert detail["import_status"] == "CASE_NOT_READY"
    assert any(c["name"] == "scan_geometry" and c["result"] == "FAIL"
               for c in detail["validation"]["checks"])
    assert client.get("/api/cases").json() == []
    assert list((iso / "cases").iterdir()) == []


# ==================================================== companion STL

def test_companion_stl_makes_geometry_ready(client, tmp_path):
    """有同主名 STL 时几何就绪，且两处 provenance 都保留。"""
    zp = zip_package(build_package(tmp_path / "ok", include_companions=True),
                     tmp_path / "ok.zip")
    body = _upload(client, PREVIEW, zp).json()

    assert body["can_commit"] is True
    assert body["import_status"] == "CASE_READY"
    assert _row(body, "扫描几何")["status"] == "PASS"
    # 格式行仍说明原件是加密的 —— 格式与几何是两件事
    assert "加密" in _row(body, "扫描格式")["detail"]

    by_path = {s["resolved_path"]: s for s in body["scans"]}
    prep = next(s for s in by_path.values() if s["role"] == "PREPARATION")
    ant = next(s for s in by_path.values() if s["role"] == "ANTAGONIST")
    for s in (prep, ant):
        assert s["original_geometry_source"] == "HPS_CE_ENCRYPTED"
        assert s["preferred_geometry_source"] == "COMPANION_STL"
        assert s["companion_binding"] == "SIBLING_SAME_STEM"
        assert s["geometry_state"] == "GEOMETRY_READY"


def test_import_creates_case_with_jaw_assets(client, iso, tmp_path):
    """建病例：预备体进它自己的颌，对颌进另一侧 —— 由颌位决定，不按顺序塞。"""
    zp = zip_package(build_package(tmp_path / "ok", include_companions=True),
                     tmp_path / "ok.zip")
    r = _upload(client, IMPORT, zp)
    assert r.status_code == 200, r.text
    body = r.json()

    assert body["design_created"] is False, "导入不得自动延伸出设计"
    cid = body["case_id"]
    kinds = {a["kind"]: a for a in body["assets"]}
    assert set(kinds) == {"lower", "upper"}, kinds
    assert kinds["lower"]["logical_path"].endswith("Scans/Lower/PreparationScan.stl")
    assert kinds["upper"]["logical_path"].endswith("Scans/Upper/AntagonistScan.stl")
    for a in kinds.values():
        assert a["source"] == "COMPANION_STL"
        assert a["original_geometry_source"] == "HPS_CE_ENCRYPTED"
        assert a["mesh"]["vertices"] > 0

    # 病例与资产都落库了，几何文件真的在磁盘上
    detail = client.get(f"/api/cases/{cid}").json()
    assert detail["case"]["id"] == cid
    order=json.loads(detail["case"]["order_json"])
    assert order['prescription']==body['prescription']
    assert order['configuration_status']=='UNCONFIRMED'
    assert {a["kind"] for a in detail["assets"]} == {"lower", "upper"}
    for a in detail["assets"]:
        assert Path(a["path"]).is_file()
        assert Path(a["path"]).parent == iso / "cases" / cid

    # 资产能通过既有出口取回
    got = client.get(f"/api/cases/{cid}/assets/lower/file")
    assert got.status_code == 200


def test_import_writes_traceable_record(client, iso, tmp_path):
    """病例目录里留下导入凭据 —— "这个病例从哪来"必须可追溯。"""
    zp = zip_package(build_package(tmp_path / "ok", include_companions=True),
                     tmp_path / "ok.zip")
    cid = _upload(client, IMPORT, zp).json()["case_id"]

    rec_path = iso / "cases" / cid / "three_shape_import.json"
    assert rec_path.is_file()
    rec = json.loads(rec_path.read_text(encoding="utf-8"))
    assert [r["label"] for r in rec["panel"]] == PANEL_LABELS
    assert rec["validation"]["ready"] is True
    assert rec["prescription"]["target_fdi"] == 46
    assert {a["kind"] for a in rec["assets"]} == {"lower", "upper"}
    # G0：默认 PSEUDONYMIZED —— 原始订单包**不留在工作区**（包内含订单 XML 原文）。
    # 技术 provenance 改由 source_sha256 / importer_version 承担（来源可追、身份不可追）。
    assert not (iso / "cases" / cid / "source_package.zip").exists()
    assert rec["source_sha256"] and rec["phi_policy"] == "PSEUDONYMIZED"
    assert rec["importer_version"]
    assert "phi_fields" not in rec, "原始 PHI 值不得落进凭据文件"

    back = client.get(f"/api/importers/three-shape/records/{cid}").json()
    assert back["has_record"] is True
    assert back["record"]["panel"] == rec["panel"]

    listed = client.get("/api/importers/three-shape/records").json()
    assert [x["case_id"] for x in listed] == [cid]
    assert listed[0]["prescription"]["target_fdi"] == 46


def test_default_case_name_carries_no_phi(client, tmp_path):
    """默认病例名不含患者身份字段。"""
    zp = zip_package(build_package(tmp_path / "ok", include_companions=True),
                     tmp_path / "ok.zip")
    body = _upload(client, IMPORT, zp).json()
    assert body["case_name"] == "3Shape Zirkon CROWN FDI46"
    assert PATIENT.lower() not in body["case_name"].lower()


def test_case_name_supplied_is_sanitized_under_default_policy(client, tmp_path):
    """G0：默认档下，操作员给的名字里**姓名类 CJK 串被替换**（原值不落库/不落盘）。

    这不是"不让命名"，而是保证工作区里不出现患者姓名；FDI、修复体类型这些
    非身份信息保留，操作员仍然分得清哪个病例。
    """
    zp = zip_package(build_package(tmp_path / "ok", include_companions=True),
                     tmp_path / "ok.zip")
    body = _upload(client, IMPORT, zp, case_name="张三 46 全锆冠").json()
    assert "张三" not in body["case_name"]
    assert body["case_name"].endswith("46 case") or "46" in body["case_name"]
    assert body["phi_policy"] == "PSEUDONYMIZED"


# ==================================================== 输入形态的拒绝

def test_non_zip_upload_rejected(client, tmp_path):
    stl = tmp_path / "scan.stl"
    stl.write_bytes(b"solid x\nendsolid x\n")
    with stl.open("rb") as f:
        r = client.post(PREVIEW, files={"file": ("scan.stl", f, "model/stl")})
    assert r.status_code == 400
    assert "ZIP" in r.json()["detail"]


def test_zip_without_alias_file_rejected(client, tmp_path):
    """缺 AliasFilenames.xml 必须明确报错，不能靠文件名猜。"""
    loose = tmp_path / "loose"
    loose.mkdir()
    (loose / "Scans").mkdir()
    (loose / "Scans" / "PreparationScan.stl").write_bytes(b"solid x\nendsolid x\n")
    zp = zip_package(loose, tmp_path / "loose.zip")

    r = _upload(client, PREVIEW, zp)
    assert r.status_code == 400
    assert "AliasFilenames.xml" in r.json()["detail"]
