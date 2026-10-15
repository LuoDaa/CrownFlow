"""P0-14：模板上传必须是"临时文件 → 校验 → 原子替换"。

非法模板失败后：正式模板字节不变、仍可加载（旧实现是"先覆盖正式模板，校验失败再删"）。
"""
import hashlib
import io
import zipfile

import trimesh

from app.api import routes


def template_dir():
    """⚠️ 必须从 ``routes`` 模块现取：隔离 fixture 改的是模块里的 ``TEMPLATE_DIR``，
    在测试模块顶层 import 会绑死工作区真实路径（conftest 已经警告过这个坑）。"""
    return routes.TEMPLATE_DIR


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def valid_stl():
    return trimesh.creation.icosphere(subdivisions=2, radius=3).export(file_type="stl")


def test_invalid_template_upload_keeps_official_template(client):
    created = client.post("/api/templates/46", files={"file": ("good.stl", valid_stl(), "model/stl")})
    assert created.status_code == 200, created.text
    official = template_dir() / "FDI_46.stl"
    before = sha(official)

    broken = client.post("/api/templates/46", files={"file": ("broken.stl", b"not an stl at all", "model/stl")})
    assert broken.status_code == 400, broken.text
    assert official.is_file(), "非法上传把正式模板删掉了"
    assert sha(official) == before, "非法上传改动了正式模板字节"
    assert client.get("/api/templates/46/file").status_code == 200
    assert not list(template_dir().glob("_tmp_upload_*.stl")), "临时文件没有清理"


def test_zip_import_stages_before_replace_and_keeps_untouched_fdi(client):
    from app.api.routes import UNIVERSAL_TO_FDI
    good_fdi, bad_fdi = UNIVERSAL_TO_FDI[16], UNIVERSAL_TO_FDI[15]
    # 先给"无效条目"对应的牙位放一个合法模板，用来验证非法条目**不会**动它
    assert client.post(f"/api/templates/{bad_fdi}",
                       files={"file": ("keep.stl", valid_stl(), "model/stl")}).status_code == 200
    before = sha(template_dir() / f"FDI_{bad_fdi}.stl")

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("Tooth_16.stl", valid_stl())
        archive.writestr("Tooth_15.stl", b"corrupt payload")
    response = client.post("/api/templates/import-universal-zip",
                           files={"file": ("lib.zip", buffer.getvalue(), "application/zip")})
    assert response.status_code == 200, response.text
    body = response.json()
    assert [item["fdi"] for item in body["imported"]] == [good_fdi], body
    assert body["rejected"] and body["rejected"][0]["fdi"] == bad_fdi, body
    assert sha(template_dir() / f"FDI_{bad_fdi}.stl") == before, "无效条目改动了该牙位已有模板"
    assert not list(template_dir().glob("_tmp_zip_*.stl")), "临时文件没有清理"
    assert client.get(f"/api/templates/{good_fdi}/file").status_code == 200
    assert client.get(f"/api/templates/{bad_fdi}/file").status_code == 200
