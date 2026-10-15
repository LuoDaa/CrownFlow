"""导入入口的 HTTP 层。

    POST /api/importers/three-shape/preview   上传 ZIP -> 解析结果面板（零副作用）
    POST /api/importers/three-shape/import    上传 ZIP -> 建病例 + 写几何
    GET  /api/cases/{id}/three-shape-import   读回已落库的导入凭据

**只收 ZIP。** 不收"服务器上的某个路径"—— 那会开出一个任意文件读取的口子，
而操作员本来拿到的就是 ZIP。

**闸门不过就 409，不建病例。** 见 ``services/three_shape_import.commit``。
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from ..db import db
from ..services.three_shape_import import (
    ImportRejected,
    commit,
    preview,
    repreview,
)

router = APIRouter(prefix="/api/importers")

#: 上传上限。真实包（两份加密扫描）只有几 MB，这里的余量给的是"解包后"的网格。
MAX_UPLOAD_BYTES = 1024 * 1024 * 1024


def _spool_zip(file: UploadFile) -> Path:
    """把上传落成临时 ZIP。调用方负责清理所在目录。"""
    name = (file.filename or "").lower()
    if not name.endswith(".zip"):
        raise HTTPException(
            400, "请上传 3Shape 订单包 ZIP（含 AliasFilenames.xml）。"
                 "单独的 .dcm/.stl 请走病例的扫描上传。")
    tmp_dir = Path(tempfile.mkdtemp(prefix="crownflow_3shape_"))
    dest = tmp_dir / "package.zip"
    total = 0
    with dest.open("wb") as out:
        while True:
            chunk = file.file.read(1024 * 1024)
            if not chunk:
                break
            total += len(chunk)
            if total > MAX_UPLOAD_BYTES:
                out.close()
                shutil.rmtree(tmp_dir, ignore_errors=True)
                raise HTTPException(413, f"包超过上限 {MAX_UPLOAD_BYTES // (1024*1024)} MB")
            out.write(chunk)
    return dest


@router.post("/three-shape/preview")
async def three_shape_preview(file: UploadFile = File(...)):
    """解析 + 验收，返回面板。**不建病例、不写网格、不落库。**"""
    zip_path = _spool_zip(file)
    try:
        return preview(zip_path)
    except ValueError as exc:
        # import_package 对"不是 3Shape 包"给出明确原因（缺 AliasFilenames.xml 等）
        raise HTTPException(400, str(exc))
    except Exception as exc:
        raise HTTPException(400, f"解析失败：{type(exc).__name__}: {exc}")
    finally:
        shutil.rmtree(zip_path.parent, ignore_errors=True)


@router.post("/three-shape/import")
async def three_shape_import(file: UploadFile = File(...),
                            case_name: str | None = Form(default=None),
                            phi_policy: str | None = Form(default=None)):
    """建病例。闸门不全过就 409，并把失败明细一起返回。

    ``phi_policy`` 可以不传（默认 ``PSEUDONYMIZED``：工作区不含原始 PHI）。
    只有确实需要保留原始订单包复核来源时，才显式传 ``IDENTIFIED``。
    """
    from ..services.phi_policy import PhiPolicyError, resolve_policy

    try:
        resolved = resolve_policy(phi_policy)
    except PhiPolicyError as exc:
        raise HTTPException(422, str(exc)) from exc
    zip_path = _spool_zip(file)
    try:
        return commit(zip_path, case_name=case_name, policy=resolved)
    except PhiPolicyError as exc:
        raise HTTPException(409, str(exc)) from exc
    except ImportRejected as exc:
        raise HTTPException(409, {
            "message": str(exc),
            "import_status": "CASE_NOT_READY",
            "validation": exc.report.as_dict(),
        })
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    except Exception as exc:
        raise HTTPException(400, f"导入失败：{type(exc).__name__}: {exc}")
    finally:
        shutil.rmtree(zip_path.parent, ignore_errors=True)


@router.get("/three-shape/records/{case_id}")
def get_record(case_id: str):
    """读回某个病例的导入凭据（含面板与闸门快照）。"""
    return repreview(case_id)


@router.get("/three-shape/records")
def list_records():
    """读回所有已落库的导入凭据（按病例）。"""
    with db() as c:
        rows = c.execute("SELECT id,name,created_at FROM cases ORDER BY created_at DESC"
                         ).fetchall()
    out = []
    for r in rows:
        rec = repreview(r["id"])
        if rec.get("has_record"):
            out.append({"case_id": r["id"], "name": r["name"],
                        "created_at": r["created_at"],
                        "prescription": rec["record"].get("prescription")})
    return out
