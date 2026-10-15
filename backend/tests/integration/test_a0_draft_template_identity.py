"""A0 回归：草稿生成必须消费**用户确认的那个模板**（审计 P0-2 的核心用例）。

场景（就是审计描述的最坏情况）：

    TEMPLATE_DIR/FDI_46.stl 存在（诱饵 B，很矮）
    用户确认了 A（很高，另存别处，landmarks['_selected_template']）
    → initialize_draft_outer
    → 产物必须来自 A（Z 向高度是判据：X/Y 会被颈缘 footprint 重新缩放，Z 不会）
    → 报告里的 template_sha256 必须 == sha256(A)（真实读入的字节）
"""
from __future__ import annotations

import hashlib
import json
import math

import numpy as np
import pytest
import trimesh
from fastapi.testclient import TestClient

from app import config
from app.api.routes import design_row, initialize_draft_outer
from app.db import db
from app.models import AutoOuterRequest


def _stl_bytes(mesh) -> bytes:
    return mesh.export(file_type="stl")


def _ring(n=48, a=3.0, b=2.0):
    theta = np.linspace(0, 2 * math.pi, n, endpoint=False)
    return [[a * math.cos(t), b * math.sin(t), 0.0] for t in theta]


@pytest.fixture()
def chosen_and_decoy(tmp_path, monkeypatch):
    """诱饵 B 放进设计目录；用户选择的 A 放在别处。"""
    design_dir = tmp_path / "design_templates"
    design_dir.mkdir()
    decoy = design_dir / "FDI_46.stl"
    decoy.write_bytes(_stl_bytes(trimesh.creation.box(extents=(1.2, 0.8, 2.0))))
    monkeypatch.setattr(config, "TEMPLATE_DIR", design_dir)

    chosen = tmp_path / "chosen_A.stl"
    chosen.write_bytes(_stl_bytes(trimesh.creation.box(extents=(6.0, 4.0, 12.0))))
    return chosen, decoy


def _design_with_selection(chosen):
    client = TestClient(__import__("app.main", fromlist=["app"]).app)
    case = client.post("/api/cases", json={"name": "a0-template-identity"}).json()["id"]
    design = client.post("/api/designs", json={"case_id": case, "target_fdi": 46,
                                              "support_type": "prep", "draft": True}).json()["id"]
    landmarks = {"_selected_template": {
        "source_path": str(chosen),
        "sha256": hashlib.sha256(chosen.read_bytes()).hexdigest(),
        "library": None, "proposal_id": "prop-A", "geometry_frame_revision_id": "rev-frame-A"}}
    with db() as c:
        c.execute("UPDATE designs SET landmarks_json=? WHERE id=?",
                  (json.dumps(landmarks, ensure_ascii=False), design))
    return design


def test_draft_consumes_selected_template_not_design_dir_decoy(isolated_db, chosen_and_decoy):
    chosen, decoy = chosen_and_decoy
    design = _design_with_selection(chosen)
    d = design_row(design)
    request = AutoOuterRequest(margin_points=_ring(), insertion_axis=[0.0, 0.0, 1.0])
    # A1：没有确认帧时 live 生成必须直接拒绝，不能拿世界轴凑一个方向
    from fastapi import HTTPException
    with pytest.raises(HTTPException) as refusal:
        initialize_draft_outer(d, request)
    assert refusal.value.status_code == 409 and "确认" in str(refusal.value.detail)
    # 显式 legacy/研究模式才允许 world-X 回退，且必须如实标注
    result, out = initialize_draft_outer(d, request, allow_world_frame=True)

    source = result["template_source"]
    assert source["source_kind"] == "USER_SELECTED", "有确认时必须用确认件，不能走回退链"
    assert source["sha256"] == hashlib.sha256(chosen.read_bytes()).hexdigest(), \
        "报告里的模板 sha256 必须等于真正读入的字节哈希"
    assert source["proposal_id"] == "prop-A" and source["selection_revision_id"] == "rev-frame-A"

    mesh = trimesh.load(out, process=True)
    z_extent = float(mesh.extents[2])
    decoy_z = float(trimesh.load(decoy, process=True).extents[2])
    assert z_extent > 5.0, "产物应来自高模板 A（Z≈12），实际 Z=%.2f" % z_extent
    assert abs(z_extent - decoy_z) > 3.0, "产物明显不来自诱饵 B（Z≈2）"
    assert result["frame_source"] == "WORLD_X_FALLBACK_EXPLICIT"
    assert result["placement_path"] == "INITIALIZED_ONCE"


def test_tampered_selected_template_blocks_draft(isolated_db, chosen_and_decoy):
    chosen, _ = chosen_and_decoy
    design = _design_with_selection(chosen)
    chosen.write_bytes(_stl_bytes(trimesh.creation.box(extents=(6.0, 4.0, 11.0))))  # 字节变了

    from fastapi import HTTPException
    d = design_row(design)
    with pytest.raises(HTTPException) as err:
        initialize_draft_outer(d, AutoOuterRequest(margin_points=_ring(),
                                                  insertion_axis=[0.0, 0.0, 1.0]),
                               allow_world_frame=True)
    assert "TEMPLATE_HASH_MISMATCH" in str(err.value.detail)
