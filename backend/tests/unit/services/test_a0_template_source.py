"""A0 回归：**唯一 Template Source Truth**。

对应审计的 P0-2 —— 用户确认了牙库 A，真正参与生成的几何绝不能来自 B。
这里直接测 ``resolve_template_source`` 的四种语义：

1. 有确认 + 字节未变 → ``USER_SELECTED``，sha256 = **磁盘真实字节**的哈希；
2. 有确认但文件被改 → **TEMPLATE_HASH_MISMATCH 拒绝**（不许静默换模板）；
3. 有确认但文件不存在 → ``TEMPLATE_SOURCE_MISSING``；
4. 无确认 + 设计目录有默认件 → ``FALLBACK_DESIGN_DIR``（如实标注是回退，不是用户选择）。
"""
from __future__ import annotations

import json
import hashlib
from pathlib import Path

import pytest

from app import config
from app.services.template_source import (
    FALLBACK_DESIGN_DIR, USER_SELECTED, TemplateSourceError, resolve_template_source,
)


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _design(tmp_path, fdi=46, landmarks=None):
    return {"target_fdi": fdi, "landmarks_json": json.dumps(landmarks or {}, ensure_ascii=False)}


def _selected(path: Path, **extra):
    return {"_selected_template": {"source_path": str(path), "sha256": _sha(path),
                                   "library": None, "proposal_id": "prop-1",
                                   "geometry_frame_revision_id": "rev-frame-1", **extra}}


def test_selected_template_is_authoritative_and_hashed_from_disk(tmp_path):
    a = tmp_path / "A.stl"
    a.write_bytes(b"AAA-template-A")
    resolved = resolve_template_source(_design(tmp_path, landmarks=_selected(a)))
    assert resolved.source_kind == USER_SELECTED
    assert resolved.path == a
    assert resolved.sha256 == _sha(a), "sha256 必须来自磁盘真实字节，而不是记录值"
    assert resolved.proposal_id == "prop-1"
    assert resolved.selection_revision_id == "rev-frame-1"
    assert resolved.as_report()["source_kind"] == USER_SELECTED


def test_modified_selected_template_fails_loud(tmp_path):
    a = tmp_path / "A.stl"
    a.write_bytes(b"AAA-template-A")
    record = _selected(a)
    a.write_bytes(b"BBB-tampered-B")          # 记录之后被改
    with pytest.raises(TemplateSourceError) as err:
        resolve_template_source(_design(tmp_path, landmarks=record))
    assert err.value.code == "TEMPLATE_HASH_MISMATCH"


def test_missing_selected_template_fails_loud(tmp_path):
    record = {"_selected_template": {"source_path": str(tmp_path / "gone.stl"),
                                     "sha256": "0" * 64}}
    with pytest.raises(TemplateSourceError) as err:
        resolve_template_source(_design(tmp_path, landmarks=record))
    assert err.value.code == "TEMPLATE_SOURCE_MISSING"


def test_fallback_is_labelled_and_prefers_design_dir(tmp_path, monkeypatch):
    design_dir = tmp_path / "templates"
    design_dir.mkdir()
    default = design_dir / "FDI_46.stl"
    default.write_bytes(b"default-template")
    monkeypatch.setattr(config, "TEMPLATE_DIR", design_dir)
    resolved = resolve_template_source(_design(tmp_path))     # 无确认记录
    assert resolved.source_kind == FALLBACK_DESIGN_DIR
    assert resolved.path == default
    assert resolved.sha256 == _sha(default)
    assert not resolved.is_user_selected


def test_no_template_anywhere_reports_no_template_available(tmp_path, monkeypatch):
    (tmp_path / "templates").mkdir()
    monkeypatch.setattr(config, "TEMPLATE_DIR", tmp_path / "templates")
    monkeypatch.setattr(config, "DATA_DIR", tmp_path / "data")
    monkeypatch.setattr(config, "TOOTH_LIBRARY_ROOT", tmp_path / "nolib")
    with pytest.raises(TemplateSourceError) as err:
        resolve_template_source(_design(tmp_path))
    assert err.value.code == "NO_TEMPLATE_AVAILABLE"
