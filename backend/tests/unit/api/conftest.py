"""API 层测试的隔离 fixture。

**必须同时改绑三处 ``CASE_DIR``/``TEMPLATE_DIR``**：

``app.db`` 是运行时读 ``config.DB_PATH`` 的，monkeypatch ``config`` 就够；
但 ``routes.py`` / ``services.three_shape_import`` 写的是
``from ..config import CASE_DIR`` —— import 时就把名字绑死了，
只改 ``config.CASE_DIR`` 它们仍指向工作区里的 ``backend/data/cases``。
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app import config, db
from app.api import routes
from app.services import three_shape_import


@pytest.fixture()
def iso(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    cases = data_dir / "cases"
    templates = data_dir / "templates"
    cases.mkdir(parents=True)
    templates.mkdir(parents=True)

    monkeypatch.setattr(config, "DATA_DIR", data_dir)
    monkeypatch.setattr(config, "CASE_DIR", cases)
    monkeypatch.setattr(config, "TEMPLATE_DIR", templates)
    monkeypatch.setattr(config, "DB_PATH", data_dir / "test.sqlite3")

    monkeypatch.setattr(routes, "CASE_DIR", cases)
    monkeypatch.setattr(routes, "TEMPLATE_DIR", templates)
    monkeypatch.setattr(three_shape_import, "CASE_DIR", cases)

    db.init_db()
    return data_dir


@pytest.fixture()
def client(iso):
    from app.main import app
    with TestClient(app) as c:
        yield c
