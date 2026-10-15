"""集成测试 fixture：把数据库与数据目录指到临时路径。

**绝不允许碰工作区里的真实病例库**（backend/data/crownflow.sqlite3 含真实病例记录）。
"""

from __future__ import annotations

import pytest

from app import config


@pytest.fixture()
def isolated_db(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    (data_dir / "cases").mkdir(parents=True)
    (data_dir / "templates").mkdir(parents=True)
    monkeypatch.setattr(config, "DATA_DIR", data_dir)
    monkeypatch.setattr(config, "CASE_DIR", data_dir / "cases")
    monkeypatch.setattr(config, "TEMPLATE_DIR", data_dir / "templates")
    monkeypatch.setattr(config, "DB_PATH", data_dir / "test.sqlite3")

    from app.db import init_db
    init_db()
    # Modules imported before this fixture hold copied path constants; isolate those too.
    import sys
    for name,module in list(sys.modules.items()):
        if name.startswith('app.'):
            for key,relative in (('CASE_DIR','cases'),('TEMPLATE_DIR','templates'),('DATA_DIR','')):
                if hasattr(module,key):monkeypatch.setattr(module,key,data_dir/relative)
    return data_dir
