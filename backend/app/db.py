import sqlite3
from contextlib import contextmanager

from . import config

SCHEMA = """
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS cases(
  id TEXT PRIMARY KEY,
  name TEXT NOT NULL,
  created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS assets(
  id TEXT PRIMARY KEY,
  case_id TEXT NOT NULL,
  kind TEXT NOT NULL,
  path TEXT NOT NULL,
  original_name TEXT NOT NULL,
  mesh_info_json TEXT,
  created_at TEXT NOT NULL,
  FOREIGN KEY(case_id) REFERENCES cases(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS designs(
  id TEXT PRIMARY KEY,
  case_id TEXT NOT NULL,
  target_fdi INTEGER NOT NULL,
  support_type TEXT NOT NULL,
  generator_mode TEXT NOT NULL DEFAULT 'template',
  parameters_json TEXT NOT NULL,
  margin_json TEXT,
  axis_json TEXT,
  landmarks_json TEXT,
  outer_path TEXT,
  cutter_path TEXT,
  final_path TEXT,
  qc_json TEXT,
  status TEXT NOT NULL DEFAULT 'created',
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  current_margin_revision_id TEXT,
  approved_by TEXT,
  FOREIGN KEY(case_id) REFERENCES cases(id) ON DELETE CASCADE
);

-- 设计修订链。修订不可变：每次变更插入新行，不更新旧行。
-- 存在理由：旧版 suggest-margin 会静默覆盖 designs.margin_json，
-- 而人工绘制的 Margin 前端从不落库（api.saveMargin 是死代码）。
CREATE TABLE IF NOT EXISTS design_revisions(
  id TEXT PRIMARY KEY,
  design_id TEXT NOT NULL,
  revision_no INTEGER NOT NULL,
  kind TEXT NOT NULL,
  algorithm_version TEXT,
  parameter_set_json TEXT,
  margin_source TEXT,
  operator TEXT,
  approved_by TEXT,
  operator_actions_json TEXT,
  artifacts_json TEXT,
  payload_json TEXT NOT NULL,
  created_at TEXT NOT NULL,
  UNIQUE(design_id, revision_no),
  FOREIGN KEY(design_id) REFERENCES designs(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_design_revisions_design
  ON design_revisions(design_id, revision_no);
"""

#: 建表之后才加入的列。SQLite 的 CREATE TABLE IF NOT EXISTS 不会给已存在的表补列，
#: 因此已有数据库需要显式迁移（工作区里已有一个含病例数据的 crownflow.sqlite3）。
_MIGRATIONS: tuple[tuple[str, str, str], ...] = (
    ("designs", "sculpt_state_json", "TEXT"),
    ("designs", "indication", "TEXT"),
    ("designs", "support_conditioning_json", "TEXT"),
    ("designs", "autofit_frame_json", "TEXT"),
    ("designs", "design_revision", "INTEGER NOT NULL DEFAULT 0"),
    ("designs", "confirmed_margin_revision_id", "TEXT"),
    ("designs", "current_axis_revision_id", "TEXT"),
    ("designs", "current_profile_revision_id", "TEXT"),
    ("cases", "order_json", "TEXT"),
    ("designs", "order_source_json", "TEXT"),
    ("designs", "neighbors_json", "TEXT"),
    ("designs", "bite_json", "TEXT"),
    ("designs", "outer_dependency_digest", "TEXT"),
    ("designs", "outer_output_sha256", "TEXT"),
    ("designs", "profile_json", "TEXT"),
    ("designs", "final_dependency_digest", "TEXT"),
    ("designs", "current_margin_revision_id", "TEXT"),
    ("designs", "approved_by", "TEXT"),
)


def _migrate(conn: sqlite3.Connection) -> None:
    for table, column, decl in _MIGRATIONS:
        existing = {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}
        if column not in existing:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {decl}")


def _db_path():
    """运行时读取路径，而不是 import 时绑定。

    这样测试可以 monkeypatch ``config.DB_PATH`` 把数据库指向临时目录，
    不会碰到工作区里的真实病例库。
    """
    return config.DB_PATH


def init_db():
    with sqlite3.connect(_db_path()) as conn:
        conn.executescript(SCHEMA)
        _migrate(conn)
        conn.execute("""CREATE TRIGGER IF NOT EXISTS design_version_on_update
            AFTER UPDATE ON designs WHEN NEW.design_revision=OLD.design_revision
            BEGIN UPDATE designs SET design_revision=OLD.design_revision+1 WHERE id=NEW.id; END""")

@contextmanager
def db():
    conn = sqlite3.connect(_db_path())
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()
