from pathlib import Path
import os

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.environ.get("CROWNFLOW_DATA_DIR", ROOT / "data")).resolve()
CASE_DIR = DATA_DIR / "cases"
#: CrownOuter 病例包仓库（真实病例：case_uuid 命名 + case.json + 各 STL）；
#: 用 CROWNFLOW_CROWN_CASES_DIR 指向数据盘，不复制、不进 Git。
CROWN_CASES_DIR = Path(os.environ.get(
    "CROWNFLOW_CROWN_CASES_DIR", ROOT / "data" / "crown_outer_cases")).resolve()
#: 后台流水线产物目录（含分层 GLB / QC JSON）。
CROWN_RUNS_DIR = Path(os.environ.get(
    "CROWNFLOW_CROWN_RUNS_DIR", DATA_DIR / "crown_runs")).resolve()
TEMPLATE_DIR = DATA_DIR / "templates"
# 模板牙库根：显式 env 优先；否则**项目目录内自带副本**（整个项目目录可直接拷到别的机器跑）；
# 都没有时回退历史位置（项目目录的上一级）。
_LIBRARY_IN_PROJECT = ROOT.parent / "32牙位基础解剖模板库"
_TEMPLATE_LIBRARY_BASE = Path(os.environ.get("CROWNFLOW_TEMPLATE_LIBRARY_ROOT")
                              or (_LIBRARY_IN_PROJECT if _LIBRARY_IN_PROJECT.is_dir()
                                  else ROOT.parents[1] / "32牙位基础解剖模板库"))
ANATOMY_DIR = Path(os.environ.get("CROWNFLOW_ANATOMY_DIR", _TEMPLATE_LIBRARY_BASE / "32位牙列-分")).resolve()
TOOTH_LIBRARY_ROOT = Path(os.environ.get('CROWNFLOW_TOOTH_LIBRARY_ROOT',_TEMPLATE_LIBRARY_BASE/'牙齿数据库')).resolve()
MODEL_DIR = Path(os.environ.get("CROWNFLOW_MODEL_DIR", ROOT / "ml" / "checkpoints")).resolve()
DB_PATH = DATA_DIR / "crownflow.sqlite3"

for p in (DATA_DIR, CASE_DIR, TEMPLATE_DIR, MODEL_DIR, CROWN_RUNS_DIR):
    p.mkdir(parents=True, exist_ok=True)

AI_CHECKPOINT = os.environ.get("CROWNFLOW_AI_CHECKPOINT", str(MODEL_DIR / "crown_deformer.ts") if (MODEL_DIR / "crown_deformer.ts").exists() else "")
