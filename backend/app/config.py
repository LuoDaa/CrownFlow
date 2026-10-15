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
ANATOMY_DIR = Path(os.environ.get("CROWNFLOW_ANATOMY_DIR", ROOT.parents[1] / "32牙位基础解剖模板库" / "32位牙列-分")).resolve()
TOOTH_LIBRARY_ROOT = Path(os.environ.get('CROWNFLOW_TOOTH_LIBRARY_ROOT',ROOT.parents[1]/'32牙位基础解剖模板库'/'牙齿数据库')).resolve()
MODEL_DIR = Path(os.environ.get("CROWNFLOW_MODEL_DIR", ROOT / "ml" / "checkpoints")).resolve()
DB_PATH = DATA_DIR / "crownflow.sqlite3"

for p in (DATA_DIR, CASE_DIR, TEMPLATE_DIR, MODEL_DIR, CROWN_RUNS_DIR):
    p.mkdir(parents=True, exist_ok=True)

AI_CHECKPOINT = os.environ.get("CROWNFLOW_AI_CHECKPOINT", str(MODEL_DIR / "crown_deformer.ts") if (MODEL_DIR / "crown_deformer.ts").exists() else "")
