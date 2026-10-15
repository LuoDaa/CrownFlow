"""CrownOuter 流水线 API：病例包登记 / 后台任务 / 产物与分层下载。

参考 3Shape 的工作流：上传 → 后台处理 → 结果页（3D 视图 + 分层显示 + QC 报告）。
本路由只做编排，几何计算在 ``app.services.crown_outer_pipeline``（引擎全部已验收）。
"""

from __future__ import annotations

import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from .. import config
from ..db import db
from ..services.crown_outer_pipeline import (
    DEFAULT_PARAMS,
    EXPERIMENTAL_STAGES,
    MAIN_STAGES,
    STAGES,
    PipelineError,
    load_bundle,
)
from ..services.crown_outer_runner import (
    artifact_path,
    get_run,
    list_runs,
    start_run,
)

router = APIRouter(prefix="/api/crown", tags=["crown"])

#: 病例包仓库：`CROWNFLOW_CROWN_CASES_DIR`（默认 `<DATA_DIR>/crown_outer_cases`）。
#  真实病例包（case_uuid 命名、PHI-free）通常放在数据盘上，用环境变量指过去，
#  不复制、不进 Git。
CASES_ROOT = config.CROWN_CASES_DIR


class RegisterCase(BaseModel):
    case_uuid: str
    name: str | None = None


#: 病例包目录名白名单（`cf_<16hex>` 之类；禁止分隔符与 ..）
_CASE_UUID_RE = re.compile(r"[A-Za-z0-9._-]{1,64}")


class RunRequest(BaseModel):
    stages: list[str] = Field(default_factory=lambda: list(MAIN_STAGES))
    params: dict = Field(default_factory=dict)
    case_uuid: str | None = None
    design_id: str | None = None


@router.get("/stages")
def stages():
    """可选阶段与默认参数（前端据此渲染参数面板；默认值带 provenance 标注）。"""
    return {
        "stages": list(STAGES),
        "main_stages": list(MAIN_STAGES),
        "experimental_stages": list(EXPERIMENTAL_STAGES),
        "default_params": DEFAULT_PARAMS,
        "note": ("默认参数只是 fallback，不是临床建议；病例自带设计参数时"
                 "（.modelInfo MinThickness / 近颈缘规则）应以病例为准"),
        "stage_labels": {
            "outer_h1": "冠外形 H.1（模板 → Margin）",
            "intaglio": "内表面（预备体 + cement gap）",
            "contact": "邻牙接触 Contact v2 + I.2",
            "occlusion": "咬合 Occlusion v1",
            "thickness_audit": "壁厚审计 + 冲突归因",
            "envelope_j1": "J.1 包络约束（实验）",
            "collar_body": "J.2.1 Collar+Body（实验）",
        },
    }


@router.get("/cases")
def list_bundles():
    """可用的病例包（服务器上的 crown_outer_cases 目录，PHI-free 命名）。"""
    out = []
    if CASES_ROOT.is_dir():
        for d in sorted(CASES_ROOT.iterdir()):
            cj = d / "case.json"
            if not (d.is_dir() and cj.exists()):
                continue
            try:
                meta = json.loads(cj.read_text(encoding="ascii"))
            except Exception:                                  # noqa: BLE001
                meta = {}
            has = {p.name for p in d.glob("*.stl")}
            out.append({
                "case_uuid": d.name,
                "fdi": meta.get("fdi"),
                "reconstruction_type": meta.get("reconstruction_type"),
                "n_margin_points": len(meta.get("margin_points") or []),
                "assets": {
                    "prep": "prep.stl" in has, "die": "die.stl" in has,
                    "mesial": "mesial_neighbor_patch.stl" in has,
                    "distal": "distal_neighbor_patch.stl" in has,
                    "antagonist": "antagonist_patch.stl" in has,
                    "gt_crown": "final_crown.stl" in has,
                },
            })
    return out


@router.post("/cases/register")
def register_bundle(p: RegisterCase):
    """把一个病例包登记成 app 内的病例（幂等：同名不重复建）。

    PHI 纪律：只用 case_uuid 作名字，不写原文件名/患者信息。
    """
    # case_uuid 是**目录名**：必须先按白名单校验，不能把路径安全寄托在 case.json 是否存在上。
    if not _CASE_UUID_RE.fullmatch(p.case_uuid or ""):
        raise HTTPException(400, "case_uuid 只能由字母/数字/._- 组成，长度 1–64")
    d = CASES_ROOT / p.case_uuid
    if not (d / "case.json").exists():
        raise HTTPException(404, "病例包不存在：%s" % p.case_uuid)
    try:
        bundle = load_bundle(d)
    except PipelineError as exc:
        raise HTTPException(400, str(exc)) from exc
    with db() as c:
        row = c.execute("SELECT id FROM cases WHERE name=?", (p.case_uuid,)).fetchone()
        if row:
            case_id = row["id"]
        else:
            case_id = uuid4().hex
            c.execute("INSERT INTO cases(id,name,created_at) VALUES(?,?,?)",
                      (case_id, p.case_uuid,
                       datetime.now(timezone.utc).isoformat()))
    return {"case_id": case_id, "case_uuid": bundle.case_uuid,
            "fdi": bundle.fdi,
            "assets": {k: str(v) for k, v in (
                ("prep", bundle.prep_path), ("die", bundle.die_path),
                ("mesial", bundle.mesial_path), ("distal", bundle.distal_path),
                ("antagonist", bundle.antagonist_path)) if v is not None}}


@router.post("/runs")
def create_run(p: RunRequest):
    """起一个后台任务。``case_uuid`` 或 ``design_id`` 二选一。"""
    case_uuid = p.case_uuid
    if case_uuid is None and p.design_id:
        with db() as c:
            row = c.execute("SELECT case_id FROM designs WHERE id=?",
                            (p.design_id,)).fetchone()
        if not row:
            raise HTTPException(404, "设计不存在")
        with db() as c:
            crow = c.execute("SELECT name FROM cases WHERE id=?",
                             (row["case_id"],)).fetchone()
        case_uuid = crow["name"] if crow else None
    if not case_uuid:
        raise HTTPException(400, "需要 case_uuid（或带病例的 design_id）")
    d = CASES_ROOT / case_uuid
    if not (d / "case.json").exists():
        raise HTTPException(404, "病例包不存在：%s" % case_uuid)
    try:
        st = start_run(d, stages=p.stages, params=p.params,
                       design_id=p.design_id)
    except PipelineError as exc:
        raise HTTPException(400, str(exc)) from exc
    return _snapshot(st)


def _snapshot(state: dict) -> dict:
    cur = state.get("current_stage")
    done = [s for s, v in state.get("stage_status", {}).items() if v == "ok"]
    return {
        "run_id": state["run_id"],
        "case_uuid": state.get("case_uuid"),
        "status": state["status"],
        "current_stage": cur,
        "stages": state.get("stage_status", {}),
        "done": done,
        "elapsed_s": round(time.time() - state.get("started_at", 0), 1),
        "log": state.get("log", [])[-40:],
        "error": state.get("error"),
    }


@router.get("/runs")
def runs():
    return list_runs()


@router.get("/runs/{run_id}")
def run_detail(run_id: str):
    """任务快照 + 各阶段结果（前端轮询用；done 后带 QC/表格/分层）。"""
    st = get_run(run_id)
    if st is None:
        raise HTTPException(404, "任务不存在")
    snap = _snapshot(st)
    snap["params"] = st.get("params")
    snap["params_provenance"] = st.get("params_provenance")
    snap["bundle_dir"] = st.get("bundle_dir")
    out = {}
    for name, s in (st.get("stages") or {}).items():
        item = {k: v for k, v in s.items() if k != "report"}
        if name == "thickness_audit" and "report" in s:
            item["report"] = s["report"]
        out[name] = item
    snap["results"] = out
    return snap


@router.get("/runs/{run_id}/artifact/{name:path}")
def run_artifact(run_id: str, name: str):
    try:
        p = artifact_path(run_id, name)
    except PipelineError as exc:
        raise HTTPException(404, str(exc)) from exc
    media = ("model/stl" if p.suffix == ".stl"
             else "model/gltf-binary" if p.suffix == ".glb"
             else "application/json" if p.suffix == ".json"
             else "application/octet-stream")
    return FileResponse(str(p), media_type=media, filename=p.name)
