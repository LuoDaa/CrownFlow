"""CrownOuter 流水线的**后台任务**运行器（线程 + 落盘 manifest）。

为什么需要：整链（H.1 → 接触 → 咬合 → 审计）在真实例上是分钟级，HTTP 请求等不起。
任务状态既在内存（进度实时）也落盘 `manifest.json`（重启后历史仍在、可复盘）。

用法::

    rid = start_run(bundle_dir, stages=["contact","occlusion"], params={...},
                    runs_root=Path("data/crown_runs"))
    ... 轮询 get_run(rid) / list_runs()
"""

from __future__ import annotations

import json
import threading
import time
import traceback
import uuid
from pathlib import Path

from .crown_outer_pipeline import (
    EXPERIMENTAL_STAGES,
    MAIN_STAGES,
    PipelineError,
    _STAGE_FN,
    load_bundle,
    merged_params,
    resolve_stages,
)

_LOCK = threading.Lock()
_RUNS: dict[str, dict] = {}


def _runs_root(root: Path | str | None = None) -> Path:
    if root is not None:
        return Path(root)
    from .. import config
    return Path(config.CROWN_RUNS_DIR)


def _persist(state: dict):
    d = Path(state["run_dir"])
    d.mkdir(parents=True, exist_ok=True)
    tmp = d / "manifest.json.tmp"
    tmp.write_text(json.dumps(state, ensure_ascii=False, indent=1),
                   encoding="utf-8")
    tmp.replace(d / "manifest.json")


def _log(state: dict, msg: str):
    state["log"].append({"at": time.strftime("%H:%M:%S"), "msg": msg})


def _execute(state: dict, bundle_dir: Path, stages: list[str], params: dict):
    out_dir = Path(state["run_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    prev: dict = {}
    try:
        bundle = load_bundle(bundle_dir)
        state["case_uuid"] = bundle.case_uuid
        _log(state, "载入病例包 %s（FDI %d，%d 个 STL 资产）"
             % (bundle.case_uuid, bundle.fdi,
                sum(x is not None for x in (bundle.prep_path, bundle.die_path,
                                            bundle.mesial_path, bundle.distal_path,
                                            bundle.antagonist_path))))
        # 已完成的阶段产物可从磁盘续跑（resume）
        for s in stages:
            if s in state["stages"] and state["stages"][s].get("artifact") \
                    and Path(state["stages"][s]["artifact"]).exists():
                prev[s] = state["stages"][s]
                _log(state, "阶段 %s 已有产物，跳过" % s)
                continue
            state["current_stage"] = s
            state["stage_status"][s] = "running"
            t0 = time.time()
            _log(state, "▶ %s 开始" % s)
            _persist(state)
            try:
                res = _STAGE_FN[s](bundle, out_dir, params, prev)
            except Exception as exc:                          # noqa: BLE001
                state["stage_status"][s] = "failed"
                state["stages"][s] = {"error": "%s: %s" % (type(exc).__name__, exc),
                                      "traceback": traceback.format_exc()[-2000:]}
                _log(state, "✗ %s 失败：%s" % (s, exc))
                raise PipelineError("阶段 %s 失败：%s" % (s, exc)) from exc
            res["elapsed_s"] = round(time.time() - t0, 1)
            state["stages"][s] = res
            state["stage_status"][s] = "ok"
            prev[s] = res
            _log(state, "✓ %s 完成（%.1fs）%s" % (
                s, res["elapsed_s"],
                "｜" + str(res.get("status")) if res.get("status") else ""))
            _persist(state)
            # **Gate 纪律（用户 2026-10-08 规格）**：模块不过 Gate，下游一律不跑。
            # 阶段只要给出 gate.status 且不是 PASS，就在这里停住并如实记账，
            # 而不是让下游拿一个已知不合格的输入继续算。
            g = res.get("gate") or {}
            if g.get("status") and g["status"] != "PASS":
                state["status"] = "blocked_by_gate"
                state["blocked_at"] = s
                state["blocked_reason"] = g
                _log(state, "⛔ %s 未过 Gate（%s）—— 下游阶段一律不跑"
                     % (s, g["status"]))
                _persist(state)
                return
        state["status"] = "done"
        _log(state, "流水线完成")
    except Exception as exc:                                  # noqa: BLE001
        state["status"] = "failed"
        state["error"] = str(exc)
        _log(state, "✗ 流水线失败：%s" % exc)
    finally:
        state["current_stage"] = None
        state["finished_at"] = time.time()
        _persist(state)


def start_run(bundle_dir: Path | str, *, stages=None, params=None,
              runs_root: Path | str | None = None,
              design_id: str | None = None) -> dict:
    """起一个后台任务，返回其状态快照（含 run_id）。"""
    stages = resolve_stages(stages)
    p, provenance = merged_params(params)
    run_id = time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
    run_dir = _runs_root(runs_root) / run_id
    state = {
        "run_id": run_id,
        "design_id": design_id,
        "bundle_dir": str(bundle_dir),
        "run_dir": str(run_dir),
        "stages_requested": stages,
        "main_stages": [s for s in stages if s in MAIN_STAGES],
        "experimental_stages": [s for s in stages if s in EXPERIMENTAL_STAGES],
        "params": p,
        "params_provenance": provenance,
        "status": "running",
        "stage_status": {},
        "stages": {},
        "current_stage": None,
        "log": [],
        "started_at": time.time(),
    }
    with _LOCK:
        _RUNS[run_id] = state
    _persist(state)
    th = threading.Thread(target=_execute, args=(state, Path(bundle_dir),
                                                 stages, p), daemon=True)
    th.start()
    return state


def get_run(run_id: str, runs_root: Path | str | None = None) -> dict | None:
    with _LOCK:
        if run_id in _RUNS:
            return _RUNS[run_id]
    p = _runs_root(runs_root) / run_id / "manifest.json"
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return None


def list_runs(runs_root: Path | str | None = None) -> list[dict]:
    """历史任务（磁盘上的 manifest，最新在前）。"""
    root = _runs_root(runs_root)
    out = []
    if root.is_dir():
        for d in sorted(root.iterdir(), reverse=True):
            f = d / "manifest.json"
            if f.exists():
                try:
                    s = json.loads(f.read_text(encoding="utf-8"))
                except Exception:                             # noqa: BLE001
                    continue
                out.append({"run_id": s.get("run_id", d.name),
                            "case_uuid": s.get("case_uuid"),
                            "status": s.get("status"),
                            "stages": list(s.get("stages", {})),
                            "started_at": s.get("started_at")})
    return out


def artifact_path(run_id: str, name: str,
                  runs_root: Path | str | None = None) -> Path:
    """取产物路径（**限定在 run 目录内**，防目录穿越）。"""
    root = (_runs_root(runs_root) / run_id).resolve()
    p = (root / name).resolve()
    # 不能用字符串前缀：`/runs/r10/x` 以 `/runs/r1` 为前缀 ⇒ r1 可读到兄弟 run。
    if not p.is_relative_to(root):
        raise PipelineError("非法产物路径")
    if not p.exists():
        raise PipelineError("产物不存在：%s" % name)
    return p
