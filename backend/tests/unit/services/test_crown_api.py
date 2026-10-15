"""CrownOuter 封装层单测：病例包解析 / 参数 provenance / 阶段依赖 / API / 路径安全。

真链路的几何计算在 ``tools/run_crown_api_e2e.py``（分钟级，跑真实例）；
这里只钉**封装语义**，不重复跑几何。
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from app.services.crown_outer_pipeline import (                # noqa: E402
    DEFAULT_PARAMS,
    EXPERIMENTAL_STAGES,
    MAIN_STAGES,
    STAGES,
    PipelineError,
    load_bundle,
    merged_params,
    resolve_stages,
)
from app.services.crown_outer_runner import (                  # noqa: E402
    artifact_path,
    get_run,
    list_runs,
    start_run,
)


def _bundle_dir(tmp_path: Path, **over) -> Path:
    d = tmp_path / "cf_testcase0000001"
    d.mkdir(parents=True, exist_ok=True)
    meta = {
        "case_uuid": d.name, "fdi": 46,
        "margin_points": [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        "insertion_axis": [0.0, 0.0, 1.0],
        "axis_mesial": [1.0, 0.0, 0.0],
        "axis_buccal": [0.0, 1.0, 0.0],
        "center": [0.0, 0.0, 0.0],
    }
    meta.update(over)
    (d / "case.json").write_text(json.dumps(meta), encoding="ascii")
    return d


def test_load_bundle_requires_margin_and_axis(tmp_path):
    d = _bundle_dir(tmp_path)
    b = load_bundle(d)
    assert b.case_uuid == d.name and b.fdi == 46
    assert b.prep_path is None and b.mesial_path is None
    bad = _bundle_dir(tmp_path / "b", margin_points=[])
    with pytest.raises(PipelineError, match="margin_points"):
        load_bundle(bad)


def test_param_provenance_marks_defaults():
    p, prov = merged_params({"min_thickness_mm": 0.5})
    assert p["min_thickness_mm"] == 0.5
    assert p["cement_gap_mm"] == DEFAULT_PARAMS["cement_gap_mm"]
    assert prov["given"] == ["min_thickness_mm"]
    assert "cement_gap_mm" in prov["defaulted"]
    assert "min_thickness_mm" not in prov["defaulted"]


def test_resolve_stages_fills_deps_and_keeps_order():
    got = resolve_stages(["occlusion"])
    # occlusion 依赖 outer_h1 + contact；内表面不依赖它，不应被塞进来
    assert got == ["outer_h1", "contact", "occlusion"]
    assert resolve_stages(["thickness_audit"]) == [
        "outer_h1", "intaglio", "contact", "occlusion", "thickness_audit"]
    assert resolve_stages(None) == list(MAIN_STAGES)
    assert "envelope_j1" in EXPERIMENTAL_STAGES
    with pytest.raises(PipelineError, match="未知阶段"):
        resolve_stages(["nope"])
    assert all(s in STAGES for s in
               resolve_stages(list(STAGES)))


def test_artifact_path_blocks_traversal(tmp_path):
    root = tmp_path / "runs"
    run = root / "r1"
    run.mkdir(parents=True)
    (run / "ok.stl").write_bytes(b"x")
    (tmp_path / "secret.txt").write_text("nope")
    assert artifact_path("r1", "ok.stl", runs_root=root).name == "ok.stl"
    for bad in ("../secret.txt", "..\\secret.txt", "sub/../../secret.txt"):
        with pytest.raises(PipelineError):
            artifact_path("r1", bad, runs_root=root)
    with pytest.raises(PipelineError, match="不存在"):
        artifact_path("r1", "missing.stl", runs_root=root)


def test_api_stages_and_cases_listing(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    from app.main import app

    monkeypatch.setattr("app.config.CROWN_CASES_DIR", tmp_path)
    import app.api.crown as crown_api
    monkeypatch.setattr(crown_api, "CASES_ROOT", tmp_path)
    d = _bundle_dir(tmp_path)
    (d / "prep.stl").write_bytes(b"solid x\n")

    c = TestClient(app)
    r = c.get("/api/crown/stages")
    assert r.status_code == 200
    body = r.json()
    assert set(body["main_stages"]) <= set(body["stages"])
    assert body["stage_labels"]["outer_h1"]

    r = c.get("/api/crown/cases")
    assert r.status_code == 200
    rows = r.json()
    assert len(rows) == 1 and rows[0]["case_uuid"] == d.name
    assert rows[0]["assets"]["prep"] is True
    assert rows[0]["assets"]["antagonist"] is False

    r = c.post("/api/crown/cases/register", json={"case_uuid": "nope"})
    assert r.status_code == 404
    r = c.post("/api/crown/runs", json={"case_uuid": "nope"})
    assert r.status_code == 404
    r = c.get("/api/crown/runs/does-not-exist")
    assert r.status_code == 404


def test_runner_persists_and_lists(tmp_path, monkeypatch):
    """跑一个**必然失败**的任务（病例包缺网格）：状态要如实失败、manifest 落盘。"""
    monkeypatch.setattr("app.config.CROWN_RUNS_DIR", tmp_path / "runs")
    d = _bundle_dir(tmp_path)
    st = start_run(d, stages=["outer_h1"], params={}, runs_root=tmp_path / "runs")
    rid = st["run_id"]
    import time
    for _ in range(60):
        cur = get_run(rid, runs_root=tmp_path / "runs")
        if cur["status"] in ("done", "failed"):
            break
        time.sleep(0.5)
    cur = get_run(rid, runs_root=tmp_path / "runs")
    assert cur["status"] == "failed"                    # 没有模板库/网格 → 必须失败
    assert cur["stages"]["outer_h1"]["error"]
    assert (tmp_path / "runs" / rid / "manifest.json").exists()
    listed = list_runs(runs_root=tmp_path / "runs")
    assert any(x["run_id"] == rid for x in listed)
