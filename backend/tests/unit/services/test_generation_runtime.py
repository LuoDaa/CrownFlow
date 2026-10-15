import json

import trimesh

import app.services.generation_runtime as runtime


def test_intaglio_cache_ignores_contact_targets_but_tracks_gap_settings():
    base = dict(
        support_sha256="support",
        margin=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        axis=[0.0, 0.0, 1.0],
        prepared_support={"status": "READY"},
    )
    a = runtime.intaglio_cache_payload(
        **base,
        parameters={"cement_gap_mm": 0.06, "proximal_target_mm": -0.02, "occlusal_target_mm": 0.02},
    )
    b = runtime.intaglio_cache_payload(
        **base,
        parameters={"cement_gap_mm": 0.06, "proximal_target_mm": -0.05, "occlusal_target_mm": 0.08},
    )
    c = runtime.intaglio_cache_payload(
        **base,
        parameters={"cement_gap_mm": 0.08, "proximal_target_mm": -0.02, "occlusal_target_mm": 0.02},
    )
    assert runtime.stable_key(a) == runtime.stable_key(b)
    assert runtime.stable_key(a) != runtime.stable_key(c)


def test_disk_mesh_cache_reuses_invariant_geometry(tmp_path, monkeypatch):
    monkeypatch.setattr(runtime, "CASE_DIR", tmp_path)
    calls = {"count": 0}

    def builder():
        calls["count"] += 1
        return trimesh.creation.box(extents=[1.0, 2.0, 3.0]), {"build": calls["count"]}

    payload = {"version": 1, "source": "same"}
    first, first_meta, first_hit, first_path = runtime.cached_mesh("case", "demo", payload, builder)
    second, second_meta, second_hit, second_path = runtime.cached_mesh("case", "demo", payload, builder)

    assert calls["count"] == 1
    assert first_hit is False
    assert second_hit is True
    assert first_path == second_path
    assert first_meta == second_meta == {"build": 1}
    assert first is not second
    assert first.vertices.shape == second.vertices.shape


def test_generation_trace_is_append_only_and_keeps_stage_seconds(tmp_path, monkeypatch):
    monkeypatch.setattr(runtime, "CASE_DIR", tmp_path)
    trace = runtime.GenerationTrace("case", "design", baseline_sha256="baseline")
    with trace.span("PROXIMAL_SOLVE"):
        pass
    trace.cache("intaglio_preview", True, "cache-key")
    snapshot = trace.snapshot(tmp_path / "trace.json", extra={"candidate_status": "PASS_ENGINEERING"})

    lines = [json.loads(line) for line in trace.audit_path.read_text(encoding="utf-8").splitlines()]
    assert any(event["stage"] == "PROXIMAL_SOLVE" and event["status"] == "DONE" for event in lines)
    assert any(event["stage"] == "CACHE_INTAGLIO_PREVIEW" and event["status"] == "HIT" for event in lines)
    assert lines[-1]["stage"] == "RUN_END"

    payload = json.loads(snapshot.read_text(encoding="utf-8"))
    assert payload["summary"]["stage_elapsed_s"]["PROXIMAL_SOLVE"] >= 0.0
    assert payload["summary"]["cache_hits"] >= 1
    assert payload["extra"]["candidate_status"] == "PASS_ENGINEERING"
