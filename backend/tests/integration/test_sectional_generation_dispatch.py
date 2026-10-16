from types import SimpleNamespace
import json

import numpy as np
import trimesh


def test_full_auto_uses_sectional_red_green_blue_baseline(tmp_path, monkeypatch):
    """BOTH context must bypass template placement and feed Sectional v1 into fine/W3 AutoFit."""
    from app.services import generation_workflow as workflow
    from app.api import routes, auto_fit
    from app.geometry import io as geometry_io
    from app.geometry import prep_margin_diagnostics
    from app.geometry import nonplanar_crown
    from app.geometry.crown_outer.patient_frame import PatientFrame
    from app.services import abutment_support, auto_fit as auto_fit_service
    from app.services import sectional_generation, contact_measurement

    support_path = tmp_path / "support.stl"
    support_mesh = trimesh.creation.cylinder(radius=2.0, height=4.0, sections=24)
    support_mesh.export(support_path)
    sectional_path = tmp_path / "sectional.stl"
    sectional_mesh = trimesh.creation.cylinder(radius=3.2, height=6.0, sections=32)
    sectional_mesh.export(sectional_path)
    sectional_report_path = tmp_path / "sectional_report.json"
    sectional_report_path.write_text("{}", encoding="utf-8")
    fine_path = tmp_path / "candidate.stl"
    sectional_mesh.export(fine_path)

    margin = [[3.0 * np.cos(t), 2.5 * np.sin(t), 0.0]
              for t in np.arange(32) * 2.0 * np.pi / 32]
    margin_rev = SimpleNamespace(payload={"points": margin}, id="margin-rev")
    axis_rev = SimpleNamespace(payload={"axis": [0.0, 0.0, 1.0]}, id="axis-rev")
    profile_rev = SimpleNamespace(payload={}, id="profile-rev", parameter_set={})
    frame = PatientFrame(
        origin=np.zeros(3), z=np.array([0.0, 0.0, 1.0]),
        x=np.array([1.0, 0.0, 0.0]), y=np.array([0.0, 1.0, 0.0]),
        revision_id="frame-rev")
    d = {
        "id": "design-26", "case_id": "case-26", "target_fdi": 26,
        "support_type": "prep",
        "parameters_json": json.dumps({"proximal_target_mm": 0.02,
                                       "occlusal_target_mm": 0.05,
                                       "minimum_thickness_mm": 0.5}),
        "autofit_frame_json": json.dumps({"revision_id": "frame-rev"}),
    }

    monkeypatch.setattr(workflow, "eligibility", lambda _d: {
        "blockers": [], "automatic_mode": "BOTH", "sectional_v1_ready": True})
    monkeypatch.setattr(workflow, "confirmed_inputs", lambda _d: {
        "margin": margin_rev, "axis": axis_rev, "profile": profile_rev})
    monkeypatch.setattr(routes, "asset", lambda _case, kind: {
        "id": "support-asset", "path": str(support_path)} if kind == "support" else None)
    monkeypatch.setattr(geometry_io, "load_mesh", lambda _path: support_mesh.copy())
    monkeypatch.setattr(prep_margin_diagnostics, "diagnose", lambda *a, **k: ({"codes": []}, None))
    monkeypatch.setattr(nonplanar_crown, "fitting_support_above_margin",
                        lambda *a, **k: (support_mesh.copy(), {"status": "OK"}))
    monkeypatch.setattr(abutment_support, "prepare_design_support", lambda _d: (None, None))
    monkeypatch.setattr(auto_fit, "frame_from", lambda _d: (frame, {"revision_id": "frame-rev"}))

    called = {"sectional": 0, "fine": 0}

    def fake_sectional(_d, got_frame, frame_ref, got_support, got_margin):
        called["sectional"] += 1
        assert got_frame is frame
        assert frame_ref["revision_id"] == "frame-rev"
        assert len(got_margin) == 32
        return sectional_mesh.copy(), {
            "status": "PASS_ENGINEERING",
            "template_source": {"source_kind": "WEAK_ANATOMY_NOT_USED_FOR_ENVELOPE"},
            "placement_path": "SECTIONAL_RED_GREEN_BLUE_V1",
        }, {"outer": str(sectional_path), "sectional_report": str(sectional_report_path)}

    def fake_propose(_d, got_frame, frame_ref, mode, baseline_path=None):
        called["fine"] += 1
        assert got_frame is frame
        assert mode == "BOTH"
        assert str(baseline_path) == str(sectional_path)
        return sectional_mesh.copy(), {
            "status": "PASS_ENGINEERING",
            "stages": [
                {"stage": "PROXIMAL", "status": "PASS_ENGINEERING"},
                {"stage": "OCCLUSAL", "status": "PASS_ENGINEERING"},
            ],
            "w3_preflight": {"status": "CONSTRUCTED_ENGINEERING"},
        }, {"outer": str(fine_path)}

    monkeypatch.setattr(sectional_generation, "build_sectional_baseline", fake_sectional)
    monkeypatch.setattr(auto_fit_service, "propose", fake_propose)
    monkeypatch.setattr(contact_measurement, "measure_confirmed_inputs",
                        lambda *a, **k: {"status": "MEASURED_ENGINEERING"})
    monkeypatch.setattr(contact_measurement, "serializable_measurement", lambda value: value)

    report, artifacts = workflow.generate_candidate(d, "AUTO")

    assert called == {"sectional": 1, "fine": 1}
    assert report["generator"] == "SECTIONAL_RED_GREEN_BLUE_V1+AUTOFIT_FINE"
    assert report["placement_path"] == "SECTIONAL_RED_GREEN_BLUE_V1"
    assert report["sectional_v1"]["status"] == "PASS_ENGINEERING"
    assert artifacts["baseline"] == str(sectional_path)
    assert artifacts["sectional_baseline"] == str(sectional_path)
    assert artifacts["sectional_report"] == str(sectional_report_path)
    assert artifacts["outer"] == str(fine_path)
