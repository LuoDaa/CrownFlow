from pathlib import Path
from types import SimpleNamespace

import numpy as np
import trimesh

from app.geometry.crown_outer.collar_body_outer import _boundary_loops
from app.v1_outer_closure_intaglio_guard import (
    _phase_metrics,
    strip_proven_artificial_basal_closure,
)


ROOT = Path(__file__).resolve().parents[2] / "app"


def _unequal_zipper(na, nb):
    a = np.arange(na, dtype=np.int64)
    b = np.arange(nb, dtype=np.int64) + na
    sa = np.linspace(0.0, 1.0, na + 1)
    sb = np.linspace(0.0, 1.0, nb + 1)
    faces = []
    i = j = 0
    while i < na or j < nb:
        next_a = sa[i + 1] if i < na else np.inf
        next_b = sb[j + 1] if j < nb else np.inf
        if next_a <= next_b:
            faces.append((a[i % na], a[(i + 1) % na], b[j % nb]))
            i += 1
        else:
            faces.append((a[i % na], b[(j + 1) % nb], b[j % nb]))
            j += 1
    return faces


def _closed_crown_with_artificial_basal_fan(na=192, nb=12):
    ta = np.linspace(0.0, 2.0 * np.pi, na, endpoint=False)
    tb = np.linspace(0.0, 2.0 * np.pi, nb, endpoint=False)
    bottom = np.c_[4.0 * np.cos(ta), 4.0 * np.sin(ta), np.full(na, 0.8)]
    top = np.c_[3.2 * np.cos(tb), 3.2 * np.sin(tb), np.full(nb, 6.0)]
    # Deliberately place the basal fan apex near one side of the Margin.  This is
    # the same topology pattern seen in closed morphology-prior crown templates:
    # one extraordinary-valence vertex seals the entire basal opening.
    basal_apex = np.array([[3.3, 0.0, 0.25]])
    top_center = np.array([[0.0, 0.0, 6.4]])
    vertices = np.vstack([bottom, top, basal_apex, top_center])
    basal_id = na + nb
    top_id = basal_id + 1
    faces = _unequal_zipper(na, nb)
    for i in range(na):
        j = (i + 1) % na
        faces.append((i, basal_id, j))  # outward normal points down
    for i in range(nb):
        j = (i + 1) % nb
        faces.append((na + i, na + j, top_id))
    return trimesh.Trimesh(vertices, np.asarray(faces, np.int64), process=False)


def test_proven_template_basal_fan_is_removed_only_from_derived_w3_skin():
    source = _closed_crown_with_artificial_basal_fan()
    source_vertices = np.asarray(source.vertices).copy()
    source_faces = np.asarray(source.faces).copy()
    assert source.is_watertight
    assert len(_boundary_loops(source.faces)) == 0

    t = np.linspace(0.0, 2.0 * np.pi, 128, endpoint=False)
    margin = np.c_[3.5 * np.cos(t), 3.5 * np.sin(t), np.zeros(128)]
    curtain = SimpleNamespace(
        points=margin,
        frame=SimpleNamespace(z=np.array([0.0, 0.0, 1.0])),
    )

    derived, report = strip_proven_artificial_basal_closure(source, curtain)

    assert report["status"] == "REMOVED_DERIVED_W3_ONLY"
    assert report["candidate_apex_valence"] == 192
    assert report["derived_component_count"] == 1
    assert report["derived_boundary_loop_count"] == 1
    assert len(_boundary_loops(derived.faces)) == 1
    assert not derived.is_watertight

    # The accepted outer is a morphology asset and must remain byte-for-byte
    # unchanged by the W3/intaglio guard.
    assert np.array_equal(np.asarray(source.vertices), source_vertices)
    assert np.array_equal(np.asarray(source.faces), source_faces)


def test_partial_outer_loop_cannot_masquerade_as_full_margin_cycle():
    t = np.linspace(0.0, 2.0 * np.pi, 128, endpoint=False)
    margin = np.c_[3.5 * np.cos(t), 3.5 * np.sin(t), np.zeros(128)]
    # Only one side of the tooth: a short arc deliberately resembling the bad
    # FDI36 transition loop that used to be stretched over all 128 Margin points.
    u = np.linspace(-0.45 * np.pi, 0.45 * np.pi, 96, endpoint=False)
    partial = np.c_[3.8 * np.cos(u), 3.8 * np.sin(u), np.full(96, 0.8)]
    phase = _phase_metrics(partial, margin)
    assert phase["max_mm"] > 3.5


def test_guard_loads_after_margin_owned_policy_but_before_api_graph():
    source = (ROOT / "main.py").read_text(encoding="utf-8")
    policy = source.index("v1_margin_owned_intaglio")
    guard = source.index("v1_outer_closure_intaglio_guard")
    api_graph = source.index("from .api.crown")
    assert policy < guard < api_graph


def test_design_select_view_uses_completed_crown_instead_of_closed_raw_outer():
    source = (
        ROOT.parents[1] / "frontend" / "src" / "app" / "ViewerStage.jsx"
    ).read_text(encoding="utf-8")
    assert 'mode === "outer"' in source
    assert 'wf.designTool === "select"' in source
    assert 'const renderMode = stableCompletePreview ? "final" : mode' in source
    assert 'const renderEditorScene = stableCompletePreview ? null : editorScene' in source
