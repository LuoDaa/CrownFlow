"""Static 1.0 UI contract checks.

The frontend has no JS test runner in this repository.  These lightweight source
checks keep two product-critical semantics from silently regressing:
* Boolean cutter must never become the displayed intaglio.
* one-click generation must not immediately launch a second legacy finalization
  pass after AUTO acceptance.
"""
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]


def _source(relative: str) -> str:
    return (ROOT / relative).read_text(encoding="utf-8")


def test_viewer_never_selects_boolean_cutter_as_candidate_inner():
    source = _source("frontend/src/app/ViewerStage.jsx")
    selector = source.split("const candidateInnerLabel", 1)[1].split("const candidateInner", 1)[0]
    assert "intaglio_surface" in selector
    assert "intaglio_preview" in selector
    assert "candidateArtifacts.cutter" not in selector
    assert "candidateInner || semanticInner || wf.innerCrown" in source


def test_one_click_generation_does_not_run_second_finalize_pass():
    source = _source("frontend/src/app/pages/SetupPage.jsx")
    body = source.split("const generateCompleteCrown", 1)[1].split("return <div", 1)[0]
    assert "generateInitialCrown" in body
    assert "confirmCrownCandidate" in body
    assert "adapter.finalize" not in body
    assert "Hollow stitching is deferred to export" in body
