"""Raw scan-pair triage for private Crown 1.0 regression cases.

This module deliberately does *not* decide a clinical tooth identity or a margin.
It only separates connected components and identifies a plausible detached
prep/die component when the scanner export contains one.  The heuristic is
intended for local regression tooling, not for live automatic tooth localization.

Why this exists
---------------
Real scanner exports often contain a full arch plus a detached die/prep mesh and
small annotation specks in the same STL.  Feeding the whole STL to prep/margin
code makes the downstream failure hard to attribute.  Triage gives the caller a
small, explicit report and either a detached support candidate or ``None``.
"""
from __future__ import annotations

import numpy as np


DEFAULT_MIN_MAJOR_FACES = 100
DEFAULT_MIN_MAJOR_FRACTION = 0.005
MIN_TOOTH_EXTENT_MM = 2.0
MAX_TOOTH_EXTENT_MM = 20.0
MAX_SECONDARY_FACE_RATIO = 0.75
MIN_COMPACTNESS = 0.20
REFERENCE_MEDIAN_EXTENT_MM = 8.0


def component_record(mesh, rank: int) -> dict:
    """Return a JSON-safe geometric summary without retaining patient vertices."""
    ext = np.asarray(mesh.extents, dtype=float)
    return {
        "rank": int(rank),
        "faces": int(len(mesh.faces)),
        "vertices": int(len(mesh.vertices)),
        "watertight": bool(mesh.is_watertight),
        "extents_mm": [float(v) for v in ext],
        "max_extent_mm": float(ext.max()) if len(ext) else 0.0,
        "min_extent_mm": float(ext.min()) if len(ext) else 0.0,
    }


def major_components(mesh, *, min_faces: int = DEFAULT_MIN_MAJOR_FACES,
                     min_fraction: float = DEFAULT_MIN_MAJOR_FRACTION):
    """Split a scan and keep connected components large enough for regression use."""
    components = sorted(
        mesh.split(only_watertight=False),
        key=lambda part: len(part.faces),
        reverse=True,
    )
    threshold = max(int(min_faces), int(np.ceil(len(mesh.faces) * float(min_fraction))))
    major = [part for part in components if len(part.faces) >= threshold]
    return major, components, threshold


def support_candidate_score(mesh, *, largest_faces: int) -> float | None:
    """Score a secondary component as a detached prep/die candidate.

    The bounds are intentionally broad and are *engineering triage* bounds only.
    They are not anatomical acceptance criteria.  Returning ``None`` means the
    component should not be auto-selected by the private regression runner.
    """
    ext = np.sort(np.asarray(mesh.extents, dtype=float))
    if len(ext) != 3 or not np.all(np.isfinite(ext)):
        return None
    if ext[0] < MIN_TOOTH_EXTENT_MM or ext[-1] > MAX_TOOTH_EXTENT_MM:
        return None

    ratio = float(len(mesh.faces)) / max(float(largest_faces), 1.0)
    if ratio > MAX_SECONDARY_FACE_RATIO:
        return None

    compactness = float(ext[0] / max(ext[-1], 1e-9))
    if compactness < MIN_COMPACTNESS:
        return None

    score = 0.0
    score += 4.0 if mesh.is_watertight else 0.0
    score += min(2.0, 5.0 * ratio)
    score += min(1.0, compactness)
    score -= abs(float(np.median(ext)) - REFERENCE_MEDIAN_EXTENT_MM) / REFERENCE_MEDIAN_EXTENT_MM
    return float(score)


def select_separate_support(mesh) -> tuple[object | None, dict]:
    """Return ``(candidate_mesh, report)`` for a scanner STL.

    The largest major component is treated as the arch.  Only later major
    components are eligible for detached-support selection; tiny specks are
    still counted in ``component_count`` but cannot become the candidate.
    """
    major, components, threshold = major_components(mesh)
    report = {
        "component_count": int(len(components)),
        "major_component_count": int(len(major)),
        "major_face_threshold": int(threshold),
        "major_components": [component_record(part, rank) for rank, part in enumerate(major)],
        "status": "NOT_FOUND",
        "selected_rank": None,
        "score": None,
    }

    if len(major) < 2:
        report["reason"] = "NO_SECOND_MAJOR_COMPONENT"
        return None, report

    largest_faces = len(major[0].faces)
    scored = []
    for rank, part in enumerate(major[1:], start=1):
        score = support_candidate_score(part, largest_faces=largest_faces)
        if score is not None:
            scored.append((score, rank, part))

    if not scored:
        report["reason"] = "NO_TOOTH_SCALE_SECONDARY_COMPONENT"
        return None, report

    score, rank, part = max(scored, key=lambda item: item[0])
    report.update(
        status="CANDIDATE_FOUND",
        selected_rank=int(rank),
        score=float(score),
        reason="SECONDARY_TOOTH_SCALE_COMPONENT",
    )
    return part, report


def pair_report(target_jaw_mesh, antagonist_mesh) -> dict:
    """Summarize a target-jaw/antagonist pair for local regression provenance."""
    support, target = select_separate_support(target_jaw_mesh)
    ant_major, ant_all, ant_threshold = major_components(antagonist_mesh)
    antagonist = {
        "component_count": int(len(ant_all)),
        "major_component_count": int(len(ant_major)),
        "major_face_threshold": int(ant_threshold),
        "major_components": [component_record(part, rank) for rank, part in enumerate(ant_major)],
    }
    return {
        "target_jaw": target,
        "antagonist": antagonist,
        "support_candidate_found": support is not None,
    }
