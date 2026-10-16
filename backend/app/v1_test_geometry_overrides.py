"""CrownFlow 1.0 test-stage geometry overrides.

Keep generation visible and conservative while the patient-space route is being
validated on real cases.  These overrides intentionally tighten M/D expansion
and prevent anatomy fairing from publishing locally inverted/collapsed faces.
They are imported before the API/service graph in app.main.
"""
from __future__ import annotations

import numpy as np

from .geometry.crown_outer import anatomy_refinement as _anatomy
from .geometry.crown_outer import patient_envelope as _envelope
from .geometry.crown_outer import space_morph as _space


_EnvelopeConfig = _envelope.PatientEnvelopeConfig
_SpaceMorphConfig = _space.SpaceMorphConfig
_AnatomyConfig = _anatomy.AnatomyRefineConfig
_refine = _anatomy.refine_crown_anatomy


def _patient_envelope_config(*args, **kwargs):
    # M/D belongs to the available interproximal space.  The old 3 mm halo cap
    # was intentionally generous and can create an over-wide molar on broad
    # automatic neighbour patches.  Keep enough room for a molar but stop the
    # halo from becoming crown-size truth.
    kwargs.setdefault("neighbor_md_extra_from_margin_mm", 2.10)
    kwargs.setdefault("neighbor_inner_quantile", 0.08)
    kwargs.setdefault("max_half_span_mm", 6.25)
    return _EnvelopeConfig(*args, **kwargs)


def _space_morph_config(*args, **kwargs):
    # 1.0 test stage: material-floor correction may improve clearance, but it
    # must not perform multi-millimetre broad shell deformation.
    kwargs.setdefault("envelope_iterations", 2)
    kwargs.setdefault("envelope_max_push_per_pass_mm", 0.25)
    return _SpaceMorphConfig(*args, **kwargs)


def _safe_anatomy_config(*args, **kwargs):
    # Anatomy should refine an already valid patient-space crown, not reshape it.
    kwargs.setdefault("anatomy_amplitude_mm", 0.12)
    kwargs.setdefault("fairing_iterations", 2)
    kwargs.setdefault("fairing_lambda", 0.14)
    kwargs.setdefault("body_weight", 0.30)
    kwargs.setdefault("occlusal_weight", 0.10)
    kwargs.setdefault("max_vertex_move_mm", 0.08)
    return _AnatomyConfig(*args, **kwargs)


def _face_vectors(vertices, faces):
    tri = np.asarray(vertices, float)[np.asarray(faces, int)]
    return np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])


def _safe_refine_crown_anatomy(mesh, frame, margin, *, fdi: int, config=None):
    cfg = config or _safe_anatomy_config()
    refined, report = _refine(mesh, frame, margin, fdi=fdi, config=cfg)

    before_v = np.asarray(mesh.vertices, float)
    after_v = np.asarray(refined.vertices, float)
    faces = np.asarray(mesh.faces, int)
    if before_v.shape != after_v.shape or np.asarray(refined.faces).shape != faces.shape:
        fallback = mesh.copy()
        return fallback, {
            **dict(report or {}),
            "status": "REVIEW_REQUIRED",
            "reason": "ANATOMY_TOPOLOGY_CHANGED_ROLLBACK",
            "published_geometry": "PRE_ANATOMY",
        }

    b = _face_vectors(before_v, faces)
    a = _face_vectors(after_v, faces)
    bn = np.linalg.norm(b, axis=1)
    an = np.linalg.norm(a, axis=1)
    valid = (bn > 1e-10) & (an > 1e-10)
    flip_count = 0
    collapsed_count = int(np.sum(an <= np.maximum(bn * 0.12, 1e-10)))
    if np.any(valid):
        cosine = np.einsum("ij,ij->i", b[valid], a[valid]) / (bn[valid] * an[valid])
        flip_count = int(np.sum(cosine <= 0.0))

    # Any inverted face can appear as a visible hole with back-face culling.
    # For the test phase, preserve the pre-anatomy crown rather than publishing
    # a damaged-looking refinement.
    if flip_count or collapsed_count:
        fallback = mesh.copy()
        return fallback, {
            **dict(report or {}),
            "status": "REVIEW_REQUIRED",
            "reason": "ANATOMY_FACE_INVERSION_ROLLBACK",
            "face_flip_count": flip_count,
            "collapsed_face_count": collapsed_count,
            "published_geometry": "PRE_ANATOMY",
        }

    report = dict(report or {})
    report.update({
        "algorithm": "REGION_WEIGHTED_ANATOMY_FAIRING_V1_SAFE",
        "face_flip_count": 0,
        "collapsed_face_count": 0,
        "published_geometry": "REFINED",
    })
    return refined, report


# Patch the module globals before API/service modules import these symbols.
_envelope.PatientEnvelopeConfig = _patient_envelope_config
_space.PatientEnvelopeConfig = _patient_envelope_config
_space.SpaceMorphConfig = _space_morph_config
_anatomy.AnatomyRefineConfig = _safe_anatomy_config
_anatomy.refine_crown_anatomy = _safe_refine_crown_anatomy
