"""Patient-driven CAD-1 outer-crown adaptation.

This module owns the *global* patient-space adaptation that must happen before
local contact refinement:

1. keep the Margin/cervical region authoritative;
2. fit mesiodistal body width to available proximal space;
3. fit overall crown height to the antagonist while preserving template relief;
4. explicitly pull the cervical rim onto the patient Margin with a smooth band.

The tooth library remains the anatomy prior.  Surrounding scans provide envelopes
and dimensions, not replacement surfaces.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

from ..nonplanar_crown import MarginCurtain
from .zones import build_crown_zones


@dataclass(frozen=True)
class AdaptationConfig:
    cervical_band_mm: float = 1.20
    cervical_hard_band_mm: float = 0.22
    body_transition_mm: float = 1.60
    md_scale_min: float = 0.72
    md_scale_max: float = 1.32
    height_scale_min: float = 0.72
    height_scale_max: float = 1.45
    max_height_delta_mm: float = 3.0
    contact_quantile: float = 0.18


def _basis(frame) -> tuple[np.ndarray, np.ndarray]:
    origin = np.asarray(frame.origin, dtype=float)
    matrix = np.asarray(frame.matrix(), dtype=float)
    if origin.shape != (3,) or matrix.shape != (3, 3):
        raise ValueError("patient frame 必须包含三维 origin 与 3x3 matrix")
    if not np.isfinite(origin).all() or not np.isfinite(matrix).all():
        raise ValueError("patient frame 含非有限数")
    return origin, matrix


def _local(points, frame) -> np.ndarray:
    origin, matrix = _basis(frame)
    return (np.asarray(points, dtype=float) - origin) @ matrix


def _world(local, frame) -> np.ndarray:
    origin, matrix = _basis(frame)
    return np.asarray(local, dtype=float) @ matrix.T + origin


def _smoothstep01(value) -> np.ndarray:
    x = np.clip(np.asarray(value, dtype=float), 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def _finite_mesh(mesh) -> None:
    vertices = np.asarray(mesh.vertices, dtype=float)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not len(vertices):
        raise ValueError("冠形网格没有有效顶点")
    if not np.isfinite(vertices).all():
        raise ValueError("冠形适配产生 NaN/Inf，已中止")


def _neighbor_face_x(surface, frame, side: str) -> float | None:
    """Return the neighbor X coordinate facing the target tooth.

    Auto-neighbor patches are already side-labelled.  In the confirmed patient
    frame the mesial patch should be on +X and distal on -X.  The quantile nearest
    the crown is deliberately robust to patch edges/outliers.
    """
    local = _local(surface.mesh.vertices, frame)
    x = local[:, 0]
    x = x[np.isfinite(x)]
    if not len(x):
        return None
    if side == "mesial":
        positive = x[x > 0]
        values = positive if len(positive) >= max(8, len(x) // 10) else x
        return float(np.quantile(values, 0.08))
    if side == "distal":
        negative = x[x < 0]
        values = negative if len(negative) >= max(8, len(x) // 10) else x
        return float(np.quantile(values, 0.92))
    raise ValueError(f"未知邻接侧: {side}")


def global_patient_prefit(mesh, frame, margin, *, proximal_surfaces=None,
                          proximal_target_mm=None, antagonist_surface=None,
                          occlusal_target_mm=None, config: AdaptationConfig | None = None):
    """Fit whole-crown width/height before local Laplacian contact refinement.

    The cervical band is blended out, so the Margin remains the pivot of the
    transform.  X is scaled/translated from mesial/distal available space.  Z is
    scaled from antagonist clearance error; cusp/fossa relief is not flattened.
    """
    cfg = config or AdaptationConfig()
    _finite_mesh(mesh)
    current = mesh.copy()
    vertices = np.asarray(current.vertices, dtype=float)
    local = _local(vertices, frame)
    margin_local = _local(margin, frame)
    margin_z = float(np.median(margin_local[:, 2]))
    height_above = MarginCurtain(margin, frame.z).axial_above(vertices)
    body_weight = _smoothstep01(height_above / max(cfg.body_transition_mm, 1e-6))

    report = {
        "stage": "GLOBAL_PREFIT",
        "status": "PASS_ENGINEERING",
        "md": {"applied": False},
        "height": {"applied": False},
    }

    # ---- Mesiodistal available-space fit -------------------------------------------------
    surfaces = proximal_surfaces or {}
    target = None if proximal_target_mm is None else float(proximal_target_mm)
    if surfaces and target is not None:
        crown_non_cervical = height_above > cfg.cervical_hard_band_mm
        crown_x = local[crown_non_cervical, 0] if np.any(crown_non_cervical) else local[:, 0]
        current_min = float(np.quantile(crown_x, 0.03))
        current_max = float(np.quantile(crown_x, 0.97))
        current_center = 0.5 * (current_min + current_max)
        current_span = max(current_max - current_min, 1e-6)

        mesial_x = _neighbor_face_x(surfaces["mesial"], frame, "mesial") if "mesial" in surfaces else None
        distal_x = _neighbor_face_x(surfaces["distal"], frame, "distal") if "distal" in surfaces else None
        desired_max = mesial_x - target if mesial_x is not None else None
        desired_min = distal_x + target if distal_x is not None else None

        if desired_min is not None and desired_max is not None and desired_max > desired_min + 0.5:
            desired_center = 0.5 * (desired_min + desired_max)
            desired_span = desired_max - desired_min
            scale = float(np.clip(desired_span / current_span, cfg.md_scale_min, cfg.md_scale_max))
            translation = desired_center - current_center
        elif desired_max is not None:
            scale = 1.0
            translation = desired_max - current_max
        elif desired_min is not None:
            scale = 1.0
            translation = desired_min - current_min
        else:
            scale = 1.0
            translation = 0.0

        full_x = (local[:, 0] - current_center) * scale + current_center + translation
        local[:, 0] += body_weight * (full_x - local[:, 0])
        report["md"] = {
            "applied": bool(abs(scale - 1.0) > 1e-6 or abs(translation) > 1e-6),
            "scale": scale,
            "translation_mm": float(translation),
            "current_span_mm": current_span,
            "desired_min_x_mm": desired_min,
            "desired_max_x_mm": desired_max,
            "mesial_face_x_mm": mesial_x,
            "distal_face_x_mm": distal_x,
        }

    # ---- Overall height fit ---------------------------------------------------------------
    if antagonist_surface is not None and occlusal_target_mm is not None:
        temp = current.copy()
        temp.vertices = _world(local, frame)
        zones = build_crown_zones(temp, frame, margin)
        ids = np.flatnonzero(zones.occlusal)
        if len(ids):
            _cp, _faces, signed, _distance, valid = antagonist_surface.query(np.asarray(temp.vertices)[ids])
            valid = np.asarray(valid, bool) & np.isfinite(signed)
            if np.any(valid):
                errors = np.asarray(signed[valid], float) - float(occlusal_target_mm)
                # Use the closest fraction of the antagonist envelope to establish
                # crown height.  This prevents deep fossae from forcing a global
                # raise while still allowing the whole crown to reach occlusion.
                ordered = np.sort(errors)
                take = max(1, int(np.ceil(len(ordered) * cfg.contact_quantile)))
                delta = float(np.median(ordered[:take]))
                delta = float(np.clip(delta, -cfg.max_height_delta_mm, cfg.max_height_delta_mm))
                occ_z = local[ids, 2]
                top = float(np.quantile(occ_z, 0.90))
                crown_height = max(top - margin_z, 0.5)
                scale = float(np.clip((crown_height + delta) / crown_height,
                                      cfg.height_scale_min, cfg.height_scale_max))
                full_z = margin_z + (local[:, 2] - margin_z) * scale
                local[:, 2] += body_weight * (full_z - local[:, 2])
                report["height"] = {
                    "applied": bool(abs(scale - 1.0) > 1e-6),
                    "scale": scale,
                    "requested_delta_mm": delta,
                    "reference_height_mm": crown_height,
                    "sample_count": int(valid.sum()),
                    "target_clearance_mm": float(occlusal_target_mm),
                }
            else:
                report["height"] = {"applied": False, "reason": "NO_VALID_ANTAGONIST_SAMPLES"}
        else:
            report["height"] = {"applied": False, "reason": "NO_OCCLUSAL_ZONE"}

    current.vertices = _world(local, frame)
    _finite_mesh(current)
    return current, report


def cervical_fit_to_margin(mesh, frame, margin, *, config: AdaptationConfig | None = None):
    """Pull the template cervical rim onto the confirmed patient Margin.

    Only the outer rim portion of the cervical band is moved strongly.  Interior
    cap vertices are excluded by a radial test, avoiding the historical failure
    mode where a closed library cap was collapsed toward the Margin loop.
    """
    cfg = config or AdaptationConfig()
    _finite_mesh(mesh)
    current = mesh.copy()
    vertices = np.asarray(current.vertices, dtype=float)
    local = _local(vertices, frame)
    margin_local = _local(margin, frame)
    center = np.mean(margin_local[:, :2], axis=0)
    mxy = margin_local[:, :2] - center
    mtheta = np.arctan2(mxy[:, 1], mxy[:, 0])
    mradius = np.linalg.norm(mxy, axis=1)

    height = MarginCurtain(margin, frame.z).axial_above(vertices)
    band = np.asarray((height >= -0.15) & (height <= cfg.cervical_band_mm), bool)
    vxy = local[:, :2] - center
    vtheta = np.arctan2(vxy[:, 1], vxy[:, 0])
    vradius = np.linalg.norm(vxy, axis=1)

    # Nearest angular Margin sample; closed-loop angular distance.
    diff = np.abs(np.angle(np.exp(1j * (vtheta[:, None] - mtheta[None, :]))))
    nearest = np.argmin(diff, axis=1)
    target_local = margin_local[nearest]
    target_radius = mradius[nearest]
    rim = band & (vradius >= np.maximum(0.40 * target_radius, target_radius - 2.0))

    if not np.any(rim):
        return current, {"stage": "CERVICAL_FIT", "status": "REVIEW_REQUIRED", "reason": "NO_CERVICAL_RIM"}

    # Hard fit at the bottom, smooth emergence transition above it.
    h = np.clip(height / max(cfg.cervical_band_mm, 1e-6), 0.0, 1.0)
    weight = (1.0 - _smoothstep01(h)) ** 1.25
    weight[height <= cfg.cervical_hard_band_mm] = 1.0
    weight *= rim.astype(float)

    before_tree = cKDTree(np.asarray(margin, dtype=float))
    before_distance = before_tree.query(vertices[rim])[0]
    desired = local.copy()
    desired[rim] = target_local[rim]
    local += weight[:, None] * (desired - local)
    current.vertices = _world(local, frame)
    _finite_mesh(current)

    after_distance = before_tree.query(np.asarray(current.vertices)[rim])[0]
    report = {
        "stage": "CERVICAL_FIT",
        "status": "PASS_ENGINEERING",
        "rim_vertex_count": int(rim.sum()),
        "before_mm": {
            "median": float(np.median(before_distance)),
            "p95": float(np.quantile(before_distance, 0.95)),
            "max": float(np.max(before_distance)),
        },
        "after_mm": {
            "median": float(np.median(after_distance)),
            "p95": float(np.quantile(after_distance, 0.95)),
            "max": float(np.max(after_distance)),
        },
    }
    return current, report
