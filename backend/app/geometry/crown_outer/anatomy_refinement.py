"""Constraint-aware morphology refinement for patient-space crowns.

This is a soft morphology prior, not a hard size owner.  It preserves the
cervical root and applies conservative anatomy plus weighted fairing.  Hard
proximal/occlusal constraints are re-solved after this stage.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from ..nonplanar_crown import MarginCurtain


REFERENCE_NORMAL_ANGLE_PRIOR = {
    "source": "six standard CAD crown references",
    "median_deg_range": [1.9, 3.0],
    "p90_deg_range": [7.0, 11.0],
    "interpretation": "soft visual prior only; high-curvature anatomy is preserved",
}


@dataclass(frozen=True)
class AnatomyRefineConfig:
    cervical_lock_mm: float = 0.55
    occlusal_start_fraction: float = 0.62
    anatomy_amplitude_mm: float = 0.22
    fairing_iterations: int = 4
    fairing_lambda: float = 0.18
    body_weight: float = 0.42
    occlusal_weight: float = 0.15
    max_vertex_move_mm: float = 0.18


def _local(points, frame):
    origin = np.asarray(frame.origin, float)
    matrix = np.asarray(frame.matrix(), float)
    return (np.asarray(points, float) - origin) @ matrix


def _world(points, frame):
    origin = np.asarray(frame.origin, float)
    matrix = np.asarray(frame.matrix(), float)
    return np.asarray(points, float) @ matrix.T + origin


def _adjacency(mesh):
    out = [set() for _ in range(len(mesh.vertices))]
    for a, b in np.asarray(mesh.edges_unique, int):
        out[int(a)].add(int(b)); out[int(b)].add(int(a))
    return out


def _normal_angle_stats(mesh):
    if len(mesh.face_adjacency) == 0:
        return {"count": 0}
    n = np.asarray(mesh.face_normals, float)
    a = np.asarray(mesh.face_adjacency, int)
    dot = np.einsum("ij,ij->i", n[a[:, 0]], n[a[:, 1]])
    deg = np.degrees(np.arccos(np.clip(dot, -1.0, 1.0)))
    return {
        "count": int(len(deg)),
        "median_deg": float(np.median(deg)),
        "p90_deg": float(np.quantile(deg, .90)),
        "p99_deg": float(np.quantile(deg, .99)),
    }


def _family(fdi):
    tooth = int(fdi) % 10
    if tooth <= 3:
        return "ANTERIOR"
    if tooth <= 5:
        return "PREMOLAR"
    return "MOLAR"


def _gauss(u, v, cu, cv, su, sv):
    return np.exp(-.5 * (((u - cu) / su) ** 2 + ((v - cv) / sv) ** 2))


def _anatomy_field(u, v, family):
    if family == "MOLAR":
        cusps = (
            1.00 * _gauss(u, v, -.44, -.38, .24, .22)
            + .92 * _gauss(u, v, -.42, .38, .24, .22)
            + .88 * _gauss(u, v, .42, -.36, .24, .22)
            + .82 * _gauss(u, v, .40, .36, .24, .22)
        )
        fossa = 1.15 * _gauss(u, v, 0.0, 0.0, .28, .24)
        groove = .45 * np.exp(-.5 * (u / .11) ** 2) * np.exp(-.5 * (v / .72) ** 2)
        raw = cusps - fossa - groove
    elif family == "PREMOLAR":
        cusps = (
            1.05 * _gauss(u, v, 0.0, -.34, .30, .22)
            + .92 * _gauss(u, v, 0.0, .34, .30, .22)
        )
        groove = .85 * _gauss(u, v, 0.0, 0.0, .34, .16)
        raw = cusps - groove
    else:
        ridge = _gauss(u, v, 0.0, 0.0, .72, .28)
        raw = .45 * ridge
    raw -= float(np.median(raw))
    scale = max(float(np.quantile(np.abs(raw), .95)), 1e-8)
    return np.clip(raw / scale, -1.0, 1.0)


def refine_crown_anatomy(mesh, frame, margin, *, fdi: int,
                         config: AnatomyRefineConfig | None = None):
    cfg = config or AnatomyRefineConfig()
    out = mesh.copy()
    before = np.asarray(out.vertices, float).copy()
    local = _local(before, frame)
    curtain = MarginCurtain(margin, frame.z)
    height = curtain.axial_above(before)
    finite_h = height[np.isfinite(height)]
    if not len(finite_h):
        return out, {"stage": "ANATOMY_REFINEMENT", "status": "NOT_ASSESSED",
                     "reason": "NO_VALID_AXIAL_HEIGHT"}
    h_top = max(float(np.quantile(finite_h, .98)), 1e-6)
    h_frac = np.clip(height / h_top, 0.0, 1.0)

    body = np.isfinite(height) & (h_frac > .35)
    cx = float(np.median(local[body, 0])) if np.any(body) else float(np.median(local[:, 0]))
    cy = float(np.median(local[body, 1])) if np.any(body) else float(np.median(local[:, 1]))
    sx = max(float(np.quantile(np.abs(local[body, 0] - cx), .92)) if np.any(body) else 1.0, .5)
    sy = max(float(np.quantile(np.abs(local[body, 1] - cy), .92)) if np.any(body) else 1.0, .5)
    u = np.clip((local[:, 0] - cx) / sx, -1.25, 1.25)
    v = np.clip((local[:, 1] - cy) / sy, -1.25, 1.25)

    family = _family(fdi)
    start = float(cfg.occlusal_start_fraction)
    anatomy_weight = np.clip((h_frac - start) / max(1.0 - start, 1e-6), 0.0, 1.0)
    anatomy_weight = anatomy_weight * anatomy_weight * (3.0 - 2.0 * anatomy_weight)
    anatomy_weight[height <= float(cfg.cervical_lock_mm)] = 0.0
    field = _anatomy_field(u, v, family)
    local[:, 2] += anatomy_weight * field * float(cfg.anatomy_amplitude_mm)
    out.vertices = _world(local, frame)

    adjacency = _adjacency(out)
    vertices = np.asarray(out.vertices, float).copy()
    fair_weight = np.full(len(vertices), float(cfg.body_weight), float)
    fair_weight[h_frac >= start] = float(cfg.occlusal_weight)
    fair_weight[height <= float(cfg.cervical_lock_mm)] = 0.0

    for _ in range(max(0, int(cfg.fairing_iterations))):
        nxt = vertices.copy()
        for i, neigh in enumerate(adjacency):
            if fair_weight[i] <= 0.0 or not neigh:
                continue
            mean = vertices[np.fromiter(neigh, dtype=int)].mean(axis=0)
            delta = (mean - vertices[i]) * float(cfg.fairing_lambda) * fair_weight[i]
            norm = float(np.linalg.norm(delta))
            if norm > float(cfg.max_vertex_move_mm):
                delta *= float(cfg.max_vertex_move_mm) / norm
            nxt[i] += delta
        vertices = nxt

    locked = height <= float(cfg.cervical_lock_mm)
    vertices[locked] = before[locked]
    out.vertices = vertices

    displacement = np.linalg.norm(vertices - before, axis=1)
    report = {
        "stage": "ANATOMY_REFINEMENT",
        "status": "PASS_ENGINEERING",
        "algorithm": "REGION_WEIGHTED_ANATOMY_FAIRING_V1",
        "family": family,
        "template_used": False,
        "cervical_lock_mm": float(cfg.cervical_lock_mm),
        "locked_vertex_count": int(locked.sum()),
        "anatomy_amplitude_mm": float(cfg.anatomy_amplitude_mm),
        "fairing_iterations": int(cfg.fairing_iterations),
        "max_displacement_mm": float(displacement.max(initial=0.0)),
        "p95_displacement_mm": float(np.quantile(displacement, .95)),
        "normal_angle_before": _normal_angle_stats(mesh),
        "normal_angle_after": _normal_angle_stats(out),
        "reference_prior": dict(REFERENCE_NORMAL_ANGLE_PRIOR),
        "hard_constraints_must_be_rechecked_after": True,
    }
    return out, report
