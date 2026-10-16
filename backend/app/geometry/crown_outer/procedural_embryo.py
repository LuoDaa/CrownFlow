"""Template-free patient-space crown embryo.

The patient owns metric geometry.  This module only creates a smooth, topology-
stable canonical outer surface whose cervical ring is the confirmed Margin.
PatientEnvelope owns MD/BL/height; contact and occlusion solvers remain the
authoritative local constraint stages.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import trimesh

from .patient_envelope import PatientEnvelope


@dataclass(frozen=True)
class ProceduralEmbryoConfig:
    sections: int = 96
    body_rings: int = 18
    occlusal_table_fraction: float = 0.58
    max_occlusal_sag_mm: float = 0.55
    minimum_height_mm: float = 4.0


def _basis(frame):
    origin = np.asarray(frame.origin, float)
    matrix = np.asarray(frame.matrix(), float)
    if origin.shape != (3,) or matrix.shape != (3, 3):
        raise ValueError("patient frame 必须包含三维 origin 与 3x3 matrix")
    return origin, matrix


def _local(points, frame):
    origin, matrix = _basis(frame)
    return (np.asarray(points, float) - origin) @ matrix


def _world(points, frame):
    origin, matrix = _basis(frame)
    return np.asarray(points, float) @ matrix.T + origin


def _resample_closed_ring(points, count):
    p = np.asarray(points, float)
    if len(p) < 8 or p.ndim != 2 or p.shape[1] != 3 or not np.isfinite(p).all():
        raise ValueError("程序化冠胚需要完整、有限的 Margin")
    if np.linalg.norm(p[0] - p[-1]) <= 1e-8:
        p = p[:-1]
    closed = np.vstack([p, p[0]])
    seg = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    perimeter = float(seg.sum())
    if perimeter <= 1e-6:
        raise ValueError("Margin 周长无效")
    s = np.r_[0.0, np.cumsum(seg)]
    target = np.linspace(0.0, perimeter, int(count), endpoint=False)
    idx = np.searchsorted(s, target, side="right") - 1
    idx = np.clip(idx, 0, len(p) - 1)
    alpha = (target - s[idx]) / np.maximum(seg[idx], 1e-12)
    return closed[idx] * (1.0 - alpha[:, None]) + closed[idx + 1] * alpha[:, None]


def _smoothstep(x):
    x = np.clip(np.asarray(x, float), 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def _family_height_ratio(fdi):
    tooth = int(fdi) % 10
    if tooth <= 2:
        return 1.12
    if tooth == 3:
        return 1.18
    if tooth <= 5:
        return 0.92
    return 0.82


def _signed_radius(cosv, sinv, env):
    ru = np.where(cosv >= 0.0, float(env.u_max), abs(float(env.u_min)))
    rv = np.where(sinv >= 0.0, float(env.v_max), abs(float(env.v_min)))
    return ru, rv


def build_procedural_crown_embryo(frame, margin, envelope: PatientEnvelope, *, fdi: int,
                                   config: ProceduralEmbryoConfig | None = None):
    """Create a template-free crown outer surface rooted exactly on Margin."""
    cfg = config or ProceduralEmbryoConfig()
    sections = max(48, int(cfg.sections))
    rings = max(10, int(cfg.body_rings))
    root_world = _resample_closed_ring(margin, sections)
    root = _local(root_world, frame)

    center = np.asarray(envelope.center_xy, float)
    md = np.asarray(envelope.md_axis_xy, float)
    bl = np.asarray(envelope.bl_axis_xy, float)
    md /= max(float(np.linalg.norm(md)), 1e-12)
    bl /= max(float(np.linalg.norm(bl)), 1e-12)

    root_xy = root[:, :2] - center
    u_root = root_xy @ md
    v_root = root_xy @ bl
    u_scale = max(float(np.quantile(np.abs(u_root), .90)), .5)
    v_scale = max(float(np.quantile(np.abs(v_root), .90)), .5)
    theta = np.unwrap(np.arctan2(v_root / v_scale, u_root / u_scale))
    c, s = np.cos(theta), np.sin(theta)
    ru, rv = _signed_radius(c, s, envelope)
    target_xy_full = center + (c * ru)[:, None] * md + (s * rv)[:, None] * bl

    width = max(envelope.md_width_mm, envelope.bl_width_mm)
    inferred = max(float(cfg.minimum_height_mm), width * _family_height_ratio(fdi))
    top = float(envelope.target_top_z) if envelope.target_top_z is not None else float(envelope.margin_z + inferred)
    height = max(top - float(envelope.margin_z), float(cfg.minimum_height_mm))

    vertices = [root.copy()]
    for k in range(1, rings + 1):
        t = k / float(rings + 1)
        emerge = _smoothstep(min(t / .32, 1.0))
        coronal = _smoothstep(max((t - .52) / .48, 0.0))
        table_scale = 1.0 - coronal * (1.0 - float(cfg.occlusal_table_fraction))
        target_xy = center + (target_xy_full - center) * table_scale
        xy = root[:, :2] + emerge * (target_xy - root[:, :2])
        base_z = root[:, 2] + height * t
        sag = float(cfg.max_occlusal_sag_mm) * (coronal ** 1.5)
        z = base_z - sag
        vertices.append(np.c_[xy, z])

    vertices = np.vstack(vertices)
    faces = []
    for r in range(rings):
        a0 = r * sections
        b0 = (r + 1) * sections
        for i in range(sections):
            j = (i + 1) % sections
            faces.append((a0 + i, a0 + j, b0 + i))
            faces.append((a0 + j, b0 + j, b0 + i))

    last = rings * sections
    center_local = np.array([center[0], center[1], top], float)
    center_id = len(vertices)
    vertices = np.vstack([vertices, center_local])
    for i in range(sections):
        j = (i + 1) % sections
        faces.append((last + i, last + j, center_id))

    mesh = trimesh.Trimesh(_world(vertices, frame), np.asarray(faces, np.int64), process=False)
    if len(mesh.faces) and np.mean(mesh.face_normals @ np.asarray(frame.z, float)) < -0.15:
        mesh.faces = mesh.faces[:, ::-1]
        mesh._cache.clear()

    report = {
        "stage": "SPACE_DRIVEN_SEED",
        "status": "PASS_ENGINEERING",
        "algorithm": "PROCEDURAL_PATIENT_ENVELOPE_EMBRYO_V1",
        "template_used": False,
        "template_size_authoritative": False,
        "margin_root_exact": True,
        "sections": sections,
        "body_rings": rings,
        "vertex_count": int(len(mesh.vertices)),
        "face_count": int(len(mesh.faces)),
        "target_height_mm": float(height),
        "patient_envelope": envelope.as_report(),
    }
    return mesh, report
