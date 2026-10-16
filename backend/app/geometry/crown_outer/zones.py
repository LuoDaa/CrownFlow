"""Stable semantic zones for CAD-1 crown deformation.

The tooth library supplies the anatomy prior.  This module partitions the placed
crown into functional regions that survive vertex-only deformation:

* cervical: margin protection / emergence transition;
* mesial + distal: proximal morphology/contact control;
* occlusal: antagonist-driven height and relief control;
* body: anatomy-preserving transition surface.

The v2 defaults are intentionally broader than the legacy masks.  The previous
very narrow masks often made AutoFit report success while visually leaving the
library tooth almost unchanged.  Broader semantic zones plus local Laplacian
falloff let the constraint solver move enough anatomy to matter without turning
the whole crown into a negative copy of surrounding teeth.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..nonplanar_crown import MarginCurtain


ZONE_MODEL_VERSION = "crown_semantic_zones/2"


@dataclass(frozen=True)
class CrownZoneConfig:
    """Geometry thresholds for CAD-1 functional zones.

    These are template/anatomy-domain thresholds, not clinical clearance values.
    Clearance/contact targets continue to come from the design profile.
    """

    cervical_height_mm: float = 0.50
    occlusal_height_quantile: float = 0.62
    occlusal_normal_min: float = -0.05
    proximal_half_width_fraction: float = 0.45
    proximal_normal_min: float = -0.10

    def validate(self) -> None:
        if not np.isfinite(self.cervical_height_mm) or self.cervical_height_mm < 0:
            raise ValueError("cervical_height_mm 必须是非负有限值")
        if not 0.0 <= self.occlusal_height_quantile <= 1.0:
            raise ValueError("occlusal_height_quantile 必须位于 [0, 1]")
        if not -1.0 <= self.occlusal_normal_min <= 1.0:
            raise ValueError("occlusal_normal_min 必须位于 [-1, 1]")
        if not 0.0 <= self.proximal_half_width_fraction <= 1.5:
            raise ValueError("proximal_half_width_fraction 超出合理范围")
        if not -1.0 <= self.proximal_normal_min <= 1.0:
            raise ValueError("proximal_normal_min 必须位于 [-1, 1]")


@dataclass(frozen=True)
class CrownZones:
    """Disjoint per-vertex semantic masks for one crown topology."""

    cervical: np.ndarray
    body: np.ndarray
    mesial: np.ndarray
    distal: np.ndarray
    occlusal: np.ndarray
    diagnostics: dict = field(default_factory=dict)

    @property
    def n_vertices(self) -> int:
        return int(len(self.cervical))

    def validate(self, n_vertices: int | None = None) -> None:
        expected = self.n_vertices if n_vertices is None else int(n_vertices)
        masks = {
            "cervical": self.cervical,
            "body": self.body,
            "mesial": self.mesial,
            "distal": self.distal,
            "occlusal": self.occlusal,
        }
        for name, mask in masks.items():
            arr = np.asarray(mask)
            if arr.shape != (expected,):
                raise ValueError(f"{name} zone 与冠形顶点数不匹配")
            if arr.dtype != np.bool_:
                raise ValueError(f"{name} zone 必须是 bool mask")
        stacked = np.stack([np.asarray(mask, bool) for mask in masks.values()], axis=0)
        membership = stacked.sum(axis=0)
        if np.any(membership != 1):
            raise ValueError("CrownZones 必须对每个顶点恰好分配一个语义区域")

    def mask_for_side(self, side: str) -> np.ndarray:
        if side == "mesial":
            return self.mesial
        if side == "distal":
            return self.distal
        raise ValueError(f"未知邻接侧: {side}")

    def as_report(self) -> dict:
        return {
            "model": ZONE_MODEL_VERSION,
            "counts": {
                "cervical": int(np.sum(self.cervical)),
                "body": int(np.sum(self.body)),
                "mesial": int(np.sum(self.mesial)),
                "distal": int(np.sum(self.distal)),
                "occlusal": int(np.sum(self.occlusal)),
            },
            **dict(self.diagnostics),
        }


def build_crown_zones(mesh, frame, margin, config: CrownZoneConfig | None = None) -> CrownZones:
    """Classify a placed crown in the confirmed patient coordinate frame.

    The returned masks are topology-stable.  As long as subsequent AutoFit stages
    move vertices without remeshing, exactly the same vertex ids can be reused.
    """
    cfg = config or CrownZoneConfig()
    cfg.validate()

    reference = np.asarray(mesh.vertices, dtype=float)
    if reference.ndim != 2 or reference.shape[1:] != (3,) or len(reference) == 0:
        raise ValueError("冠形网格没有有效顶点")
    normals = np.asarray(mesh.vertex_normals, dtype=float)
    if normals.shape != reference.shape or not np.isfinite(normals).all():
        raise ValueError("冠形顶点法向无效，无法建立语义区域")

    matrix = np.asarray(frame.matrix(), dtype=float)
    if matrix.shape != (3, 3):
        raise ValueError("patient frame matrix 必须是 3x3")
    origin = np.asarray(frame.origin, dtype=float)
    if origin.shape != (3,):
        raise ValueError("patient frame origin 必须是三维点")

    local = (reference - origin) @ matrix
    height = MarginCurtain(margin, frame.z).axial_above(reference)
    cervical = np.asarray(height <= cfg.cervical_height_mm, dtype=bool)
    non_cervical = ~cervical

    occlusal = np.zeros(len(reference), dtype=bool)
    occlusal_cut = None
    if np.any(non_cervical):
        occlusal_cut = float(np.quantile(local[non_cervical, 2], cfg.occlusal_height_quantile))
        occlusal = (
            non_cervical
            & (local[:, 2] >= occlusal_cut)
            & ((normals @ np.asarray(frame.z, float)) > cfg.occlusal_normal_min)
        )

    margin_local_x = (np.asarray(margin, dtype=float) - origin) @ np.asarray(frame.x, float)
    half_width = max(float(np.ptp(margin_local_x)) / 2.0, 1e-6)
    proximal_x = cfg.proximal_half_width_fraction * half_width
    x_normal = normals @ np.asarray(frame.x, float)

    mesial = (
        non_cervical
        & ~occlusal
        & (local[:, 0] > proximal_x)
        & (x_normal > cfg.proximal_normal_min)
    )
    distal = (
        non_cervical
        & ~occlusal
        & (-local[:, 0] > proximal_x)
        & (-x_normal > cfg.proximal_normal_min)
    )
    body = ~(cervical | occlusal | mesial | distal)

    zones = CrownZones(
        cervical=np.asarray(cervical, dtype=bool),
        body=np.asarray(body, dtype=bool),
        mesial=np.asarray(mesial, dtype=bool),
        distal=np.asarray(distal, dtype=bool),
        occlusal=np.asarray(occlusal, dtype=bool),
        diagnostics={
            "config": {
                "cervical_height_mm": float(cfg.cervical_height_mm),
                "occlusal_height_quantile": float(cfg.occlusal_height_quantile),
                "occlusal_normal_min": float(cfg.occlusal_normal_min),
                "proximal_half_width_fraction": float(cfg.proximal_half_width_fraction),
                "proximal_normal_min": float(cfg.proximal_normal_min),
            },
            "occlusal_local_z_cut_mm": occlusal_cut,
            "margin_half_width_mm": float(half_width),
            "proximal_local_x_cut_mm": float(proximal_x),
        },
    )
    zones.validate(len(reference))
    return zones
