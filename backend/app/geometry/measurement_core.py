"""Shared geometric sign/validity primitives for contact, occlusion and thickness."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

SIGN_SOLID_CONTAINMENT = "SOLID_CONTAINMENT"
SIGN_REVIEWED_NORMAL_LOCAL = "REVIEWED_NORMAL_LOCAL"
SIGN_SMOOTHED_VERTEX_NORMAL = "SMOOTHED_VERTEX_NORMAL"
SIGN_UNSIGNED_ONLY = "UNSIGNED_ONLY"
SIGN_GLOBAL_INSIDE_OUTSIDE = SIGN_SOLID_CONTAINMENT
SIGN_REVIEWED_LOCAL_NORMAL = SIGN_REVIEWED_NORMAL_LOCAL

DEFAULT_CONTAINMENT_EPSILON_MM = 1e-06
LOCAL_NORMAL_TANGENCY_EPS = 0.05

VALIDITY_BOUNDARY_AND_CREASE = "BOUNDARY_AND_CREASE"
VALIDITY_BOUNDARY_ONLY = "BOUNDARY_ONLY"
VALIDITY_NONE = "NONE"

DOMAIN_CONTACT = "CONTACT"
DOMAIN_OCCLUSION = "OCCLUSION"
DOMAIN_THICKNESS = "THICKNESS"


def stats(values) -> dict | None:
    values = np.asarray(values, float)
    if not len(values):
        return None
    return {"p50_mm": float(np.quantile(values, .5)),
            "p95_mm": float(np.quantile(values, .95)),
            "max_mm": float(values.max())}


def validity_counts(valid) -> dict:
    valid = np.asarray(valid, bool)
    return {"valid_count": int(valid.sum()), "ambiguous_count": int((~valid).sum())}


def is_closed_oriented_solid(mesh) -> bool:
    try:
        return bool(mesh.is_watertight and mesh.is_winding_consistent and mesh.volume > 0)
    except Exception:  # noqa: BLE001
        return False


@dataclass
class SignSource:
    """Single sign primitive shared by contact and occlusion.

    For open reviewed patches the old implementation used one global normal sign
    and then rejected every closest face whose normal had dot(outward) <= .5.
    That is incorrect for a curved proximal surface: a valid wall can naturally
    rotate far away from the one reference vector.  We now orient *each closest
    face normal* into the reviewed outward hemisphere and compute the sign locally.
    Only almost-tangent faces (|dot| < 0.05) remain ambiguous.
    """

    method: str
    mesh: object = None
    patch: object = None
    outward: object = None
    epsilon_mm: float = DEFAULT_CONTAINMENT_EPSILON_MM

    @classmethod
    def for_mesh(cls, mesh, outward=None, *, epsilon_mm: float = DEFAULT_CONTAINMENT_EPSILON_MM):
        from .crown_outer.contact import NeighborPatch
        patch = NeighborPatch(np.asarray(mesh.vertices), np.asarray(mesh.faces))
        if is_closed_oriented_solid(mesh):
            return cls(SIGN_SOLID_CONTAINMENT, mesh=mesh, patch=patch, epsilon_mm=epsilon_mm)
        normal = _unit(outward)
        if normal is None:
            return cls(SIGN_UNSIGNED_ONLY, mesh=mesh, patch=patch, epsilon_mm=epsilon_mm)
        # Keep the legacy aggregate orientation for callers that inspect
        # patch.normal_sign, but query() uses per-face orientation below.
        patch.normal_sign = 1 if float(np.median(patch.normals @ normal)) > 0 else -1
        return cls(SIGN_REVIEWED_NORMAL_LOCAL, mesh=mesh, patch=patch, outward=normal,
                   epsilon_mm=epsilon_mm)

    @classmethod
    def for_patch(cls, patch, outward=None, *, epsilon_mm: float = DEFAULT_CONTAINMENT_EPSILON_MM):
        import trimesh
        mesh = trimesh.Trimesh(np.asarray(patch.vertices), np.asarray(patch.faces), process=False)
        return cls.for_mesh(mesh, outward, epsilon_mm=epsilon_mm)

    @property
    def signed_available(self) -> bool:
        return self.method != SIGN_UNSIGNED_ONLY

    def oriented_normals(self, face_ids):
        """Normals oriented locally toward the reviewed outward hemisphere."""
        ids = np.asarray(face_ids, dtype=int)
        normals = np.asarray(self.patch.normals[ids], dtype=float).copy()
        if self.method == SIGN_REVIEWED_NORMAL_LOCAL and self.outward is not None:
            dots = normals @ np.asarray(self.outward, float)
            normals[dots < 0.0] *= -1.0
        elif self.patch is not None:
            normals *= float(getattr(self.patch, "normal_sign", 1.0))
        norm = np.linalg.norm(normals, axis=1)
        normals /= np.maximum(norm[:, None], 1e-12)
        return normals

    def query(self, points):
        points = np.asarray(points, float)
        if points.ndim == 1:
            points = points[None, :]
        boundary = 0.0 if self.method == SIGN_SOLID_CONTAINMENT else 0.5
        cp, fids, _signed, distance, valid = self.patch.closest_with_validity(
            points, boundary_safety_mm=boundary)
        cp = np.asarray(cp, float)
        fids = np.asarray(fids, int)
        distance = np.asarray(distance, float)
        valid = np.asarray(valid, bool)

        if self.method == SIGN_SOLID_CONTAINMENT:
            signed = np.where(self.mesh.contains(points), -distance, distance)
            valid = np.ones(len(points), bool)
        elif self.method == SIGN_REVIEWED_NORMAL_LOCAL:
            raw_normals = np.asarray(self.patch.normals[fids], float)
            facing = raw_normals @ np.asarray(self.outward, float)
            normals = raw_normals.copy()
            normals[facing < 0.0] *= -1.0
            normals /= np.maximum(np.linalg.norm(normals, axis=1)[:, None], 1e-12)
            signed = np.einsum("ij,ij->i", points - cp, normals)
            # A curved patch is allowed to rotate relative to the single reviewed
            # outward vector.  Only a nearly perpendicular face has indeterminate
            # hemisphere orientation and is removed from the decision set.
            valid = valid & (np.abs(facing) >= LOCAL_NORMAL_TANGENCY_EPS)
        else:
            signed = np.full(len(points), np.nan)

        ambiguous = distance < float(self.epsilon_mm)
        valid = valid & (~ambiguous) & np.isfinite(distance)
        if self.method != SIGN_UNSIGNED_ONLY:
            valid &= np.isfinite(signed)
        return cp, fids, np.asarray(signed, float), distance, valid

    def ambiguous_mask(self, points) -> np.ndarray:
        _cp, _fids, _signed, distance, _valid = self.query(points)
        return distance < float(self.epsilon_mm)

    def as_report(self) -> dict:
        payload = {"sign_method": self.method, "signed_available": self.signed_available,
                   "epsilon_mm": float(self.epsilon_mm)}
        if self.method == SIGN_REVIEWED_NORMAL_LOCAL:
            payload.update({
                "orientation": "PER_CLOSEST_FACE_HEMISPHERE",
                "tangency_epsilon_abs_dot": float(LOCAL_NORMAL_TANGENCY_EPS),
            })
        return payload


def _unit(vector):
    if vector is None:
        return None
    array = np.asarray(vector, float)
    norm = float(np.linalg.norm(array))
    if array.shape != (3,) or not np.isfinite(array).all() or norm < 1e-8:
        return None
    return array / norm


@dataclass(frozen=True)
class DomainContract:
    domain: str
    sign_method: str
    sign_semantics: str
    validity: str
    penetration_meaning: str
    owns_crossing_audit: bool
    input_contract: str
    implementation: str

    def as_report(self) -> dict:
        return {"domain": self.domain, "sign_method": self.sign_method,
                "sign_semantics": self.sign_semantics, "validity_convention": self.validity,
                "penetration_meaning": self.penetration_meaning,
                "owns_crossing_audit": self.owns_crossing_audit,
                "input_contract": self.input_contract, "implementation": self.implementation}


CONTACT_CONTRACT = DomainContract(
    domain=DOMAIN_CONTACT,
    sign_method=f"{SIGN_GLOBAL_INSIDE_OUTSIDE} | {SIGN_REVIEWED_LOCAL_NORMAL} | {SIGN_UNSIGNED_ONLY}",
    sign_semantics="positive=gap_to_reference, negative=penetration_into_reference, |signed|=euclidean_distance",
    validity=VALIDITY_BOUNDARY_AND_CREASE,
    penetration_meaning="max(0, -signed) = penetration depth into reference surface",
    owns_crossing_audit=False,
    input_contract="closed oriented solid or open patch with reviewed outward hemisphere",
    implementation="services.contact_measurement.measure_surface",
)

OCCLUSION_CONTRACT = DomainContract(
    domain=DOMAIN_OCCLUSION,
    sign_method=f"{SIGN_SOLID_CONTAINMENT} | {SIGN_REVIEWED_NORMAL_LOCAL} | {SIGN_UNSIGNED_ONLY}",
    sign_semantics="signed<0 = antagonist penetration; deficit=max(target_gap-signed,0)",
    validity=VALIDITY_BOUNDARY_ONLY,
    penetration_meaning="target_gap - signed",
    owns_crossing_audit=False,
    input_contract="crown with cervical lock; antagonist solid or reviewed open patch",
    implementation="crown_outer.occlusion.solve_occlusion_collision",
)

THICKNESS_CONTRACT = DomainContract(
    domain=DOMAIN_THICKNESS,
    sign_method=SIGN_SMOOTHED_VERTEX_NORMAL,
    sign_semantics="signed=distance to nearest surface using smoothed normal; margin=signed-target",
    validity=VALIDITY_BOUNDARY_ONLY,
    penetration_meaning="signed - target",
    owns_crossing_audit=True,
    input_contract="outer open cervical surface + fitting inner surface",
    implementation="crown_outer.thickness.audit_thickness",
)

DOMAIN_CONTRACTS: dict[str, DomainContract] = {
    DOMAIN_CONTACT: CONTACT_CONTRACT,
    DOMAIN_OCCLUSION: OCCLUSION_CONTRACT,
    DOMAIN_THICKNESS: THICKNESS_CONTRACT,
}


def domain_report(domain: str, **extra) -> dict:
    contract = DOMAIN_CONTRACTS[domain]
    return {**contract.as_report(), **extra}


@dataclass
class DomainMeasurement:
    contract: DomainContract
    values: dict = field(default_factory=dict)
    notes: list = field(default_factory=list)

    def as_report(self) -> dict:
        return {**self.contract.as_report(), "values": dict(self.values),
                "notes": list(self.notes)}
