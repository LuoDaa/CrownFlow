import numpy as np
import pytest
import trimesh

from app.geometry.crown_outer.patient_frame import PatientFrame
from app.geometry.crown_outer.sectional_v1 import (
    SectionalOuterError,
    SectionalV1Config,
    build_sectional_outer,
)


def _frame():
    return PatientFrame(
        origin=np.zeros(3),
        x=np.array([1.0, 0.0, 0.0]),
        y=np.array([0.0, 1.0, 0.0]),
        z=np.array([0.0, 0.0, 1.0]),
        revision_id="fixture-frame",
    )


def _margin(n=64):
    theta = np.arange(n) * 2.0 * np.pi / n
    return np.c_[3.4 * np.cos(theta), 2.8 * np.sin(theta), 0.08 * np.sin(2.0 * theta)]


def _prep():
    # Explicitly place a 4-mm-high prep above the Margin plane.  Cylinder is used
    # here because trimesh cone vertex distributions make a high quantile a poor
    # deterministic statement of the intended fixture height.
    mesh = trimesh.creation.cylinder(radius=2.15, height=4.0, sections=64)
    mesh.apply_translation([0.0, 0.0, 2.0])
    return mesh


def _neighbors():
    mesial = trimesh.creation.box(extents=[1.0, 5.2, 5.5])
    mesial.apply_translation([5.35, 0.0, 3.15])
    distal = trimesh.creation.box(extents=[1.0, 5.2, 5.2])
    distal.apply_translation([-5.25, 0.0, 3.05])
    return {"mesial": mesial, "distal": distal}


def _antagonist(z0=7.15):
    x = np.linspace(-4.8, 4.8, 49)
    y = np.linspace(-4.1, 4.1, 43)
    xx, yy = np.meshgrid(x, y, indexing="xy")
    zz = z0 + 0.33 * np.cos(1.15 * xx) * np.cos(1.05 * yy)
    vertices = np.c_[xx.ravel(), yy.ravel(), zz.ravel()]
    faces = []
    ny, nx = xx.shape
    for iy in range(ny - 1):
        for ix in range(nx - 1):
            a = iy * nx + ix
            b = a + 1
            c = a + nx
            d = c + 1
            faces.extend(((a, b, c), (b, d, c)))
    return trimesh.Trimesh(vertices=vertices, faces=np.asarray(faces), process=False)


def _boundary_edges(mesh):
    edges = np.sort(np.vstack((mesh.faces[:, [0, 1]], mesh.faces[:, [1, 2]], mesh.faces[:, [2, 0]])), axis=1)
    unique, counts = np.unique(edges, axis=0, return_counts=True)
    return unique[counts == 1]


def test_sectional_v1_matches_red_green_blue_contract():
    config = SectionalV1Config(
        min_wall_mm=0.50,
        proximal_gap_mm=0.03,
        occlusal_gap_mm=0.05,
        ceiling_radius_mm=1.05,
    )
    mesh, report = build_sectional_outer(
        _margin(), _frame(), _prep(),
        neighbors=_neighbors(), antagonist_mesh=_antagonist(), fdi=26, config=config,
    )

    # 1) RED / Margin hard boundary.
    assert report["margin"]["hard_boundary"] is True
    assert report["margin"]["max_error_mm"] < 1e-9
    assert report["red_inner_constraint"]["minimum_occlusal_outer_height_mm"] >= 4.49

    # 2) GREEN / broad neighbor contact support, not one spike.
    assert report["green_lateral"]["mesial"]["broad_contact_ray_count"] >= 3
    assert report["green_lateral"]["distal"]["broad_contact_ray_count"] >= 3
    assert report["green_lateral"]["mesial"]["radial_conflict_ray_count"] == 0
    assert report["green_lateral"]["distal"]["radial_conflict_ray_count"] == 0

    # 3) BLUE / antagonist-constrained cap with visible relief and contact-like peaks.
    blue = report["blue_occlusal"]
    assert blue["status"] == "PASS_ENGINEERING"
    assert blue["ceiling_coverage"] >= 0.95
    assert blue["clearance_min_mm"] >= config.occlusal_gap_mm - 1e-6
    assert blue["contact_like_vertex_count"] > 0
    assert blue["cap_relief_p95_p05_mm"] > 0.15

    # Structured outer: one open cervical loop, consistent winding, no accidental holes.
    boundary = _boundary_edges(mesh)
    assert len(boundary) == len(_margin())
    assert report["mesh"]["boundary_all_on_margin"] is True
    assert mesh.is_winding_consistent
    assert not np.any(mesh.area_faces < 1e-10)


def test_sectional_v1_refuses_antagonist_inside_minimum_wall_floor():
    low = _antagonist(z0=4.20)
    with pytest.raises(SectionalOuterError, match="INSUFFICIENT_OCCLUSAL_SPACE"):
        build_sectional_outer(
            _margin(), _frame(), _prep(),
            neighbors=_neighbors(), antagonist_mesh=low, fdi=26,
            config=SectionalV1Config(min_wall_mm=0.50, occlusal_gap_mm=0.05),
        )


def test_sectional_v1_is_equivariant_to_rigid_world_transform():
    margin = _margin()
    prep = _prep()
    neighbors = _neighbors()
    antagonist = _antagonist()
    frame = _frame()
    config = SectionalV1Config(ceiling_radius_mm=1.05)
    reference, _ = build_sectional_outer(
        margin, frame, prep, neighbors=neighbors, antagonist_mesh=antagonist, fdi=26, config=config)

    angle = np.deg2rad(31.0)
    axis = np.array([0.3, 0.7, 0.64]); axis /= np.linalg.norm(axis)
    transform = trimesh.transformations.rotation_matrix(angle, axis)
    transform[:3, 3] = [12.0, -7.5, 3.2]
    rotation = transform[:3, :3]
    translation = transform[:3, 3]

    def points_xform(points):
        return np.asarray(points) @ rotation.T + translation

    moved_prep = prep.copy(); moved_prep.apply_transform(transform)
    moved_neighbors = {}
    for side, mesh in neighbors.items():
        moved = mesh.copy(); moved.apply_transform(transform); moved_neighbors[side] = moved
    moved_antagonist = antagonist.copy(); moved_antagonist.apply_transform(transform)
    moved_frame = PatientFrame(
        origin=frame.origin @ rotation.T + translation,
        x=frame.x @ rotation.T,
        y=frame.y @ rotation.T,
        z=frame.z @ rotation.T,
        revision_id=frame.revision_id,
    )
    moved, _ = build_sectional_outer(
        points_xform(margin), moved_frame, moved_prep,
        neighbors=moved_neighbors, antagonist_mesh=moved_antagonist, fdi=26, config=config)

    assert reference.faces.shape == moved.faces.shape
    assert np.array_equal(reference.faces, moved.faces)
    assert np.allclose(points_xform(reference.vertices), moved.vertices, atol=2e-6)
