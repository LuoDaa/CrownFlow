import numpy as np
import trimesh

from app.geometry.crown_outer.patient_frame import PatientFrame
from app.geometry.crown_outer.proximal_rounding import round_proximal_contacts
from app.geometry.crown_outer.sectional_v1 import SectionalV1Config, build_sectional_outer


def _frame():
    return PatientFrame(
        origin=np.zeros(3),
        x=np.array([1.0, 0.0, 0.0]),
        y=np.array([0.0, 1.0, 0.0]),
        z=np.array([0.0, 0.0, 1.0]),
        revision_id="proximal-rounding-fixture",
    )


def _margin(n=96):
    theta = np.arange(n) * 2.0 * np.pi / n
    return np.c_[3.35 * np.cos(theta), 2.75 * np.sin(theta), 0.05 * np.sin(2.0 * theta)]


def _prep():
    mesh = trimesh.creation.cylinder(radius=2.10, height=4.0, sections=64)
    mesh.apply_translation([0.0, 0.0, 2.0])
    return mesh


def _neighbors():
    mesial = trimesh.creation.box(extents=[1.0, 5.4, 5.7])
    mesial.apply_translation([5.55, 0.0, 3.20])
    distal = trimesh.creation.box(extents=[1.0, 5.4, 5.5])
    distal.apply_translation([-5.45, 0.0, 3.10])
    return {"mesial": mesial, "distal": distal}


def _antagonist(z0=7.25):
    x = np.linspace(-5.0, 5.0, 51)
    y = np.linspace(-4.2, 4.2, 45)
    xx, yy = np.meshgrid(x, y, indexing="xy")
    zz = z0 + 0.30 * np.cos(1.1 * xx) * np.cos(1.0 * yy)
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


def test_rounding_broadens_proximal_patch_without_touching_margin_or_blue_cap():
    margin = _margin()
    frame = _frame()
    neighbors = _neighbors()
    config = SectionalV1Config(
        min_wall_mm=0.50,
        proximal_gap_mm=0.02,
        occlusal_gap_mm=0.05,
        contact_height_ratio=0.62,
        top_inset_mm=0.20,
        neighbor_sector_cos=0.32,
        neighbor_ray_band_mm=2.0,
        max_proximal_extension_mm=3.0,
        side_levels=(0.0, 0.10, 0.22, 0.36, 0.50, 0.62, 0.73, 0.83, 0.92, 1.0),
        ceiling_radius_mm=1.05,
    )
    baseline, _ = build_sectional_outer(
        margin,
        frame,
        _prep(),
        neighbors=neighbors,
        antagonist_mesh=_antagonist(),
        fdi=26,
        config=config,
    )
    before = np.asarray(baseline.vertices, float).copy()
    rounded, report = round_proximal_contacts(
        baseline,
        margin,
        frame,
        neighbors=neighbors,
        sectional_config=config,
    )
    after = np.asarray(rounded.vertices, float)

    n = len(margin)
    level_count = len(config.side_levels)

    # RED hard boundary unchanged exactly within numerical transform tolerance.
    np.testing.assert_allclose(after[:n], before[:n], atol=1e-10)
    assert report["margin_preserved"] is True

    # Last side ring + every BLUE cap vertex are protected, so occlusion behaviour
    # cannot be changed by this GREEN morphology pass.
    protected_start = (level_count - 1) * n
    np.testing.assert_allclose(after[protected_start:], before[protected_start:], atol=1e-10)
    assert report["blue_occlusal_vertices_preserved"] is True

    # Middle side rings really moved; the pass is not a no-op.
    middle = after[n:protected_start] - before[n:protected_start]
    assert float(np.linalg.norm(middle, axis=1).max()) > 0.05

    for side in ("mesial", "distal"):
        info = report["per_side"][side]
        assert info["ray_coverage"] >= 5
        assert info["rounded_contact_ray_count"] >= 5
        assert info["longest_contiguous_contact_run"] >= 3
        assert info["maximum_outward_move_mm"] > 0.05
        assert info["angular_roundness_second_diff_p95_mm"] is not None

    assert rounded.is_winding_consistent
    assert np.array_equal(rounded.faces, baseline.faces)


def test_rounding_does_not_invent_contact_for_absent_neighbor():
    margin = _margin()
    frame = _frame()
    config = SectionalV1Config(
        neighbor_sector_cos=0.32,
        neighbor_ray_band_mm=2.0,
        side_levels=(0.0, 0.10, 0.22, 0.36, 0.50, 0.62, 0.73, 0.83, 0.92, 1.0),
        ceiling_radius_mm=1.05,
    )
    neighbors = _neighbors()
    neighbors["distal"] = None
    baseline, _ = build_sectional_outer(
        margin,
        frame,
        _prep(),
        neighbors=neighbors,
        antagonist_mesh=_antagonist(),
        fdi=26,
        config=config,
    )
    rounded, report = round_proximal_contacts(
        baseline,
        margin,
        frame,
        neighbors=neighbors,
        sectional_config=config,
    )

    assert report["per_side"]["distal"]["status"] == "NOT_ASSESSED"
    assert report["per_side"]["distal"]["reason"] == "NO_CONFIRMED_NEIGHBOR"
    assert rounded.is_winding_consistent
