import numpy as np
import trimesh
from trimesh.proximity import closest_point

from app.geometry.crown_outer.collar_body_outer import _boundary_loops
from app.geometry.nonplanar_crown import (
    MarginCurtain,
    _separate_cervical_source_surfaces,
    _single_open_surface_above_margin,
)


def _margin(radius=2.5, n=96):
    theta = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    return np.c_[radius * np.cos(theta), radius * np.sin(theta), np.zeros_like(theta)]


def test_direct_source_clip_extracts_one_outer_and_one_intaglio_open_surface():
    margin = _margin()
    curtain = MarginCurtain(margin, [0.0, 0.0, 1.0])

    outer = trimesh.creation.cylinder(radius=3.0, height=7.0, sections=128)
    outer.apply_translation([0.0, 0.0, 3.0])
    cutter = trimesh.creation.cylinder(radius=2.35, height=6.5, sections=128)
    cutter.apply_translation([0.0, 0.0, 2.8])

    outside, outer_report = _single_open_surface_above_margin(
        outer, curtain, 0.6, "OUTER", invert=False,
    )
    inside, inner_report = _single_open_surface_above_margin(
        cutter, curtain, 0.6, "INTAGLIO", invert=True,
    )

    assert len(outside.faces) > 0
    assert len(inside.faces) > 0
    assert len(_boundary_loops(outside.faces)) == 1
    assert len(_boundary_loops(inside.faces)) == 1
    assert outer_report["boundary_loop_count"] == 1
    assert inner_report["boundary_loop_count"] == 1
    assert not outer_report["inverted_for_material_orientation"]
    assert inner_report["inverted_for_material_orientation"]
    assert np.isfinite(outside.vertices).all()
    assert np.isfinite(inside.vertices).all()


def test_direct_source_clip_prefers_largest_single_loop_component():
    margin = _margin()
    curtain = MarginCurtain(margin, [0.0, 0.0, 1.0])

    main = trimesh.creation.cylinder(radius=3.0, height=7.0, sections=96)
    main.apply_translation([0.0, 0.0, 3.0])
    small = trimesh.creation.cylinder(radius=0.25, height=2.0, sections=32)
    small.apply_translation([6.0, 0.0, 2.0])
    source = trimesh.util.concatenate([main, small])

    surface, report = _single_open_surface_above_margin(
        source, curtain, 0.6, "OUTER", invert=False,
    )

    assert report["component_count"] >= 2
    assert report["selected_area_mm2"] > report["discarded_component_area_mm2"]
    assert len(_boundary_loops(surface.faces)) == 1


def test_cervical_source_separation_moves_only_outer_skin_and_increases_clearance():
    margin = _margin(radius=2.3)
    curtain = MarginCurtain(margin, [0.0, 0.0, 1.0])

    # Deliberately make the two authoritative skins almost coincident so the
    # pre-stitch separation stage has work to do.
    outer = trimesh.creation.cylinder(radius=2.39, height=7.0, sections=128)
    outer.apply_translation([0.0, 0.0, 3.0])
    cutter = trimesh.creation.cylinder(radius=2.35, height=6.8, sections=128)
    cutter.apply_translation([0.0, 0.0, 2.9])

    outside, _ = _single_open_surface_above_margin(
        outer, curtain, 0.6, "OUTER", invert=False,
    )
    inside, _ = _single_open_surface_above_margin(
        cutter, curtain, 0.6, "INTAGLIO", invert=True,
    )
    inside_before = np.asarray(inside.vertices).copy()

    height_before = curtain.axial_above(outside.vertices)
    band_before = (height_before >= 0.55) & (height_before <= 2.6)
    before_distance = closest_point(inside, np.asarray(outside.vertices)[band_before])[1]

    separated, report = _separate_cervical_source_surfaces(
        outside,
        inside,
        curtain,
        0.6,
        min_clearance_mm=0.18,
        band_mm=2.0,
        iterations=4,
    )

    height_after = curtain.axial_above(separated.vertices)
    band_after = (height_after >= 0.55) & (height_after <= 2.6)
    after_distance = closest_point(inside, np.asarray(separated.vertices)[band_after])[1]

    assert report["moved_vertices"] > 0
    assert report["intaglio_unchanged"]
    assert report["margin_curve_unchanged"]
    assert np.allclose(inside.vertices, inside_before)
    assert float(np.quantile(after_distance, 0.05)) > float(np.quantile(before_distance, 0.05))
    assert report["p05_after_mm"] > float(np.quantile(before_distance, 0.05))
    assert len(_boundary_loops(separated.faces)) == 1
    assert np.isfinite(separated.vertices).all()
