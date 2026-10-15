import numpy as np
import trimesh

from app.geometry.crown_outer.collar_body_outer import _boundary_loops
from app.geometry.nonplanar_crown import MarginCurtain, _single_open_surface_above_margin


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
