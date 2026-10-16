from __future__ import annotations

import math

import numpy as np
import trimesh

from app.geometry.crown_outer.auto_fit_adapter import fit_domains
from app.geometry.crown_outer.patient_frame import confirmed_patient_frame
from app.geometry.crown_outer.zones import ZONE_MODEL_VERSION, build_crown_zones


def _case():
    crown = trimesh.creation.cylinder(radius=2.0, height=4.0, sections=48)
    crown.apply_translation([0.0, 0.0, 2.0])
    theta = np.linspace(0.0, 2.0 * math.pi, 96, endpoint=False)
    margin = np.c_[2.0 * np.cos(theta), 2.0 * np.sin(theta), np.zeros_like(theta)]
    frame = confirmed_patient_frame(
        origin=(0.0, 0.0, 0.0),
        mesial=(1.0, 0.0, 0.0),
        buccal=(0.0, 1.0, 0.0),
        insertion=(0.0, 0.0, 1.0),
        revision_id="zones-test",
    )
    return crown, margin, frame


def test_semantic_zones_partition_every_vertex_once():
    crown, margin, frame = _case()
    zones = build_crown_zones(crown, frame, margin)
    zones.validate(len(crown.vertices))

    stacked = np.stack([
        zones.cervical,
        zones.body,
        zones.mesial,
        zones.distal,
        zones.occlusal,
    ])
    assert np.all(stacked.sum(axis=0) == 1)
    assert zones.cervical.any()
    assert zones.occlusal.any()
    assert zones.mesial.any()
    assert zones.distal.any()
    assert zones.as_report()["model"] == ZONE_MODEL_VERSION


def test_occlusal_fit_domain_consumes_shared_semantic_zone():
    crown, margin, frame = _case()
    zones = build_crown_zones(crown, frame, margin)

    masks, locked = fit_domains(
        crown,
        frame,
        margin,
        {"occlusal": object()},
        "OCCLUSAL",
        zones=zones,
    )

    assert np.array_equal(masks["occlusal"], zones.occlusal)
    assert np.array_equal(locked, zones.cervical)


def test_semantic_zone_vertex_ids_are_rigid_rotation_equivariant():
    crown, margin, frame = _case()
    base = build_crown_zones(crown, frame, margin)

    transform = trimesh.transformations.rotation_matrix(math.radians(37.0), [0.3, 0.7, 0.2])
    rotated = crown.copy()
    rotated.apply_transform(transform)
    rotation = transform[:3, :3]
    margin_rot = trimesh.transform_points(margin, transform)
    frame_rot = confirmed_patient_frame(
        origin=rotation @ frame.origin,
        mesial=rotation @ frame.x,
        buccal=rotation @ frame.y,
        insertion=rotation @ frame.z,
        revision_id="zones-test-rotated",
    )

    got = build_crown_zones(rotated, frame_rot, margin_rot)

    assert np.array_equal(got.cervical, base.cervical)
    assert np.array_equal(got.body, base.body)
    assert np.array_equal(got.mesial, base.mesial)
    assert np.array_equal(got.distal, base.distal)
    assert np.array_equal(got.occlusal, base.occlusal)
