from types import SimpleNamespace
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np
import trimesh

from app.geometry.crown_outer.auto_fit_adapter import FitSurface
from app.services.contact_measurement import neighbor_patch_diagnostics


def test_neighbor_patch_validity_checks_the_expected_side_and_signed_coverage():
    crown = trimesh.creation.icosphere(subdivisions=2, radius=2)
    neighbor = trimesh.creation.icosphere(subdivisions=2, radius=1)
    region = SimpleNamespace(axes={'mesial': np.array([1., 0, 0]), 'occlusal': np.array([0., 0, 1.])},
                             weights={'mesial': np.ones(len(crown.vertices))},
                             protection=np.ones(len(crown.vertices)))
    entry = {'confirmation_status': 'CONFIRMED', 'pollution_reviewed': True}
    neighbor.apply_translation([3, 0, 0])
    valid = neighbor_patch_diagnostics(crown, region, {'mesial': (FitSurface(neighbor), entry)})['mesial']
    assert valid['status'] == 'VALID'
    assert valid['expected_side'] and valid['patch_faces'] == len(neighbor.faces)
    assert valid['axial_overlap_mm'] > 0 and valid['facing_score'] > .25
    assert valid['valid_signed_samples'] > 0

    surface = FitSurface(neighbor)
    def unsigned_query(points):
        cp, faces, signed, distance, _ = surface.query(points)
        return cp, faces, signed, distance, np.zeros(len(points), bool)
    uncertain = SimpleNamespace(mesh=neighbor, solid=False, outward=np.array([-1., 0, 0]),
                                query=unsigned_query)
    no_sign = neighbor_patch_diagnostics(crown, region, {'mesial': (uncertain, entry)})['mesial']
    assert no_sign['status'] == 'INVALID' and 'NO_VALID_SIGNED_SAMPLES' in no_sign['reasons']

    neighbor.apply_translation([-6, 0, 0])
    wrong = neighbor_patch_diagnostics(crown, region, {'mesial': (FitSurface(neighbor), entry)})['mesial']
    assert wrong['status'] == 'INVALID' and 'WRONG_SIDE' in wrong['reasons']
