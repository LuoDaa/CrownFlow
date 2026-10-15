"""Implant localization must not turn a scan component into an approved crown support."""

import json

import numpy as np
import trimesh
from fastapi.testclient import TestClient


def _implant_mesh():
    angles = np.linspace(0, 2 * np.pi, 128, endpoint=False)
    levels = np.linspace(-5, 5, 12)
    vertices = [[2.2 * np.cos(a), 2.2 * np.sin(a), z] for z in levels for a in angles]
    faces = []
    for layer in range(len(levels) - 1):
        for i in range(len(angles)):
            a = layer * len(angles) + i
            b = layer * len(angles) + (i + 1) % len(angles)
            c = a + len(angles)
            d = b + len(angles)
            faces.extend(((a, b, c), (b, d, c)))
    vertices.extend(((0, 0, -5), (0, 0, 5)))
    for i in range(len(angles)):
        faces.append((len(vertices) - 2, (i + 1) % len(angles), i))
        faces.append((len(vertices) - 1, (len(levels) - 1) * len(angles) + i,
                      (len(levels) - 1) * len(angles) + (i + 1) % len(angles)))
    return trimesh.Trimesh(vertices=vertices, faces=faces, process=True)


def test_implant_candidate_requires_restorative_abutment_confirmation(isolated_db, monkeypatch):
    from app.api import routes
    from app.main import app

    monkeypatch.setattr(routes, 'CASE_DIR', isolated_db / 'cases')
    client = TestClient(app)
    case = client.post('/api/cases', json={'name': 'implant-review'}).json()['id']
    jaw = trimesh.creation.annulus(r_min=2.25, r_max=12, height=10, sections=512)
    scan = trimesh.util.concatenate((jaw, _implant_mesh()))
    uploaded = client.post(f'/api/cases/{case}/assets/lower',
                           files={'file': ('lower.stl', scan.export(file_type='stl'), 'model/stl')})
    assert uploaded.status_code == 200, uploaded.text

    located = client.get(f'/api/cases/{case}/implant-candidates?source=lower')
    assert located.status_code == 200, located.text
    assert located.json()['status'] == 'CANDIDATES_FOUND'
    candidate = located.json()['candidates'][0]
    assert candidate['classification'] == 'implant_component_candidate'
    design = client.post('/api/designs', json={'case_id': case, 'target_fdi': 46,
                                                'support_type': 'abutment', 'draft': True}).json()['id']
    request = {'center': candidate['center'], 'axis': candidate['axis'], 'radius_mm': 8}
    first = client.post(f'/api/designs/{design}/draft-preparation', json=request)
    assert first.status_code == 200, first.text
    assert first.json()['points'] == []
    support = client.get(f'/api/cases/{case}').json()['assets'][-1]
    assert json.loads(support['mesh_info_json'])['restorative_abutment_confirmed'] is False
    assert len(trimesh.load(support['path']).split(only_watertight=False)) == 1
    ring = [[2 * np.cos(a), 2 * np.sin(a), 2] for a in np.linspace(0, 2 * np.pi, 16, endpoint=False)]
    blocked = client.post(f'/api/designs/{design}/margin', json={'points': ring})
    assert blocked.status_code == 409

    rejected = client.post(f'/api/designs/{design}/draft-preparation',
                           json={**request, 'restorative_abutment_confirmed': True})
    assert rejected.status_code == 422, 'A cylinder without a shoulder must not receive a fabricated margin'
    final_support = trimesh.creation.revolve(
        np.array([[0,-4],[1.3,-4],[3,0],[2.2,0],[2.2,4],[0,4]]), sections=128)
    final_support.apply_transform(trimesh.transformations.rotation_matrix(np.radians(20), [1, 0, 0]))
    uploaded = client.post(f'/api/cases/{case}/assets/support',
                           files={'file': ('restorative_abutment.stl', final_support.export(file_type='stl'), 'model/stl')})
    assert uploaded.status_code == 200, uploaded.text
    second = client.post(f'/api/designs/{design}/draft-preparation',
                         json={**request, 'restorative_abutment_confirmed': True})
    assert second.status_code == 200, second.text
    assert second.json()['status'] == 'DRAFT_REVIEW_REQUIRED'
    assert len(second.json()['points']) == 128
    expected_axis = trimesh.transformations.rotation_matrix(np.radians(20), [1, 0, 0])[:3, :3] @ [0, 0, 1]
    assert abs(np.dot(second.json()['axis'], expected_axis)) > .98
    saved = client.post(f'/api/designs/{design}/margin', json={'points': second.json()['points']})
    assert saved.status_code == 200, saved.text
