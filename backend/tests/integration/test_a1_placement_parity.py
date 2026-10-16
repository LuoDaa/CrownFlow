"""A1 回归：**同一个已确认帧 + 同一颗模板 ⇒ 同一个 baseline**。

A1 之前 live 有三条互不相干的放置实现：Preview 用确认的 mesial/buccal/axis 帧、
workflow/generate 用 world-X 推断、legacy /outer 又是第三条（kabsch+RBF）。
审计 P0-1 描述的正是"预览里看到的和真正生成的不是一个东西"。

本文件把 live 路径钉在同一实现上，断言在**任何形变发生之前**逐顶点/逐字节等价：

    /auto-fit/template-preview  ↔  /workflow/generate 的 baseline
    /outer/auto                 ↔  /workflow/generate 的 baseline

TEMPLATE_ONLY 候选本身不做形变，所以"形变前"在这里就是"产物本身"。
"""

from __future__ import annotations

import json

import numpy as np
import pytest
import trimesh

from app import config
from app.services.design_dependencies import file_digest

from .test_auto_fit import ready  # noqa: F401  （fixture 复用）

LIBRARY = "洋溢-长夜同在"


def _library_margin():
    """牙库件能通过尺寸门（factor∈[0.4,2.5]）的颈缘：半径 4、带轻微非平面起伏。"""
    theta = np.arange(48) * np.pi / 24
    return np.c_[4 * np.cos(theta), 4 * np.sin(theta), 0.3 * np.sin(2 * theta)]


def _generate(client, path, kind='TEMPLATE_ONLY'):
    status = client.get(path + '/status').json()
    return client.post(path + '/workflow/generate',
                       json={'kind': kind, 'confirmed': True,
                             'expected_input_digest': status['input_digest']})


def test_outer_auto_and_generate_baseline_are_byte_identical(ready):  # noqa: F811
    """初稿入口与候选 baseline 必须是同一次放置（同一模板、同一帧、同一实现）。"""
    client, path, _, _ = ready
    design = client.get(path).json()
    draft = client.post(path + '/outer/auto', json={'margin_points': json.loads(design['margin_json']),
                                                    'insertion_axis': json.loads(design['axis_json'])})
    assert draft.status_code == 200, draft.text
    assert draft.json()['frame_source'] == 'CONFIRMED_FRAME'
    assert draft.json()['placement_path'] == 'INITIALIZED_ONCE'
    draft_outer = client.get(path).json()['outer_path']

    generated = _generate(client, path)
    assert generated.status_code == 200, generated.text
    report = generated.json()['payload']
    baseline = generated.json()['artifacts']['baseline']
    assert report['baseline_sha256'] == file_digest(baseline)
    assert file_digest(baseline) == file_digest(draft_outer), "初稿与候选 baseline 的字节必须一致"


def test_generate_report_carries_truth_source(ready):  # noqa: F811
    """候选报告必须自证：哪颗模板（sha256）、哪个确认帧（revision_id）、哪条实现。"""
    client, path, _, _ = ready
    generated = _generate(client, path)
    assert generated.status_code == 200, generated.text
    report = generated.json()['payload']
    frame_value = json.loads(client.get(path).json()['autofit_frame_json'])

    assert report['template_source']['source_kind'] == 'FALLBACK_DESIGN_DIR'
    assert report['template_source']['sha256'] == file_digest(config.TEMPLATE_DIR / 'FDI_46.stl')
    assert report['patient_frame'] == {'revision_id': frame_value['revision_id'], 'source': 'CONFIRMED'}
    assert report['frame_source'] == 'CONFIRMED_FRAME'
    assert report['placement_path'] == 'INITIALIZED_ONCE'
    assert report['baseline_sha256'] == file_digest(generated.json()['artifacts']['baseline'])


def test_template_preview_and_generate_baseline_agree_before_deformation(ready):  # noqa: F811
    """牙库路径：Preview 的放置结果必须与 generate 的 baseline 逐字节相同。"""
    from app.config import TOOTH_LIBRARY_ROOT
    if not TOOTH_LIBRARY_ROOT.is_dir():
        pytest.skip('external professional tooth library unavailable')
    client, path, _, _ = ready

    margin = _library_margin()
    assert client.post(path + '/margin', json={'points': margin.tolist()}).status_code == 200
    # A1：帧的原点取自边缘线质心，边缘线一变就必须重新确认坐标（不允许静默漂移）
    digest = client.get(path + '/status').json()['input_digest']
    assert client.put(path + '/auto-fit/frame',
                      json={'mesial': [1, 0, 0], 'buccal': [0, 1, 0], 'units': 'mm',
                            'confirmed': True, 'expected_input_digest': digest}).status_code == 200

    status = client.get(path + '/status').json()
    preview = client.post(path + '/auto-fit/template-preview',
                          json={'library': LIBRARY, 'confirmed': True,
                                'expected_input_digest': status['input_digest']})
    assert preview.status_code == 200, preview.text
    preview_outer = preview.json()['artifacts']['outer']
    accepted = client.post(path + f"/auto-fit/{preview.json()['id']}/accept",
                           json={'confirmed': True, 'confirmed_template_anatomy': True,
                                 'expected_input_digest': status['input_digest']})
    assert accepted.status_code == 200, accepted.text

    generated = _generate(client, path)
    assert generated.status_code == 200, generated.text
    report = generated.json()['payload']
    baseline = generated.json()['artifacts']['baseline']

    assert report['template_source']['source_kind'] == 'USER_SELECTED'
    assert report['template_source']['library'] == LIBRARY
    assert report['placement_path'] == 'LIBRARY_PLACED_ONCE'
    assert file_digest(baseline) == file_digest(preview_outer), \
        'Preview 与 Generate baseline 在形变前必须逐字节等价'

    left = trimesh.load(baseline, process=True)
    right = trimesh.load(preview_outer, process=True)
    assert np.allclose(left.vertices, right.vertices, atol=1e-9)
    assert np.array_equal(left.faces, right.faces)


def test_draft_does_not_wipe_the_confirmed_template_selection(ready):  # noqa: F811
    """初稿入口不得整表覆盖 landmarks_json：已确认的牙库件必须留着（否则下一次生成偷偷换模板）。"""
    from app.config import TOOTH_LIBRARY_ROOT
    if not TOOTH_LIBRARY_ROOT.is_dir():
        pytest.skip('external professional tooth library unavailable')
    client, path, _, _ = ready

    assert client.post(path + '/margin', json={'points': _library_margin().tolist()}).status_code == 200
    digest = client.get(path + '/status').json()['input_digest']
    assert client.put(path + '/auto-fit/frame',
                      json={'mesial': [1, 0, 0], 'buccal': [0, 1, 0], 'units': 'mm',
                            'confirmed': True, 'expected_input_digest': digest}).status_code == 200
    status = client.get(path + '/status').json()
    preview = client.post(path + '/auto-fit/template-preview',
                          json={'library': LIBRARY, 'confirmed': True,
                                'expected_input_digest': status['input_digest']})
    assert preview.status_code == 200, preview.text
    assert client.post(path + f"/auto-fit/{preview.json()['id']}/accept",
                       json={'confirmed': True, 'confirmed_template_anatomy': True,
                             'expected_input_digest': status['input_digest']}).status_code == 200
    selected = json.loads(client.get(path).json()['landmarks_json'])['_selected_template']

    design = client.get(path).json()
    draft = client.post(path + '/outer/auto', json={'margin_points': json.loads(design['margin_json']),
                                                    'insertion_axis': json.loads(design['axis_json'])})
    assert draft.status_code == 200, draft.text
    after = json.loads(client.get(path).json()['landmarks_json'])
    assert after.get('_selected_template') == selected, '初稿把已确认的模板选择抹掉了'
    assert after.get('mode') == 'auto_initialization'
    assert draft.json()['template_source']['source_kind'] == 'USER_SELECTED'
    assert draft.json()['template_source']['library'] == LIBRARY
    assert draft.json()['placement_path'] == 'LIBRARY_PLACED_ONCE'
    baseline = _generate(client, path).json()['artifacts']['baseline']
    assert file_digest(baseline) == file_digest(client.get(path).json()['outer_path']), \
        '采纳牙库件之后的初稿与候选 baseline 仍须逐字节一致'


def test_unconfirmed_frame_blocks_live_generation_without_guessing_axes(isolated_db, monkeypatch):
    """没有确认帧时，live 生成必须拒绝并指出缺的是坐标确认（不用 world-X 凑方向）。"""
    from app.api import routes, auto_fit
    from app.services import design_dependencies
    monkeypatch.setattr(routes, 'CASE_DIR', isolated_db / 'cases')
    monkeypatch.setattr(auto_fit, 'CASE_DIR', isolated_db / 'cases')
    monkeypatch.setattr(routes, 'TEMPLATE_DIR', isolated_db / 'templates')
    monkeypatch.setattr(design_dependencies, 'TEMPLATE_DIR', isolated_db / 'templates')
    # 这条测试只验证 FRAME blocker；先给一个合法模板，避免被“不存在模板”这一独立前置条件抢先拦截。
    template_path = isolated_db / 'templates' / 'FDI_46.stl'
    template_path.write_bytes(trimesh.creation.box(extents=(6.0, 4.0, 8.0)).export(file_type='stl'))
    from fastapi.testclient import TestClient
    from app.main import app

    client = TestClient(app)
    case = client.post('/api/cases', json={'name': 'a1-no-frame'}).json()['id']
    prep = trimesh.creation.cone(radius=2, height=4, sections=32)
    assert client.post(f'/api/cases/{case}/assets/support',
                       files={'file': ('support.stl', prep.export(file_type='stl'), 'model/stl')}).status_code == 200
    design = client.post('/api/designs', json={'draft': True, 'case_id': case, 'target_fdi': 46,
                                               'support_type': 'prep'}).json()['id']
    path = f'/api/designs/{design}'
    margin = _library_margin()[:32].tolist()
    assert client.post(path + '/margin', json={'points': margin}).status_code == 200
    assert client.post(path + '/axis', json={'axis': [0, 0, 1]}).status_code == 200
    assert client.put(path + '/profile',
                      json={'material': 'unvalidated fixture', 'coverage': 'ENGINEERING_ONLY',
                            'bite_confirmed': True, 'confirmed': True,
                            'parameters': {'machine_axes': 5, 'minimum_thickness_mm': .2}}).status_code == 200

    refusal = client.post(path + '/outer/auto', json={'margin_points': margin, 'insertion_axis': [0, 0, 1]})
    assert refusal.status_code == 409 and '确认' in refusal.json()['detail']

    state = client.get(path + '/workflow/eligibility').json()
    assert [b['stage'] for b in state['blockers']] == ['FRAME'], state
    assert state['template_allowed'] is False
    generated = _generate(client, path)
    assert generated.status_code == 409
    assert 'FRAME' in [b['stage'] for b in generated.json()['detail']['blockers']]
