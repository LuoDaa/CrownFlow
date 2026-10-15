"""B 回归（审计 #5）：四条路径说**同一句**接触结论。

审计原话是"Placement 有 signed penetration，但 Candidate Review 仍只有 unsigned"——
同一个几何在不同页面给出互相矛盾的结论。本文件把四条路径钉在同一个测量上：

    放置（placement_contact）  /workflow/generate 的报告
    AutoFit（metrics_before）   propose 的求解前度量
    候选审核（analysis）         GET /auto-fit/{id}/analysis
    最终 QC（ContactQC）        services.auto_fit.audit_final_context

同一颗冠 + 同一份已确认邻牙 ⇒ 四处的 valid_count / 有符号统计必须一致
（数值按同一实现算，容差取 1e-9：这不是"差不多"，是同一份数）。
"""

from __future__ import annotations

import json

import pytest
import trimesh

from app import config
from app.services.design_dependencies import file_digest

from .test_auto_fit import prepare, ready  # noqa: F401  （fixture 复用）

SIGNED_KEYS = ("valid_count", "ambiguous_count")
SIGNED_VALUES = ("signed_min_mm", "penetration_max_mm", "signed_p95_mm", "target_error_p95_mm", "deficit_p95_mm")


def _flatten(report):
    """把四条路径各自的报告压成同一组可比数字（都是同一实现算出来的）。"""
    return {"signed_min_mm": report["signed_min_mm"], "penetration_max_mm": report["penetration_max_mm"],
            "signed_p95_mm": report["signed_valid"]["p95_mm"],
            "target_error_p95_mm": report["target_error"]["p95_mm"],
            "deficit_p95_mm": report["deficit"]["p95_mm"],
            "valid_count": report["valid_count"], "ambiguous_count": report["ambiguous_count"]}


def _compare(left, right, label):
    left, right = _flatten(left), _flatten(right)
    for key in SIGNED_KEYS:
        assert left[key] == right[key], "%s 的 %s 不一致：%r vs %r" % (label, key, left[key], right[key])
    for key in SIGNED_VALUES:
        assert left[key] == pytest.approx(right[key], abs=1e-9), \
            "%s 的 %s 不一致：%r vs %r" % (label, key, left[key], right[key])


@pytest.fixture()
def measured_design(ready):  # noqa: F811
    """已确认帧 + 近中邻牙 + 远中缺牙 + 目标间隙的 AUTO 候选。

    模板换成**顶点加密**的圆柱：`ready` 用的粗圆柱只有上下两圈顶点，
    中间被 neck/occlusal 掩码切光后邻接接触带恒为空 —— 那样测到的是"没有采样点"，
    而不是"四条路径口径不一致"。
    """
    client, path, design, neighbors = ready
    template = trimesh.creation.cylinder(radius=3.0, height=6.0, sections=32)
    (config.TEMPLATE_DIR / 'FDI_46.stl').write_bytes(template.subdivide().subdivide().export(file_type='stl'))
    row = client.get(path).json()
    draft = client.post(path + '/outer/auto', json={'margin_points': json.loads(row['margin_json']),
                                                    'insertion_axis': json.loads(row['axis_json'])})
    assert draft.status_code == 200, draft.text
    prepare(client, path, neighbors)
    state = client.get(path + '/workflow/eligibility').json()
    assert state['automatic_mode'] == 'PROXIMAL_ONLY', state
    generated = client.post(path + '/workflow/generate',
                            json={'kind': 'AUTO', 'confirmed': True,
                                  'expected_input_digest': state['input_digest']})
    assert generated.status_code == 200, generated.text
    return client, path, design, generated.json()


def test_four_paths_report_the_same_signed_contact(measured_design):
    """四条路径两两在同一颗冠上比较：基线对基线、候选对候选，几何必须逐一对应。"""
    client, path, design, candidate = measured_design
    baseline = candidate['artifacts']['baseline']
    outer = candidate['artifacts']['outer']
    report = candidate['payload']

    placement = report['placement_contact']['proximal']['mesial']      # 放置（基线）
    autofit_before = report['metrics_before']['proximal']['mesial']     # AutoFit 求解前（基线）

    analysis = client.get(path + f"/auto-fit/{candidate['id']}/analysis?kind=neighbors")
    assert analysis.status_code == 200, analysis.text
    review = analysis.json()['stats']['per_side']['mesial']             # 候选审核（候选）
    autofit_after = report['joint_recheck']['after']['mesial']          # AutoFit 求解后（候选）

    from app.api.routes import design_row
    from app.services.auto_fit import audit_final_context
    row = design_row(design)
    final_qc = {}
    audit_final_context(row, baseline, baseline, final_qc)               # 最终 QC（基线）
    final_qc_candidate = {}
    audit_final_context(row, outer, outer, final_qc_candidate)           # 最终 QC（候选）

    _compare(autofit_before, placement, 'AutoFit↔放置（基线）')
    _compare(autofit_before, final_qc['ContactQC']['per_side']['mesial'], 'AutoFit↔最终QC（基线）')
    # 候选审核与最终 QC 面对的是同一份候选文件、同一个接触域 ⇒ 逐个数字必须相同
    _compare(review, final_qc_candidate['ContactQC']['per_side']['mesial'], '候选审核↔最终QC（候选）')

    # AutoFit 的 after 用**基线冻结的接触域**（before/after 才可比），审核与 QC 用该文件自己的域
    # ⇒ 采样集合本来就不同（域大小如实写进报告）。所以这里比的是"同一颗冠给出同一结论"的
    # 有符号标量与实现标识，而不是采样个数。
    for key in ('implementation', 'sign_method', 'sign_semantics', 'kind', 'side', 'target_mm', 'status'):
        assert review[key] == autofit_after[key], key
    for key in ('signed_min_mm', 'penetration_max_mm'):
        assert review[key] == pytest.approx(autofit_after[key], abs=1e-9), key
    assert autofit_after['domain_vertex_count'] != review['domain_vertex_count'], \
        '域来源不同这件事必须体现在报告里（frozen baseline domain vs 候选自身域）'

    # 四条路径都必须是**有符号**的，而且必须承认同一个符号来源
    entries = {'AutoFit': autofit_after, '放置': placement, '候选审核': review,
               '最终QC': final_qc_candidate['ContactQC']['per_side']['mesial']}
    for label, entry in entries.items():
        assert entry['signed_min_mm'] is not None, label + ' 没有有符号结果'
        assert entry['sign_semantics'].startswith('positive=gap_to_reference'), label
        assert entry['sign_method'] in ('REVIEWED_LOCAL_NORMAL', 'GLOBAL_INSIDE_OUTSIDE'), \
            label + ' 的符号来源标注不对：%r' % entry['sign_method']
    assert len({entry['sign_method'] for entry in entries.values()}) == 1, \
        '四条路径对同一参考面必须承认同一个符号来源：%r' % {k: v['sign_method'] for k, v in entries.items()}

    # 候选审核页面不再是无符号-only
    payload = analysis.json()
    assert payload['sign_method'] == review['sign_method']
    assert payload['signed_mm'] is not None and len(payload['signed_mm']) == len(payload['vertices'])
    assert payload['distance_mm'] is not None and len(payload['distance_mm']) == len(payload['vertices'])
    assert '穿进参考面' in payload['note'] and '不证明接触或穿透' not in payload['note']
    assert payload['candidate_outer_sha256'] == file_digest(outer)
    assert payload['implementation'].startswith('crown_outer.auto_fit_adapter.FitSurface')


def test_placement_contact_is_present_and_degrades_honestly(measured_design):
    """放置快照必须带着"哪条输入没确认"，而不是拿别的参考面凑一个有符号数。"""
    client, path, design, candidate = measured_design
    placement = candidate['payload']['placement_contact']
    assert placement['status'] == 'MEASURED'
    assert placement['skipped']['distal']['status'] == 'SKIPPED_ABSENT'
    assert 'occlusal' in placement['skipped'] and placement['skipped']['occlusal']['status'] == 'NOT_ASSESSED'
    assert placement['proximal']['mesial']['kind'] == 'PROXIMAL'
    assert placement['crown_sha256'] == file_digest(candidate['artifacts']['baseline'])


def test_analysis_refuses_to_invent_sign_when_reference_has_no_normal(ready):  # noqa: F811
    """权威落点：参考面连外向法向都没有时，只给幅值并写明原因（不许伪造有符号结论）。"""
    from app.services.contact_measurement import (unsigned_reference_distance, SIGN_UNSIGNED_ONLY,
        STATUS_NOT_ASSESSED)
    import trimesh

    mesh = trimesh.creation.box(extents=(10, 10, 10))
    result = unsigned_reference_distance(mesh.vertices, mesh, reason='REFERENCE_OUTWARD_NORMAL_UNCONFIRMED')
    assert result['sign_method'] == SIGN_UNSIGNED_ONLY
    assert result['signed_mm'] is None and result['distance_mm'] is not None
    assert result['status'] == STATUS_NOT_ASSESSED
    assert result['sign_semantics'] == 'magnitude_only_no_penetration_claim'

    # 开放表面没有外向法向时，FitSurface 直接拒绝（不允许猜符号）
    from app.geometry.crown_outer.auto_fit_adapter import FitSurface
    open_patch = trimesh.Trimesh([[-1, -1, 0], [1, -1, 0], [0, 1, 0]], [[0, 1, 2]], process=False)
    with pytest.raises(ValueError):
        FitSurface(open_patch)


def test_service_report_shape_is_the_only_one(ready):  # noqa: F811
    """求解器消费的报告字段必须仍由唯一实现产出（形状不许分叉）。"""
    from app.geometry.crown_outer.auto_fit_adapter import FitSurface, measure
    import numpy as np
    import trimesh

    crown = trimesh.creation.icosphere(subdivisions=1, radius=3)
    neighbor = trimesh.creation.box(extents=(2, 2, 2)).apply_translation([3.05, 0, 0])
    surface = FitSurface(neighbor)
    report, query = measure(np.asarray(crown.vertices), np.ones(len(crown.vertices), bool), surface, 0.05)
    for key in ('status', 'valid_count', 'ambiguous_count', 'unsigned', 'signed_valid', 'penetration',
                'deficit', 'target_error', 'effective_target_gap_mm', 'sign_method', 'kind',
                'signed_min_mm', 'penetration_max_mm', 'implementation', 'sign_semantics'):
        assert key in report, key
    ids, cp, fids, signed, distance, valid = query
    assert len(ids) == len(crown.vertices) and len(signed) == len(ids)
    assert json.dumps(report, default=float) is not None      # 报告必须可序列化
