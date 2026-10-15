"""B 阶段唯一接触口径的服务级测试（不经过 HTTP）。

钉住三件事：
1. 报告形状与符号语义（求解器/AutoFit/审核/QC 共用同一个形状）；
2. 多参考面合成规则：逐顶点取**最不利**（最小）有符号值，无效样本退回幅值并标记；
3. 符号不可得时的**显式降级**：只给幅值 + 机器可读原因，绝不伪造穿透结论。
"""

from __future__ import annotations

import numpy as np
import pytest
import trimesh

from app.geometry.crown_outer.auto_fit_adapter import FitSurface
from app.services import contact_measurement as CM


def _crown():
    return trimesh.creation.icosphere(subdivisions=2, radius=3.0)


def test_report_is_signed_and_documented():
    crown = _crown()
    neighbor = trimesh.creation.box(extents=(2, 2, 6)).apply_translation([3.5, 0, 0])
    surface = FitSurface(neighbor)
    report, query = CM.measure_surface(surface, crown.vertices,
                                       np.ones(len(crown.vertices), bool), 0.05,
                                       kind=CM.KIND_PROXIMAL, side='mesial', crown='deadbeef')
    assert report['status'] == CM.STATUS_MEASURED
    assert report['kind'] == 'PROXIMAL' and report['side'] == 'mesial'
    assert report['crown_sha256'] == 'deadbeef'
    assert report['implementation'].startswith('crown_outer.auto_fit_adapter.FitSurface')
    assert report['sign_semantics'].startswith('positive=gap_to_reference')
    assert report['sign_method'] == CM.SIGN_GLOBAL  # 闭合可定向实体
    # 有符号最小间隙 = -最大穿透深度；两者必须自洽
    assert report['signed_min_mm'] == pytest.approx(-report['penetration_max_mm']
                                                    if report['signed_min_mm'] < 0 else report['signed_min_mm'])
    assert report['signed']['min_mm'] == pytest.approx(report['signed_min_mm'])
    assert report['target_mm'] == report['effective_target_gap_mm'] == 0.05
    ids, cp, fids, signed, distance, valid = query
    assert np.allclose(distance, np.abs(signed), atol=1e-9), '幅值必须有符号的绝对值'


def test_penetration_is_negative_signed_not_unsigned():
    """穿进邻牙时必须给出负的有符号值（旧的无符号口径看不出来）。"""
    crown = _crown()
    neighbor = trimesh.creation.box(extents=(2, 2, 6)).apply_translation([2.4, 0, 0])  # 与冠体重叠
    report, _ = CM.measure_surface(FitSurface(neighbor), crown.vertices,
                                   np.ones(len(crown.vertices), bool), 0.05, kind=CM.KIND_PROXIMAL)
    assert report['signed_min_mm'] < 0, report
    assert report['penetration_max_mm'] == pytest.approx(-report['signed_min_mm'], abs=1e-9)
    assert report['penetration_max_mm'] > 0


def test_empty_mask_is_not_assessed():
    crown = _crown()
    report, query = CM.measure_surface(FitSurface(trimesh.creation.box(extents=(2, 2, 2))),
                                       crown.vertices, np.zeros(len(crown.vertices), bool), 0.05,
                                       kind=CM.KIND_PROXIMAL)
    assert query is None and report['status'] == CM.STATUS_NOT_ASSESSED


def test_combine_takes_worst_case_and_keeps_finite_magnitudes():
    items = {
        'a': {'ids': [0, 1, 2], 'signed_mm': [0.3, 0.2, -0.1], 'distance_mm': [0.3, 0.2, 0.1],
              'valid': [True, True, False]},
        'b': {'ids': [0, 1, 2], 'signed_mm': [0.5, -0.4, 0.9], 'distance_mm': [0.5, 0.4, 0.9],
              'valid': [True, True, True]},
    }
    combined = CM.combine_per_vertex(items)
    # 顶点 0：两侧都有效 → 取更小（更不利）的 0.3
    assert combined['signed_mm'][0] == pytest.approx(0.3)
    # 顶点 1：b 侧 -0.4 更不利 → 取 -0.4
    assert combined['signed_mm'][1] == pytest.approx(-0.4)
    # 顶点 2：只有 b 有效 → 取 b 的 0.9（a 的 -0.1 因无效不得参与）
    assert combined['signed_mm'][2] == pytest.approx(0.9)
    assert combined['valid'] == [True, True, True]
    assert np.isfinite(combined['distance_mm']).all()
    assert combined['combine_rule'] == 'min_signed_over_available_references'


def test_combine_falls_back_to_magnitude_for_fully_ambiguous_vertex():
    items = {'a': {'ids': [0], 'signed_mm': [-2.0], 'distance_mm': [2.0], 'valid': [False]}}
    combined = CM.combine_per_vertex(items)
    assert combined['valid'] == [False]           # 结论不可用
    assert combined['signed_mm'][0] == pytest.approx(2.0)   # 但热力图仍要有有限数
    assert combined['ambiguous_count'] == 1


def test_unsigned_only_refusal_is_explicit():
    reference = trimesh.creation.box(extents=(4, 4, 4))
    result = CM.unsigned_reference_distance(reference.vertices, reference,
                                            reason='REFERENCE_OUTWARD_NORMAL_UNCONFIRMED')
    assert result['sign_method'] == CM.SIGN_UNSIGNED_ONLY
    assert result['signed_mm'] is None and result['status'] == CM.STATUS_NOT_ASSESSED
    assert result['reason'] == 'REFERENCE_OUTWARD_NORMAL_UNCONFIRMED'
    assert result['sign_semantics'] == 'magnitude_only_no_penetration_claim'


def test_frame_from_design_rebuilds_confirmed_frame():
    import json
    margin = np.c_[np.cos(np.linspace(0, 2 * np.pi, 8, endpoint=False)) * 2,
                   np.sin(np.linspace(0, 2 * np.pi, 8, endpoint=False)) * 2, np.zeros(8)]
    design = {'margin_json': json.dumps(margin.tolist()), 'axis_json': json.dumps([0, 0, 1])}
    frame_ref = {'x': [1, 0, 0], 'y': [0, 1, 0], 'z': [0, 0, 1], 'revision_id': 'rev-1'}
    frame = CM.frame_from_design(design, frame_ref)
    assert frame.revision_id == 'rev-1'
    assert np.allclose(frame.origin, 0, atol=1e-9)
    assert np.allclose(frame.matrix(), np.eye(3), atol=1e-9)


def test_measure_confirmed_inputs_without_frame_is_not_assessed():
    design = {'margin_json': '[]', 'axis_json': '[0,0,1]', 'neighbors_json': '{}', 'bite_json': '{}'}
    result = CM.measure_confirmed_inputs(design, {}, 'whatever.stl')
    assert result['status'] == CM.STATUS_NOT_ASSESSED and result['reason'] == 'FRAME_UNCONFIRMED'
    stripped = CM.serializable_measurement(result)
    assert 'surfaces' not in stripped and 'domains' not in stripped  # 进程内对象必须被剥掉
