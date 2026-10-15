"""J.1 Intaglio-Constrained CrownOuter 单测。

合成例要能解析判分，所以用**杯壳（侧壁 + 顶盖，底部开放 = 唯一颈环）+ 杯形内表面**：
壁厚处处可算、颈环钉死可验、"落进预备体内部"能精确构造。
真实例（H.1 + 代型 intaglio）走 `tools/run_outer_v2_case.py`。

⚠️ 夹具必须是**单边界**壳：第一版拿两头开口的直筒当"冠"，两端都成了边界 →
96 个顶点被当成"锁定/刀口带"（那版全部假失败）。真实冠也只有一条边界
（Occlusion 输出 146 条边界边 = 一个环）。

守住的语义（规格六条）：
1. 颈环位移**精确 0**（共享颈缘边界）
2. 包络违反 → 0（Outer 包住 Intaglio；**带符号**距离，落在预备体内部也算违反）
3. 局部厚度 ≥ T（用 audit 的 t_closest 复验，见 test_envelope_implies_audit_measure）
5/6. 无翻面 / 无自交（硬门；过不去就 CONSTRAINT_CONFLICT，不硬推）
4. 解剖偏差最小：**未违反处位移恒 0**
"""
from __future__ import annotations

import numpy as np
import pytest

from app.geometry.crown_outer.intaglio_constrained import (
    STATUS_OK,
    OuterV2Error,
    OuterV2Params,
    build_intaglio_constrained_outer,
)
from app.geometry.crown_outer.thickness import (
    SurfaceQuery,
    ThicknessParameters,
)


def _cup(radii, zs, cap_up, n=48):
    """侧壁（每圈半径 radii[j] / 高度 zs[j]）+ 顶盖扇到 (0,0,cap_up)。

    底部开放 → **恰好一条边界**（颈环），与真实冠同构。
    """
    radii = list(radii)
    zs = list(zs)
    assert len(radii) == len(zs) >= 2
    v, f = [], []
    for k in range(n):
        a = 2 * np.pi * k / n
        for j in range(len(zs)):
            v.append((np.cos(a) * radii[j], np.sin(a) * radii[j],
                      float(zs[j])))
    v.append((0.0, 0.0, float(cap_up)))
    apex = len(v) - 1
    n_r = len(zs)
    for k in range(n):
        k2 = (k + 1) % n
        for j in range(n_r - 1):
            a0, a1 = n_r * k + j, n_r * k + j + 1
            b0, b1 = n_r * k2 + j, n_r * k2 + j + 1
            f += [(a0, b0, b1), (a0, b1, a1)]
        f.append((n_r * k + n_r - 1, n_r * k2 + n_r - 1, apex))
    return np.asarray(v, float), np.asarray(f, np.int64)


def _intaglio_cup(r=2.0, z_top=4.5, n_rings=12, n=48, dome=0.5):
    """内表面（预备体 + gap）：底部开放、顶部**穹顶**封口。

    顶盖做成穹顶（不是平盘）：真实预备体的咬合面是圆钝的；平盘会造出 90° 锐棱，
    包络投影在锐棱上收敛极慢（相邻两面互相拉扯），那不是真实工况。
    """
    zs = list(np.linspace(0.0, z_top * 0.8, n_rings))
    radii = [r] * n_rings
    for k in (1, 2, 3, 4):                      # 穹顶：半径递减、高度递增
        zs.append(z_top * 0.8 + (z_top - z_top * 0.8) * k / 4.0)
        radii.append(r * (1.0 - 0.30 * k / 4.0))
    return _cup(radii, zs, z_top + dome, n=n)


class _Mesh:
    def __init__(self, v, f):
        self.vertices = np.asarray(v, float)
        self.faces = np.asarray(f, np.int64)


def _rule(t_min=0.5, near=None, near_d=None):
    return ThicknessParameters(min_thickness_mm=t_min,
                               near_margin_thickness_mm=near,
                               near_margin_distance_mm=near_d)


def _params(t_min=0.5, safety=0.0, **kw):
    return OuterV2Params(thickness=_rule(t_min), safety_mm=safety, **kw)


ZS12 = list(np.linspace(0.0, 4.5, 12))


def _cup12(radii_fn, cap_up, zs=None, n=48):
    zs = list(zs or ZS12)
    return _cup([radii_fn(z) for z in zs], zs, cap_up, n=n)


def test_params_required():
    v, f = _cup12(lambda z: 2.2, 4.8)
    with pytest.raises(OuterV2Error):
        build_intaglio_constrained_outer(_Mesh(v, f), _Mesh(*_intaglio_cup()),
                                         params=None)
    with pytest.raises(OuterV2Error):
        build_intaglio_constrained_outer(_Mesh(v, f), _Mesh(*_intaglio_cup()),
                                         params=OuterV2Params(thickness=None))


def test_no_violation_at_all_is_noop():
    """壁厚已经处处 ≥ T：一个顶点都不许动（解剖偏差 = 0）。"""
    av, af = _cup12(lambda z: 3.5, 7.0)                  # 壁 1.5、顶盖高 2.5
    iv, if_ = _intaglio_cup(2.0, 4.5)
    res = build_intaglio_constrained_outer(_Mesh(av, af), _Mesh(iv, if_),
                                           _params(t_min=0.5))
    assert res.status == STATUS_OK
    assert res.envelope_before["n_violating_outer"] == 0
    assert res.envelope_before["n_violating_intaglio"] == 0
    assert res.anatomy_deviation["n_moved_vertices"] == 0
    v = np.asarray(res.mesh.vertices, float)
    assert np.allclose(v[:len(av)], av)


def test_violation_is_pushed_out_and_residual_is_reported():
    """壁 0.1mm → 推到 ≥T 方向；颈环精确 0；残余如实上报（不假称达标）。

    ⚠️ 断言的是**语义 + 实质改善 + 报告诚实**，不是"一次到位归零"：
    包络投影在大变形下会被几何安全门拒绝（真实例 α=1 面积翻面 18.5%），
    所以引擎按线搜索小步推进并如实报残余 —— 这是被测行为本身。
    """
    t = 0.6
    av, af = _cup12(lambda z: 2.1, 5.6)
    iv, if_ = _intaglio_cup()
    res = build_intaglio_constrained_outer(_Mesh(av, af), _Mesh(iv, if_),
                                           _params(t_min=t, safety=0.0,
                                                   max_outer_iterations=20))
    assert res.status in (STATUS_OK, "REVIEW_REQUIRED")
    b, a = res.envelope_before, res.envelope_after
    assert a["energy_mm"] < 0.5 * b["energy_mm"]        # 实质改善
    assert a["max_deficit_mm"] < b["max_deficit_mm"]
    if res.status == "REVIEW_REQUIRED":
        assert any("残余" in n for n in res.notes)       # 诚实上报
    v = np.asarray(res.mesh.vertices, float)
    loop = np.abs(v[:, 2]) < 1e-9
    assert np.allclose(np.linalg.norm(v[loop][:, :2], axis=1), 2.1)  # 颈环不动
    s = SurfaceQuery(iv, if_).query(v)[2]
    assert s[~loop].min() >= 0.0                         # 不再落进预备体内部
    assert res.safety["ok"]                              # 硬门（自交/面积翻面）


def test_anatomy_untouched_where_no_violation():
    """只有下半段薄：不违反的圈与顶盖必须一动不动。"""
    t = 0.6
    # 半径随高度从 2.1 涨到 3.0 → 下半段薄（0.1）、上半段厚（≥0.9）
    av, af = _cup12(lambda z: 2.1 + 0.9 * (z / 4.5), 6.5)
    iv, if_ = _intaglio_cup(2.0, 4.5)
    res = build_intaglio_constrained_outer(_Mesh(av, af), _Mesh(iv, if_),
                                           _params(t_min=t, safety=0.0,
                                                   max_outer_iterations=10))
    assert res.status in (STATUS_OK, "REVIEW_REQUIRED")
    v = np.asarray(res.mesh.vertices, float)[:len(av)]
    av = np.asarray(av, float)
    moved = np.linalg.norm(v - av, axis=1)
    z = av[:, 2]
    assert moved[np.abs(z) < 1e-9].max() == 0.0          # 颈环：精确 0
    assert moved[np.abs(z - 4.5) < 1e-9].max() < 1e-9    # 顶圈（厚 1.0）不动
    assert moved[z > 5.5].max() < 1e-9                   # 顶盖（不违反）不动
    assert moved[np.abs(z - 0.818) < 1e-3].max() > 0.05  # 薄的下段确实被推


def test_envelope_implies_audit_measure():
    """包络约束 ⇒ Audit 的 t_closest 改善（规格里那条可证蕴含，实测复核）。

    要求的是**单调改善到包络外**（残余由引擎如实上报），不是断言一次归零：
    大变形下 α 被硬安全门限制（见 test_violation_is_pushed_out... 的说明）。
    """
    t = 0.5
    av, af = _cup12(lambda z: 2.1, 5.5)
    iv, if_ = _intaglio_cup()
    res = build_intaglio_constrained_outer(_Mesh(av, af), _Mesh(iv, if_),
                                           _params(t_min=t, safety=0.0,
                                                   max_outer_iterations=20))
    sq_before = SurfaceQuery(av, af, boundary_safety_mm=0.0)
    d_before = sq_before.query(iv)[3]
    sq_out = SurfaceQuery(np.asarray(res.mesh.vertices, float),
                          np.asarray(res.mesh.faces, np.int64),
                          boundary_safety_mm=0.0)
    d = sq_out.query(iv)[3]
    interior = iv[:, 2] > 0.5      # 颈缘那圈与冠颈缘重合（刀口）→ 按 audit 口径排除
    assert np.median(d[interior]) > np.median(d_before[interior]) + 0.05
    assert d[interior].min() > 0.0          # 不再有内表面点落在冠外（负壁厚）


def test_inside_prep_is_detected_and_pushed_out():
    """外表面整片落在预备体**内部**：必须判为违反并推到外面（覆盖率失败场景）。"""
    av, af = _cup12(lambda z: 2.0, 4.5)                  # 与内表面重合
    iv, if_ = _intaglio_cup(2.0, 4.5)
    v0 = np.asarray(av, float)
    res = build_intaglio_constrained_outer(_Mesh(av, af), _Mesh(iv, if_),
                                           _params(t_min=0.5, safety=0.0,
                                                   max_outer_iterations=20))
    assert res.status in (STATUS_OK, "REVIEW_REQUIRED")
    v = np.asarray(res.mesh.vertices, float)
    s = SurfaceQuery(iv, if_).query(v)[2]
    loop = np.abs(v[:, 2]) < 1e-9
    # 负壁厚大幅消除（不是断言"一次清零"：要推 ~1.7mm，α 被硬安全门限制，
    # 引擎按小步推进并如实报残余 —— 真实例同现象，见 docs/OUTER_V2_J1.md §5）
    n_before = res.envelope_before["n_inside_prep"]
    n_after = res.envelope_after["n_inside_prep"]
    assert n_before > 0                       # 初始确实整片在预备体内（检测到了）
    assert n_after < n_before                 # 方向正确（往外推）
    assert res.safety["ok"]                   # 硬门：不许靠折叠换进度
    assert np.allclose(np.linalg.norm(v[loop][:, :2], axis=1), 2.0)


def test_fail_loud_when_input_orientation_wrong():
    """内表面法向整体朝内（绕制反了）→ **不自动翻号**，fail loud。"""
    iv, if_ = _intaglio_cup(2.0, 4.5)
    av, af = _cup12(lambda z: 2.1, 4.8)
    with pytest.raises(OuterV2Error):
        build_intaglio_constrained_outer(
            _Mesh(av, af), _Mesh(iv, if_[:, ::-1].copy()),
            _params(t_min=0.5))
