import sys
sys.path.insert(0, ".")
sys.path.insert(0, "tools")
from build_auto_margin_gt import (_classify_kind, ASSET_KIND_DIE,
                                  ASSET_KIND_INTAGLIO, ASSET_KIND_UNCLEAR)


def st(radial, above, ax_min=-3.0, n=5000, area=200.0):
    return dict(radial=radial, above_frac=above, ax_min=ax_min, ax_max=3.0,
                n=n, area=area)


def test_crown_shell_by_intaglio():
    # 实测：冠壳内表面 −0.83 / 外表面 +0.79
    assert _classify_kind([st(-0.83, 0.94), st(0.79, 0.96)]) == ASSET_KIND_INTAGLIO
    assert _classify_kind([st(0.26, 0.84), st(-0.829, 0.93)]) == ASSET_KIND_INTAGLIO


def test_solid_crown_body_is_not_die():
    # 实测 bridge_slm_cad：两分量 radial ≈ +0.1、above ≈ 0.5
    assert _classify_kind([st(0.118, 0.515, -2.0), st(0.096, 0.487, -4.0)]) == ASSET_KIND_UNCLEAR
    assert _classify_kind([st(0.112, 0.5, -2.0), st(0.34, 0.884, -3.0)]) == ASSET_KIND_UNCLEAR


def test_die_needs_wall_and_prep_and_depth():
    assert _classify_kind([st(0.15, 0.97, -0.2), st(0.70, 0.05, -8.0)]) == ASSET_KIND_DIE
    # 壁不够"环" → UNCLEAR
    assert _classify_kind([st(0.15, 0.97, -0.2), st(0.20, 0.05, -8.0)]) == ASSET_KIND_UNCLEAR
    # 颈方不下探 → UNCLEAR
    assert _classify_kind([st(0.15, 0.97, -0.2), st(0.70, 0.05, -0.2)]) == ASSET_KIND_UNCLEAR
    # 冠方不是"几乎全在上面" → UNCLEAR
    assert _classify_kind([st(0.15, 0.60, -0.2), st(0.70, 0.05, -8.0)]) == ASSET_KIND_UNCLEAR


def test_component_count():
    assert _classify_kind([st(0.15, 0.97)]) == ASSET_KIND_UNCLEAR
    assert _classify_kind([st(0.15, 0.97), st(0.7, 0.05), st(0.7, 0.05)]) == ASSET_KIND_UNCLEAR


def test_never_promotes_intaglio_to_die():
    """有内表面时无论别的指标多像代型，都必须判冠壳。"""
    for r in (-0.31, -0.5, -0.9):
        assert _classify_kind([st(r, 0.99, -9.0), st(0.7, 0.03, -9.0)]) == ASSET_KIND_INTAGLIO


# ================================================= closest_on_tri 退化三角形
# 真实扫描网格里有零面积三角形。旧实现 t = d1/(d1-d3) 会 ZeroDivisionError，
# 让整行被误判 REJECT（实测在距离门控投票时崩溃）。

def test_closest_on_tri_normal_cases():
    from build_auto_margin_gt import closest_on_tri
    a, b, c = (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)
    # 三角形内部的点（z=0 平面）→ 距离 0（重心插值有 ~1e-33 的浮点残差）
    assert closest_on_tri((0.2, 0.2, 0.0), a, b, c)[1] < 1e-24
    # 内部点正上方 1mm → 距离 1.0（投影落在面内）
    _q, d2 = closest_on_tri((0.2, 0.2, 1.0), a, b, c)
    assert abs(d2 ** 0.5 - 1.0) < 1e-12
    # 钝角外侧 → 最近点是顶点 b
    _q, d2 = closest_on_tri((2.0, 0.0, 0.0), a, b, c)
    assert abs(d2 ** 0.5 - 1.0) < 1e-12
    # 边 ab 外侧 → 最近点在边上 (0.5, 0, 0)
    _q, d2 = closest_on_tri((0.5, -1.0, 0.0), a, b, c)
    assert abs(d2 ** 0.5 - 1.0) < 1e-12


def test_closest_on_tri_degenerate_no_crash():
    """共线 / 三点重合的退化三角形必须返回有限值，不能抛 ZeroDivisionError。"""
    from build_auto_margin_gt import closest_on_tri
    p = (1.0, 1.0, 1.0)
    # 三点共线（面积 0）→ 退化为投影到最长边
    q, d2 = closest_on_tri(p, (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (2.0, 0.0, 0.0))
    assert abs(d2 ** 0.5 - 1.41421356) < 1e-6, (q, d2)
    # 三点重合 → 退化为点距离
    q, d2 = closest_on_tri(p, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
    assert abs(d2 ** 0.5 - 1.7320508) < 1e-6
    # 极窄条（面积接近 0 但不为 0）
    q, d2 = closest_on_tri(p, (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1e-14, 0.0))
    assert d2 ** 0.5 > 0.9
    # 查询点与某顶点重合 → 距离 0
    q, d2 = closest_on_tri((1.0, 0.0, 0.0), (0.0, 0.0, 0.0), (1.0, 0.0, 0.0),
                           (0.0, 1.0, 0.0))
    assert d2 == 0.0
