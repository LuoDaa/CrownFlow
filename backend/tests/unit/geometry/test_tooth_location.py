"""目标牙定位与 Support 分割的单元测试 —— **合成 fixture，可提交 Git**。

夹具是"牙龈平板 + 预备体凸台"，交界处是**精确 90° 的凹脊**，
所以这里能断言"区域停在解剖边界上"本身。真实病例不一定有这个条件 ——
见 ``tests/acceptance/geometry/test_tooth_location_real.py``：
真实 FDI36 病例的基台/牙龈交界**没有**凹脊（实测最大凹度 0.054）。

刻意用"环 + 缝合"显式构造网格，而不是高度场台阶：高度场必须把壁塞进一个网格
间距里，壁底与壁顶的顶点会混在一起，二面角还随分辨率漂移；显式构造下两条环的
二面角精确是 90°（凹谷强度正好 0.5），改方位向分辨率完全不影响它。

锁住的规则：

  1. 凹谷场只标凹脊，不标凸脊（牙尖、边缘嵴不能被当成分割边界）
  2. **刚性变换等变** —— 位置、面积、状态都不随坐标系改变
  3. 分辨率稳定性
  4. 低置信度 fail-loud，不返回"看起来像那么回事"的坐标
  5. 切割半径必须与"源是否为区域"配套（混用会把 support 撑大好几倍）
"""

from __future__ import annotations

import json

import numpy as np
import pytest
import trimesh

from tests.acceptance._synthetic import apply_rigid, random_rigid, transform_points

from app.geometry.tooth_location import (
    CROWN_WIDTH_MM,
    VALLEY_BLOCK,
    LocalizationStatus,
    SupportStatus,
    extract_support,
    locate_target_tooth,
    vertex_valley_strength,
)

#: 夹具用 FDI36（近远中冠宽 11.0 mm），预备体半径取冠宽的一半。
PREP_RADIUS_MM = CROWN_WIDTH_MM[36] / 2.0
PREP_HEIGHT_MM = 4.0
WALL_TAPER_MM = 0.4


def _ring(n_a: int, radius: float, z: float) -> np.ndarray:
    a = np.linspace(0.0, 2.0 * np.pi, n_a, endpoint=False)
    return np.column_stack([radius * np.cos(a), radius * np.sin(a),
                            np.full(n_a, z)])


def _sew(rings, n_a: int) -> list[list[int]]:
    faces: list[list[int]] = []
    for k in range(len(rings) - 1):
        base, nxt = k * n_a, (k + 1) * n_a
        for i in range(n_a):
            j = (i + 1) % n_a
            faces.append([base + i, base + j, nxt + j])
            faces.append([base + i, nxt + j, nxt + i])
    return faces


def _cap(rings, n_a: int, apex: np.ndarray) -> trimesh.Trimesh:
    verts = np.concatenate(list(rings) + [apex.reshape(1, 3)])
    faces = _sew(rings, n_a)
    last = (len(rings) - 1) * n_a
    for i in range(n_a):
        faces.append([last + i, last + (i + 1) % n_a, len(verts) - 1])
    return trimesh.Trimesh(vertices=verts, faces=np.asarray(faces), process=True)


def prep_plate(n_a: int = 180, half: float = 14.0, prep_r: float = PREP_RADIUS_MM,
               prep_h: float = PREP_HEIGHT_MM, taper: float = WALL_TAPER_MM,
               n_wall: int = 8):
    """牙龈平板 + 预备体凸台。

        r=prep_r, z=0       牙龈 -> 壁：**凹** 90°（区域应当停在这里）
        r=prep_r, z=prep_h  壁 -> 台面：**凸** 90°（不能被当成分割边界）

    **壁要分成多条环**（``n_wall``）：整面壁只用一个四边形环的话，三角形是
    0.2 mm 宽 × 4 mm 高的长条，法向由叉积算出、数值上几乎退化，
    凹/凸判据会在壁上给出垃圾结果，把台面与壁切断（实测：区域只剩台面）。

    返回 ``(网格, 台面中心真值点)``。
    """
    rings = [_ring(n_a, r, 0.0) for r in np.linspace(half, prep_r, 22)]
    wall_z = np.linspace(0.0, prep_h, n_wall + 1)[1:]
    wall_r = prep_r - taper * (wall_z / prep_h)
    rings += [_ring(n_a, r, z) for r, z in zip(wall_r, wall_z)]
    rings += [_ring(n_a, r, prep_h) for r in np.linspace(prep_r - taper, 0.0, 12)[1:]]
    return _cap(rings, n_a, np.array([0.0, 0.0, prep_h])), np.array([0.0, 0.0, prep_h])


def smooth_bump_plate(n_a: int = 180, half: float = 14.0, sigma: float = 3.0,
                      height: float = 4.0):
    """**没有凹脊**的平滑凸包 —— 验证"边界只能是度量切割"会被如实报出来。"""
    rings = [_ring(n_a, half, 0.0)]
    rings += [_ring(n_a, r, height * np.exp(-(r / sigma) ** 2))
              for r in np.linspace(half, 1e-3, 40)[1:]]
    return _cap(rings, n_a, np.array([0.0, 0.0, height])), np.array([0.0, 0.0, height])


# ============================================================ 凹谷场

def test_valley_field_marks_creases_not_crests():
    """凹谷场必须只标凹脊：壁底（凹 90°）标出，壁顶（凸 90°）不能标。"""
    mesh, _ = prep_plate()
    vs = vertex_valley_strength(mesh)
    V = np.asarray(mesh.vertices, float)
    r = np.hypot(V[:, 0], V[:, 1])
    h = V[:, 2]

    at_crease = np.isclose(h, 0.0) & np.isclose(r, PREP_RADIUS_MM, atol=1e-6)
    at_crest = np.isclose(h, PREP_HEIGHT_MM, atol=1e-6) & (r > PREP_RADIUS_MM - 1.0)
    on_top = (r < PREP_RADIUS_MM - 1.0) & np.isclose(h, PREP_HEIGHT_MM, atol=1e-6)

    assert at_crease.sum() > 100 and at_crest.sum() > 100, "夹具的两条环没被定位到"
    assert vs[on_top].max() < 0.05, f"台面不该有凹度：{vs[on_top].max()}"
    assert vs[at_crease].mean() > VALLEY_BLOCK + 0.03, \
        f"壁底 90° 凹角应约 0.5，实测 {vs[at_crease].mean():.3f}"
    assert vs[at_crest].max() < 0.05, \
        f"壁顶是凸角，不该被当成凹脊：max={vs[at_crest].max():.3f}"


def test_valley_field_is_rigid_equivariant():
    """凹谷场的分布不随刚体变换改变（逐点比不稳：变换后可能重合并顶点）。"""
    mesh, _ = prep_plate()
    base = vertex_valley_strength(mesh)

    def stats(f):
        return float(f.mean()), float(f.max()), int((f > VALLEY_BLOCK).sum())

    want = stats(base)
    for k in range(3):
        T = random_rigid(np.random.default_rng(100 + k))
        got = stats(vertex_valley_strength(apply_rigid(mesh, T)))
        assert got[2] == want[2], f"变换 {k}: 被屏障断开的顶点数变了 {want} -> {got}"
        assert got[0] == pytest.approx(want[0], rel=0.02), f"变换 {k}: 均值变了"
        assert got[1] == pytest.approx(want[1], rel=0.02), f"变换 {k}: 最大值变了"


# ============================================================ 定位

def test_locate_from_seed_finds_the_prep():
    """点击台面 -> 区域应当被凹脊拦住，只圈住预备体。"""
    mesh, top = prep_plate()
    res = locate_target_tooth(mesh, 36, user_seed=top)

    assert res.method == "SEED"
    assert res.status is LocalizationStatus.OK, res.diagnostics
    assert len(res.region_vertex_ids) > 500
    assert res.region_area_mm2 < 400.0, f"区域过大：{res.region_area_mm2}"
    pos = np.asarray(res.position, float)
    assert np.hypot(pos[0], pos[1]) < 1.0, f"位置偏离中心轴：{pos}"
    assert res.confidence > 0.5
    assert res.diagnostics["reasons"] == []
    assert res.diagnostics["cut_front_valley_strength_mean"] > 0.2
    assert res.diagnostics["n_blocked_edges"] > 0


def test_locate_is_rigid_equivariant():
    """位置、面积、状态都必须随刚体变换一致改变 —— 硬性要求。"""
    mesh, top = prep_plate()
    base = locate_target_tooth(mesh, 36, user_seed=top)
    for k in range(4):
        T = random_rigid(np.random.default_rng(200 + k))
        seed = transform_points(top.reshape(1, 3), T)[0]
        got = locate_target_tooth(apply_rigid(mesh, T), 36, user_seed=seed)

        assert got.status is base.status, \
            f"变换 {k}: 状态 {base.status} -> {got.status}"
        assert got.region_area_mm2 == pytest.approx(base.region_area_mm2, rel=0.02), \
            f"变换 {k}: 面积 {base.region_area_mm2:.3f} -> {got.region_area_mm2:.3f}"
        expected = transform_points(np.asarray(base.position, float).reshape(1, 3), T)[0]
        err = float(np.linalg.norm(np.asarray(got.position, float) - expected))
        assert err < 1e-6, f"变换 {k}: 位置偏差 {err:.3e} mm"


def test_locate_is_stable_under_resolution_change():
    """方位向分辨率变化不该改变结论（显式构造下壁角精确 90°，与分辨率无关）。"""
    fine, top = prep_plate(n_a=240)
    coarse, _ = prep_plate(n_a=90)
    a = locate_target_tooth(fine, 36, user_seed=top)
    b = locate_target_tooth(coarse, 36, user_seed=top)
    assert a.status is b.status, f"{a.status} vs {b.status}"
    assert a.region_area_mm2 == pytest.approx(b.region_area_mm2, rel=0.06), \
        f"面积随分辨率漂移：{a.region_area_mm2:.1f} vs {b.region_area_mm2:.1f}"


def test_locate_reviews_when_seed_is_far_from_the_prep():
    """种子落在平板远端时不能报 OK —— 区域会越界，必须明说。"""
    mesh, _ = prep_plate()
    res = locate_target_tooth(mesh, 36, user_seed=[12.0, 12.0, 0.0])
    assert res.status is not LocalizationStatus.OK
    assert res.diagnostics["reasons"], "非 OK 必须给出原因"


def test_unknown_fdi_raises_before_any_work():
    mesh, _ = prep_plate()
    with pytest.raises(ValueError, match="不在已知牙位表"):
        locate_target_tooth(mesh, 99, user_seed=[0.0, 0.0, PREP_HEIGHT_MM])


def test_auto_mode_declines_with_measured_reason():
    """自动模式当前一律明确拒绝并说明为什么，而不是猜一个坐标。"""
    mesh, _ = prep_plate()
    res = locate_target_tooth(mesh, 36)
    assert res.method == "AUTO_UNAVAILABLE"
    assert res.status is LocalizationStatus.REVIEW_REQUIRED
    assert res.confidence == 0.0
    assert res.region_vertex_ids == []
    reasons = res.diagnostics["reasons"]
    assert reasons and "判别度不足" in reasons[0]
    assert "user_seed" in res.diagnostics["required_action"]


def test_jaw_and_width_table_is_self_consistent():
    mesh, top = prep_plate()
    for fdi in CROWN_WIDTH_MM:
        assert 4.0 < CROWN_WIDTH_MM[fdi] < 15.0
    assert locate_target_tooth(mesh, 46, user_seed=top).jaw == "lower"
    assert locate_target_tooth(mesh, 16, user_seed=top).jaw == "upper"


# ============================================================ support 分割

def test_support_stops_at_the_anatomical_crease():
    """半径合适时 support 必须停在凹脊上：单连通、单边界环、目标完整。"""
    mesh, top = prep_plate()
    res = locate_target_tooth(mesh, 36, user_seed=top)
    sup = extract_support(mesh, res, target_fdi=36)

    assert sup.status is SupportStatus.OK, sup.diagnostics
    assert sup.mesh is not None
    assert sup.checks["single_connected_component"] is True
    assert sup.checks["n_boundary_loops"] == 1
    assert sup.checks["target_retention"] == pytest.approx(1.0, abs=1e-9)
    assert sup.checks["boundary_is_anatomical"] is True
    assert sup.checks["boundary_valley_ok"] is True
    assert sup.diagnostics["sources_are_region"] is True
    assert sup.diagnostics["reasons"] == []
    assert sup.mesh.area < 600.0, "切出来的是预备体，不是整块平板"


def test_support_flags_metric_cut_when_no_crease_exists():
    """**没有凹脊**时边界只能是按半径切的度量切割 —— 必须如实报出来。

    这正是真实 FDI36 病例的情形（基台/牙龈交界凹度实测仅 0.054），
    所以这条测试同时把"真实病例为什么必然 REVIEW_REQUIRED"钉住了。
    """
    mesh, top = smooth_bump_plate()
    vs = vertex_valley_strength(mesh)
    assert vs.max() < VALLEY_BLOCK, f"夹具本意是无凹脊，实测最大凹度 {vs.max():.3f}"

    res = locate_target_tooth(mesh, 36, user_seed=top)
    sup = extract_support(mesh, res, target_fdi=36)

    assert sup.checks["boundary_is_anatomical"] is False
    assert sup.checks["convex_cut_fraction"] > 0.3, "边界应当有相当比例走在凸面上"
    assert sup.status is SupportStatus.REVIEW_REQUIRED
    joined = " ".join(sup.diagnostics["reasons"])
    assert "凸面" in joined
    # 但目标本身仍要完整保留、单连通 —— 度量切割不等于切坏了
    assert sup.checks["target_retention"] == pytest.approx(1.0, abs=1e-9)
    assert sup.checks["single_connected_component"] is True


def test_support_cut_radius_matches_source_kind():
    """源是区域时半径 = 余量；源只是一个点时半径 = 冠宽/2 + 余量。

    混用会把 support 撑大好几倍（实测 112 mm² -> 662 mm²）。
    """
    mesh, top = prep_plate()
    as_region = extract_support(
        mesh, locate_target_tooth(mesh, 36, user_seed=top), target_fdi=36)
    assert as_region.diagnostics["sources_are_region"] is True
    assert as_region.cut_radius_mm == pytest.approx(as_region.diagnostics["margin_mm"])

    as_point = extract_support(mesh, top, target_fdi=36, margin_mm=1.5)
    assert as_point.diagnostics["sources_are_region"] is False
    assert as_point.cut_radius_mm == pytest.approx(CROWN_WIDTH_MM[36] / 2.0 + 1.5)
    assert as_point.mesh is not None


def test_support_rejects_empty_target():
    mesh, _ = prep_plate()
    empty = locate_target_tooth(mesh, 36)          # AUTO_UNAVAILABLE：没有区域
    sup = extract_support(mesh, empty, target_fdi=36)
    assert sup.status is SupportStatus.FAILED
    assert sup.mesh is None
    assert sup.checks["target_region_present"] is False


def test_support_accepts_point_id_and_point_cloud():
    mesh, top = prep_plate()
    V = np.asarray(mesh.vertices, float)
    by_point = extract_support(mesh, top, target_fdi=36)
    by_id = extract_support(mesh, int(np.argmin(np.linalg.norm(V - top, axis=1))),
                            target_fdi=36)
    by_cloud = extract_support(mesh, V[np.linalg.norm(V - top, axis=1) < 1.0],
                               target_fdi=36)
    for r in (by_point, by_id, by_cloud):
        assert r.mesh is not None and len(r.mesh.faces) > 20
        assert r.checks["target_retention"] == pytest.approx(1.0, abs=1e-9)


def test_supports_are_rigid_equivariant():
    mesh, top = prep_plate()
    base = extract_support(
        mesh, locate_target_tooth(mesh, 36, user_seed=top), target_fdi=36)
    for k in range(3):
        T = random_rigid(np.random.default_rng(300 + k))
        moved = apply_rigid(mesh, T)
        seed = transform_points(top.reshape(1, 3), T)[0]
        sup = extract_support(
            moved, locate_target_tooth(moved, 36, user_seed=seed), target_fdi=36)
        assert sup.status is base.status, f"变换 {k}: {base.status} -> {sup.status}"
        assert sup.checks["n_boundary_loops"] == base.checks["n_boundary_loops"]
        assert sup.checks["boundary_is_anatomical"] == \
            base.checks["boundary_is_anatomical"]
        assert len(sup.mesh.vertices) == len(base.mesh.vertices)


# ============================================================ 序列化

def test_results_serialize_to_json():
    mesh, top = prep_plate()
    res = locate_target_tooth(mesh, 36, user_seed=top)
    sup = extract_support(mesh, res, target_fdi=36)
    for payload in (res.as_dict(), sup.as_dict()):
        text = json.dumps(payload, allow_nan=False)      # NaN 会直接抛
        assert "Infinity" not in text
    assert res.as_dict()["status"] == "OK"
    assert sup.as_dict()["mesh"]["faces"] > 0
    assert "region_vertex_ids" not in sup.as_dict()
    assert res.as_dict()["diagnostics"]["n_blocked_edges"] > 0
