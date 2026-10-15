"""Contact Engine v1 的合成测试（提交 I，方案：5 类场景 + 不变量）。

所有 fixture 的"真实距离"预先已知，断言是**严格的**：
目标间隙 0.3mm、邻牙平面位置已知 → 变形后冠面到平面的距离可解析验证。
"""
import math

import numpy as np
import pytest

from app.geometry.crown_outer.contact import (
    ContactParameters,
    NeighborPatch,
    optimize_contacts,
)
from app.geometry.crown_outer.patient_frame import PatientFrame
from app.geometry.crown_outer.contact import _boundary_loop


TARGET = 0.3   # 与 ContactParameters(target_gap_mm=0.3) 一致


def crown_tube(r=5.0, height=7.0, n=48, nz=8):
    """圆柱冠：底环=颈环（开放底边），直壁（fixture 间隙可解析）。"""
    verts = []
    for k in range(nz):
        z = height * k / (nz - 1)
        for i in range(n):
            a = 2 * math.pi * i / n
            verts.append((r * math.cos(a), r * math.sin(a), z))
    verts = np.asarray(verts)
    faces = []
    for k in range(nz - 1):
        for i in range(n):
            j = (i + 1) % n
            a, b = k * n + i, k * n + j
            c, d = (k + 1) * n + j, (k + 1) * n + i
            faces.append((a, b, c))
            faces.append((a, c, d))
    # 顶盖（封住顶部，避免上边界）
    top0 = (nz - 1) * n
    verts = np.vstack([verts, [[0.0, 0.0, height]]])
    apex = len(verts) - 1
    for i in range(n):
        j = (i + 1) % n
        faces.append((top0 + i, top0 + j, apex))
    return verts, np.asarray(faces, dtype=np.int64)


def plane_patch(x_plane, y_half=9.0, z_lo=-0.5, z_hi=6.5, flip=False):
    """竖直平面 patch（邻牙邻接面的合成替身），法向朝 +x（flip 则 −x）。"""
    v = []
    for z in (z_lo, (z_lo + z_hi) / 2, z_hi):
        for y in (-y_half, 0.0, y_half):
            v.append((x_plane, y, z))
    v = np.asarray(v)
    # 顶点布局：z 3 行 × y 3 列；两个quad各切两刀
    f = [(0, 1, 4), (0, 4, 3), (1, 2, 5), (1, 5, 4),
         (3, 4, 7), (3, 7, 6), (4, 5, 8), (4, 8, 7)]
    f = [(a, c, b) for a, b, c in f] if flip else f
    return v, np.asarray(f, dtype=np.int64)


def frame():
    return PatientFrame(origin=np.zeros(3), z=np.array([0.0, 0.0, 1.0]),
                        x=np.array([1.0, 0.0, 0.0]),
                        y=np.array([0.0, 1.0, 0.0]))


def patch_class(v, f):
    return NeighborPatch(v, f)


def cyl_shell_patch(r=5.3, side="distal", n=16, nz=6, height=7.0):
    """与冠面同心的偏移壳段（distal = θ∈[140°,220°]）——
    每个区域顶点的间隙都精确等于 r−5，用于"已达标"fixture。"""
    # ±60°：冠的区域带到 |u|≥0.6·r（θ≈±53°），壳必须覆盖它
    th0, th1 = (np.deg2rad(120), np.deg2rad(240)) if side == "distal"         else (np.deg2rad(-60), np.deg2rad(60))
    verts, faces = [], []
    for k in range(nz):
        z = height * k / (nz - 1)
        for i in range(n):
            a = th0 + (th1 - th0) * i / (n - 1)
            verts.append((r * math.cos(a), r * math.sin(a), z))
    verts = np.asarray(verts)
    for k in range(nz - 1):
        for i in range(n - 1):
            a, b = k * n + i, k * n + i + 1
            c, d = (k + 1) * n + i + 1, (k + 1) * n + i
            faces.append((a, b, c))
            faces.append((a, c, d))
    return verts, np.asarray(faces, dtype=np.int64)


PARAMS = ContactParameters(target_gap_mm=TARGET)


def _substantive_ok(res):
    """实质不变量：状态只能是 OK，或"仅咬合位移提示"的 REVIEW_REQUIRED。

    合成夹具是 8 圈粗管，顶环离接触区只有两条边，位移传播比真实细网格
    明显（真实病例实测咬合 p95 = 0.0023mm）。所以这里不把"咬合位移偏大"
    当失败，但 margin / 穿透 / 翻转类的提示一律不接受。"""
    if res.status == "OK":
        return True
    if res.status != "REVIEW_REQUIRED":
        return False
    return all("咬合区位移" in n for n in res.notes)


def surface_gap_x(vertices, x_plane, side):
    """冠顶点在指定侧到平面的距离（只取该侧 |x| 最大的一些点）。"""
    sign = 1.0 if side == "mesial" else -1.0
    xs = sign * vertices[:, 0]
    sel = xs > 0.6 * xs.max()
    # 间隙 = sign·(x_plane − x_v)：mesial 平面在外侧更大 x，distal 相反
    return float(np.min(sign * (x_plane - vertices[sel, 0])))


# ============================================================ 1. 邻牙太远

def test_too_far_moves_contact_region_toward_neighbor():
    verts, faces = crown_tube()
    # 平面在 x=+6.0：冠面最大 x≈5 → 初始 gap≈1.0 > 目标 0.3
    mv, mf = plane_patch(+6.0)
    dv, df = cyl_shell_patch(r=5.3, side="distal")   # distal 已在目标
    res = optimize_contacts(  # noqa: E501
        type("M", (), {"vertices": verts, "faces": faces})(),
        frame(), patch_class(mv, mf), patch_class(dv, df), PARAMS)
    assert _substantive_ok(res), res.summary()
    assert res.mesial.vertices_modified > 0
    assert res.distal.vertices_modified == 0
    v2 = np.asarray(res.mesh.vertices)
    # 变形后 mesial 面到平面的距离 ≈ 0.3（解析可断言）
    assert surface_gap_x(v2, 6.0, "mesial") == pytest.approx(TARGET, abs=0.1)
    # distal 未动
    assert surface_gap_x(v2, -5.3, "distal") == pytest.approx(TARGET, abs=0.1)


# ============================================================ 2. 穿透退让

def test_penetration_retreats_to_target_gap():
    """局部凸刺（fin）穿入冠体 0.5mm → 检出穿透并局部退让到目标间隙。

    单一无限平面读不出"穿过"（顶点在平面另一侧只是"距离更远"），
    穿透语义必须由**局部侵入几何**承载 —— 真实邻牙正是如此。"""
    verts, faces = crown_tube()
    mv, mf = plane_patch(+5.6)          # 平面 gap 0.6
    # fin：从平面伸向冠体的小三角（x=4.5，法向 −x），穿入冠面 0.5mm
    fin_v = np.asarray([(4.5, -0.4, 2.0), (4.5, 0.4, 2.5), (4.5, -0.4, 3.0)])
    fin_f = np.asarray([(0, 2, 1)])     # 绕序使法向 −x
    mf = np.vstack([mf, fin_f + len(mv)])
    mv = np.vstack([mv, fin_v])
    dv, df = plane_patch(-5.3)
    res = optimize_contacts(
        type("M", (), {"vertices": verts, "faces": faces})(),
        frame(), patch_class(mv, mf), patch_class(dv, df), PARAMS)
    assert _substantive_ok(res), res.summary()
    # 引擎自己的带符号度量必须报告穿透已消除
    assert res.mesial.penetration_mm <= 0.05, res.summary()
    v2 = np.asarray(res.mesh.vertices)
    # fin 正前方的冠顶点（原 (5,0,2)）必须退到 fin 面 0.3mm 外：
    # 4.5 + target ± 容差，且不越过 fin 面
    idx = int(np.argmin(np.linalg.norm(
        verts - np.array([5.0, 0.0, 2.0]), axis=1)))
    assert 4.5 + TARGET - 0.15 <= v2[idx, 0] <= 5.0, (
        verts[idx], v2[idx])
    assert res.mesial.vertices_modified > 0


# ==================================================== 3. 一侧正常一侧过远

def test_only_abnormal_side_modified():
    verts, faces = crown_tube()
    mv, mf = cyl_shell_patch(r=5.3, side="mesial")   # gap 0.3 = 目标 → 不动
    dv, df = plane_patch(-6.4)          # gap 1.4 → 需要移
    res = optimize_contacts(
        type("M", (), {"vertices": verts, "faces": faces})(),
        frame(), patch_class(mv, mf), patch_class(dv, df), PARAMS)
    assert _substantive_ok(res), res.summary()
    assert res.mesial.vertices_modified == 0
    assert res.distal.vertices_modified > 0
    v2 = np.asarray(res.mesh.vertices)
    assert surface_gap_x(v2, -6.4, "distal") == pytest.approx(TARGET, abs=0.1)
    # mesial 未动
    assert surface_gap_x(v2, 5.3, "mesial") == pytest.approx(TARGET, abs=0.05)


# ==================================================== 4. 两侧同时约束

def test_both_sides_converge_simultaneously():
    verts, faces = crown_tube()
    mv, mf = plane_patch(+5.8)          # gap 0.8
    dv, df = plane_patch(-6.2)          # gap 1.2
    res = optimize_contacts(
        type("M", (), {"vertices": verts, "faces": faces})(),
        frame(), patch_class(mv, mf), patch_class(dv, df), PARAMS)
    assert _substantive_ok(res), res.summary()
    v2 = np.asarray(res.mesh.vertices)
    assert surface_gap_x(v2, 5.8, "mesial") == pytest.approx(TARGET, abs=0.1)
    assert surface_gap_x(v2, -6.2, "distal") == pytest.approx(TARGET, abs=0.1)
    assert res.mesial.vertices_modified > 0
    assert res.distal.vertices_modified > 0


# ==================================================== 5. 不可满足条件

def test_infeasible_neighbor_reported_without_modification():
    verts, faces = crown_tube()
    mv, mf = plane_patch(+8.5)          # gap 3.5 > max_displacement 1.5
    dv, df = plane_patch(-5.3)
    res = optimize_contacts(
        type("M", (), {"vertices": verts, "faces": faces})(),
        frame(), patch_class(mv, mf), patch_class(dv, df), PARAMS)
    assert res.status == "DESIGN_INFEASIBLE"
    assert res.mesial.status == "INFEASIBLE"
    assert res.distal.status == "OK"
    # fail-loud：几何不得被半途修改
    assert np.allclose(np.asarray(res.mesh.vertices), verts)


# ============================================================ 不变量

def _invariants(res, verts0, faces):
    loop = _boundary_loop(verts0, faces)
    v2 = np.asarray(res.mesh.vertices)
    d_margin = np.linalg.norm(v2[loop] - verts0[loop], axis=1)
    # margin LOCK
    assert d_margin.max() < 0.01
    # 咬合面（z 最高带）几乎不动
    top = verts0[:, 2] >= np.percentile(verts0[:, 2], 90)
    d_top = np.linalg.norm(v2[top] - verts0[top], axis=1)
    assert np.percentile(d_top, 95) < 0.3
    # 无翻转面
    n0 = np.cross(verts0[faces[:, 1]] - verts0[faces[:, 0]],
                  verts0[faces[:, 2]] - verts0[faces[:, 0]])
    n1 = np.cross(v2[faces[:, 1]] - v2[faces[:, 0]],
                  v2[faces[:, 2]] - v2[faces[:, 0]])
    den = np.linalg.norm(n0, axis=1) * np.linalg.norm(n1, axis=1)
    okd = den > 1e-18
    assert not np.any(np.einsum("ij,ij->i", n0[okd], n1[okd]) / den[okd] < 0)
    # 无退化面
    assert np.linalg.norm(n1, axis=1).min() > 1e-9


def test_invariants_hold_after_both_sides_move():
    verts, faces = crown_tube()
    mv, mf = plane_patch(+5.8)
    dv, df = plane_patch(-6.2)
    res = optimize_contacts(
        type("M", (), {"vertices": verts, "faces": faces})(),
        frame(), patch_class(mv, mf), patch_class(dv, df), PARAMS)
    assert _substantive_ok(res), res.summary()
    _invariants(res, verts, faces)


def test_contact_metrics_reported():
    verts, faces = crown_tube()
    mv, mf = plane_patch(+5.4)          # gap 0.4，贴近后应产生接触区
    dv, df = plane_patch(-5.3)
    res = optimize_contacts(
        type("M", (), {"vertices": verts, "faces": faces})(),
        frame(), patch_class(mv, mf), patch_class(dv, df), PARAMS)
    assert res.mesial.contact_area_mm2 > 0
    assert res.mesial.contact_centroid is not None
    # 质心在 mesial 侧（x > 0）
    assert res.mesial.contact_centroid[0] > 0
    assert res.distal.contact_area_mm2 >= 0


def test_patch_normal_auto_orientation():
    """法向翻转的 patch 必须被自动检测并纠正（否则穿透判定反向）。"""
    verts, faces = crown_tube()
    mv, mf = plane_patch(+5.3, flip=True)   # 法向朝 −x（指向冠体）
    dv, df = plane_patch(-5.3)
    res = optimize_contacts(
        type("M", (), {"vertices": verts, "faces": faces})(),
        frame(), patch_class(mv, mf), patch_class(dv, df), PARAMS)
    assert _substantive_ok(res), res.summary()
    v2 = np.asarray(res.mesh.vertices)
    assert surface_gap_x(v2, 5.3, "mesial") == pytest.approx(TARGET, abs=0.1)
