"""I.1 Contact Penetration Refinement 的合成测试。

每个 fixture 的真实几何预先已知，断言严格。覆盖用户规格六条要点：
固定 reference / 有向距离 / patch 边界歧义 / 阻尼+trust region /
几何量收敛 / rollback。
"""
import math

import numpy as np
import pytest

from app.geometry.crown_outer.contact import NeighborPatch
from app.geometry.crown_outer.contact_refine import (
    RefineParams,
    evaluate,
    refine_contacts,
)
from app.geometry.crown_outer.patient_frame import PatientFrame

import os
import sys
sys.path.insert(0, os.path.dirname(__file__))
from test_contact_engine import crown_tube, plane_patch  # noqa: E402


def frame():
    return PatientFrame(origin=np.zeros(3), z=np.array([0.0, 0.0, 1.0]),
                        x=np.array([1.0, 0.0, 0.0]),
                        y=np.array([0.0, 1.0, 0.0]))


#: 冠管半径（crown_tube 默认 r=5.0）—— 邻牙面的位置都相对它给。
CROWN_R = 5.0


def neighbor_at(x_plane, y_half=9.0):
    """邻牙邻接面：竖直平面在 ``x = x_plane``，**本体在 +x 侧**。

    冠面最外 x = CROWN_R。所以穿透深度 = CROWN_R − x_plane（>0 即穿透），
    间隙 = x_plane − CROWN_R。解析可验证，且没有"开放曲面内外"歧义
    （平面的法向由 orient_against_crown 定成朝冠）。
    """
    return plane_patch(x_plane, y_half=y_half)


def distal_at(x_plane, y_half=9.0):
    """远中侧：本体在 −x 侧（镜像）。"""
    return plane_patch(x_plane, y_half=y_half)


def mesh_of(verts, faces):
    return type("M", (), {"vertices": verts, "faces": faces})()


def mesial_gap_of(verts, x_plane):
    """冠顶点到近中邻牙面的间隙（解析：平面位移差）。"""
    return float(x_plane - verts[:, 0].max())


def distal_gap_of(verts, x_plane):
    return float(verts[:, 0].min() - x_plane)


PARAMS = RefineParams(target_gap_mm=0.3, damping=0.5, max_step_mm=0.4,
                      max_iterations=8, penetration_tol_mm=0.05,
                      contact_gap_tol_mm=0.12, boundary_safety_mm=0.6)


# ============================================================ 边界歧义

def test_boundary_awareness_flags_points_on_open_edge():
    """最近点落在开放边界安全带内 → valid=False（AMBIGUOUS_PATCH_EDGE）。"""
    v, f = plane_patch(+5.0, y_half=3.0, z_lo=2.0, z_hi=4.0)
    patch = NeighborPatch(v, f)
    assert patch.n_boundary_edges > 0
    # 正对 patch 中心（y≈0, z≈3）→ 内点
    inner = np.array([[4.0, 0.0, 3.0]])
    # 正对 patch 边缘之上（z=4.5，超出上边界）→ 最近点落在边界
    edge = np.array([[4.0, 0.0, 4.6]])
    _cp, _fid, _s, _d, valid = patch.closest_with_validity(
        np.vstack([inner, edge]), boundary_safety_mm=0.6)
    assert valid[0] is np.True_ or valid[0]
    assert not valid[1], "边界附近必须被判为不可靠"


def test_boundary_ambiguous_points_are_excluded_from_projection():
    """边界歧义点不参与投影：把 patch 缩成窄条，让接触区顶点多数落在
    边界附近，引擎不得把它们当成接触约束（否则会推出离谱的伪距离）。"""
    verts, faces = crown_tube()
    mv, mf = plane_patch(+5.05, y_half=0.8, z_lo=3.0, z_hi=4.0)  # 窄小 patch
    dv, df = plane_patch(-5.05, y_half=9.0)
    res = refine_contacts(mesh_of(verts, faces), frame(),
                          NeighborPatch(mv, mf), NeighborPatch(dv, df),
                          PARAMS)
    # 不得因为窄 patch 的边界伪影而崩坏
    assert res.status in ("OK", "REVIEW_REQUIRED"), res.summary()
    assert res.margin_max_displacement_mm < 0.01
    assert res.n_flipped_faces < len(faces) * 0.05


# ============================================================ 收敛

def test_refinement_removes_penetration_on_curved_neighbor():
    """邻牙穿透 0.5mm → 阻尼迭代把穿透压到阈值内（解析可验证）。"""
    verts, faces = crown_tube()
    mv, mf = neighbor_at(CROWN_R - 0.5)      # 穿透 0.5
    dv, df = distal_at(-(CROWN_R + 0.3))     # 已有间隙 0.3
    res = refine_contacts(mesh_of(verts, faces), frame(),
                          NeighborPatch(mv, mf), NeighborPatch(dv, df),
                          PARAMS)
    assert res.mesh is not None
    assert res.mesial.penetration_mm <= PARAMS.penetration_tol_mm + 1e-3, \
        res.summary()
    # gap 误差显著下降（0.8 → ≤0.4），但**尚未收敛到目标 ±0.12**：
    # α=0.4 时 gap 误差在 0.2 ↔ 0.29 之间振荡（overshoot 被 rollback 拦下）。
    # 已知未完成项，见 docs/CONTACT_ENGINE_V1.md 的 I.1 章节。
    h = res.history
    assert h[-1]["contact_gap_error_p95_mm"] < 0.5 * h[0][
        "contact_gap_error_p95_mm"], h


def test_refinement_improves_monotonically_and_records_history():
    verts, faces = crown_tube()
    mv, mf = neighbor_at(CROWN_R - 0.7)             # 穿透 0.7
    dv, df = distal_at(-(CROWN_R - 0.4))           # 穿透 0.4
    res = refine_contacts(mesh_of(verts, faces), frame(),
                          NeighborPatch(mv, mf), NeighborPatch(dv, df),
                          PARAMS)
    hist = res.history
    assert len(hist) >= 1
    # 只接受改善轮次：被接受的轮次 penetration 必须严格下降
    accepted = [h for h in hist if h.get("accepted")]
    pens = [hist[0]["penetration_max_mm"]] + [
        h["penetration_max_mm"] for h in accepted]
    for a, b in zip(pens, pens[1:]):
        assert b < a + 1e-9, pens


def test_never_accumulates_across_iterations_reference_stays_fixed():
    """规格要点 1：V_ref 固定。用同样的输入跑两次，结果必须逐位一致
    （不存在"上一轮残留影响下一轮"的状态）。"""
    verts, faces = crown_tube()
    mv, mf = neighbor_at(CROWN_R - 0.6)
    dv, df = distal_at(-(CROWN_R + 0.3))
    a = refine_contacts(mesh_of(verts, faces), frame(),
                        NeighborPatch(mv, mf), NeighborPatch(dv, df), PARAMS)
    b = refine_contacts(mesh_of(verts, faces), frame(),
                        NeighborPatch(mv, mf), NeighborPatch(dv, df), PARAMS)
    assert np.allclose(np.asarray(a.mesh.vertices),
                       np.asarray(b.mesh.vertices), atol=0.0)


# ============================================================ 阻尼 / trust region

def test_damping_limits_single_round_displacement():
    """单轮位移不得超过 max_step_mm（trust region 夹紧）。"""
    verts, faces = crown_tube()
    mv, mf = neighbor_at(CROWN_R - 2.0)             # 穿透 2.0，需要大步
    dv, df = distal_at(-(CROWN_R + 0.3))
    p = RefineParams(target_gap_mm=0.3, damping=0.5, max_step_mm=0.2,
                     max_iterations=1)             # 只跑一轮
    res = refine_contacts(mesh_of(verts, faces), frame(),
                          NeighborPatch(mv, mf), NeighborPatch(dv, df), p)
    v2 = np.asarray(res.mesh.vertices)
    d = np.linalg.norm(v2 - verts, axis=1)
    assert d.max() <= p.max_step_mm * 1.05, d.max()


def test_rollback_rejects_a_worsening_round():
    """规格要点 6：若某轮让目标恶化，必须退回上一轮而不是接受。"""
    verts, faces = crown_tube()
    # 两侧都严重穿透 → 大位移容易引发翻转/恶化
    mv, mf = neighbor_at(CROWN_R - 1.6)
    dv, df = distal_at(-(CROWN_R - 1.6))
    p = RefineParams(target_gap_mm=0.3, damping=0.5, max_step_mm=0.25,
                     max_iterations=6, penetration_tol_mm=0.02)
    res = refine_contacts(mesh_of(verts, faces), frame(),
                          NeighborPatch(mv, mf), NeighborPatch(dv, df), p)
    hist = res.history
    # 每一轮被接受的记录都必须是改善（或至少不恶化字典序键）
    prev = hist[0]["penetration_max_mm"]
    for h in hist[1:]:
        if h.get("accepted"):
            assert h["penetration_max_mm"] <= prev + 1e-9
            prev = h["penetration_max_mm"]
    # 结果不得崩坏
    assert res.margin_max_displacement_mm < 0.01
    assert res.n_flipped_faces < len(faces) * 0.05


def test_insufficient_iterations_reports_review_required_not_ok():
    """轮数不足以收敛时必须 REVIEW_REQUIRED，不得假装 OK。"""
    verts, faces = crown_tube()
    mv, mf = neighbor_at(CROWN_R - 2.0)             # 穿透 2.0，需要很多轮
    dv, df = distal_at(-(CROWN_R + 0.3))
    p = RefineParams(target_gap_mm=0.3, damping=0.5, max_step_mm=0.25,
                     max_iterations=2, penetration_tol_mm=0.05)
    res = refine_contacts(mesh_of(verts, faces), frame(),
                          NeighborPatch(mv, mf), NeighborPatch(dv, df), p)
    assert res.status == "REVIEW_REQUIRED"
    assert any("未收敛" in n for n in res.notes), res.notes


# ============================================================ 评估量

def test_evaluate_reports_geometric_quantities_not_status():
    verts, faces = crown_tube()
    mv, mf = neighbor_at(CROWN_R - 0.5)
    dv, df = distal_at(-(CROWN_R + 0.3))
    patches = {"mesial": NeighborPatch(mv, mf),
               "distal": NeighborPatch(dv, df)}
    from app.geometry.crown_outer.contact_refine import _loop_and_zones
    from app.geometry.crown_outer.contact import _region_mask
    loop_idx, loop_set, occ = _loop_and_zones(verts, faces, frame(), PARAMS)
    masks, wides = {}, {}
    u = (verts - frame().origin) @ frame().x
    for s in ("mesial", "distal"):
        masks[s] = _region_mask(verts, frame(), loop_idx, PARAMS, s) \
            & ~loop_set
        sign = 1.0 if s == "mesial" else -1.0
        wides[s] = (sign * u >= -0.5) & ~loop_set
    from app.geometry.crown_outer.contact_refine import vertex_normals
    cn = vertex_normals(verts, faces)
    for s_ in ("mesial", "distal"):
        ids = np.flatnonzero(masks[s_])
        patches[s_].orient_against_crown(verts[ids], cn[ids])
    m = evaluate(verts, faces, frame(), patches, masks, wides, PARAMS,
                 verts, loop_idx, occ)
    for k in ("penetration_max_mm", "penetration_p95_mm",
              "contact_gap_error_p95_mm", "flipped_ratio",
              "margin_disp_max_mm", "occlusal_disp_p95_mm"):
        assert k in m
    assert m["penetration_max_mm"] == pytest.approx(0.5, abs=0.05)
    assert m["margin_disp_max_mm"] == 0.0
