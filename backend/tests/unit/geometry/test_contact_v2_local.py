"""Contact v2 —— 局部带状求解的专测。

核心不变量（与 v1 的本质区别）：

    **咬合面与颈缘的位移由构造精确为 0**（不是"弱锚尽量小"）。

背景：v1/I.1 用全局系统（咬合弱锚 α=300 对硬约束 W=1e4 无抵抗力），真实例实测
咬合被拖 1.08mm，违反"Occlusal ≈ 0"规格。v2 把自由区收缩到接触带 + 过渡环，
其余顶点是固定边界 —— 位移恒等于 0，且接触种子若落进锁定区则本迭代不动
（fail-safe：咬合接触是 Occlusion Engine 的职责）。
"""
from __future__ import annotations

import sys
import os

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(__file__))
from test_contact_engine import crown_tube                     # noqa: E402
from test_contact_refine import (CROWN_R, PARAMS, distal_at,   # noqa: E402
                                 frame, neighbor_at)
from app.geometry.crown_outer.contact import NeighborPatch     # noqa: E402
from app.geometry.crown_outer.contact_refine import (          # noqa: E402
    RefineParams, _loop_and_zones, _transition_rings, _vertex_adjacency,
    refine_contacts)


def _capped_tube():
    """开放管 + **顶盖**：盖面法向朝轴 → occ_table 非空，测得出真"咬合面"。"""
    verts, faces = crown_tube()
    v = np.asarray(verts, dtype=float)
    top_z = v[:, 2].max()
    top_ring = np.flatnonzero(np.abs(v[:, 2] - top_z) < 1e-9)
    center = len(v)
    v = np.vstack([v, [v[top_ring, 0].mean(), top_z, v[top_ring, 2].mean()]])
    faces = list(map(list, np.asarray(faces)))
    for i in range(len(top_ring)):
        a, b = top_ring[i], top_ring[(i + 1) % len(top_ring)]
        faces.append([a, b, center])
    return v, np.asarray(faces, dtype=np.int64)


def _run(**over):
    verts, faces = _capped_tube()
    mv, mf = neighbor_at(CROWN_R - 0.5)          # 近中穿透 0.5mm
    dv, df = distal_at(-(CROWN_R + 0.3))         # 远中间隙 0.3mm
    p = RefineParams(target_gap_mm=0.3, damping=0.5, max_step_mm=0.4,
                     max_iterations=8, **over)
    res = refine_contacts(mesh(verts, faces), frame(), NeighborPatch(mv, mf),
                          NeighborPatch(dv, df), p)
    return verts, faces, res


def mesh(verts, faces):
    return type("M", (), {"vertices": verts, "faces": faces})()


def _zones(verts, faces):
    loop_idx, loop_set, _occ = _loop_and_zones(verts, faces, frame(),
                                               RefineParams())
    from app.geometry.crown_outer.contact_refine import occ_table_mask
    occ = np.flatnonzero(occ_table_mask(verts, faces, frame(), RefineParams()))
    return loop_idx, loop_set, occ


def test_occlusal_and_margin_displacement_are_exactly_zero():
    """**v2 的定义性不变量**：修邻接绝不移动咬合/颈缘（精确 0，非"尽量小"）。"""
    verts, faces, res = _run()
    assert res.mesh is not None
    v = np.asarray(res.mesh.vertices)
    loop_idx, _ls, occ = _zones(verts, faces)
    assert np.abs(v[loop_idx] - verts[loop_idx]).max() == 0.0, \
        "颈缘顶点被移动 —— 局部带状求解失效"
    assert np.abs(v[occ] - verts[occ]).max() == 0.0, \
        "咬合顶点被移动 —— 局部带状求解失效"
    assert res.margin_max_displacement_mm == 0.0
    assert res.occlusal_p95_displacement_mm == 0.0


def test_penetration_removed_without_flips():
    verts, faces, res = _run()
    assert res.mesial.penetration_mm <= PARAMS.penetration_tol_mm, res.summary()
    assert res.n_flipped_faces == 0


def test_transition_rings_never_reach_locked_zone():
    """过渡环必须被锁定区挡住（回归：set(bool_array) 迭代出 {True,False} 的坑）。"""
    verts, faces = crown_tube()
    loop_idx, loop_set, occ = _zones(verts, faces)
    locked = loop_set.copy()
    locked[occ] = True
    adj = _vertex_adjacency(faces, len(verts))
    seeds = set(range(len(verts))) - set(int(i) for i in np.flatnonzero(locked))
    free = _transition_rings(seeds, locked, adj, 8)
    assert not (free & set(int(i) for i in np.flatnonzero(loop_set)))
    assert not (free & set(int(i) for i in np.flatnonzero(occ)))


def test_seeds_inside_locked_zone_do_not_move():
    """fail-safe：接触种子落在锁定区时，该顶点本迭代不动（目标被忽略）。"""
    verts, faces = crown_tube()
    loop_idx, loop_set, occ = _zones(verts, faces)
    locked = loop_set.copy()
    locked[occ] = True
    locked_ids = np.flatnonzero(locked)
    adj = _vertex_adjacency(faces, len(verts))
    # 种子 = 全部锁定区顶点 → 过渡环应为空（锁定区外无种子可达）
    free = _transition_rings(locked_ids, locked, adj, 3)
    assert len(free) == 0


def test_v1_global_path_still_available():
    """local_band=False 保留 v1 全局路径（对照/回归用），且新旧结果都合法。"""
    verts, faces, res = _run(local_band=False)
    assert res.mesh is not None
    # v1 路径只保证"弱锚尽量小"，不保证 0 —— 只断言合法性与穿透
    assert res.mesial.penetration_mm <= PARAMS.penetration_tol_mm + 1e-3
    assert res.n_flipped_faces == 0
