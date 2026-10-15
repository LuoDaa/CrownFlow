"""Occlusion v1（Collision Resolution Engine）的合成测试。

夹具：带顶盖的冠管（盖 = 咬合面）+ **从上方压入的对颌板**
（板体在对颌侧/上方，表面朝下）。冠盖顶被压入对颌 ~0.8mm。

断言核心（用户 2026-10-02 规格）：
    occlusal 穿透 p95 单调大幅下降 / max 不恶化
    margin 位移精确 0
    proximal core（非咬合面）位移精确 0
    翻面不恶化
    Contact 保持性复测存在（handoff 语义）
"""
from __future__ import annotations

import sys
import os

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(__file__))
from test_contact_i2 import _capped_tube                       # noqa: E402
from test_contact_refine import frame, neighbor_at, CROWN_R    # noqa: E402
from app.geometry.crown_outer.contact import NeighborPatch     # noqa: E402
from app.geometry.crown_outer.occlusion import (               # noqa: E402
    OcclusionParams, solve_occlusion_collision)


def _antagonist_plate(z_plane, half=12.0, n=48):
    """对颌：水平板在 z = z_plane，**板体在上方**（表面法向朝下/朝冠）。"""
    verts = []
    faces = []
    grid = []
    for i in range(n + 1):
        row = []
        for j in range(n + 1):
            x = -half + 2.0 * half * i / n
            y = -half + 2.0 * half * j / n
            row.append(len(verts))
            verts.append((x, y, z_plane))
        grid.append(row)
    for i in range(n):
        for j in range(n):
            a, b = grid[i][j], grid[i][j + 1]
            c, d = grid[i + 1][j], grid[i + 1][j + 1]
            # 法向朝下（-z）：绕序按右手定则排
            faces.append([a, c, b])
            faces.append([b, c, d])
    return np.asarray(verts, dtype=float), np.asarray(faces, dtype=np.int64)


def _run(press=0.8, **over):
    verts, faces = _capped_tube()
    top_z = verts[:, 2].max()
    av, af = _antagonist_plate(top_z - press)     # 压入冠盖 press mm
    mv, mf = neighbor_at(-(CROWN_R + 0.3))        # 远中侧正常间隙（Contact 保持）
    p = OcclusionParams(**over)
    # O1：符号来源必须**显式声明**（开放裁片没有闭合实体内外可判）。
    # 对颌板本体在上方、表面朝下；两块邻牙面本体在 +x 侧、表面朝 −x（见邻居夹具 docstring）。
    res = solve_occlusion_collision(
        type("M", (), {"vertices": verts, "faces": faces})(), frame(),
        NeighborPatch(av, af),
        contact_patches={"distal": NeighborPatch(mv, mf),
                         "mesial": NeighborPatch(*neighbor_at(CROWN_R + 0.3))},
        params=p,
        antagonist_outward=[0.0, 0.0, -1.0],
        contact_outward={"distal": [-1.0, 0.0, 0.0], "mesial": [-1.0, 0.0, 0.0]})
    return verts, faces, res


def test_occlusal_penetration_drops_dramatically():
    verts, faces, res = _run(press=0.8)
    assert res.mesh is not None
    b = res.history[0]
    a = res.history[-1]
    assert b["p95"] > 0.3, "夹具前提：初始咬合干涉应显著（%.3f）" % b["p95"]
    assert a["p95"] < b["p95"] * 0.25, \
        "occlusal p95 应下降 ≥75%%（%.3f → %.3f）" % (b["p95"], a["p95"])
    assert a["max"] <= b["max"] + 1e-9, "max 不得恶化"


def test_margin_and_proximal_core_never_move():
    """**硬不变量**：颈环与一切非咬合面顶点位移精确 0。"""
    verts, faces, res = _run(press=0.8)
    v = np.asarray(res.mesh.vertices)
    assert res.margin_max_displacement_mm == 0.0
    assert res.proximal_core_max_displacement_mm == 0.0, \
        "proximal core 被咬合修拉动 —— 局部化失效"
    # 咬合面以外区域逐点验证
    from app.geometry.crown_outer.occlusion import _classify
    from test_contact_refine import frame as F
    _loop, _ls, occ_t, locked = _classify(verts, faces, F(), OcclusionParams())
    assert np.abs(v[locked] - verts[locked]).max() == 0.0


def test_history_monotone_and_flips_guarded():
    verts, faces, res = _run(press=0.8)
    pens = [h["p95"] for h in res.history]
    for a, b in zip(pens, pens[1:]):
        assert b <= a + 1e-9, pens
    assert res.n_flipped_faces <= 0.01 * len(faces)


def test_contact_preservation_reported():
    verts, faces, res = _run(press=0.8)
    assert "before" in res.contact_preservation and \
        "after" in res.contact_preservation
    for side in res.contact_preservation["before"]:
        b = res.contact_preservation["before"][side]["p95"]
        a = res.contact_preservation["after"][side]["p95"]
        assert a <= b + 0.02 + 1e-9, \
            "Contact %s 被咬合修拉坏（%.3f → %.3f）" % (side, b, a)
