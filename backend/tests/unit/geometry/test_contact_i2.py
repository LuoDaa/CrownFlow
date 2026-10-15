"""Contact I.2 —— SDF 投影 + 局部带 blend + line search 的合成测试。

夹具：带顶盖的冠管（盖 = 咬合面）+ 深插鳍（近中穿透 ~1mm 级）。
断言的核心不变量（用户 2026-10-08 **硬化后**的口径）：

    editable 穿透单调下降，**但接受步必须过几何硬门**
      （零新增自交 / 零非流形 / 零翻面 / 锁定区位移精确 0）
    咬合面/颈缘位移精确为 0
    occlusal locked 穿透不被当作 solver failure（handoff 记账）
    **清不掉的穿透如实报 CONSTRAINT_CONFLICT —— 不许以折叠换清零**
      （旧口径"翻面只要别超过 +0.2%"已删除）
"""
from __future__ import annotations

import sys
import os

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(__file__))
from test_contact_engine import crown_tube                     # noqa: E402
from test_contact_refine import frame, neighbor_at, CROWN_R    # noqa: E402
from app.geometry.crown_outer.contact import NeighborPatch     # noqa: E402
from app.geometry.crown_outer.contact_i2 import (I2Params,     # noqa: E402
                                                 measure,
                                                 solve_proximal_projection)


def _capped_tube():
    verts, faces = crown_tube()
    v = np.asarray(verts, dtype=float)
    top_z = v[:, 2].max()
    top_ring = np.flatnonzero(np.abs(v[:, 2] - top_z) < 1e-9)
    v = np.vstack([v, [v[top_ring, 0].mean(), top_z, v[top_ring, 2].mean()]])
    faces = [list(map(int, f)) for f in np.asarray(faces)]
    for i in range(len(top_ring)):
        a, b = top_ring[i], top_ring[(i + 1) % len(top_ring)]
        faces.append([a, b, len(v) - 1])
    return v, np.asarray(faces, dtype=np.int64)


def _run(deep=1.2, **over):
    verts, faces = _capped_tube()
    mv, mf = neighbor_at(CROWN_R - deep)         # 近中深穿透
    dv, df = neighbor_at(-(CROWN_R + 0.3))       # 远中间隙 0.3
    p = I2Params(**over)
    res = solve_proximal_projection(
        type("M", (), {"vertices": verts, "faces": faces})(), frame(),
        NeighborPatch(mv, mf), NeighborPatch(dv, df), p)
    return verts, faces, res


def test_deep_penetration_respects_geometry_gate():
    """新契约：清穿透**不许以折叠为代价**。

    夹具前提：初始 editable p95 显著（>0.5mm）。深穿透下这条夹具"清干净必然
    要折叠"，所以正确结果不是"压到 tolerance"，而是：
      ① 每一个被接受的步都过了几何硬门；
      ② 被接受的序列在 editable p95 上单调下降；
      ③ 若目标不可达 → 如实 CONSTRAINT_CONFLICT（而不是折叠换清零）。
    """
    verts, faces, res = _run()
    assert res.mesh is not None
    base = max(res.history[0]["mesial_editable"]["p95"],
               res.history[0]["distal_editable"]["p95"])
    assert base > 0.5, "夹具前提：初始 editable p95 应显著（%.3f）" % base
    acc = [h for h in res.history if h.get("accepted")]
    assert acc, "至少应有一个可接受的小步"
    for h in acc:
        assert h["gate"]["ok"] is True, h
        m = h["gate"]["metrics"]
        assert m["flipped"]["n_flipped_severe"] == 0, h
        assert m["self_intersections_focus"]["new"] == 0, h
    chain = [base] + [max(h["mesial_editable"]["p95"],
                          h["distal_editable"]["p95"]) for h in acc]
    for a, b in zip(chain, chain[1:]):
        assert b < a + 1e-9, chain
    if res.constraint_conflict:
        assert res.status == "CONSTRAINT_CONFLICT"
        assert chain[-1] > 0.05, "报冲突时不该已经清干净"
    else:
        assert chain[-1] <= 0.05


def test_locked_surfaces_never_move():
    """**硬不变量**：咬合盖与颈环位移精确 0（不管穿透多深）。"""
    verts, faces, res = _run(deep=2.0)
    v = np.asarray(res.mesh.vertices)
    loop = np.asarray(__import__(
        "app.geometry.crown_outer.contact_i2", fromlist=["_loop_of"]
    )._loop_of(verts, faces))
    assert np.abs(v[loop] - verts[loop]).max() == 0.0
    # 咬合盖 = z 最高的中心顶点（法向朝轴）
    top = int(np.argmax(v[:, 2]))
    assert np.abs(v[top] - verts[top]).max() == 0.0


def test_occlusal_conflict_is_handoff_not_failure():
    """锁定区残留穿透必须记账为 handoff，而不是判定 solver 失败。"""
    verts, faces, res = _run(deep=2.0)
    assert res.handoff_to_occlusion["note"]
    # handoff 账里必须有 occlusal 穿透统计（dict 含 count/p95 等）
    for side in ("mesial", "distal"):
        assert isinstance(res.handoff_to_occlusion[side], dict)
        assert "p95" in res.handoff_to_occlusion[side]
    # 最终状态只由 editable 穿透决定；锁定区穿透存在也允许 OK。
    # 硬化后多一种合法结局：几何安全的前提下清不掉 → CONSTRAINT_CONFLICT。
    assert res.status in ("OK", "REVIEW_REQUIRED", "CONSTRAINT_CONFLICT")


def test_flips_do_not_worsen():
    verts, faces, res = _run()
    assert res.n_flipped_faces == 0, res.n_flipped_faces


def test_history_is_monotone_on_editable():
    """**被接受**的步必须让 editable p95 单调不升（被拒绝的 trial 不进这条链）。"""
    verts, faces, res = _run()
    acc = [h for h in res.history if h.get("accepted")]
    if not acc:
        assert res.constraint_conflict
        return
    pens = [max(res.history[0]["mesial_editable"]["p95"],
                res.history[0]["distal_editable"]["p95"])] +         [max(h["mesial_editable"]["p95"], h["distal_editable"]["p95"])
         for h in acc]
    for a, b in zip(pens, pens[1:]):
        assert b <= a + 1e-9, (pens, res.alphas_accepted)
