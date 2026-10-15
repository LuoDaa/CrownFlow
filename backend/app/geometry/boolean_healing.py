"""boolean 输出后的**安全愈合**（crown_minus_cutter() 之后、transition clip 之前）。

为什么在这里：逐阶段校验已定位——outer/cutter 输入都是 0 duplicate / 水密，
而 `crown_minus_cutter` 的输出首次出现 duplicate（Case1：1,466 对 <0.01mm，
最小边 1.2e-05mm，是布尔精确算术换算产生的数值碎片）。

为什么不是简单 merge vertices：实测把 1.2e-05 的碎片直接合并会让布尔壳不再水密
（boundary 仍为 0，但 `is_watertight=False`，绕向/闭合被破坏）。因此这里做
**逐条 tiny edge 的 edge collapse**，并且：

  * 只折叠**内部边**（恰好被 2 个面共享）⇒ 边界环天然不变；
  * 折叠前检查 **link condition**（u、v 的公共邻点恰好是该边的两个对顶点）
    ⇒ 不产生非流形、不产生夹点；
  * 折叠会让两个面退化 ⇒ 删除（B），随后重建 vertex-face 邻接（C）与绕向（D）；
  * 每一轮之后校验 boundary_edges / nonmanifold_edges / components / watertight，
    任一变化立即回退到本轮输入（整体失败则回退到原始输入）。
"""
from __future__ import annotations

import numpy as np
import trimesh

from .topology_validation import record_topology, validate_topology

TINY_EDGE_MM = 0.002
MAX_ROUNDS = 6


def _face_index(faces):
    return [set(int(i) for i in tri) for tri in faces]


def _edge_faces(faces):
    """边 → 面索引集合。"""
    table = {}
    for index, tri in enumerate(faces):
        a, b, c = (int(tri[0]), int(tri[1]), int(tri[2]))
        for u, v in ((a, b), (b, c), (c, a)):
            key = (u, v) if u < v else (v, u)
            table.setdefault(key, []).append(index)
    return table


def _link_condition_ok(adjacency, faces, u, v, shared):
    """link condition：u 与 v 的公共邻点集合 == 该边的两个对顶点。"""
    neighbours_u = set()
    for face_index in adjacency[u]:
        neighbours_u.update(int(x) for x in faces[face_index])
    neighbours_v = set()
    for face_index in adjacency[v]:
        neighbours_v.update(int(x) for x in faces[face_index])
    common = (neighbours_u & neighbours_v) - {u, v}
    if len(shared) != 2:
        return False
    opposite = set()
    for face_index in shared:
        opposite.add(int(set(faces[face_index]) - {u, v}).pop())
    return common == opposite


def _apply_collapses(vertices, faces, pairs, *, tiny_edge_mm):
    """对互不共享顶点的 (u, v) 对做一轮独立折叠。返回 (vertices, faces, stats)。"""
    adjacency = [set() for _ in range(len(vertices))]
    for index, tri in enumerate(faces):
        for vid in tri:
            adjacency[int(vid)].add(index)
    edge_table = _edge_faces(faces)
    used = set()
    accepted = []
    rejected = {"boundary_or_nonmanifold": 0, "link_condition": 0, "duplicate_face": 0}
    for a, b in pairs:
        if a in used or b in used:
            continue
        key = (a, b) if a < b else (b, a)
        shared = edge_table.get(key, [])
        if len(shared) != 2:
            rejected["boundary_or_nonmanifold"] += 1
            continue
        if not _link_condition_ok(adjacency, faces, a, b, shared):
            rejected["link_condition"] += 1
            continue
        # 折叠后是否会出现重复面（夹点）
        new_sets = []
        for face_index in adjacency[b]:
            if face_index in shared:
                continue
            remapped = [a if int(x) == b else int(x) for x in faces[face_index]]
            new_sets.append(tuple(sorted(remapped)))
        existing = {tuple(sorted(int(x) for x in tri)) for index, tri in enumerate(faces)
                    if a in set(int(x) for x in tri)}
        if any(s in existing for s in new_sets):
            rejected["duplicate_face"] += 1
            continue
        used.add(a); used.add(b)
        accepted.append((a, b, set(shared)))
    if not accepted:
        return vertices, faces, {"accepted": 0, "rejected": rejected, "dropped_faces": 0}
    remap = np.arange(len(vertices))
    drop = set()
    for a, b, shared in accepted:
        remap[b] = a
        drop |= shared
    new_faces = []
    for index, tri in enumerate(faces):
        if index in drop:
            continue
        new_faces.append([int(remap[int(x)]) for x in tri])
    new_faces = [tri for tri in new_faces if len(set(tri)) == 3]
    return (np.asarray(vertices, float)[np.unique(remap)],
            np.asarray(new_faces, np.int64), {"accepted": len(accepted), "rejected": rejected,
                                              "dropped_faces": len(drop)})


def boolean_healing_stage(mesh, *, tiny_edge_mm: float = TINY_EDGE_MM, max_rounds: int = MAX_ROUNDS,
                          folder=None):
    """返回 ``(healed_mesh, report)``。任何不变量变化 ⇒ 回退，绝不劣化。"""
    before = validate_topology(mesh)
    if before.get("empty"):
        return mesh, {"status": "SKIPPED_EMPTY", "before": before}
    current = trimesh.Trimesh(np.asarray(mesh.vertices, float), np.asarray(mesh.faces, np.int64), process=False)
    rounds = []
    from scipy.spatial import cKDTree
    reverted = None
    for round_index in range(max_rounds):
        vertices = np.asarray(current.vertices, float)
        faces = np.asarray(current.faces, np.int64)
        pairs = sorted((int(a), int(b)) for a, b in cKDTree(vertices).query_pairs(float(tiny_edge_mm)))
        if not pairs:
            break
        candidate_vertices, candidate_faces, stats = _apply_collapses(vertices, faces, pairs,
                                                                     tiny_edge_mm=tiny_edge_mm)
        if stats["accepted"] == 0:
            rounds.append({**stats, "round": round_index + 1})
            break
        candidate = trimesh.Trimesh(candidate_vertices, candidate_faces, process=False)
        # C/D：重建邻接 + 重新定向法向
        candidate.remove_unreferenced_vertices()
        try:
            trimesh.repair.fix_normals(candidate, multibody=False)
        except Exception as exc:      # 绕向修复是 best-effort；不变量校验才是真门
            rounds_note = {"winding_repair_error": str(exc)[:120]}
        after = validate_topology(candidate)
        invariants_ok = (after["boundary_edges"] == before["boundary_edges"]
                         and after["nonmanifold_edges"] == before["nonmanifold_edges"]
                         and after["components"] == before["components"]
                         and after["watertight"] == before["watertight"])
        rounds.append({**stats, "round": round_index + 1, "after": after,
                       "invariants_ok": bool(invariants_ok)})
        if not invariants_ok:
            reverted = {"round": round_index + 1, "reason": "INVARIANT_CHANGED", "after": after}
            break
        current = candidate
    final = validate_topology(current)
    report = {"algorithm": "boolean_healing/link_condition_collapse/1",
              "tiny_edge_mm": tiny_edge_mm, "before": before, "after": final, "rounds": rounds,
              "status": "REVERTED" if reverted else ("APPLIED" if final != before else "NO_CHANGE"),
              "reverted": reverted,
              "target": {"near_duplicate_before": before.get("near_duplicate_pairs_below_0.01mm"),
                         "near_duplicate_after": final.get("near_duplicate_pairs_below_0.01mm"),
                         "min_edge_before_mm": before.get("min_edge_mm"),
                         "min_edge_after_mm": final.get("min_edge_mm")}}
    if reverted:
        return mesh, report
    record_topology("03b_boolean_healing", current, folder=folder, extra={"healing": report})
    return current, report
