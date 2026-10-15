"""生成出口的拓扑校验（只读 + 可选的最小焊接）。

目的：定位 duplicate vertex **首次出现**的阶段（template deformation / subdivision /
boolean / export），而不是在最后一步用大范围修复把问题盖掉。

`validate_topology` 只测量；`record_topology` 把结果按阶段追加到一份 JSON，
这样即使后续阶段（stitch）失败，证据仍然留在磁盘上。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import trimesh

EPS_MM = 0.01


def validate_topology(mesh, *, eps_mm: float = EPS_MM, aspect_max: float = 10.0) -> dict:
    """返回该网格的拓扑/质量指标（不修改输入）。"""
    vertices = np.asarray(mesh.vertices, float)
    faces = np.asarray(mesh.faces, np.int64)
    report = {"vertices": int(len(vertices)), "faces": int(len(faces))}
    if not len(vertices) or not len(faces):
        return {**report, "empty": True}
    # 精确重复（逐坐标相同）
    unique, inverse, counts = np.unique(vertices, axis=0, return_inverse=True, return_counts=True)
    report["exact_duplicate_vertices"] = int(len(vertices) - len(unique))
    # 近重复（欧氏距离 < eps）
    from scipy.spatial import cKDTree
    try:
        pairs = cKDTree(vertices).query_pairs(float(eps_mm))
        report["near_duplicate_pairs_below_%.2fmm" % eps_mm] = int(len(pairs))
    except (ValueError, MemoryError):
        report["near_duplicate_pairs_below_%.2fmm" % eps_mm] = None
    # 边界边 / 非流形边
    from ..geometry.crown_outer.thickness import _edge_table
    counts_by_edge, _ori = _edge_table(faces)
    report["boundary_edges"] = int(sum(1 for n in counts_by_edge.values() if n == 1))
    report["nonmanifold_edges"] = int(sum(1 for n in counts_by_edge.values() if n > 2))
    # 连通分量（按面邻接）
    labels = trimesh.graph.connected_component_labels(trimesh.Trimesh(vertices, faces, process=False).face_adjacency,
                                                      node_count=len(faces))
    report["components"] = int(len(np.unique(labels)))
    # 面质量
    tri = vertices[faces]
    edges = np.stack([np.linalg.norm(tri[:, 1] - tri[:, 2], axis=1),
                      np.linalg.norm(tri[:, 2] - tri[:, 0], axis=1),
                      np.linalg.norm(tri[:, 0] - tri[:, 1], axis=1)], axis=1)
    area = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    aspect = edges.max(axis=1) / np.maximum(edges.min(axis=1), 1e-12)
    report["degenerate_faces"] = int((area <= 1e-12).sum())
    report["faces_aspect_above_%.0f" % aspect_max] = int((aspect > aspect_max).sum())
    report["min_edge_mm"] = round(float(edges.min()), 6)
    report["aspect_max"] = round(float(aspect.max()), 3)
    report["watertight"] = bool(trimesh.Trimesh(vertices, faces, process=False).is_watertight)
    return report


def record_topology(stage: str, mesh, *, folder=None, extra=None) -> dict:
    """测量并把结果按阶段追加到 ``<DATA_DIR>/debug/topology_validation.json``。"""
    report = validate_topology(mesh)
    if extra:
        report.update(extra)
    report["stage"] = stage
    try:
        from ..config import DATA_DIR
        target = Path(folder or (Path(DATA_DIR) / "debug")) / "topology_validation.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        data = json.loads(target.read_text(encoding="utf-8")) if target.is_file() else {}
        data[stage] = report
        target.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    except (OSError, ValueError):
        pass
    return report


def merge_near_duplicates(mesh, *, eps_mm: float = 0.02):
    """**出口最小修复**：按**距离**合并重复顶点（<eps）并重建邻接；丢弃因此退化的面。

    不变量（任一不满足就原样回退）：boundary / nonmanifold / 组件数 / watertight 全部不变。
    """
    before = validate_topology(mesh)
    vertices = np.asarray(mesh.vertices, float)
    faces = np.asarray(mesh.faces, np.int64)
    from scipy.spatial import cKDTree
    pairs = sorted(cKDTree(vertices).query_pairs(float(eps_mm)))
    if not pairs:
        return mesh, {"merge_applied": False, "reason": "NO_NEAR_DUPLICATE", "before": before}
    parent = list(range(len(vertices)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    merged_count = 0
    for a, b in pairs:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
            merged_count += 1
    remap = np.array([find(i) for i in range(len(vertices))], np.int64)
    new_faces = remap[faces]
    degenerate = ((new_faces[:, 0] == new_faces[:, 1]) | (new_faces[:, 1] == new_faces[:, 2])
                  | (new_faces[:, 0] == new_faces[:, 2]))
    candidate = trimesh.Trimesh(vertices=vertices, faces=new_faces[~degenerate], process=False)
    candidate.remove_unreferenced_vertices()
    # 合并会留下"边仍成对但绕向同向"的局部不一致（boundary=0 而 watertight=False 就是它），
    # 用标准绕向修复重建邻接后再校验。
    try:
        trimesh.repair.fix_normals(candidate, multibody=False)
    except (ValueError, IndexError):
        pass
    after = validate_topology(candidate)
    ok = (after["boundary_edges"] == before["boundary_edges"]
          and after["nonmanifold_edges"] == before["nonmanifold_edges"]
          and after["components"] == before["components"]
          and after["watertight"] == before["watertight"]
          and after["degenerate_faces"] == 0)
    if not ok:
        return mesh, {"merge_applied": False, "reason": "INVARIANT_CHANGED", "before": before,
                      "after": after, "candidate_pairs": int(len(pairs))}
    return candidate, {"merge_applied": True, "eps_mm": eps_mm, "candidate_pairs": int(len(pairs)),
                       "merged_vertices": int(merged_count), "dropped_degenerate_faces": int(degenerate.sum()),
                       "before": before, "after": after}
