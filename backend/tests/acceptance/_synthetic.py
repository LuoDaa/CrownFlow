"""合成几何 fixture —— 就位方向求解器的验收基础。

和 FDI36 真实病例不同，这里**真值是构造出来的**，因此可以做严格断言。
这很重要：FDI36 基台的临床正确就位方向无法从数据独立确定（见
``test_fdi36_golden.py::test_insertion_axis_agrees_with_fitted_wall_axis`` 的说明），
所以真实病例上只能做"不劣于平凡解 + 与几何自洽"的断言，而"精度"只能在合成几何上验。
"""

from __future__ import annotations

import math

import numpy as np
import trimesh


def frustum(
    r_bottom: float = 3.0,
    r_top: float = 2.2,
    height: float = 8.0,
    *,
    sections: int = 120,
    tilt_deg: float = 0.0,
    bulge: float = 1.0,
    bulge_at: float = 0.5,
) -> tuple[trimesh.Trimesh, np.ndarray]:
    """向上收缩的圆台（可选倾斜、可选外凸环）。

    返回 ``(mesh, base_ring)``。``base_ring`` 就是精确的 Margin 环 —— 保证
    Margin 相关断言不会被"环提取不准"这种无关噪声污染。

    ``bulge > 1`` 在中段加一个外凸环，相对任何轴都形成倒凹；``tilt_deg`` 整体倾斜。
    """
    theta = np.linspace(0.0, 2.0 * np.pi, sections, endpoint=False)
    zs = [0.0, height * bulge_at, height]
    radii = [r_bottom, (r_bottom + r_top) / 2.0 * bulge, r_top]
    verts = np.vstack([
        np.c_[r * np.cos(theta), r * np.sin(theta), np.full(sections, z)]
        for r, z in zip(radii, zs)
    ])
    faces: list[list[int]] = []
    for k in range(2):
        a, b = k * sections, (k + 1) * sections
        for i in range(sections):
            j = (i + 1) % sections
            faces += [[a + i, a + j, b + j], [a + i, b + j, b + i]]

    center_bottom = len(verts)
    verts = np.vstack([verts, [[0.0, 0.0, 0.0]]])
    center_top = len(verts)
    verts = np.vstack([verts, [[0.0, 0.0, height]]])
    for i in range(sections):
        j = (i + 1) % sections
        faces += [[center_bottom, j, i],
                  [center_top, 2 * sections + i, 2 * sections + j]]

    mesh = trimesh.Trimesh(vertices=verts, faces=np.array(faces), process=True)
    base_ring = verts[:sections].copy()

    if tilt_deg:
        T = trimesh.transformations.rotation_matrix(math.radians(tilt_deg), [0, 1, 0])
        mesh.apply_transform(T)
        base_ring = (T[:3, :3] @ base_ring.T).T + T[:3, 3]

    return mesh, base_ring


def true_axis_for_tilt(tilt_deg: float) -> np.ndarray:
    """``frustum(tilt_deg=...)`` 的真值轴（= 倾斜后的 +Z）。"""
    return np.array([math.sin(math.radians(tilt_deg)), 0.0, math.cos(math.radians(tilt_deg))])


def random_rigid(rng: np.random.Generator, *, translation_scale: float = 80.0):
    """随机刚体变换矩阵（正交 + 平移）。"""
    A = rng.normal(size=(3, 3))
    Q, _ = np.linalg.qr(A)
    if np.linalg.det(Q) < 0:
        Q[:, 0] = -Q[:, 0]
    T = np.eye(4)
    T[:3, :3] = Q
    T[:3, 3] = rng.normal(0.0, translation_scale, 3)
    return T


def apply_rigid(mesh: trimesh.Trimesh, T: np.ndarray) -> trimesh.Trimesh:
    out = mesh.copy()
    out.apply_transform(T)
    return out


def transform_points(points: np.ndarray, T: np.ndarray) -> np.ndarray:
    return (T[:3, :3] @ np.asarray(points).T).T + T[:3, 3]
