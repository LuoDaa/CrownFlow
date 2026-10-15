"""R_spin 残余可观测性 sanity check（解析圆柱 A/B/C 三种 correspondence）。

目的：判定"均匀圆柱上 R_spin≈10.8 的残余敏感度"到底来自
  1) 真实几何  2) 离散 mesh 采样  3) 最近邻匹配  4) 法向实现
方法：不依赖任何 mesh fixture，直接构造**解析圆柱**（完整 360°、无端盖、无起伏、无噪声、
半径均匀、轴明确），只用点+法向数组做三种 correspondence：

  A fixed        : source i ↔ target i（原始对应）
  B nn-rematch   : 旋转后 source 重新找 target 最近邻，**法向取 rematched target 的**
  C analytic     : theta_target = theta_source + Δθ（按理论角解析对应）

理论预期：A 的 point residual > 0 且法向角 ≈ 旋转角；
C 在理想连续圆柱上 rmse→0 且法向角→0；
B 的残差量级由**角向网格间距**决定（离散采样 artifact），随网格加密而下降。
"""

from __future__ import annotations

import numpy as np

#: 与 `_pair_quality` 一致的口径（只换 correspondence 来源）
CORR_CAP_MM = 2.0
RMSE_FLOOR_MM = 0.05
INLIER_FLOOR = 0.01
NORMAL_ANGLE_FLOOR_DEG = 0.5


def analytic_cylinder(*, radius=9.0, height=5.0, sections=720, rings=9):
    theta = np.linspace(0.0, 2.0 * np.pi, sections, endpoint=False)
    z = np.linspace(-height / 2.0, height / 2.0, rings)
    tt, zz = np.meshgrid(theta, z, indexing="ij")
    points = np.c_[radius * np.cos(tt).ravel(), radius * np.sin(tt).ravel(), zz.ravel()]
    normals = np.c_[np.cos(tt).ravel(), np.sin(tt).ravel(), np.zeros(tt.size)]
    return points, normals, theta


def rotate_about_z(points, normals, degrees):
    angle = np.radians(degrees)
    cos, sin = np.cos(angle), np.sin(angle)
    rotation = np.array([[cos, -sin, 0.0], [sin, cos, 0.0], [0.0, 0.0, 1.0]])
    return points @ rotation.T, normals @ rotation.T


def _sensitivity(rmse_mm, inlier, normal_deg, base):
    drop = base["inlier_ratio"] - inlier
    rise = rmse_mm - base["rmse_mm"]
    normal_rise = normal_deg - base["normal_agreement_max_deg"]
    return max(drop / INLIER_FLOOR, rise / RMSE_FLOOR_MM, normal_rise / NORMAL_ANGLE_FLOOR_DEG)


def _measure(distance, normal_deg):
    inlier = float((distance <= CORR_CAP_MM).mean())
    return float(np.sqrt(np.mean(distance ** 2))), inlier, normal_deg


def probe(points, normals, theta, degrees, mode):
    moved, moved_normals = rotate_about_z(points, normals, degrees)
    if mode == "fixed":
        # A：原始第 i 点对第 i 点（不重匹配）
        distance = np.linalg.norm(moved - points, axis=1)
        dots = np.abs(np.einsum("ij,ij->i", moved_normals, normals))
    elif mode == "nn":
        # B：旋转后重新找最近邻；**法向取 rematched target 的法向**
        from scipy.spatial import cKDTree
        distance, index = cKDTree(points).query(moved, k=1)
        dots = np.abs(np.einsum("ij,ij->i", moved_normals, normals[index]))
    elif mode == "analytic":
        # C：按理论角解析对应（theta_target = theta_source + Δθ，环绕）
        from scipy.spatial import cKDTree
        angle = np.radians(degrees)
        target_theta = np.mod(np.arctan2(points[:, 1], points[:, 0]) + angle, 2.0 * np.pi)
        # 同一 z 环内按角解析定位
        rings = np.unique(np.round(points[:, 2], 9))
        index = np.empty(len(points), dtype=int)
        for ring in rings:
            z_mask = np.isclose(points[:, 2], ring)
            ring_ids = np.flatnonzero(z_mask)
            ring_theta = np.mod(np.arctan2(points[ring_ids, 1], points[ring_ids, 0]), 2.0 * np.pi)
            order = np.argsort(ring_theta)
            grid = ring_theta[order]
            delta = np.abs(np.mod(target_theta[z_mask][:, None] - grid[None, :] + np.pi, 2 * np.pi) - np.pi)
            index[ring_ids] = ring_ids[order][np.argmin(delta, axis=1)]
        distance = np.linalg.norm(moved - points[index], axis=1)
        dots = np.abs(np.einsum("ij,ij->i", moved_normals, normals[index]))
    else:
        raise ValueError(mode)
    normal_deg = float(np.degrees(np.arccos(np.clip(dots, -1.0, 1.0))).max())
    rmse, inlier, normal_deg = _measure(distance, normal_deg)
    return {"rmse_mm": rmse, "inlier_ratio": inlier, "normal_agreement_max_deg": normal_deg}


def run_probe(sections=720, radius=9.0):
    points, normals, theta = analytic_cylinder(sections=sections, radius=radius)
    rows = []
    for mode in ("fixed", "nn", "analytic"):
        base = probe(points, normals, theta, 0.0, mode)
        for degrees in (0.5, 1.0, 2.0, 4.0):
            measured = probe(points, normals, theta, degrees, mode)
            rows.append({"probe": mode, "degrees": degrees, **measured,
                         "sensitivity": _sensitivity(measured["rmse_mm"], measured["inlier_ratio"],
                                                     measured["normal_agreement_max_deg"], base)})
    return rows


def test_analytic_cylinder_correspondence_probes():
    rows = run_probe(sections=720)
    print("\n%-9s %-6s %-12s %-8s %-14s %s" % ("probe", "angle", "rmse_mm", "inlier", "normal_deg", "sensitivity"))
    for row in rows:
        print("%-9s %-6.1f %-12.6f %-8.3f %-14.6f %.3f" % (
            row["probe"], row["degrees"], row["rmse_mm"], row["inlier_ratio"],
            row["normal_agreement_max_deg"], row["sensitivity"]))

    analytic = [r for r in rows if r["probe"] == "analytic"]
    for row in analytic:
        assert row["rmse_mm"] < 1e-9, "解析对应必须精确：rmse=%.3e" % row["rmse_mm"]
        assert row["normal_agreement_max_deg"] < 1e-4, \
            "解析对应法向角必须为 0：%.3e" % row["normal_agreement_max_deg"]
    fixed = [r for r in rows if r["probe"] == "fixed"]
    for row in fixed:
        expected = 2.0 * 9.0 * np.sin(np.radians(row["degrees"]) / 2.0)   # 弦长
        assert abs(row["rmse_mm"] - expected) < 1e-6, "A 的 point residual 应等于弦长"
        assert abs(row["normal_agreement_max_deg"] - row["degrees"]) < 1e-6, "A 的法向角应等于旋转角"


def test_nn_rematch_residual_is_a_sampling_artifact():
    """把角向网格加密，NN 残差应按比例下降 ⇒ 残余来自离散采样而非几何。"""
    coarse = {r["degrees"]: r for r in run_probe(sections=48)}
    fine = {r["degrees"]: r for r in run_probe(sections=960)}
    for degrees in (2.0, 4.0):
        c, f = coarse[degrees], fine[degrees]
        print("[sampling] %s° nn: coarse(sections=48) normal=%.6f rmse=%.6f | fine(960) normal=%.6f rmse=%.6f"
              % (degrees, c["normal_agreement_max_deg"], c["rmse_mm"],
                 f["normal_agreement_max_deg"], f["rmse_mm"]))
        assert f["normal_agreement_max_deg"] < c["normal_agreement_max_deg"] / 5.0, \
            "加密网格后 NN 法向残差必须显著下降（否则不是采样 artifact）"
