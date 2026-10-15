"""Canonical World 夹具 + **冻结的帧约定/刚性契约**独立测试。

与旧文件的分工（按规格"重建 Canonical World fixture，不修旧 fixture"）：
- 本文件是**新的、干净的**夹具与契约测试；
- 旧的 `test_bite_registration_synthetic.py` 保持原样（其中的失败已被归类为 LEGACY_FIXTURE，
  等 12 类 case 在新夹具上重建后一起处理）。

World 构造（GT 由**独立帧**给出，不再用被测公式反推）：

    Upper_world / Lower_world          —— 在 World 里就是正确咬合位姿
    A_upper / A_lower / A_bite         —— 三个**独立**的 world→scanner 刚体变换
    Upper_input = A_upper @ Upper_world
    Lower_input = A_lower @ Lower_world
    Bite_input  = A_bite @ Upper_bite_patch_world  ∪  A_bite @ Lower_bite_patch_world

    GT_T_upper_to_bite = A_bite @ inv(A_upper)
    GT_T_lower_to_bite = A_bite @ inv(A_lower)
    GT_T_lower_to_upper = A_upper @ inv(A_lower)
"""

from __future__ import annotations

import numpy as np
import pytest
import trimesh

from app.geometry.bite_registration import (
    joint_consistency_qc,
    lower_to_upper_from_bite,
    register,
    rigid_transform_report,
)


# ============================================================ Canonical World 夹具
def _rigid(axis, angle_deg, translation):
    axis = np.asarray(axis, float)
    axis = axis / np.linalg.norm(axis)
    angle = np.radians(angle_deg)
    skew = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    rotation = np.eye(3) + np.sin(angle) * skew + (1 - np.cos(angle)) * (skew @ skew)
    matrix = np.eye(4)
    matrix[:3, :3] = rotation
    matrix[:3, 3] = np.asarray(translation, float)
    return matrix


def _feature_field(theta, z, *, side, contact_z, band_half, seed):
    """band 内可观测特征：两条反向螺旋脊（轴向非对称）+ 6 个不同高度/角度/幅度的牙尖。"""
    rng = np.random.default_rng(seed)
    dz = (z - contact_z) * side
    inside = np.abs(dz) <= band_half
    t = theta / np.pi
    field = np.zeros_like(theta)
    for slope, amp in ((1.0, 0.80), (-0.75, 0.55)):
        field += amp * np.exp(-((t - slope * (dz / max(band_half, 1e-6)) * 0.30) / 0.055) ** 2)
    for _ in range(6):
        t0 = rng.uniform(-0.45, 0.45)
        d0 = rng.uniform(0.15, 0.95) * band_half
        amp = rng.uniform(0.25, 0.85) * (0.55 + 0.45 * d0 / band_half)
        field += amp * np.exp(-(((t - t0) / rng.uniform(0.035, 0.075)) ** 2
                                + ((dz - d0) / (rng.uniform(0.10, 0.28) * band_half)) ** 2))
    return field * inside


def jaw_solid(*, radius=9.0, length=7.0, sections=64, rings=24, seed=1, z_center=0.0,
              feature_side=+1, band_half=2.0, feature_seed=11, cusp=0.55):
    """闭合实体牙弓块（可水密 ⇒ signed 穿透才有权威性），band 内带可观测特征。"""
    rng = np.random.default_rng(seed)
    theta = np.linspace(-np.pi, np.pi, sections, endpoint=False)
    z = np.linspace(-length / 2, length / 2, rings) + z_center
    tt, zz = np.meshgrid(theta, z, indexing="ij")
    r = np.full_like(tt, radius)
    for a, b, amp, w in zip(rng.uniform(-0.42, 0.42, 8), rng.uniform(-0.42, 0.42, 8),
                            rng.uniform(0.45, 1.0, 8) * cusp, rng.uniform(0.05, 0.11, 8)):
        r += amp * np.exp(-(((tt / np.pi - a) / w) ** 2 + ((zz - z_center) / length - b) ** 2 / (w * 1.4) ** 2))
    contact_z = z_center - feature_side * length / 2.0
    r += cusp * _feature_field(tt, zz, side=feature_side, contact_z=contact_z,
                               band_half=band_half, seed=feature_seed)
    vertices = np.c_[ (r * np.sin(tt)).ravel(), (r * np.cos(tt)).ravel(), zz.ravel() ]
    index = lambda i, j: (i % sections) * rings + (j % rings)
    faces = []
    for i in range(sections):
        for j in range(rings - 1):
            a, b, c, d = index(i, j), index(i, j + 1), index(i + 1, j + 1), index(i + 1, j)
            faces += [[a, b, c], [a, c, d]]
    bottom = len(vertices); vertices = np.vstack([vertices, [0.0, 0.0, z[0]]])
    top = len(vertices); vertices = np.vstack([vertices, [0.0, 0.0, z[-1]]])
    for i in range(sections):
        faces.append([bottom, index(i + 1, 0), index(i, 0)])
        faces.append([top, index(i, rings - 1), index(i + 1, rings - 1)])
    mesh = trimesh.Trimesh(vertices, np.asarray(faces, int), process=True)
    mesh.fix_normals()
    return mesh


def band_of(mesh, z_lo, z_hi):
    """颊侧一段壳带（**开放裁片**，模拟 bite scan 的处境）。"""
    vertices = np.asarray(mesh.vertices, float)
    faces = np.asarray(mesh.faces, int)
    centres = vertices[faces].mean(axis=1)
    radius = float(np.percentile(np.linalg.norm(vertices[:, :2], axis=1), 90))
    keep = (centres[:, 2] >= z_lo) & (centres[:, 2] <= z_hi) & (centres[:, 1] >= 0.2 * radius)
    sub = mesh.submesh([np.flatnonzero(keep)])
    if not sub:
        raise ValueError("band(%.1f, %.1f) 取不到面：夹具区间错了" % (z_lo, z_hi))
    return sub[0]


def canonical_world(*, gap=0.30, band=(-2.5, 2.5), upper_length=7.0, lower_length=7.0,
                    lower_overlap_mm=0.0, upper_seed=1, lower_seed=2):
    """World 里正确咬合的两块牙弓 + 从 World 裁下的两条 bite patch。"""
    upper_world = jaw_solid(radius=9.0, length=upper_length, seed=upper_seed,
                            z_center=gap / 2.0 + upper_length / 2.0, feature_side=+1, feature_seed=11)
    lower_world = jaw_solid(radius=9.4, length=lower_length, seed=lower_seed,
                            z_center=-gap / 2.0 - lower_length / 2.0 + lower_overlap_mm,
                            feature_side=-1, feature_seed=23)
    return {"upper_world": upper_world, "lower_world": lower_world,
            "upper_patch_world": band_of(upper_world, *band),
            "lower_patch_world": band_of(lower_world, *band)}


def scanner_inputs(world, *, A_upper=np.eye(4), A_lower=np.eye(4), A_bite=np.eye(4),
                   with_bite=True):
    """按三个**独立**帧生成输入与 GT（GT 由帧定义给出，不由被测公式反推）。"""
    def moved(mesh, A):
        out = mesh.copy()
        out.apply_transform(np.asarray(A, float))
        return out

    A_upper, A_lower, A_bite = (np.asarray(A, float) for A in (A_upper, A_lower, A_bite))
    inputs = {"upper": moved(world["upper_world"], A_upper),
              "lower": moved(world["lower_world"], A_lower)}
    if with_bite:
        inputs["bite"] = trimesh.util.concatenate([moved(world["upper_patch_world"], A_bite),
                                                   moved(world["lower_patch_world"], A_bite)])
    else:
        inputs["bite"] = None
    gts = {"T_upper_to_bite": A_bite @ np.linalg.inv(A_upper),
           "T_lower_to_bite": A_bite @ np.linalg.inv(A_lower),
           "T_lower_to_upper": A_upper @ np.linalg.inv(A_lower)}
    return inputs, gts


def describe_band(mesh):
    """可观测性诊断（防夹具退化成近对称光滑壳）。"""
    v = np.asarray(mesh.vertices, float)
    centred = v - v.mean(axis=0)
    values = np.sort(np.linalg.eigvalsh(centred.T @ centred / max(len(v), 1)))[::-1]
    normals = np.asarray(mesh.face_normals, float)
    return {"pca_eigen_ratio": float(values[1] / max(values[0], 1e-12)),
            "surface_area_mm2": float(np.asarray(mesh.area_faces).sum()),
            "normal_variation_deg": float(np.degrees(np.arccos(np.clip(
                np.abs(normals @ np.array([0.0, 1.0, 0.0])), -1, 1))).std()),
            "z_extent_mm": float(np.ptp(v[:, 2]))}


# ============================================================ ① 帧约定（冻结）
def test_frame_convention_matrix_equality():
    world = canonical_world()
    _, gts = scanner_inputs(world, A_upper=_rigid([1, 0, 0], 7, [1, -2, 3]),
                            A_lower=_rigid([0.3, 0.8, -0.2], -11, [-4, 1, 2]),
                            A_bite=_rigid([0.2, -0.6, 0.7], 23, [6, -3, 5]))
    combined = lower_to_upper_from_bite(gts["T_upper_to_bite"], gts["T_lower_to_bite"])
    assert np.allclose(combined, gts["T_lower_to_upper"], atol=1e-9), \
        "矩阵层不成立：inv(T_upper_to_bite) @ T_lower_to_bite != T_lower_to_upper"


def test_frame_convention_point_path_equality():
    """point-level 路径一致性：世界→upper→bite 与 世界→bite 必须同点。"""
    world = canonical_world()
    A_upper = _rigid([1, 1, 0], 14, [2, -1, 4])
    A_lower = _rigid([0, 1, 0.4], -9, [-3, 2, 1])
    A_bite = _rigid([0.5, 0.2, -0.9], 31, [7, 3, -2])
    inputs, gts = scanner_inputs(world, A_upper=A_upper, A_lower=A_lower, A_bite=A_bite)

    p_world = trimesh.transform_points(np.asarray(world["upper_world"].vertices, float)[:256], np.eye(4))
    p_upper = trimesh.transform_points(p_world, A_upper)                       # upper scanner 坐标
    p_bite_direct = trimesh.transform_points(p_world, A_bite)                  # 世界→bite 直达
    p_bite_via = trimesh.transform_points(p_upper, gts["T_upper_to_bite"])     # 世界→upper→bite
    assert np.allclose(p_bite_via, p_bite_direct, atol=1e-9), "路径一致性不成立（upper 侧）"

    p_lower = trimesh.transform_points(p_world, A_lower)
    p_bite_via_lower = trimesh.transform_points(p_lower, gts["T_lower_to_bite"])
    assert np.allclose(p_bite_via_lower, p_bite_direct, atol=1e-9), "路径一致性不成立（lower 侧）"


def test_frame_convention_round_trip():
    world = canonical_world()
    _, gts = scanner_inputs(world, A_upper=_rigid([0, 0, 1], 5, [1, 1, 1]),
                            A_lower=_rigid([1, 0, 0], -6, [-1, 0, 2]),
                            A_bite=_rigid([0.1, 0.9, 0.3], 40, [15, -8, 6]))
    points = np.asarray(world["upper_patch_world"].vertices, float)
    bite = trimesh.transform_points(points, gts["T_upper_to_bite"])
    back = trimesh.transform_points(bite, np.linalg.inv(gts["T_upper_to_bite"]))
    assert np.allclose(back, points, atol=1e-9), "往返不闭合"


def test_frame_convention_lower_to_upper_point_test():
    """lower→upper 必须在**点**层面成立（不只看矩阵）。"""
    world = canonical_world(lower_overlap_mm=0.0)
    A_upper = _rigid([0.2, 0.3, 0.9], 8, [2, 3, -4])
    A_lower = _rigid([0.7, -0.1, 0.2], -13, [-5, 1, 6])
    inputs, gts = scanner_inputs(world, A_upper=A_upper, A_lower=A_lower, A_bite=np.eye(4))
    p_lower = np.asarray(inputs["lower"].vertices, float)[:256]
    via = trimesh.transform_points(p_lower, gts["T_lower_to_upper"])            # 矩阵路径
    direct = trimesh.transform_points(
        trimesh.transform_points(p_lower, np.linalg.inv(A_lower)), A_upper)     # 世界中转直算
    assert np.allclose(via, direct, atol=1e-9), "lower→upper 点层面不一致"


# ============================================================ ② 刚性契约
def test_rigid_contract_positive():
    k_in, k_out = np.array([[0.0, 0, 0], [3, 0, 0], [0, 4, 0], [0, 0, 5]]), None
    k_out = trimesh.transform_points(k_in, _rigid([0.3, 0.5, 0.8], 21, [1, 2, 3]))
    cases = {
        "identity": np.eye(4),
        "translation": _rigid([0, 0, 1], 0, [1, 2, 3]),
        "rotation": _rigid([0.2, 0.4, 1.0], 33, [0, 0, 0]),
        "rigid": _rigid([0.5, -0.2, 0.7], 17, [-2, 5, 1]),
        "kabsch": None,
    }
    from app.geometry.bite_registration import three_point_transform
    cases["kabsch"] = three_point_transform(k_in, k_out)
    for name, matrix in cases.items():
        report = rigid_transform_report(matrix)
        assert report["rigid_pose_valid"], "%s 被误判为非刚性：%s" % (name, report)
        assert report["no_scale_no_shear"], name


def test_rigid_contract_negative():
    scale = np.eye(4); scale[:3, :3] = np.diag([1.0, 1.05, 1.0])
    shear = np.eye(4); shear[0, 1] = 0.1
    reflection = np.eye(4); reflection[0, 0] = -1.0          # det = -1（镜像）
    bad_bottom = np.eye(4); bad_bottom[3, 3] = 0.999
    for name, matrix in (("scale", scale), ("shear", shear),
                         ("reflection", reflection), ("bad_bottom_row", bad_bottom)):
        report = rigid_transform_report(matrix)
        assert not report["rigid_pose_valid"], "%s 被误判为刚性：%s" % (name, report)


# ============================================================ Bite Frame 一致性契约
def test_no_cross_surface_rmse_in_result_or_joint_qc():
    """结果与 JointConsistencyQC 都不得含跨物理表面的 RMSE。"""
    world = canonical_world()
    inputs, _ = scanner_inputs(world, A_upper=_rigid([0, 0, 1], 3, [1, 0, 0]))
    result = register(inputs["bite"], inputs["upper"], inputs["lower"])
    assert "jaw_pose_point_rmse_mm" not in result, "live result 不得带 GT 型指标"
    joint = result.get("joint_consistency") or {}
    assert joint.get("no_cross_surface_rmse") is True or joint.get("status") == "NOT_ASSESSED", joint
    from app.geometry.bite_registration import BiteRegistrationProfile
    qc = joint_consistency_qc({}, {}, collision={}, penetration={},
                              rigid=rigid_transform_report(np.eye(4)),
                              profile=BiteRegistrationProfile())
    assert "no_cross_surface_rmse" in qc


def test_canonical_world_is_observable():
    """新夹具的 band 必须信息充足（防退化成近对称光滑壳）。"""
    world = canonical_world()
    info = describe_band(world["upper_patch_world"])
    print("[canonical band]", info)
    assert info["surface_area_mm2"] > 50.0 and info["pca_eigen_ratio"] < 0.95
    assert info["normal_variation_deg"] > 5.0 and info["z_extent_mm"] > 2.0


def test_original_inputs_unchanged_by_register():
    world = canonical_world()
    inputs, _ = scanner_inputs(world, A_upper=_rigid([0, 1, 0], 4, [0, 0, 1]))
    before = {k: np.asarray(v.vertices, float).copy() for k, v in inputs.items() if v is not None}
    register(inputs["bite"], inputs["upper"], inputs["lower"])
    for key, mesh in inputs.items():
        if mesh is None:
            continue
        assert np.allclose(before[key], np.asarray(mesh.vertices, float)), "%s 被改动" % key


def test_missing_bite_scan_contract():
    world = canonical_world()
    inputs, _ = scanner_inputs(world, with_bite=False)
    result = register(None, inputs["upper"], inputs["lower"])
    assert result["status"] == "FAILED" and result["reason"] == "BITE_SCAN_MISSING"


if __name__ == "__main__":
    pytest.main([__file__, "-q", "-s"])
