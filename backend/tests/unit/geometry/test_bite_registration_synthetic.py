"""咬合配准：合成 GT 验收（§A）。

每个 case 都必须报：translation error / rotation error / surface RMSE / status。

合成构造（与真实 3D 扫描同构）：
- `upper`：一块带颊侧隆起的曲面（半圆柱壳），代表上颌；
- `lower`：同样形状的副本，按 **已知 GT** 放到咬合位置（所以 lower→upper 的 GT 是已知的）；
- `bite`：两侧扫描**贴在同一张咬合带上**取下的一条壳带 —— 关键点是 bite 用**自己的坐标系**
  （bite frame），它与 upper / lower 的关系用已知的 T_b2u、T_b2l 表示。
  这正是真实 bite scan 的处境（一张扫描，同时贴到上下颌）。

因此 solver 应当：恢复 T_b2u、T_b2l → 组合出 T_lower_to_upper ≈ inv(GT)。
"""

from __future__ import annotations

import numpy as np
import pytest
import trimesh

from app.geometry.bite_registration import (
    BiteRegistrationProfile,
    STATUS_FAILED,
    STATUS_PASS,
    STATUS_REVIEW,
    combine_lower_to_upper,
    register,
    rotation_deg,
    three_point_transform,
    translation_mm,
)


# ------------------------------------------------------------------ 合成夹具
def _rigid(axis, angle_deg, translation):
    """刚体 4×4（列向量右乘）。"""
    axis = np.asarray(axis, float); axis = axis / np.linalg.norm(axis)
    angle = np.radians(angle_deg)
    skew = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    rotation = np.eye(3) + np.sin(angle) * skew + (1 - np.cos(angle)) * (skew @ skew)
    matrix = np.eye(4); matrix[:3, :3] = rotation; matrix[:3, 3] = np.asarray(translation, float)
    return matrix


def _feature_field(theta, z, *, side: int, contact_z: float, band_half: float, seed: int):
    """**band 内**的可观测特征场：两条反向螺旋脊 + 6 个离散牙尖。

    为什么要这样设计（上一版失败的根因）：光滑圆柱片段沿轴向滑动几乎不改变残差，
    于是 ICP 收敛到"RMSE 0.002mm、位姿差 2.5mm/6°"的伪解 —— band 的**可观测性不足**。
    - 螺旋脊：角位置随深度 `dz` 变化 ⇒ **轴向非对称**，同时钉住旋转与轴向平移；
    - 离散牙尖：角度/深度/幅度/宽度各不相同，深度越深幅度越大（再破一次对称）；
    - 全部乘 `inside` 掩码 ⇒ 特征只存在于 band 内，band 外仍是光滑面（不破坏咬合间隙逻辑）。
    """
    rng = np.random.default_rng(seed)
    dz = (z - contact_z) * side                     # 0 = 咬合端，向牙体内部为正
    inside = np.abs(dz) <= band_half
    t = theta / np.pi
    field = np.zeros_like(theta)
    for slope, amp in ((1.0, 0.80), (-0.75, 0.55)):  # 两条反向螺旋脊
        centre = slope * (dz / max(band_half, 1e-6)) * 0.30
        field += amp * np.exp(-((t - centre) / 0.055) ** 2)
    for _ in range(6):                               # 多高度/多角度/多幅度的离散牙尖
        t0 = rng.uniform(-0.45, 0.45)
        d0 = rng.uniform(0.15, 0.95) * band_half
        amp = rng.uniform(0.25, 0.85) * (0.55 + 0.45 * d0 / band_half)
        wt = rng.uniform(0.035, 0.075)
        wz = rng.uniform(0.10, 0.28) * band_half
        field += amp * np.exp(-(((t - t0) / wt) ** 2 + ((dz - d0) / wz) ** 2))
    return field * inside


def _jaw_solid(radius: float = 9.0, length: float = 14.0, sections: int = 64, rings: int = 24,
               seed: int = 1, cusp_amp: float = 0.55, z_offset: float = 0.0,
               feature_side: int = 0, band_half: float = 2.0, feature_seed: int = 11):
    """**闭合实体**牙弓块：闭合圆柱 + **非对称**牙尖样凸起。

    为什么必须这样建（第一版踩到的三件事）：
    - 开放壳 ⇒ 穿模只能给 unsigned 诊断，signed 穿透结论不成立；
    - 光滑圆柱壳 ⇒ 沿面滑动自由，ICP 有孔径问题（实测平移/旋转恢复差 0.27mm/6.5°、3.9°）；
    - 上下颌同形 ⇒ "哪条带贴上颌"几何上真的分不出（8/10 case 直接 BITE_SIDE_AMBIGUOUS）。
    径向位移不改变拓扑，所以实体仍然水密；`seed` 不同 ⇒ 牙尖布局不同 ⇒ 上下颌可分辨。
    """
    rng = np.random.default_rng(seed)
    count = 8
    t0 = rng.uniform(-0.42, 0.42, count)
    z0 = rng.uniform(-0.42, 0.42, count)
    amps = rng.uniform(0.45, 1.0, count) * cusp_amp
    widths = rng.uniform(0.05, 0.11, count)
    theta = np.linspace(-np.pi, np.pi, sections, endpoint=False)
    z = np.linspace(-length / 2, length / 2, rings) + z_offset
    tt, zz = np.meshgrid(theta, z, indexing="ij")
    r = np.full_like(tt, radius)
    for a, b, amp, w in zip(t0, z0, amps, widths):          # 全身低频起伏（保持上下颌不同形）
        r += amp * np.exp(-(((tt / np.pi - a) / w) ** 2 + ((zz / length - b) / (w * 1.4)) ** 2))
    if feature_side:                                        # band 内的可观测特征（按咬合端定位）
        contact_z = z_offset - feature_side * length / 2.0
        r += cusp_amp * _feature_field(tt, zz, side=feature_side, contact_z=contact_z,
                                       band_half=band_half, seed=feature_seed)
    vertices = np.c_[ (r * np.sin(tt)).ravel(), (r * np.cos(tt)).ravel(), zz.ravel() ]
    index = lambda i, j: (i % sections) * rings + (j % rings)
    faces = []
    for i in range(sections):
        for j in range(rings - 1):
            a, b, c, d = index(i, j), index(i, j + 1), index(i + 1, j + 1), index(i + 1, j)
            faces += [[a, b, c], [a, c, d]]
    bottom = len(vertices); vertices = np.vstack([vertices, [0.0, 0.0, -length / 2 + z_offset]])
    top = len(vertices); vertices = np.vstack([vertices, [0.0, 0.0, length / 2 + z_offset]])
    for i in range(sections):
        faces.append([bottom, index(i + 1, 0), index(i, 0)])
        faces.append([top, index(i, rings - 1), index(i + 1, rings - 1)])
    mesh = trimesh.Trimesh(vertices, np.asarray(faces, int), process=True)
    mesh.fix_normals()
    return mesh


def _band(mesh, z_lo, z_hi, buccal_only=True):
    """颊侧一段壳带（模拟 bite scan 只扫到咬合带附近的颊面）——扫描是**开放裁片**，这正是它的处境。"""
    vertices = np.asarray(mesh.vertices, float)
    faces = np.asarray(mesh.faces, int)
    centres = vertices[faces].mean(axis=1)
    keep = (centres[:, 2] >= z_lo) & (centres[:, 2] <= z_hi)
    if buccal_only:
        radius = float(np.percentile(np.linalg.norm(vertices[:, :2], axis=1), 90))
        keep &= centres[:, 1] >= 0.2 * radius
    sub = mesh.submesh([np.flatnonzero(keep)])
    if not sub:
        raise ValueError("band(%.1f, %.1f) 在这块网格上取不到任何面——夹具区间选错了" % (z_lo, z_hi))
    return sub[0]


def _case(*, lower_to_upper_gt, bite_to_upper=np.eye(4), band=(-2.5, 2.5),
          upper_length=7.0, lower_length=7.0, lower_overlap_mm=0.0, mating_gap_mm=0.30):
    """构造 (bite, upper, lower, GT)，GT 的含义由**物理构造**给出而不是由公式反推。

    - `upper`：上颌实体，占 z ∈ [0, L_upper]；
    - `lower_mated`：下颌实体，占 z ∈ [-L_lower, 0] ⇒ **GT=I 就是"已咬合"**（接触于 z=0）；
    - 交给求解器的 `lower = inv(GT) @ lower_mated` —— 所以求解器看到的下颌在**它自己的帧**里，
      它必须把 `T_lu` 恢复成 GT（这正是真实处境：下颌按自己的扫描帧给出）；
    - bite：从两侧**已咬合**的接触带取下，再按 `T_bl = T_bu @ GT` 放回 bite 帧（推导见 combine 文档）。

    `lower_overlap_mm` > 0 时把下颌**做进上颌里**（夹具故意造穿模）。
    """
    # 咬合面固定在 z=0：上颌占 [0, L]、下颌占 [-L, 0] ⇒ identity GT 就是"已咬合、面接触"
    # 已咬合位留 `mating_gap_mm`：否则两侧牙尖在 GT 位置就互相嵌进去，
    # 于是**每个** case 都被穿模门判 GROSS（夹具缺陷，不是真实咬合）。
    upper = _jaw_solid(radius=9.0, length=upper_length, seed=1,
                       z_offset=upper_length / 2.0 + mating_gap_mm / 2.0,
                       feature_side=+1, band_half=2.0, feature_seed=11)
    lower_mated = _jaw_solid(radius=9.4, length=lower_length, seed=2,
                             z_offset=-lower_length / 2.0 - mating_gap_mm / 2.0 + lower_overlap_mm,
                             feature_side=-1, band_half=2.0, feature_seed=23)   # 不同 seed ⇒ 上下颌特征布局不同
    t_b2u = np.asarray(bite_to_upper, float)
    # ⚠️ 物理事实：bite 是**一次刚性扫描**，两侧带都在**同一个世界帧**里被拍到，
    # 所以两侧都用同一个世界→bite 变换 `inv(T_bu)` 放回 bite 帧。
    # 第一版给下颌用了 `inv(T_bl)`（T_bl = T_bu @ GT），那要求 T_bl² = GT 才自洽 ——
    # 于是夹具实际在告诉求解器"这个下颌已经在正确位置了"，实测位姿误差恒等于 GT 本身
    # （2.5059 vs 2.5 / 5.9957 vs 6.0）就是它的症状。
    upper_side = _band(upper, *band).copy()
    upper_side.apply_transform(np.linalg.inv(t_b2u))
    lower_side = _band(lower_mated, *band).copy()
    lower_side.apply_transform(np.linalg.inv(t_b2u))
    # 求解器应恢复出的 T_bl（bite→handed lower，用于文档与断言）：
    #   p_B = inv(T_bu) @ p_mated_lower = inv(T_bu) @ GT @ p_own  ⇒  T_bl = inv(T_bu) @ GT
    t_b2l = np.linalg.inv(t_b2u) @ np.asarray(lower_to_upper_gt, float)
    bite = trimesh.util.concatenate([upper_side, lower_side])
    lower = lower_mated.copy()
    lower.apply_transform(np.linalg.inv(np.asarray(lower_to_upper_gt, float)))
    return bite, upper, lower, np.asarray(lower_to_upper_gt, float)


def _errors(estimate, ground_truth):
    """位姿误差：平移（mm）与旋转（deg），用 ΔT = est @ inv(gt) 度量。"""
    delta = np.asarray(estimate, float) @ np.linalg.inv(np.asarray(ground_truth, float))
    return translation_mm(delta), rotation_deg(delta)


def _surface_rmse(mesh_a, matrix_a, mesh_b, matrix_b, samples=4000):
    """把两个 mesh 各自按自己的位姿摆到公共帧后，双向最近点 RMSE（mm）。"""
    from scipy.spatial import cKDTree
    pts_a = trimesh.transform_points(mesh_a.sample(samples), np.asarray(matrix_a, float))
    pts_b = trimesh.transform_points(mesh_b.sample(samples), np.asarray(matrix_b, float))
    d_ab, _ = cKDTree(pts_b).query(pts_a)
    d_ba, _ = cKDTree(pts_a).query(pts_b)
    return float(np.sqrt(np.mean(np.r_[d_ab, d_ba] ** 2)))


def describe_bite_band(mesh) -> dict:
    """**可观测性诊断**（夹具/debug 用，不是产品功能）。

    目的：防止以后测试又退化成"近对称光滑壳"——那样 ICP 有规范自由度，
    残差极小但位姿错，测试会给出误导性的"通过"。
    """
    vertices = np.asarray(mesh.vertices, float)
    faces = np.asarray(mesh.faces, int)
    centre = vertices.mean(axis=0)
    centred = vertices - centre
    values = np.linalg.eigvalsh(centred.T @ centred / max(len(centred), 1))
    values = np.sort(values)[::-1]
    normals = np.asarray(mesh.face_normals, float)
    spread = np.ptp(vertices, axis=0)
    return {
        "bbox_mm": spread.tolist(),
        "surface_area_mm2": float(np.asarray(mesh.area_faces).sum()),
        "pca_eigen_ratio": float(values[1] / max(values[0], 1e-12)),   # 圆柱/平面 → 接近 1
        "pca_smallest_ratio": float(values[2] / max(values[0], 1e-12)),
        "normal_variation_deg": float(np.degrees(np.arccos(np.clip(
            np.abs(normals @ np.array([0.0, 1.0, 0.0])), -1, 1))).std()),
        "feature_spread_mm": {"x": float(spread[0]), "y": float(spread[1]), "z": float(spread[2]),
                              "radial": float(np.ptp(np.linalg.norm(vertices[:, :2], axis=1)))},
    }


def test_bite_band_is_observable():
    """band 必须**信息充足**：不能是近圆柱光滑壳（否则规范自由度会让残差骗人）。"""
    bite, upper, lower, _ = _case(lower_to_upper_gt=np.eye(4))
    band = _band(upper, -2.5, 2.5)
    info = describe_bite_band(band)
    print("[observability] band", info)
    print("[observability] bite(两侧合计)", describe_bite_band(bite))
    assert info["surface_area_mm2"] > 50.0, info
    assert info["pca_eigen_ratio"] < 0.95, "band 仍接近轴对称（圆柱），ICP 会有规范自由度：%s" % info
    assert info["feature_spread_mm"]["radial"] > 0.15, "band 内径向特征幅度不足：%s" % info
    assert info["feature_spread_mm"]["z"] > 2.0, info
    assert info["normal_variation_deg"] > 5.0, "法向变化太小 ⇒ 近乎光滑壳：%s" % info


def _report(name, result, gt, *, expect):
    te, re = (None, None)
    if result.get("T_lower_to_upper") is not None:
        te, re = _errors(result["T_lower_to_upper"], gt)
    line = ("%s status=%s reason=%s penetration_status=%s translation_err=%s rotation_err=%s upper_rmse=%s lower_rmse=%s"
            % (name, result["status"], result["reason"],
               (result.get("penetration") or {}).get("penetration_status"),
               "n/a" if te is None else "%.4f mm" % te,
               "n/a" if re is None else "%.4f deg" % re,
               result.get("upper_match_rmse_mm"), result.get("lower_match_rmse_mm")))
    print(line)
    assert result["status"] in expect, "%s 状态不符：%s / %s" % (name, result["status"], result["reason"])
    return te, re


# ------------------------------------------------------------------ 1 identity
def test_identity_case():
    gt = np.eye(4)
    bite, upper, lower, gt = _case(lower_to_upper_gt=gt)
    result = register(bite, upper, lower)
    te, re = _report("identity", result, gt, expect=(STATUS_PASS,))
    assert te < 0.05 and re < 0.5, "identity 必须被还原：%.4f mm / %.4f deg" % (te, re)


# ------------------------------------------------------------------ 2 已知平移
def test_known_translation():
    gt = _rigid([0, 0, 1], 0.0, [0, 0, -2.5])
    bite, upper, lower, gt = _case(lower_to_upper_gt=gt, bite_to_upper=_rigid([1, 0, 0], 12, [1.0, -0.5, 0.3]))
    result = register(bite, upper, lower)
    te, re = _report("translation", result, gt, expect=(STATUS_PASS,))
    assert te < 0.10, "纯平移必须还原到 0.1mm 内：%.4f" % te


# ------------------------------------------------------------------ 3 已知旋转
def test_known_rotation():
    gt = _rigid([1, 0, 0], 6.0, [0, 0, 0])
    bite, upper, lower, gt = _case(lower_to_upper_gt=gt, bite_to_upper=_rigid([0, 1, 0], -9, [0.4, 0.2, -0.6]))
    result = register(bite, upper, lower)
    te, re = _report("rotation", result, gt, expect=(STATUS_PASS,))
    assert re < 1.0, "纯旋转必须还原到 1° 内：%.4f" % re


# ------------------------------------------------------------------ 4 一般刚体
def test_known_rigid_transform():
    gt = _rigid([0.3, 0.7, -0.4], 7.5, [0.8, -1.2, -2.0])
    bite, upper, lower, gt = _case(lower_to_upper_gt=gt, bite_to_upper=_rigid([1, 1, 0], 15, [-1.5, 0.9, 1.1]))
    result = register(bite, upper, lower)
    te, re = _report("rigid", result, gt, expect=(STATUS_PASS,))
    assert te < 0.15 and re < 1.5, "刚体必须还原：%.4f mm / %.4f deg" % (te, re)


# ------------------------------------------------------------------ 5 局部扫描
def test_partial_bite_scan():
    gt = _rigid([0, 0, 1], 4.0, [0.5, 0, -1.5])
    bite, upper, lower, gt = _case(lower_to_upper_gt=gt, bite_to_upper=_rigid([1, 0, 0], 8, [0.6, 0.4, 0]), band=(-1.2, 1.2))
    result = register(bite, upper, lower)
    te, re = _report("partial", result, gt, expect=(STATUS_PASS, STATUS_REVIEW))
    if result["status"] == STATUS_PASS:
        assert te < 0.25 and re < 2.0


# ------------------------------------------------------------------ 6 噪声
def test_noisy_scan():
    gt = _rigid([0.2, -0.5, 0.8], 5.0, [0.4, 0.6, -1.8])
    bite, upper, lower, gt = _case(lower_to_upper_gt=gt, bite_to_upper=_rigid([0, 1, 0], 10, [0.3, -0.4, 0.5]))
    rng = np.random.default_rng(7)
    bite = bite.copy()
    bite.vertices = np.asarray(bite.vertices) + rng.normal(0, 0.04, np.asarray(bite.vertices).shape)
    result = register(bite, upper, lower)
    te, re = _report("noise", result, gt, expect=(STATUS_PASS, STATUS_REVIEW))
    if result["status"] == STATUS_PASS:
        assert te < 0.30


# ------------------------------------------------------------------ 7 局外点
def test_outliers_do_not_break_registration():
    gt = _rigid([0, 1, 0], 3.5, [0.2, -0.3, -1.6])
    bite, upper, lower, gt = _case(lower_to_upper_gt=gt, bite_to_upper=_rigid([1, 0, 1], 6, [0.5, 0.5, -0.2]))
    rng = np.random.default_rng(11)
    spikes = trimesh.creation.icosphere(subdivisions=2, radius=1.2)
    spikes.apply_translation([0, 0, 26])            # 明显离群的孤岛
    bite = trimesh.util.concatenate([bite, spikes])
    result = register(bite, upper, lower)
    te, re = _report("outlier", result, gt, expect=(STATUS_PASS, STATUS_REVIEW))
    if result["status"] == STATUS_PASS:
        assert te < 0.30, "局外点不得把结果拽走：%.4f" % te


# ------------------------------------------------------------------ 8 歧义侧
def test_ambiguous_bite_sides_are_reported_not_guessed():
    """两侧形状/位置对称 ⇒ 无法区分谁贴上颌，必须 REVIEW_REQUIRED / BITE_SIDE_AMBIGUOUS。"""
    bite, upper, lower, _ = _case(lower_to_upper_gt=np.eye(4))
    band = _band(upper, -2.5, 2.5)
    bite = trimesh.util.concatenate([band.copy(), band.copy()])   # 两份一模一样 ⇒ 刻意歧义
    result = register(bite, upper, lower)
    _report("ambiguous", result, np.eye(4), expect=(STATUS_REVIEW, STATUS_FAILED))
    assert result["reason"] == "BITE_SIDE_AMBIGUOUS" or result["status"] == STATUS_FAILED
    assert result["T_lower_to_upper"] is None or result["status"] != STATUS_PASS


# ------------------------------------------------------------------ 9 错配
def test_wrong_alignment_is_failed_or_review():
    """给一个形状完全不同/位置离谱的对颌 ⇒ 不得 PASS。"""
    # 错配：夹具本身自洽，但**把 bite 整体挪到一个与两侧都不符的位置**（模拟咬合扫描不可信）
    bite, upper, lower, _ = _case(lower_to_upper_gt=np.eye(4))
    bite = bite.copy()
    bite.apply_transform(_rigid([0.3, 0.6, 0.7], 40.0, [15.0, -8.0, 6.0]))
    result = register(bite, upper, lower)
    _report("wrong_alignment", result, np.eye(4), expect=(STATUS_FAILED, STATUS_REVIEW))
    if result["status"] == STATUS_PASS:
        pytest.fail("错误配准不得判 PASS")


# ------------------------------------------------------------------ 10 大面积穿模
def test_gross_jaw_penetration_is_gated():
    """上下颌被解成大面积互穿 ⇒ FAILED / GROSS_JAW_INTERSECTION，不得进入下游。"""
    profile = BiteRegistrationProfile(gross_penetration_max_mm=0.2, gross_penetration_area_mm2=5.0,
                                      gross_penetration_ratio=0.02)
    # 夹具**故意**把下颌做进上颌 6mm：bite 与它自洽，于是求解器恢复出的位姿本身就是穿透位姿
    bite, upper, lower, _ = _case(lower_to_upper_gt=np.eye(4), lower_overlap_mm=6.0)
    result = register(bite, upper, lower, profile)
    _report("penetration", result, np.eye(4), expect=(STATUS_FAILED,))
    assert result["reason"] == "GROSS_JAW_INTERSECTION", result["reason"]
    assert result["gross_penetration_max_mm"] > 0.2


# ------------------------------------------------------------------ 单位测试：坐标约定
def test_matrix_convention_is_pinned():
    """组合位姿的左右乘顺序必须被 GT 还原（不要照抄顺序）。"""
    t_b2u = _rigid([0, 0, 1], 10.0, [1.0, 2.0, 3.0])
    lower_to_upper = _rigid([1, 0, 0], -5.0, [0.5, -0.25, 1.5])
    # 由物理关系 T_bl = T_bu @ T_lu（不是由 combine 的公式反推）
    t_b2l = t_b2u @ lower_to_upper
    combined = combine_lower_to_upper(t_b2u, t_b2l)
    te, re = _errors(combined, lower_to_upper)
    assert te < 1e-9 and re < 1e-9, "组合约定错误：%.3e mm / %.3e deg" % (te, re)


def test_three_point_assist_recovers_known_transform():
    """三点辅助：由 3 对对应点求出的刚体必须还原已知 GT。"""
    rng = np.random.default_rng(3)
    source = rng.normal(0, 5, (3, 3))
    gt = _rigid([0.2, 0.5, 0.8], 12.0, [1.5, -2.0, 0.75])
    target = trimesh.transform_points(source, gt)
    estimated = three_point_transform(source, target)
    te, re = _errors(estimated, gt)
    assert te < 1e-6 and re < 1e-6


def test_missing_bite_scan_is_failed_not_guessed():
    """无 bite scan ⇒ BITE_SCAN_MISSING，不许猜（§22）。"""
    bite, upper, lower, _ = _case(lower_to_upper_gt=np.eye(4))
    result = register(None, upper, lower)
    assert result["status"] == STATUS_FAILED
    assert result["reason"] == "BITE_SCAN_MISSING"
    assert result["T_lower_to_upper"] is None


def test_original_meshes_are_not_modified():
    """预处理不得改动输入网格（原始字节由调用方校验）。"""
    gt = _rigid([0, 0, 1], 2.0, [0, 0, -1.0])
    bite, upper, lower, _ = _case(lower_to_upper_gt=gt)
    before = {name: np.asarray(mesh.vertices).copy() for name, mesh in
              (("bite", bite), ("upper", upper), ("lower", lower))}
    register(bite, upper, lower)
    for name, mesh in (("bite", bite), ("upper", upper), ("lower", lower)):
        assert np.allclose(before[name], np.asarray(mesh.vertices)), "%s 被改动了" % name
