"""共享夹具：WELL_POSED_CANONICAL 与 DEGENERATE_AXIAL_SYMMETRY 两个世界。

为什么必须拆（实测依据）：
- 均匀圆柱 band 沿轴滑动是**精确对称**（同 RMSE / 同 inlier）⇒ 唯一恢复 GT pose 的要求不成立；
- 因此"唯一恢复 GT"的测试（identity/translation/rotation/rigid/三帧/large_init/solver-output/equivariance）
  必须跑在**打破轴向与旋转对称**的 well-posed 世界上；
- 退化世界只用于检测"弱约束/多解必须判 REVIEW"。

WELL_POSED 的对称破坏手段（全部沿轴变化、非周期）：
1. 轴向锥度 taper（band 内半径随轴向位置变化）；
2. 单侧非周期 groove，**深度沿轴变化**；
3. 3 个 cusp，轴向位置/角度/幅度互不相同。
`assert_well_posed()` 会主动验证：沿源主轴 ±0.5/1/2/3 mm、绕主轴 0.5–4° 的扰动**必须**明显降低匹配质量，
否则夹具不合格（防止以后又退回近对称光滑壳）。
"""

from __future__ import annotations

import numpy as np
import trimesh


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


def axis_rotation(axis, angle):
    return _rigid(axis, np.degrees(angle), [0, 0, 0])[:3, :3]


def jaw_solid(*, radius=9.0, length=7.0, sections=40, rings=20, seed=1, z_center=0.0,
              feature_side=+1, band_half=2.0, feature_seed=11,
              taper=0.0, groove_amp=0.0, cusp_amp=0.0, ripple_amp=0.30):
    """闭合实体牙弓块（水密 ⇒ signed 穿透才有权威性）。

    `taper/groove_amp/cusp_amp` 全为 0 时就是**均匀圆柱**（退化世界）；给正值即 WELL_POSED。
    """
    rng = np.random.default_rng(feature_seed)
    theta = np.linspace(-np.pi, np.pi, sections, endpoint=False)
    z = np.linspace(-length / 2, length / 2, rings) + z_center
    tt, zz = np.meshgrid(theta, z, indexing="ij")
    z_contact = z_center - feature_side * length / 2.0
    dz = (zz - z_contact) * feature_side                     # 0 = 咬合端，向牙体内部为正
    # ⚠️ 不能 clip 到 band_half：那样 dz>band_half 的牙体变成**完全均匀圆柱**，
    # 于是"沿轴滑到那段"的位姿能拿到 rmse=0/inlier=1（实测 10.46°/6.36mm 的错位姿却是完美证据）。
    # 让特征沿**全长**变化（taper/groove 深度都不饱和）⇒ 不存在"几何相同的另一段"。
    band = np.clip(dz, 0.0, length) / max(length, 1e-9)

    r = np.full_like(tt, radius)
    if taper:
        r += taper * band                                    # 轴向锥度（沿轴变化）
    if groove_amp:
        groove_theta = rng.uniform(-0.30, 0.30)
        r -= groove_amp * np.exp(-(((tt / np.pi) - groove_theta) / 0.18) ** 2) * (0.35 + 0.65 * band)
        # 第二条 groove：**不同方位 + 不同轴向变化率** ⇒ 打破"绕主轴旋转"的弱约束（实测 Rz 曾=0.7）
        groove2_theta = groove_theta + rng.uniform(0.55, 1.35)
        r -= 0.45 * groove_amp * np.exp(-(((tt / np.pi) - groove2_theta) / 0.10) ** 2) * (0.80 - 0.55 * band)
        # 窄而深的方位特征：点面距离/inlier 对**切向滑动**本来不敏感（滑动仍贴面），
        # 只有"窄特征"能把几百微米的切向位移变成可测的表面距离 ⇒ Rz 才可观测。
        for extra_theta, extra_depth, extra_width in ((groove_theta - 0.45, 0.85, 0.045),
                                                      (groove2_theta + 0.50, 0.70, 0.040)):
            r -= extra_depth * groove_amp * np.exp(-(((tt / np.pi) - extra_theta) / extra_width) ** 2)
    if cusp_amp:
        for _ in range(3):
            t0 = rng.uniform(-0.5, 0.5)
            d0 = rng.uniform(0.10, 0.90) * band_half
            amp = cusp_amp * rng.uniform(0.35, 0.90)
            wt = rng.uniform(0.05, 0.10)
            wz = rng.uniform(0.08, 0.20) * band_half
            r += amp * np.exp(-(((tt / np.pi - t0) / wt) ** 2 + ((dz - d0) / wz) ** 2))
    # band 外的低频起伏：只用于区分上下颌，不参与可观测性
    outer_seed = np.random.default_rng(seed)
    for a, b, amp, w in zip(outer_seed.uniform(-0.42, 0.42, 4), outer_seed.uniform(-0.42, 0.42, 4),
                            ripple_amp * outer_seed.uniform(0.67, 1.5, 4), outer_seed.uniform(0.06, 0.12, 4)):
        r += amp * np.exp(-(((tt / np.pi - a) / w) ** 2 + ((dz - length / 2) / length / (w * 1.4)) ** 2))

    vertices = np.c_[(r * np.sin(tt)).ravel(), (r * np.cos(tt)).ravel(), zz.ravel()]
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
    """颊侧一段壳带（**开放裁片**，与真实 bite scan 的处境一致）。"""
    vertices = np.asarray(mesh.vertices, float)
    faces = np.asarray(mesh.faces, int)
    centres = vertices[faces].mean(axis=1)
    radius = float(np.percentile(np.linalg.norm(vertices[:, :2], axis=1), 90))
    keep = (centres[:, 2] >= z_lo) & (centres[:, 2] <= z_hi) & (centres[:, 1] >= 0.2 * radius)
    sub = mesh.submesh([np.flatnonzero(keep)])
    if not sub:
        raise ValueError("band(%.1f, %.1f) 取不到面：夹具区间错了" % (z_lo, z_hi))
    return sub[0]


def moved(mesh, matrix):
    out = mesh.copy()
    out.apply_transform(np.asarray(matrix, float))
    return out


WELL_POSED_KW = dict(taper=0.90, groove_amp=0.55, cusp_amp=0.62)
DEGENERATE_KW = dict(taper=0.0, groove_amp=0.0, cusp_amp=0.0, ripple_amp=0.0)  # 真·均匀圆柱（带内无任何起伏）


def canonical_world(*, gap=0.30, band=(-2.5, 2.5), upper_length=7.0, lower_length=7.0,
                    lower_overlap_mm=0.0, upper_seed=1, lower_seed=2, well_posed=True):
    """World 里正确咬合的两块牙弓 + 从 World 裁下的两条 bite patch。"""
    kw = WELL_POSED_KW if well_posed else DEGENERATE_KW
    upper = jaw_solid(radius=9.0, length=upper_length, seed=upper_seed,
                      z_center=gap / 2.0 + upper_length / 2.0, feature_side=+1, feature_seed=11, **kw)
    lower = jaw_solid(radius=9.4, length=lower_length, seed=lower_seed,
                      z_center=-gap / 2.0 - lower_length / 2.0 + lower_overlap_mm,
                      feature_side=-1, feature_seed=23, **kw)
    return {"upper_world": upper, "lower_world": lower,
            "upper_patch_world": band_of(upper, *band), "lower_patch_world": band_of(lower, *band)}


def scanner_inputs(world, *, A_upper=np.eye(4), A_lower=np.eye(4), A_bite=np.eye(4), with_bite=True):
    """按三个**独立**帧生成输入与 GT（GT 由帧定义给出，不由被测公式反推）。"""
    A_upper, A_lower, A_bite = (np.asarray(A, float) for A in (A_upper, A_lower, A_bite))
    inputs = {"upper": moved(world["upper_world"], A_upper), "lower": moved(world["lower_world"], A_lower),
              "bite": None}
    if with_bite:
        inputs["bite"] = trimesh.util.concatenate([moved(world["upper_patch_world"], A_bite),
                                                   moved(world["lower_patch_world"], A_bite)])
    gts = {"T_upper_to_bite": A_bite @ np.linalg.inv(A_upper),
           "T_lower_to_bite": A_bite @ np.linalg.inv(A_lower),
           "T_lower_to_upper": A_upper @ np.linalg.inv(A_lower)}
    return inputs, gts


# ------------------------------------------------------------------ 夹具自校验
ABS_RMSE_FLOOR_MM = 0.05   # ENGINEERING_DEFAULT_UNVALIDATED
ABS_INLIER_FLOOR = 0.01  # ENGINEERING_DEFAULT_UNVALIDATED
ABS_NORMAL_ANGLE_FLOOR_DEG = 0.5  # ENGINEERING_DEFAULT_UNVALIDATED（deg/deg）


def quality_sensitivity(patch, jaw, matrix, profile, spin_axis=None):
    """在局部主轴系里扰动 6DoF：返回每个自由度的**最大相对质量退化**（<1 视为弱约束）。"""
    from app.geometry.bite_registration import _pair_quality, _sample
    centres, _, _ = _sample(patch, profile.sample_points, profile)
    centre = centres.mean(axis=0)
    values, axes = np.linalg.eigh((centres - centre).T @ (centres - centre) / len(centres))
    axes = axes[:, np.argsort(values)[::-1]]
    if np.linalg.det(axes) < 0:
        axes[:, 2] *= -1.0
    base = _pair_quality(patch, jaw, matrix, profile)
    base_inlier, base_rmse = base.get("inlier_ratio") or 0.0, base.get("rmse_mm") or 1e-9

    base_normal_deg = base.get("normal_agreement_max_deg") or 0.0

    def delta(quality):
        # 绝对变化口径（base≈0 时比值会炸成 1e16，实测踩到）+ 法向一致角证据
        drop = base_inlier - (quality.get("inlier_ratio") or 0.0)
        rmse = quality.get("rmse_mm")
        rise_mm = (rmse - base_rmse) if rmse is not None else 0.0
        normal_deg = quality.get("normal_agreement_max_deg") or 0.0
        return max(drop / ABS_INLIER_FLOOR, rise_mm / ABS_RMSE_FLOOR_MM,
                   (normal_deg - base_normal_deg) / ABS_NORMAL_ANGLE_FLOOR_DEG)

    out = {}
    for index in range(3):
        worst = 0.0
        for step in (0.5, 1.0, 2.0, 3.0):
            for sign in (1.0, -1.0):
                shift = np.eye(4); shift[:3, 3] = axes[:, index] * step * sign
                worst = max(worst, delta(_pair_quality(patch, jaw, shift @ np.asarray(matrix, float), profile)))
        out["T%s" % "xyz"[index]] = worst
    for index in range(3):
        worst = 0.0
        for deg in (0.5, 1.0, 2.0, 4.0):
            for sign in (1.0, -1.0):
                spin = np.eye(4); spin[:3, :3] = axis_rotation(axes[:, index], np.radians(deg * sign))
                worst = max(worst, delta(_pair_quality(patch, jaw, spin @ np.asarray(matrix, float), profile)))
        out["R%s" % "xyz"[index]] = worst
    if spin_axis is not None:
        # 显式探针：绕**给定轴**（例如 jaw 自身的对称轴）旋转。patch 的主轴 ≠ 圆柱轴，
        # 所以上面的 Rx/Ry/Rz 并不等于"绕就位轴自转"——这个探针才是。
        worst = 0.0
        for deg in (0.5, 1.0, 2.0, 4.0):
            for sign in (1.0, -1.0):
                spin = np.eye(4); spin[:3, :3] = axis_rotation(np.asarray(spin_axis, float), np.radians(deg * sign))
                worst = max(worst, delta(_pair_quality(patch, jaw, spin @ np.asarray(matrix, float), profile)))
        out["R_spin"] = worst
    return out


def symmetry_axis(mesh):
    # 取自 mesh 自身协方差的最大特征向量（均匀圆柱 ⇒ 对称轴；patch 上不成立）
    v = np.asarray(mesh.vertices, float); c = v.mean(axis=0)
    values, axes = np.linalg.eigh((v - c).T @ (v - c) / len(v))
    axis = axes[:, int(np.argmax(values))]
    return axis / np.linalg.norm(axis)
