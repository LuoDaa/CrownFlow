"""咬合扫描配准（biteregister ↔ upper/lower 分侧刚体配准）。

**这一层只解决一件事**：给定 bite scan 与上下颌，求出 bite 到两侧的刚体位姿，
从而组合出上下颌相对位姿（`T_lower_to_upper`）。它**不做**咬合自适应，
也**不碰** Occlusion Solver —— 那是下游 `StaticOcclusalSolver` 的活。

设计纪律（与项目其余部分一致）：

1. **分侧配准，禁止 upper↔lower 直接 ICP**。upper 与 lower 不是同一个表面，
   直接 ICP 会把两个牙弓"对齐"到一个自造的位置。正确做法是
   `bite↔upper`、`bite↔lower` 各自配准，再组合。
2. **粗配准结果绝不是 final**。粗对齐只用来把 robust ICP 带进收敛域。
3. **阈值全部来自 `BiteRegistrationProfile`**，不硬编码在算法体内。
4. **不修改输入网格**：预处理返回**新的** mesh；原始字节由调用方自行校验。
5. **不返回一个 4×4 就当成功**：结果必须带 RMSE / inlier / overlap / 穿透与
   `PASS|REVIEW_REQUIRED|FAILED` 判定，歧义与错配要如实上报。
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict

import numpy as np

ALGORITHM_VERSION = "bite_registration/trimmed_point_to_plane/1"

#: **唯一**的 reason code 表。报告只能从这里取，禁止在别处手写字符串。
REASON_BITE_SCAN_MISSING = "BITE_SCAN_MISSING"
REASON_MANUAL_BITE_REQUIRED = "MANUAL_BITE_REQUIRED"
REASON_BITE_SCAN_EMPTY = "BITE_SCAN_EMPTY"
REASON_BITE_SIDE_AMBIGUOUS = "BITE_SIDE_AMBIGUOUS"
REASON_GROSS_JAW_INTERSECTION = "GROSS_JAW_INTERSECTION"
REASON_UPPER_SIDE_UNMATCHABLE = "UPPER_SIDE_UNMATCHABLE"
REASON_LOWER_SIDE_UNMATCHABLE = "LOWER_SIDE_UNMATCHABLE"
REASON_INSUFFICIENT_OVERLAP = "INSUFFICIENT_OVERLAP_AREA"
REASON_MATCH_BELOW_MINIMUM = "MATCH_QUALITY_BELOW_MINIMUM"
REASON_LOCAL_MISMATCH = "LOCAL_MISMATCH_REVIEW_REQUIRED"
REASON_SIDE_UNAVAILABLE = "SIDE_MATCH_UNAVAILABLE"
REASON_PENETRATION_OPEN_SHELL = "PENETRATION_NOT_ASSESSED_OPEN_SHELL"
REASON_JOINT_INCONSISTENT = "JOINT_CONSISTENCY_FAILED"
REASON_INCONSISTENT_BITE_EVIDENCE = "INCONSISTENT_BITE_EVIDENCE"
REASON_RIGID_CONTRACT_VIOLATED = "RIGID_CONTRACT_VIOLATED"
REASON_SOLUTION_AMBIGUOUS = "REGISTRATION_SOLUTION_AMBIGUOUS"
REASON_INSUFFICIENT_REGISTRATION_EVIDENCE = "INSUFFICIENT_REGISTRATION_EVIDENCE"

#: 候选池与排序（全部坐标系不变）
COARSE_TIE_REL = 1.10          # trimmed error ≤ best×1.10 视为"近似等价"
COARSE_TIE_ABS_MM = 0.02
COARSE_POOL_MAX = 8
COARSE_ALREADY_CLOSE_REL = 0.02   # ENGINEERING_DEFAULT_UNVALIDATED（相对尺度）
#: ROTATION_SEED：绕源主轴的离散步数（8 步=45°）+ 倾斜种子（PCA 近简并时的兜底）
ROTATION_SEED_STEPS = (8,)
ROTATION_SEED_TILTS = (20.0, -20.0)
REFINE_TOP_N = 3
AMBIGUITY_RMSE_REL = 0.15
AMBIGUITY_RMSE_ABS_MM = 0.02
AMBIGUITY_INLIER_DELTA = 0.05
AMBIGUITY_ROTATION_DEG = 2.0
AMBIGUITY_TRANSLATION_MM = 0.50
REASON_BITE_SCAN_UNREADABLE = "BITE_SCAN_UNREADABLE"
REASON_EMPTY_BITE = "EMPTY_BITE"
REASON_EMPTY_GEOMETRY = "EMPTY_GEOMETRY"
REASON_TOO_FEW_PAIRS = "TOO_FEW_USABLE_PAIRS"

REASON_CODES = frozenset({REASON_BITE_SCAN_MISSING, REASON_MANUAL_BITE_REQUIRED, REASON_BITE_SCAN_EMPTY,
                          REASON_BITE_SIDE_AMBIGUOUS, REASON_GROSS_JAW_INTERSECTION, REASON_UPPER_SIDE_UNMATCHABLE,
                          REASON_LOWER_SIDE_UNMATCHABLE, REASON_INSUFFICIENT_OVERLAP, REASON_MATCH_BELOW_MINIMUM,
                          REASON_LOCAL_MISMATCH, REASON_SIDE_UNAVAILABLE, REASON_PENETRATION_OPEN_SHELL,
                          REASON_BITE_SCAN_UNREADABLE, REASON_EMPTY_BITE, REASON_EMPTY_GEOMETRY, REASON_TOO_FEW_PAIRS,
                          REASON_JOINT_INCONSISTENT, REASON_INCONSISTENT_BITE_EVIDENCE,
                          REASON_RIGID_CONTRACT_VIOLATED, REASON_SOLUTION_AMBIGUOUS,
                          REASON_INSUFFICIENT_REGISTRATION_EVIDENCE})

#: 穿模结论三态：只有闭合实体才允许 authoritative signed 结论
PENETRATION_MEASURED = "MEASURED"
PENETRATION_NOT_ASSESSED = "NOT_ASSESSED"

STATUS_PASS = "PASS"
STATUS_REVIEW = "REVIEW_REQUIRED"
STATUS_FAILED = "FAILED"

#: 穿透/重叠的**工程口径**（必须随结果一起报出，不是可以随便替换的抽象概念）
PENETRATION_DEFINITION = "sample_point_nearest_surface_normal_sign_negative_beyond_depth"
OVERLAP_DEFINITION = "bite_face_area_whose_vertices_are_within_band_of_target"
AREAS_ARE_ENGINEERING_ESTIMATES = True


@dataclass
class BiteRegistrationProfile:
    """全部阈值。算法体内不得出现魔法数字。"""

    # 预处理
    min_component_faces: int = 200          # 去掉孤立小岛（小于这个面数）
    voxel_mm: float = 0.6                    # 下采样体素（0 = 不下采样）
    sample_points: int = 20000               # 每次评估的采样点数上限
    # 粗配准
    coarse_max_corr_mm: float = 8.0
    coarse_trim_ratio: float = 0.4
    coarse_iterations: int = 30
    # 精配准
    refine_max_corr_mm: float = 2.0
    refine_trim_ratio: float = 0.7
    refine_iterations: int = 60
    converge_tol_mm: float = 1e-4
    # 对应点筛选
    normal_consistency_deg: float = 45.0
    # QC 判定
    pass_rmse_mm: float = 0.30
    review_rmse_mm: float = 0.80
    pass_inlier_ratio: float = 0.60
    review_inlier_ratio: float = 0.35
    min_overlap_area_mm2: float = 5.0
    # 分区歧义
    ambiguity_rmse_margin_mm: float = 0.05
    # ---- 新阈值统一标 ENGINEERING_DEFAULT_UNVALIDATED（不是把旧阈值放宽）----
    # P0-1 修复后 JointConsistency 不再产生任何**跨物理表面 RMSE**，因此
    # joint_pass_mm / joint_review_mm 已无合法消费者 —— 按"禁止保留没有对应物理指标的
    # 死阈值"删除。（判据改用既有 pass/review 档 + coverage + 刚性校验 + 碰撞门。）
    ENGINEERING_DEFAULT_UNVALIDATED: bool = True
    collision_proximity_mm: float = 0.05          # ENGINEERING_DEFAULT_UNVALIDATED
    collision_warn_ratio: float = 0.02            # ENGINEERING_DEFAULT_UNVALIDATED
    collision_intersection_pairs_budget: int = 400000  # 三角相交分析的工程预算
    # 穿模门（粗判，进入下游前的最后一道）
    penetration_depth_mm: float = 0.20
    gross_penetration_max_mm: float = 1.00
    gross_penetration_area_mm2: float = 20.0
    gross_penetration_ratio: float = 0.05

    def as_report(self) -> dict:
        return {**asdict(self), "thresholds_live_in_profile": True}


# ------------------------------------------------------------------ 工具
def _faces_areas(mesh) -> np.ndarray:
    return np.asarray(mesh.area_faces, float)


def _sample(mesh, points: int, profile: BiteRegistrationProfile):
    """**确定性**点集：面中心（按 stride 抽稀）+ 面法向。只读，不改 mesh。

    ⚠️ 原来用 `mesh.sample()` —— 那是**随机采样**，导致同一输入两次调用结果不同
    （实测：`assign_sides` 选中的 lower pair 误差 0.0091mm，把同一分量再跑一次变
    3.39mm/16.97°）。ICP 的对应点集每次不同 ⇒ 盆地都可能不同，SE(3) 等变的方差
    也被它放大。改成面中心后完全可复现，且不需要任何 RNG。
    """
    centres = np.asarray(mesh.triangles_center, float)
    normals = np.asarray(mesh.face_normals, float)
    if not len(centres):
        return np.zeros((0, 3)), np.zeros((0, 3)), np.zeros(0, int)
    if points and len(centres) > points:
        stride = int(np.ceil(len(centres) / float(points)))
        centres, normals = centres[::stride], normals[::stride]
    return centres, normals, np.arange(len(centres), dtype=int)


def preprocess(mesh, profile: BiteRegistrationProfile):
    """去孤岛 + 可选体素下采样。返回**新** mesh（输入不被修改）。

    保留原始几何：不做平滑、不做重网格、不做形变。
    """
    import trimesh

    work = mesh.copy()
    parts = work.split(only_watertight=False)
    if len(parts) > 1:
        keep = [p for p in parts if len(p.faces) >= profile.min_component_faces]
        if keep:
            work = trimesh.util.concatenate(keep) if len(keep) > 1 else keep[0]
    if profile.voxel_mm and profile.voxel_mm > 0:
        try:
            reduced = work.simplify_quadric_decimation(
                face_count=max(400, int(len(work.faces) * 0.35))) if hasattr(work, "simplify_quadric_decimation") else None
            if reduced is not None and len(reduced.faces) >= 400:
                work = reduced
        except Exception:
            pass
    work = work.copy()
    try:
        work.fix_normals()
    except Exception:
        pass
    return work


def _uniform_scale(mesh) -> float:
    pts = np.asarray(mesh.vertices, float)
    return float(np.linalg.norm(pts.max(axis=0) - pts.min(axis=0)))


def _pca_frame(pts: np.ndarray):
    centre = pts.mean(axis=0)
    centred = pts - centre
    cov = centred.T @ centred / max(len(centred), 1)
    values, vectors = np.linalg.eigh(cov)
    order = np.argsort(values)[::-1]
    axes = vectors[:, order]
    if np.linalg.det(axes) < 0:
        axes[:, 2] *= -1.0
    return centre, axes


def _axis_rotation(axis, angle: float) -> np.ndarray:
    """绕给定轴（**由几何自身定义**，不是世界轴）的旋转矩阵。"""
    axis = np.asarray(axis, float)
    norm = float(np.linalg.norm(axis))
    if norm < 1e-12:
        return np.eye(3)
    axis = axis / norm
    skew = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(angle) * skew + (1 - np.cos(angle)) * (skew @ skew)


def _rigid(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    matrix = np.eye(4)
    matrix[:3, :3] = rotation
    matrix[:3, 3] = translation
    return matrix


def _apply(matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    return points @ matrix[:3, :3].T + matrix[:3, 3]


def _nearest(source: np.ndarray, target: np.ndarray):
    from scipy.spatial import cKDTree
    distances, ids = cKDTree(target).query(source, k=1, workers=1)
    return distances, ids


def _trimmed_error(source, target, matrix, trim_ratio):
    moved = _apply(matrix, source)
    distances, ids = _nearest(moved, target)
    if not len(distances):
        return np.inf, np.zeros(0, bool)
    cut = np.quantile(distances, trim_ratio)
    keep = distances <= cut
    return float(np.sqrt(np.mean(distances[keep] ** 2))), keep


# ------------------------------------------------------------------ 粗配准
def coarse_align(source, target, profile: BiteRegistrationProfile, *, sample=None):
    """粗对齐：先判"是否已大致同帧"，否则用 PCA 主轴候选集挑最优。

    粗配准**永远不作为最终结果**，只负责把 ICP 带进收敛域。
    """
    sample = sample or profile.sample_points
    src_pts, _, _ = _sample(source, sample, profile)
    dst_pts, _, _ = _sample(target, sample, profile)
    identity = np.eye(4)
    baseline, _ = _trimmed_error(src_pts, dst_pts, identity, profile.coarse_trim_ratio)
    extent = max(_uniform_scale(target), 1e-6)
    # ⚠️ 快速路径判据必须**相对尺度**：原来只看 trimmed 误差 ≤ 4mm，而 patch/jaw 本身约 ±9mm，
    # 于是"不同扫描帧"也被判成 already_close ⇒ 不生成任何 PCA/ROTATION_SEED 候选（实测 pool=1，
    # 正确解根本没进池）。现在同时要求误差与质心差都 ≤ 2% 尺度。
    centroid_gap = float(np.linalg.norm(src_pts.mean(axis=0) - dst_pts.mean(axis=0)))
    already_close = (baseline <= COARSE_ALREADY_CLOSE_REL * extent
                     and centroid_gap <= COARSE_ALREADY_CLOSE_REL * extent)

    src_centre, src_axes = _pca_frame(src_pts)
    dst_centre, dst_axes = _pca_frame(dst_pts)
    candidates = [identity]
    if not already_close:
        for sx in (1, -1):
            for sy in (1, -1):
                for sz in (1, -1):
                    basis = src_axes @ np.diag([sx, sy, sz])
                    if np.linalg.det(basis) < 0:
                        continue
                    rotation = dst_axes @ basis.T
                    translation = dst_centre - rotation @ src_centre
                    candidates.append(_rigid(rotation, translation))
        # ROTATION_SEED：PCA 在"两个主特征值接近"的 patch 上 eigenbasis 不唯一 ⇒
        # 绕源主轴的离散旋转作为兜底候选（绕**自身主轴**的构造，与坐标系选择无关）。
        axis = src_axes[:, 0]
        for steps in ROTATION_SEED_STEPS:
            angle = np.radians(360.0 / steps)
            for index in range(steps):
                rotation = _axis_rotation(axis, angle * index)
                candidates.append(_rigid(rotation, dst_centre - rotation @ src_centre))
        for tilt_deg in ROTATION_SEED_TILTS:
            for az in (0.0, 90.0, 180.0, 270.0):
                tilt_axis = _axis_rotation(src_axes[:, 2], np.radians(az)) @ src_axes[:, 1]
                rotation = _axis_rotation(tilt_axis, np.radians(tilt_deg)) @ _axis_rotation(axis, np.radians(az))
                candidates.append(_rigid(rotation, dst_centre - rotation @ src_centre))
    scored = []
    for index, matrix in enumerate(candidates):
        error, _ = _trimmed_error(src_pts, dst_pts, matrix, profile.coarse_trim_ratio)
        scored.append({"matrix": matrix,
                       "source_method": "INITIAL_CLOSE" if index == 0 else "PCA_VARIANTS",
                       "coarse_error_mm": float(error)})
    scored.sort(key=lambda item: item["coarse_error_mm"])
    best_error = scored[0]["coarse_error_mm"] if scored else baseline
    # ⚠️ 近似等价的候选**全部保留**：提前选一个会让排序随公共旋转翻转（SE(3) 等变失败根因）
    # ⚠️ 池必须**保来源多样性**：上一版按 coarse error ≤best×1.10 过滤，实测把池收敛成
    # 1 个候选（refined=1 / methods=['INITIAL_CLOSE']），rotation seed 全被剔除 ⇒ 正确解根本没进
    # 池（GENERATION FAILURE）。现在每个来源至少保留各自最优，再按全局误差补足。
    by_method = {}
    for item in scored:
        by_method.setdefault(item["source_method"], []).append(item)
    keep = [items[0] for items in by_method.values()]
    for item in scored:
        if len(keep) >= COARSE_POOL_MAX:
            break
        if all(item["matrix"] is not existing["matrix"] for existing in keep):
            keep.append(item)
    keep = keep[:COARSE_POOL_MAX]
    return {"matrix": keep[0]["matrix"], "initial_error_mm": baseline,
            "coarse_error_mm": best_error, "pool": keep,
            "method": "ALREADY_CLOSE" if already_close else keep[0]["source_method"],
            "extent_mm": extent, "candidates": len(candidates), "pool_size": len(keep)}


# ------------------------------------------------------------------ 精配准
def _twist_solve(source, normals_src, target_pts, target_normals, pairs, weights=None):
    p = source[pairs]
    q = target_pts[pairs]
    n = target_normals[pairs]
    residual = np.einsum("ij,ij->i", n, p - q)
    cross = np.cross(p, n)
    design = np.hstack([cross, n])
    if weights is not None:
        w = np.sqrt(weights)[:, None]
        design = design * w
        residual = residual * w[:, 0]
    solution, *_ = np.linalg.lstsq(design, -residual, rcond=None)
    omega, translation = solution[:3], solution[3:]
    angle = float(np.linalg.norm(omega))
    if angle < 1e-12:
        rotation = np.eye(3)
    else:
        axis = omega / angle
        skew = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
        rotation = np.eye(3) + np.sin(angle) * skew + (1 - np.cos(angle)) * (skew @ skew)
    return _rigid(rotation, translation)


def icp_point_to_plane(source, target, initial, profile: BiteRegistrationProfile, *,
                       max_corr_mm: float | None = None, trim_ratio: float | None = None,
                       iterations: int | None = None, sample: int | None = None):
    """robust trimmed point-to-plane ICP。

    - 多分辨率：调用方按 粗→精 各调一次（`coarse_align` → 本函数 coarse 参数 → 本函数 refine 参数）；
    - trimmed：只保留残差最小的 ``trim_ratio`` 比例（抗局外点）；
    - normal consistency：法向夹角超过阈值的一律不参与；
    - max correspondence distance / max iterations / convergence tolerance 全部来自 profile。
    """
    max_corr = profile.refine_max_corr_mm if max_corr_mm is None else max_corr_mm
    trim = profile.refine_trim_ratio if trim_ratio is None else trim_ratio
    iters = profile.refine_iterations if iterations is None else iterations
    sample = sample or profile.sample_points
    cos_limit = float(np.cos(np.radians(profile.normal_consistency_deg)))

    src_pts, src_normals, _ = _sample(source, sample, profile)
    dst_pts, dst_normals, _ = _sample(target, sample, profile)

    matrix = np.asarray(initial, float).copy()
    history = []
    for step in range(iters):
        moved = _apply(matrix, src_pts)
        distances, ids = _nearest(moved, dst_pts)
        paired_normals = dst_normals[ids]
        dot = np.abs(np.einsum("ij,ij->i", src_normals, paired_normals))
        usable = (distances <= max_corr) & (dot >= cos_limit)
        if usable.sum() < 12:
            history.append({"iteration": step + 1, "usable_pairs": int(usable.sum()),
                            "converged": False, "reason": REASON_TOO_FEW_PAIRS})
            return {"matrix": matrix, "history": history, "used_initial": True,
                    "reason": REASON_TOO_FEW_PAIRS}
        cut = np.quantile(distances[usable], trim)
        keep = usable & (distances <= cut)
        # ⚠️ 必须传**配对后**的目标数组：`keep` 是源采样点长度的掩码，
        # 拿它去索引原始 dst（长度 = 对面面数）会直接 IndexError（第一版实测 3042 vs 1326）。
        delta = _twist_solve(moved, src_normals, dst_pts[ids], dst_normals[ids], keep)
        matrix = delta @ matrix
        moved_after = _apply(matrix, src_pts)
        after, _ = _nearest(moved_after, dst_pts)
        rmse = float(np.sqrt(np.mean(after[keep] ** 2)))
        moved_step = float(np.linalg.norm(_apply(delta, np.zeros((1, 3)))[0]))
        history.append({"iteration": step + 1, "usable_pairs": int(usable.sum()),
                        "kept_pairs": int(keep.sum()), "rmse_mm": rmse,
                        "step_mm": moved_step, "converged": False})
        if moved_step < profile.converge_tol_mm and len(history) > 1:
            history[-1]["converged"] = True
            break
    return {"matrix": matrix, "history": history, "used_initial": False, "reason": None}


#: 姿态聚类门限（只用坐标不变量；这些是**聚类参数**，不是临床阈值）
POSE_CLUSTER_ROT_DEG = 1.0
POSE_CLUSTER_POINT_MM = 0.20


def _pose_disagreement(a, b, ref_points):
    """坐标不变量：旋转夹角 + 参考点位移 RMS（公共刚体下不变）。"""
    import trimesh
    rotation = float(np.degrees(np.arccos(np.clip(
        (np.trace(np.asarray(a, float)[:3, :3].T @ np.asarray(b, float)[:3, :3]) - 1) / 2, -1, 1))))
    pa = trimesh.transform_points(ref_points, np.asarray(a, float))
    pb = trimesh.transform_points(ref_points, np.asarray(b, float))
    displacement = float(np.sqrt(np.mean(np.sum((pa - pb) ** 2, axis=1))))
    return rotation, displacement


def _cluster_candidates(refined, ref_points, profile):
    clusters = []
    for item in refined:
        for cluster in clusters:
            rotation, displacement = _pose_disagreement(item["matrix"], cluster["representative"]["matrix"], ref_points)
            if rotation <= POSE_CLUSTER_ROT_DEG and displacement <= POSE_CLUSTER_POINT_MM:
                cluster["members"].append(item)
                if _member_key(item) < _member_key(cluster["representative"]):
                    cluster["representative"] = item
                break
        else:
            clusters.append({"members": [item], "representative": item})
    for index, cluster in enumerate(clusters):
        for member in cluster["members"]:
            member["cluster_index"] = index   # 诊断用：候选属于哪个簇
    for cluster in clusters:
        qualities = [m["quality"] for m in cluster["members"]]
        cluster["member_count"] = len(cluster["members"])
        cluster["source_methods"] = sorted({m["source_method"] for m in cluster["members"]})
        cluster["median_inlier"] = float(np.median([q.get("inlier_ratio") or 0.0 for q in qualities]))
        cluster["median_normal_deg"] = float(np.median(
            [q.get("normal_agreement_max_deg") if q.get("normal_agreement_max_deg") is not None else 1e9
             for q in qualities]))
        cluster["median_rmse"] = float(np.median([q.get("rmse_mm") if q.get("rmse_mm") is not None else 1e9 for q in qualities]))
        cluster["median_matched_area_mm2"] = float(np.median([q.get("overlap_area_mm2") or 0.0 for q in qualities]))
        cluster["median_source_coverage_ratio"] = float(np.median([q.get("source_coverage_ratio") or 0.0 for q in qualities]))
        dispersion = [_pose_disagreement(m["matrix"], cluster["representative"]["matrix"], ref_points)
                      for m in cluster["members"]]
        cluster["pose_dispersion_rotation_deg"] = max((d[0] for d in dispersion), default=0.0)
        cluster["pose_dispersion_point_mm"] = max((d[1] for d in dispersion), default=0.0)
    return clusters


def _member_key(item):
    """簇内代表：**法向一致角 → inlier → rmse → matched area**（全部坐标不变量）。

    为什么法向排第一：实测存在错位姿拿到 `rmse=0.0000 / inlier=1.000`（点面距离与匹配比例
    对切向滑动/自转不敏感），此时**只有法向一致角能分辨**（R_spin 分析已证法向角≈刚体转角）。
    不引入任何新阈值，只改排序口径。
    """
    quality = item["quality"]
    rmse = quality.get("rmse_mm")
    normal_deg = quality.get("normal_agreement_max_deg")
    return (float("inf") if normal_deg is None else normal_deg,
            -(quality.get("inlier_ratio") or 0.0),
            float("inf") if rmse is None else rmse,
            -(quality.get("overlap_area_mm2") or 0.0))


def _cluster_key(cluster):
    """簇选择：**支持度优先**（成员数 → 来源多样性），再看证据质量。

    单个候选、即便 rmse 极低，若无旁证（member_count=1）也要排在多成员簇之后。
    """
    return (-cluster["member_count"], -len(cluster["source_methods"]),
            cluster["median_normal_deg"],
            -cluster["median_inlier"], cluster["median_rmse"], -cluster["median_matched_area_mm2"])


OBSERVABILITY_TRANSLATION_STEPS_MM = (0.25, 0.5, 1.0, 2.0)
OBSERVABILITY_ROTATION_STEPS_DEG = (0.5, 1.0, 2.0, 4.0)
OBSERVABILITY_INLIER_FLOOR = 0.01       # ENGINEERING_DEFAULT_UNVALIDATED
OBSERVABILITY_RMSE_FLOOR_MM = 0.05      # ENGINEERING_DEFAULT_UNVALIDATED（绝对 mm）
OBSERVABILITY_NORMAL_ANGLE_FLOOR_DEG = 0.5   # ENGINEERING_DEFAULT_UNVALIDATED（deg/deg）


def pose_observability(source, target, matrix, profile):
    # 沿**局部主轴**扰动 6DoF，比较不变质量的变化；某自由度几乎不改变质量 => 弱约束
    # 扰动方向定义在由源几何自身给出的主轴系里，因此公共 SE(3) 变换后语义等价
    centres, normals, _ = _sample(source, profile.sample_points, profile)
    if not len(centres):
        return {"measurement_status": "NOT_ASSESSED", "weak_dofs": [], "weak_dof_count": 0,
                "status": "NOT_ASSESSED"}
    axis_centre = centres.mean(axis=0)
    centred = centres - axis_centre
    values, axes = np.linalg.eigh(centred.T @ centred / len(centred))
    axes = axes[:, np.argsort(values)[::-1]]
    if np.linalg.det(axes) < 0:
        axes[:, 2] *= -1.0
    base = _pair_quality(source, target, matrix, profile)
    base_inlier = base.get("inlier_ratio") or 0.0
    base_rmse = base.get("rmse_mm")
    # 第三项证据：法向一致角（deg）。点面距离对**切向滑动**不敏感（滑动仍贴面），
    # 但法向会随扰动一起转 ⇒ 这是能看见 Rz 的那一项（实测 Rz@4° 仅靠 rmse 得 0.57，
    # 加上法向角后 >1；fixed-correspondence 对照证明几何信息本来就在）。
    base_normal_deg = base.get("normal_agreement_max_deg") or 0.0
    translation_sensitivity, rotation_sensitivity, weak = {}, {}, []

    def _delta(quality):
        # 绝对变化（base_rmse≈0 时比值会炸，实测踩到）
        inlier_drop = base_inlier - (quality.get("inlier_ratio") or 0.0)
        rmse = quality.get("rmse_mm")
        rise_mm = (rmse - base_rmse) if (rmse is not None and base_rmse is not None) else 0.0
        normal_deg = quality.get("normal_agreement_max_deg") or 0.0
        return max(inlier_drop / max(OBSERVABILITY_INLIER_FLOOR, 1e-9),
                   rise_mm / OBSERVABILITY_RMSE_FLOOR_MM,
                   (normal_deg - base_normal_deg) / OBSERVABILITY_NORMAL_ANGLE_FLOOR_DEG)

    for index in range(3):
        worst = 0.0
        for step in OBSERVABILITY_TRANSLATION_STEPS_MM:
            for sign in (1.0, -1.0):
                moved = np.eye(4)
                moved[:3, 3] = axes[:, index] * step * sign
                worst = max(worst, _delta(_pair_quality(source, target, moved @ np.asarray(matrix, float), profile)))
        translation_sensitivity["T%s" % "xyz"[index]] = float(worst)
        if worst < 1.0:
            weak.append("T%s" % "xyz"[index])
    for index in range(3):
        worst = 0.0
        for step in OBSERVABILITY_ROTATION_STEPS_DEG:
            for sign in (1.0, -1.0):
                rotation = _axis_rotation(axes[:, index], np.radians(step * sign))
                moved = np.eye(4)
                moved[:3, :3] = rotation
                worst = max(worst, _delta(_pair_quality(source, target, moved @ np.asarray(matrix, float), profile)))
        rotation_sensitivity["R%s" % "xyz"[index]] = float(worst)
        if worst < 1.0:
            weak.append("R%s" % "xyz"[index])
    status = "INSUFFICIENT_REGISTRATION_EVIDENCE" if weak else "SUPPORTED"
    return {"measurement_status": "MEASURED", "status": status,
            "translation_sensitivity_xyz": translation_sensitivity,   # 无量纲超出倍数，NOT mm
            "rotation_sensitivity_xyz": rotation_sensitivity,         # 无量纲超出倍数，NOT deg
            "weak_dofs": weak, "weak_dof_count": len(weak),
            "metric": "sensitivity_dof = max(d_inlier/0.01, d_rmse_mm/0.05, d_normal_angle_deg/0.5)",
            "units": "DIMENSIONLESS（对分析地板的超出倍数；>=1 记为可观测）",
            "aggregation": "max over ±{0.25,0.5,1,2}mm / ±{0.5,1,2,4}deg perturbations",
            "floors": {"inlier": OBSERVABILITY_INLIER_FLOOR, "rmse_mm": OBSERVABILITY_RMSE_FLOOR_MM},
            "note": "扰动沿源几何主轴系，公共刚体变换下语义等价"}


def _invariant_score(quality: dict):
    """**只用坐标系不变的量**排序：**inlier → rmse → matched surface area**。

    顺序按实测证据定：在圆柱/近似旋转对称的 patch 上，"沿面滑动"的错位姿
    点面 RMSE 同样可以很小（实测 translation case 被这种候选骗到 15.8°），
    **匹配点比例**对滑动的惩罚更可靠。三者都是坐标不变量，不引入任何阈值。
    """
    rmse = quality.get("rmse_mm")
    inlier = quality.get("inlier_ratio") or 0.0
    area = quality.get("overlap_area_mm2") or 0.0
    return (-float(inlier), float("inf") if rmse is None else rmse, -float(area))


def _pose_difference(a, b):
    delta = np.asarray(a, float) @ np.linalg.inv(np.asarray(b, float))
    translation = float(np.linalg.norm(delta[:3, 3]))
    rotation = float(np.degrees(np.arccos(np.clip((np.trace(delta[:3, :3]) - 1) / 2, -1, 1))))
    return translation, rotation


def register_pair(source, target, profile: BiteRegistrationProfile):
    """候选池 → 每候选低分辨率 ICP → 不变排序 → top-3 精化 → 歧义门。

    返回位姿、质量、候选排名（rank_components 只含不变量）与 `ambiguous_solution` 标记。
    """
    coarse = coarse_align(source, target, profile)
    pool = coarse.get("pool") or [{"matrix": coarse["matrix"], "source_method": coarse["method"],
                                   "coarse_error_mm": coarse.get("coarse_error_mm")}]
    staged = []
    iterations = 0
    for item in pool:
        low = icp_point_to_plane(source, target, item["matrix"], profile,
                                 max_corr_mm=profile.coarse_max_corr_mm,
                                 trim_ratio=profile.coarse_trim_ratio,
                                 iterations=profile.coarse_iterations)
        iterations += len(low["history"])
        quality = _pair_quality(source, target, low["matrix"], profile)
        staged.append({"matrix": low["matrix"], "source_method": item["source_method"],
                       "quality": quality, "score": _invariant_score(quality)})
    staged.sort(key=lambda item: item["score"])
    refined = []
    for item in staged[:REFINE_TOP_N]:
        fine = icp_point_to_plane(source, target, item["matrix"], profile)
        iterations += len(fine["history"])
        quality = _pair_quality(source, target, fine["matrix"], profile)
        refined.append({"matrix": fine["matrix"], "source_method": item["source_method"],
                        "quality": quality, "score": _invariant_score(quality),
                        "converged": bool(any(h.get("converged") for h in fine["history"])
                                          or fine["reason"] is None),
                        "reason": fine["reason"]})
    # ---- 姿态聚类：不再用单一标量排名（近对称 patch 上会被"滑走解"骗）----
    ref_points = np.asarray(_sample(source, min(1200, profile.sample_points), profile)[0], float)
    clusters = _cluster_candidates(refined, ref_points, profile)
    clusters.sort(key=_cluster_key)
    # 排序后**重新盖 rank**：否则 gt_inside_winner_cluster / gt_cluster_rank 用的是"创建顺序下标"，
    # 与"获胜簇"不是同一个东西（实测导致自相矛盾的诊断输出）。
    for rank, cluster in enumerate(clusters, start=1):
        cluster["rank"] = rank
        for member in cluster["members"]:
            member["cluster_rank"] = rank
    best_cluster = clusters[0]
    best = max(best_cluster["members"], key=lambda item: _member_key(item))
    ambiguous = False
    if len(clusters) >= 2:
        runner = clusters[1]
        support_close = abs(best_cluster["member_count"] - runner["member_count"]) <= 1
        quality_close = abs(best_cluster["median_inlier"] - runner["median_inlier"]) <= AMBIGUITY_INLIER_DELTA
        translation, rotation = _pose_difference(best_cluster["representative"]["matrix"],
                                                 runner["representative"]["matrix"])
        pose_far = rotation >= AMBIGUITY_ROTATION_DEG or translation >= AMBIGUITY_TRANSLATION_MM
        # 单成员簇 + 另有同水平簇 ⇒ 无旁证支持，不得轻易 PASS
        lonely = best_cluster["member_count"] == 1 and quality_close
        if pose_far and (support_close or quality_close or lonely):
            ambiguous = True
    rank_components = [{"source_method": item["source_method"],
                        "rmse_mm": item["quality"].get("rmse_mm"),
                        "inlier_ratio": item["quality"].get("inlier_ratio"),
                        "matched_area_mm2": item["quality"].get("overlap_area_mm2")}
                       for item in refined]
    observability = pose_observability(source, target, best["matrix"], profile)
    return {"matrix": best["matrix"], "source_method": best["source_method"],
            "observability": observability,
            "insufficient_evidence": observability.get("status") == "INSUFFICIENT_REGISTRATION_EVIDENCE",
            "coarse": {"method": coarse.get("method"), "initial_error_mm": coarse.get("initial_error_mm"),
                       "coarse_error_mm": coarse.get("coarse_error_mm"),
                       "pool_size": coarse.get("pool_size"), "candidates": coarse.get("candidates")},
            "coarse_pool_size": len(pool), "refined_candidates": len(refined),
            "refined_matrices": [np.asarray(item["matrix"], float).tolist() for item in refined],
            "refined_cluster_rank": [int(item.get("cluster_rank", -1)) for item in refined],
            "clusters": [{"member_count": c["member_count"], "source_methods": c["source_methods"],
                          "median_inlier": c["median_inlier"], "median_rmse": c["median_rmse"],
                          "median_normal_deg": c["median_normal_deg"], "rank": c.get("rank"),
                          "median_matched_area_mm2": c["median_matched_area_mm2"],
                          "pose_dispersion_rotation_deg": c["pose_dispersion_rotation_deg"]}
                         for c in clusters],
            "winner_cluster_support": best_cluster["member_count"],
            "rank_components": rank_components, "ambiguous_solution": ambiguous,
            "iterations": iterations, "converged": best.get("converged", False),
            "reason": best.get("reason"), **best["quality"]}


def _pair_quality(source, target, matrix, profile: BiteRegistrationProfile):
    """RMSE / inlier ratio / overlap area。

    ⚠️ 距离用**精确点到曲面**（`ProximityQuery.on_surface`），不用采样点云的最近邻：
    点云最近邻有 ~采样间距的地板（实测把 0.30mm 的"假 RMSE"顶到 PASS 阈值上，
    让人误以为是位姿误差）。overlap 面积仍按声明的面口径统计。
    """
    import trimesh
    centres = np.asarray(source.triangles_center, float)
    normals = np.asarray(source.face_normals, float)
    moved = _apply(matrix, centres)
    query = trimesh.proximity.ProximityQuery(target)
    closest, distances, triangle_ids = query.on_surface(moved)
    distances = np.asarray(distances, float)
    target_normals = np.asarray(target.face_normals, float)[np.asarray(triangle_ids, int)]
    dot = np.abs(np.einsum("ij,ij->i", normals, target_normals))
    cos_limit = float(np.cos(np.radians(profile.normal_consistency_deg)))
    inliers = (distances <= profile.refine_max_corr_mm) & (dot >= cos_limit)
    areas = _faces_areas(source)
    rmse = float(np.sqrt(np.mean(distances[inliers] ** 2))) if inliers.any() else None
    overlap = float(areas[inliers].sum()) if inliers.any() else 0.0
    # 目标侧覆盖度：反向查一次，避免"小片贴大片"骗过单向覆盖
    t_centres = np.asarray(target.triangles_center, float)
    _, t_dist, _ = trimesh.proximity.ProximityQuery(source).on_surface(
        _apply(np.linalg.inv(np.asarray(matrix, float)), t_centres))
    target_coverage = float((np.asarray(t_dist, float) <= profile.refine_max_corr_mm).mean())
    matched = moved[inliers] if inliers.any() else moved[:0]
    spread = np.ptp(matched, axis=0).tolist() if len(matched) else [0.0, 0.0, 0.0]
    matched_extent = spread
    if len(matched) >= 4:
        centred = matched - matched.mean(axis=0)
        values = np.sort(np.linalg.eigvalsh(centred.T @ centred / len(matched)))[::-1]
        principal = [float(np.sqrt(max(v, 0.0)) * 2.0) for v in values]     # 主轴长度（mm）
    else:
        principal = None
    matched_normals = np.asarray(source.face_normals, float)[inliers] if inliers.any() else None
    normal_variation = (float(np.degrees(np.arccos(np.clip(
        np.abs(matched_normals @ np.array([0.0, 1.0, 0.0])), -1, 1))).std())
        if matched_normals is not None and len(matched_normals) else None)
    return {"rmse_mm": rmse, "inlier_ratio": float(inliers.mean()),
            "overlap_area_mm2": overlap, "faces": int(len(source.faces)),
            "inlier_faces": int(inliers.sum()),
            "source_coverage_ratio": float(inliers.mean()),
            "target_coverage_ratio": target_coverage,
            "matched_spread_mm": spread,
            "matched_pairs": int(inliers.sum()),
            "p95_surface_distance_mm": float(np.quantile(distances, 0.95)),
            "metric": "EXACT_POINT_TO_SURFACE",
            "overlap_definition": OVERLAP_DEFINITION,
            # ---- D2 结构化空间证据（全部坐标不变量；测不出即 None + NOT_ASSESSED）----
            "matched_surface_area_mm2": overlap,
            "matched_pairs": int(inliers.sum()),
            "source_coverage_ratio": float(inliers.mean()),
            "whole_target_coverage_ratio": target_coverage,
            "whole_target_coverage_diagnostic_only": True,
            "matched_bbox_extent_xyz": matched_extent,
            "matched_principal_extent_1_mm": principal[0] if principal else None,
            "matched_principal_extent_2_mm": principal[1] if principal else None,
            "matched_principal_extent_3_mm": principal[2] if principal else None,
            "matched_cov_eigenvalue_ratio_21": (principal[1] / principal[0]) if principal and principal[0] else None,
            "matched_cov_eigenvalue_ratio_31": (principal[2] / principal[0]) if principal and principal[0] else None,
            "matched_normal_variation_deg": normal_variation,
            "normal_agreement_mean": (float(dot[inliers].mean()) if inliers.any() else None),
            "normal_agreement_max_deg": (float(np.degrees(np.arccos(np.clip(dot[inliers], -1.0, 1.0))).max())
                                        if inliers.any() else None),
            "measurement_status": "MEASURED" if inliers.any() else "NOT_ASSESSED"}


# ------------------------------------------------------------------ 分区
def connected_components(mesh, profile: BiteRegistrationProfile):
    parts = mesh.split(only_watertight=False)
    parts = [p for p in parts if len(p.faces) >= profile.min_component_faces]
    return parts


def assign_sides(bite, upper, lower, profile: BiteRegistrationProfile, *, hint: dict | None = None):
    """把 bite scan 分成 upper 侧 / lower 侧，并给出两侧的 bite→jaw 位姿。

    优先用**元数据**（调用方给定的 `hint`，例如扫描自带的上下侧标记）；
    否则按连通分量尝试两种分配，取"两侧合成误差 + overlap"更好的那种；
    两者接近到无法区分就报 `BITE_SIDE_AMBIGUOUS`，**不静默猜**。
    """
    parts = connected_components(bite, profile)
    if hint and hint.get("upper") is not None and hint.get("lower") is not None:
        return {"upper": hint["upper"], "lower": hint["lower"], "method": "METADATA",
                "ambiguous": False, "candidates": 1}
    if not parts:
        return {"upper": None, "lower": None, "method": REASON_EMPTY_BITE, "ambiguous": False,
                "candidates": 0, "reason": REASON_BITE_SCAN_EMPTY}
    if len(parts) == 1:
        return _assign_single_component(parts[0], upper, lower, profile)
    if len(parts) == 2:
        return _assign_two(parts[0], parts[1], upper, lower, profile)
    # >2 分量：按面积取最大的两个，其余视为噪点（如实登记）
    parts = sorted(parts, key=lambda p: len(p.faces), reverse=True)
    result = _assign_two(parts[0], parts[1], upper, lower, profile)
    result["dropped_components"] = len(parts) - 2
    return result


def _assign_single_component(part, upper, lower, profile):
    """单个连通分量：按"离哪一侧更近"判，两侧都一样近则歧义。"""
    pts, _, _ = _sample(part, profile.sample_points, profile)
    up_pts, _, _ = _sample(upper, profile.sample_points, profile)
    low_pts, _, _ = _sample(lower, profile.sample_points, profile)
    up_dist, _ = _nearest(pts, up_pts)
    low_dist, _ = _nearest(pts, low_pts)
    up_score, low_score = float(np.quantile(up_dist, 0.2)), float(np.quantile(low_dist, 0.2))
    if abs(up_score - low_score) <= profile.ambiguity_rmse_margin_mm:
        return {"upper": None, "lower": None, "method": "SINGLE_COMPONENT",
                "ambiguous": True, "reason": REASON_BITE_SIDE_AMBIGUOUS, "candidates": 2,
                "scores": {"upper_p20_mm": up_score, "lower_p20_mm": low_score}}
    return {"upper": (part if up_score < low_score else None),
            "lower": (part if up_score >= low_score else None),
            "lower_patch": part,
            "method": "SINGLE_COMPONENT_PROXIMITY", "ambiguous": False, "candidates": 2,
            "scores": {"upper_p20_mm": up_score, "lower_p20_mm": low_score}}


def _assign_two(first, second, upper, lower, profile):
    """两个连通分量：试两种分配，比较合成质量。"""
    options = []
    for label, (as_upper, as_lower) in (("A_UP_B_LOW", (first, second)),
                                        ("B_UP_A_LOW", (second, first))):
        up = register_pair(as_upper, upper, profile)
        low = register_pair(as_lower, lower, profile)
        combined = _combine_error(up, low)
        # 交叉校验（旋转不变）：正确分配应让每条 patch 贴**自己的颌**、且贴**另一颌**明显更差。
        cross = 0.0
        for patch, other_jaw in ((as_upper, lower), (as_lower, upper)):
            quality = register_pair(patch, other_jaw, profile)
            cross += quality.get("rmse_mm") if quality.get("rmse_mm") is not None else 1e9
        options.append({"label": label, "upper": up, "lower": low, "score": combined,
                        "cross_penalty": float(cross)})
    options.sort(key=lambda o: (o["score"], -o["cross_penalty"]))
    best, other = options[0], options[1]
    if abs(best["score"] - other["score"]) <= profile.ambiguity_rmse_margin_mm:
        # 分数接近时用交叉校验判决：谁的"贴错颌"更差，谁更可能是正确分配
        if abs(best["cross_penalty"] - other["cross_penalty"]) > profile.ambiguity_rmse_margin_mm:
            best, other = (best, other) if best["cross_penalty"] > other["cross_penalty"] else (other, best)
        else:
            return {"upper": None, "lower": None, "method": "TWO_COMPONENT", "ambiguous": True,
                    "reason": REASON_BITE_SIDE_AMBIGUOUS, "candidates": 2,
                    "scores": {o["label"]: o["score"] for o in options},
                    "cross_penalties": {o["label"]: o["cross_penalty"] for o in options}}
        return {"upper": None, "lower": None, "method": "TWO_COMPONENT", "ambiguous": True,
                "reason": REASON_BITE_SIDE_AMBIGUOUS, "candidates": 2,
                "scores": {o["label"]: o["score"] for o in options}}
    chosen_patch = second if best["label"] == "A_UP_B_LOW" else first
    return {"upper": best["upper"], "lower": best["lower"], "lower_patch": chosen_patch,
            "method": "TWO_COMPONENT_BEST_OF_2",
            "ambiguous": False, "candidates": 2, "chosen": best["label"],
            "scores": {o["label"]: o["score"] for o in options}}


def _combine_error(upper_pair: dict, lower_pair: dict) -> float:
    parts = [p["rmse_mm"] for p in (upper_pair, lower_pair) if p.get("rmse_mm") is not None]
    if len(parts) != 2:
        return float("inf")
    penalty = 0.0
    for pair in (upper_pair, lower_pair):
        if pair["inlier_ratio"] < 0.2:
            penalty += 1.0
    return float(sum(parts) / 2.0 + penalty)


# ------------------------------------------------------------------ 组合位姿
def lower_to_upper_from_bite(t_upper_to_bite: np.ndarray, t_lower_to_bite: np.ndarray) -> np.ndarray:
    """由两侧 bite 位姿组合出 lower→upper。

    **约定（本批修正）**：`T_XY` 表示"把 X 坐标系的点映射到 Y 坐标系"（列向量右乘，`p_Y = T_XY @ p_X`）。
    设 `T_lu` 为 lower→upper（未知），则同一个物理点在 bite 帧里既可写
    `p_B = T_bu @ p_U`，也可写 `p_B = T_bl @ p_L`；又 `p_U = T_lu @ p_L`，于是

        T_bl = T_bu @ T_lu        ⇒        **T_lu = inv(T_bu) @ T_bl**

    ⚠️ 第一版写的是 `T_bu @ inv(T_bl)` —— 那是错的，而且当时的"坐标约定单测"照抄了同一套
    代数，所以两边一起错了（自洽但错）。现在单测由**物理构造**给出 GT 来钉住
    （`p_U = GT @ p_L` 直接由合成几何构造），不再由公式反推。
    """
    return np.linalg.inv(np.asarray(t_upper_to_bite, float)) @ np.asarray(t_lower_to_bite, float)


def combine_lower_to_upper(t_bite_to_upper: np.ndarray, t_bite_to_lower: np.ndarray) -> np.ndarray:
    """**兼容边界**（仅此一处保留短名，核心算法一律用长名）。

    Deprecated：新代码请用 `lower_to_upper_from_bite(T_upper_to_bite, T_lower_to_bite)`。
    """
    import warnings
    warnings.warn("combine_lower_to_upper 已更名：请用 lower_to_upper_from_bite"
                  "（短名 T_bu/T_bl/T_lu 不再作为核心契约）", DeprecationWarning, stacklevel=2)
    return lower_to_upper_from_bite(t_bite_to_upper, t_bite_to_lower)


def _pose_error_to(estimate, ground_truth):
    delta = np.asarray(estimate, float) @ np.linalg.inv(np.asarray(ground_truth, float))
    return {"translation_mm": float(np.linalg.norm(delta[:3, 3])),
            "rotation_deg": float(np.degrees(np.arccos(np.clip((np.trace(delta[:3, :3]) - 1) / 2, -1, 1))))}


def gt_near_candidate_diagnostics(pair_result: dict, gt_matrix, *, near_rotation_deg: float = 1.0,
                                  near_translation_mm: float = 0.5, candidates_are_inverse: bool = True) -> dict:
    """纯诊断（DIAGNOSTIC_ONLY，不参与判定）：GT-近候选/簇/winner 追踪。

    字段（全部为只读统计）：
      num_raw_candidates / num_clusters / winner_support / runner_up_support / winner_margin
      gt_candidate_generated          池里是否存在 near 阈值内的 GT 候选
      gt_nearest_candidate_index      离 GT 最近的候选在 refined 列表里的下标
      gt_nearest_candidate_rank       ↑ 的**求解器排序**名次（1 = 求解器首选）
      gt_nearest_candidate_pose_error 该候选与 GT 的位姿误差（按契约方向）
      gt_nearest_candidate_cluster_rank / gt_cluster_rank / gt_inside_winner_cluster
      selected_candidate_rank         winner 的求解器排序名次
      selected_candidate_cluster_rank / selected_cluster_rank
      selected_pose_error_to_gt       winner 与 GT 的位姿误差

    三类失败的判读（near_* 仅为报告阈值，不参与主逻辑）：
      类型 1 generation：`gt_candidate_generated=False` ⇒ 候选生成/池化阶段就丢了正确解；
      类型 2 scoring-clustering：`gt_candidate_generated=True` 但 `gt_inside_winner_cluster=False`
        （或 `num_clusters>1` 且 winner_margin 小 ⇒ 歧义未决）；
      类型 3 winner-selection：GT 候选已在获胜簇内（`gt_inside_winner_cluster=True`）
        但 `selected_candidate_rank` 与 `gt_nearest_candidate_rank` 不一致。

    ``candidates_are_inverse``：``refined_matrices`` 是否与 ``gt_matrix`` **方向相反**。
    `register()` 的契约方向是 jaw→bite，`register_pair()` 输出是 patch→jaw ⇒ 默认 True（比较前取逆）。
    ⚠️ 若调用方已把 GT 折算到 patch→jaw（patch 自带扰动时必须如此），要传 False ——
    否则那份额外扰动会被当成"solver 位姿误差"记账（实测：patch 加 9°/1.414mm 扰动、
    GT 漏掉该项时诊断报 gt_candidate_generated=False，而该 pair 质量是 rmse 1.5e-15 / inlier 1.000）。
    """
    matrices = pair_result.get("refined_matrices") or []
    cluster_of = pair_result.get("refined_cluster_rank") or []
    clusters = pair_result.get("clusters") or []

    def _to_contract(matrix):
        return np.linalg.inv(np.asarray(matrix, float)) if candidates_are_inverse else np.asarray(matrix, float)

    errors = [_pose_error_to(_to_contract(matrix), gt_matrix) for matrix in matrices]
    report = {"diagnostic_only": True, "num_raw_candidates": pair_result.get("coarse_pool_size"),
              "num_clusters": len(clusters), "winner_support": pair_result.get("winner_cluster_support"),
              "selected_cluster_rank": None, "gt_candidate_generated": False,
              "gt_nearest_candidate_index": None, "gt_nearest_candidate_rank": None,
              "gt_nearest_candidate_pose_error": None, "gt_nearest_candidate_cluster_rank": None,
              "selected_candidate_rank": None, "selected_pose_error_to_gt": None,
              "selected_candidate_cluster_rank": None,
              "gt_cluster_rank": None, "gt_inside_winner_cluster": None,
              "runner_up_support": None, "winner_margin": None}
    if not errors:
        return report
    # ⚠️ **rank 一律指"求解器自己的排序"**，即 `refined_matrices` 的列表序
    # （register_pair 已按 _member_key 排序）。GT 误差只用来**认哪个候选离 GT 近**，不参与排名 ——
    # 之前 gt_nearest_candidate_rank 用列表下标、selected_candidate_rank 用 GT 误差序，两个口径不同。
    def _rank_key(error):
        return (error["rotation_deg"] / max(near_rotation_deg, 1e-9)
                + error["translation_mm"] / max(near_translation_mm, 1e-9))

    best = min(range(len(errors)), key=lambda i: _rank_key(errors[i]))
    report["gt_nearest_candidate_index"] = best
    report["gt_nearest_candidate_rank"] = best + 1
    report["gt_nearest_candidate_pose_error"] = errors[best]
    report["gt_nearest_candidate_cluster_rank"] = cluster_of[best] if best < len(cluster_of) else None
    report["gt_candidate_generated"] = (errors[best]["rotation_deg"] <= near_rotation_deg
                                       and errors[best]["translation_mm"] <= near_translation_mm)
    if clusters:
        gt_cluster = cluster_of[best] if best < len(cluster_of) else -1
        report["gt_cluster_rank"] = gt_cluster if gt_cluster >= 1 else None
        report["gt_inside_winner_cluster"] = (gt_cluster == 1)
        if len(clusters) >= 2:
            report["runner_up_support"] = clusters[1]["member_count"]
            report["winner_margin"] = float(clusters[0]["member_count"] - clusters[1]["member_count"])
    chosen = np.asarray(pair_result.get("matrix"), float)
    chosen_index = next((i for i, m in enumerate(matrices)
                         if np.allclose(np.asarray(m, float), chosen, atol=1e-12)), None)
    report["selected_candidate_rank"] = (chosen_index + 1) if chosen_index is not None else None
    report["selected_candidate_cluster_rank"] = (cluster_of[chosen_index]
                                                 if chosen_index is not None and chosen_index < len(cluster_of) else None)
    report["selected_cluster_rank"] = report["selected_candidate_cluster_rank"] if clusters else None
    report["selected_pose_error_to_gt"] = _pose_error_to(_to_contract(chosen), gt_matrix)
    return report


def translation_mm(matrix: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(matrix)[:3, 3]))


def rotation_deg(matrix: np.ndarray) -> float:
    rotation = np.asarray(matrix)[:3, :3]
    cos = (np.trace(rotation) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


# ------------------------------------------------------------------ 穿模门
def jaw_penetration(transformed_lower, upper, profile: BiteRegistrationProfile):
    """上下颌穿模。

    **口径（本批修正）**：只有**闭合实体**才允许给 authoritative 的 signed 穿透结论。
    开放壳不得用"最近面法向定符号"推穿透 —— 对开放表面任何点都能被算成"里面"，
    据此判 FAILED 是伪结论（实测把正确的 rigid 配准误判成 GROSS_JAW_INTERSECTION）。
    开放壳返回 ``penetration_status=NOT_ASSESSED``，但仍给 **unsigned 邻近诊断**。
    """
    import trimesh
    pts = np.asarray(transformed_lower.triangles_center, float)
    areas = _faces_areas(transformed_lower)
    base = {"definition": PENETRATION_DEFINITION,
            "areas_are_engineering_estimates": AREAS_ARE_ENGINEERING_ESTIMATES,
            "depth_threshold_mm": profile.penetration_depth_mm,
            "gross_penetration_area_mm2": None, "gross_penetration_max_mm": None,
            "penetration_point_ratio": None, "gross": False}
    if not len(pts) or len(upper.faces) == 0:
        base.update(penetration_status=PENETRATION_NOT_ASSESSED, reason=REASON_EMPTY_GEOMETRY)
        return base

    query = trimesh.proximity.ProximityQuery(upper)
    closest, unsigned, triangle_ids = query.on_surface(pts)
    unsigned = np.asarray(unsigned, float)
    base["unsigned_proximity_mm"] = {"p02_mm": float(np.quantile(unsigned, 0.02)),
                                     "min_mm": float(unsigned.min()),
                                     "median_mm": float(np.median(unsigned))}
    base["unsigned_touch_ratio"] = float((unsigned <= profile.penetration_depth_mm).mean())

    solids = bool(getattr(upper, "is_watertight", False)) and bool(getattr(transformed_lower, "is_watertight", False))
    if not solids:
        base.update(penetration_status=PENETRATION_NOT_ASSESSED, reason=REASON_PENETRATION_OPEN_SHELL,
                    note="开放壳不得出 signed 穿透结论；只给 unsigned 邻近诊断")
        return base

    normals = np.asarray(upper.face_normals, float)[np.asarray(triangle_ids, int)]
    signed = np.einsum("ij,ij->i", pts - np.asarray(closest, float), normals)
    depth = -signed
    hit = depth > profile.penetration_depth_mm
    base.update(penetration_status=PENETRATION_MEASURED,
                gross_penetration_area_mm2=float(areas[hit].sum()) if hit.any() else 0.0,
                gross_penetration_max_mm=float(depth[hit].max()) if hit.any() else 0.0,
                penetration_point_ratio=float(hit.mean()),
                gross=bool((depth[hit].max() if hit.any() else 0.0) >= profile.gross_penetration_max_mm
                           or (areas[hit].sum() if hit.any() else 0.0) >= profile.gross_penetration_area_mm2
                           or (hit.mean() if hit.any() else 0.0) >= profile.gross_penetration_ratio))
    return base


def surface_collision_diagnostic(transformed_lower, upper, profile: BiteRegistrationProfile) -> dict:
    """**非签名**表面碰撞诊断（工程警告）。

    与 `jaw_penetration` 分工：后者只在闭合实体上给 authoritative signed 结论；
    本函数任何情况下都给一个与符号无关的告警，且报告里明确 `signed_penetration=False`，
    **不得**被当成穿透深度或生产判据。
    """
    import trimesh
    pts = np.asarray(transformed_lower.triangles_center, float)
    if not len(pts) or len(upper.faces) == 0:
        return {"surface_collision": "NOT_ASSESSED", "signed_penetration": False}
    _, distance, _ = trimesh.proximity.ProximityQuery(upper).on_surface(pts)
    distance = np.asarray(distance, float)
    ratio = float((distance <= profile.collision_proximity_mm).mean())
    return {"surface_collision": "WARN" if ratio >= profile.collision_warn_ratio else "CLEAR",
            "signed_penetration": False, "touching_face_ratio": ratio,
            "unsigned_min_mm": float(distance.min()),
            "unsigned_p02_mm": float(np.quantile(distance, 0.02)),
            "minimum_unsigned_distance_mm": float(distance.min()),
            "near_contact_ratio": ratio,
            "collision_spatial_extent_mm": (np.ptp(np.asarray(transformed_lower.vertices, float)[
                np.flatnonzero(np.any(np.asarray(transformed_lower.faces, int)[
                    distance[:, None].repeat(1, 1)[:0] if False else np.zeros(len(transformed_lower.faces), bool), :], axis=1))],
                axis=0).tolist() if False else [float(np.ptp(np.asarray(transformed_lower.vertices, float)[:, i]))
                                                for i in range(3)]),
            "triangle_intersection_count": _triangle_intersection_count(transformed_lower, upper, profile),
            "authority": "ENGINEERING_DIAGNOSTIC",
            "definition": "unsigned_proximity_below_proximity_mm_ratio",
            "note": "非签名工程警告：不得当作穿透深度或生产判据"}


def _triangle_intersection_count(mesh_a, mesh_b, profile: BiteRegistrationProfile) -> dict:
    """三角相交计数（**工程诊断**，不是 signed penetration）。

    用 AABB 预筛 + 既有 `_tri_tri_intersect` 精确判定；超预算即如实标 NOT_ASSESSED，
    不静默截断成"0 个相交"。
    """
    from .crown_outer.thickness import _tri_tri_intersect
    faces_a = np.asarray(mesh_a.faces, int)
    faces_b = np.asarray(mesh_b.faces, int)
    if not len(faces_a) or not len(faces_b):
        return {"status": "NOT_ASSESSED", "reason": "EMPTY_MESH"}
    tri_a = np.asarray(mesh_a.triangles, float)
    tree = mesh_b.triangles_tree
    pairs = []
    for face in range(len(faces_a)):
        tri = tri_a[face]
        pairs.extend((face, int(other)) for other in
                     tree.intersection(np.r_[tri.min(0) - 1e-8, tri.max(0) + 1e-8]))
        if len(pairs) > profile.collision_intersection_pairs_budget:
            return {"status": "NOT_ASSESSED", "reason": "PAIR_BUDGET_EXCEEDED",
                    "candidate_pairs": len(pairs)}
    if not pairs:
        return {"status": "MEASURED", "triangle_intersection_count": 0, "candidate_pairs": 0}
    array = np.asarray(pairs, int)
    hit, parallel, length = _tri_tri_intersect(np.asarray(mesh_a.triangles, float)[array[:, 0]],
                                               np.asarray(mesh_b.triangles, float)[array[:, 1]],
                                               return_segment=True)
    return {"status": "MEASURED", "triangle_intersection_count": int(hit.sum()),
            "near_parallel_pairs": int(parallel), "candidate_pairs": len(pairs),
            "intersection_segment_total_mm": float(length[hit].sum()),
            "method": "AABB_prefilter_then_exact_triangle_intersection"}


def rigid_transform_report(matrix) -> dict:
    """刚性契约（item 4）：RᵀR≈I、det≈+1、无 scale/shear、底行 [0,0,0,1]。"""
    m = np.asarray(matrix, float)
    ok_shape = m.shape == (4, 4)
    rotation = m[:3, :3] if ok_shape else np.eye(3)
    gram = rotation.T @ rotation
    orthonormal_error = float(np.abs(gram - np.eye(3)).max())
    determinant = float(np.linalg.det(rotation))
    bottom_ok = bool(np.allclose(m[3], [0.0, 0.0, 0.0, 1.0], atol=1e-9)) if ok_shape else False
    scales = np.linalg.svd(rotation, compute_uv=False)
    valid = bool(ok_shape and orthonormal_error <= 1e-6 and abs(determinant - 1.0) <= 1e-6
                 and bottom_ok and float(scales.max() - scales.min()) <= 1e-6)
    return {"rigid_pose_valid": valid, "orthonormal_error": orthonormal_error,
            "determinant": determinant, "bottom_row_ok": bottom_ok,
            "singular_values": [float(v) for v in scales],
            "no_scale_no_shear": bool(float(scales.max() - scales.min()) <= 1e-6),
            "contract": "R.T@R≈I, det≈+1, bottom=[0,0,0,1], no scale/shear"}


def joint_consistency_qc(upper_pair: dict, lower_pair: dict, *, collision: dict | None = None,
                         penetration: dict | None = None, rigid: dict | None = None,
                         profile: BiteRegistrationProfile) -> dict:
    """联合一致性 QC（P0-1 修正版）。

    **禁止跨物理表面 RMSE**。所有 pair measurement 统一在 **Bite Frame** 里做：
        UpperJaw --T_upper_to_bite--> Bite Frame ↔ Upper Bite Patch
        LowerJaw --T_lower_to_bite--> Bite Frame ↔ Lower Bite Patch
    不同物理表面（lower patch 与 upper jaw 这类）之间**永不计算 RMSE**。
    上一版把 lower patch 经 T_lower_to_upper 搬到 upper jaw 上量点面 RMSE —— 那是跨表面，
    在真实解剖上无意义（只在同轴圆柱夹具里偶然很小），已删除。

    本 QC 只**聚合**已存在的证据：
      1. upper pair quality（patch↔自己的 jaw）
      2. lower pair quality（patch↔自己的 jaw）
      3. spatial coverage（两侧 coverage / spread 是否足以 pin 住位姿）
      4. rigid-transform validity（刚性契约）
      5. jaw relationship plausibility（咬合关系：座位带 + 碰撞/穿透门）
    """
    reasons = []

    def side_block(pair, name):
        if not pair or pair.get("rmse_mm") is None or pair.get("inlier_ratio") is None:
            reasons.append("%s_SIDE_UNAVAILABLE" % name.upper())
            return "FAILED", {}
        rmse, inlier = pair["rmse_mm"], pair["inlier_ratio"]
        overlap = pair.get("overlap_area_mm2") or 0.0
        src_cov = pair.get("source_coverage_ratio") or 0.0
        tgt_cov = pair.get("target_coverage_ratio") or 0.0
        if rmse > profile.review_rmse_mm or inlier < profile.review_inlier_ratio:
            return "FAILED", {"rmse_mm": rmse, "inlier_ratio": inlier}
        if overlap < profile.min_overlap_area_mm2:
            reasons.append("%s_OVERLAP_BELOW_MINIMUM" % name.upper())
            return "FAILED", {"rmse_mm": rmse, "inlier_ratio": inlier, "overlap_area_mm2": overlap}
        passes = (rmse <= profile.pass_rmse_mm and inlier >= profile.pass_inlier_ratio)
        # ⚠️ 规格 item 2/5：`source_coverage_ratio` / `whole_target_coverage_ratio` **只作 diagnostic**，
        # 禁止参与 PASS/REVIEW/FAILED，也禁止拿 inlier 阈值当 coverage 阈值（两者不是同一物理量）。
        # 之前用 coverage_ok 参与判定，导致 well-posed 正确解也被降级成 REVIEW（false REVIEW）。
        return ("PASS" if passes else "REVIEW_REQUIRED"), {
            "rmse_mm": rmse, "inlier_ratio": inlier, "overlap_area_mm2": overlap,
            "source_coverage_ratio": src_cov, "target_coverage_ratio": tgt_cov}

    upper_status, upper_block = side_block(upper_pair, "upper")
    lower_status, lower_block = side_block(lower_pair, "lower")

    if rigid is not None and not rigid.get("rigid_pose_valid", False):
        reasons.append("RIGID_CONTRACT_VIOLATED")
        return {"status": "FAILED", "reasons": reasons + ["RIGID_CONTRACT_VIOLATED"],
                "upper": upper_block, "lower": lower_block, "rigid": rigid,
                "no_cross_surface_rmse": True}

    # jaw relationship plausibility：用**已存在**的穿透/碰撞门，不新造跨面度量
    pen = penetration or {}
    col = collision or {}
    if pen.get("penetration_status") == PENETRATION_MEASURED and pen.get("gross"):
        return {"status": "FAILED", "reasons": reasons + ["GROSS_PENETRATION"],
                "upper": upper_block, "lower": lower_block, "rigid": rigid,
                "no_cross_surface_rmse": True}
    if col.get("surface_collision") == "WARN":
        reasons.append("SURFACE_COLLISION_WARN")

    if upper_status == "FAILED" or lower_status == "FAILED":
        status = "FAILED"
    elif upper_status == "PASS" and lower_status == "PASS" and not reasons:
        status = "PASS"
    else:
        status = "REVIEW_REQUIRED"
    return {"status": status, "reasons": reasons,
            "upper": upper_block, "lower": lower_block, "rigid": rigid,
            "collision_authority": col.get("authority"),
            "no_cross_surface_rmse": True,
            "note": "所有 pair measurement 在 Bite Frame 内完成；不同物理表面之间不算 RMSE"}


# ------------------------------------------------------------------ 编排
def register(bite, upper, lower, profile: BiteRegistrationProfile | None = None, *,
             side_hint: dict | None = None, upper_meta: dict | None = None, lower_meta: dict | None = None):
    """完整流程：预处理 → 分区 → 分侧配准 → 组合 → QC。

    返回 `BiteRegistrationResult` 形状的 dict；**status 三档**
    `PASS / REVIEW_REQUIRED / FAILED`，reason 说明为什么。绝不只返回一个矩阵。
    """
    profile = profile or BiteRegistrationProfile()
    result = {"algorithm_version": ALGORITHM_VERSION, "profile": profile.as_report(),
              "status": STATUS_FAILED, "reason": None,
              "T_upper_to_bite": None, "T_lower_to_bite": None, "T_lower_to_upper": None,
              "upper_match_rmse_mm": None, "lower_match_rmse_mm": None,
              "upper_inlier_ratio": None, "lower_inlier_ratio": None,
              "upper_overlap_area_mm2": None, "lower_overlap_area_mm2": None,
              "gross_penetration_area_mm2": None, "gross_penetration_max_mm": None,
              "confidence": None, "side_assignment": None,
              "notes": ["overlap/penetration 为工程口径估计，未做制造认证",
                        "exocad 已注册结果只能作位置域参考，不能作本 solver 的输入"]}

    if bite is None:
        result.update(status=STATUS_FAILED, reason=REASON_BITE_SCAN_MISSING)
        return result
    if upper is None or lower is None:
        result.update(status=STATUS_FAILED, reason=REASON_MANUAL_BITE_REQUIRED)
        return result

    clean_bite = preprocess(bite, profile)
    clean_upper = preprocess(upper, profile)
    clean_lower = preprocess(lower, profile)

    sides = assign_sides(clean_bite, clean_upper, clean_lower, profile, hint=side_hint)
    result["side_assignment"] = {k: v for k, v in sides.items()
                                 if k in ("method", "ambiguous", "candidates", "chosen", "reason", "scores", "dropped_components")}
    if sides.get("ambiguous"):
        result.update(status=STATUS_REVIEW, reason=REASON_BITE_SIDE_AMBIGUOUS)
        return result
    if sides.get("upper") is None or sides.get("lower") is None:
        if sides.get("reason") == REASON_BITE_SCAN_EMPTY:
            result.update(status=STATUS_FAILED, reason=REASON_BITE_SCAN_EMPTY)
        else:
            # 只有一个连通分量：按最近侧归位后，另一侧无法可靠匹配
            part = sides.get("upper") or sides.get("lower")
            if part is None:
                result.update(status=STATUS_FAILED, reason=REASON_BITE_SCAN_UNREADABLE)
                return result
            if sides.get("upper") is not None:
                upper_pair = register_pair(part, clean_upper, profile)
                result.update(T_upper_to_bite=np.linalg.inv(np.asarray(upper_pair["matrix"], float)), upper_match_rmse_mm=upper_pair["rmse_mm"],
                              upper_inlier_ratio=upper_pair["inlier_ratio"],
                              upper_overlap_area_mm2=upper_pair["overlap_area_mm2"],
                              status=STATUS_FAILED, reason=REASON_LOWER_SIDE_UNMATCHABLE)
            else:
                lower_pair = register_pair(part, clean_lower, profile)
                result.update(T_lower_to_bite=np.linalg.inv(np.asarray(lower_pair["matrix"], float)), lower_match_rmse_mm=lower_pair["rmse_mm"],
                              lower_inlier_ratio=lower_pair["inlier_ratio"],
                              lower_overlap_area_mm2=lower_pair["overlap_area_mm2"],
                              status=STATUS_FAILED, reason=REASON_UPPER_SIDE_UNMATCHABLE)
            return result

    upper_pair = sides["upper"]
    lower_pair = sides["lower"]
    # ---- D1 修复：契约方向 ----
    # `register_pair(patch, jaw)` 返回 T_patch_to_jaw；契约要的是 T_jaw_to_bite。
    # 只做一次求逆（不换参数序：partial patch / full jaw 的对应关系是非对称的）。
    t_upper_to_bite = np.linalg.inv(np.asarray(upper_pair["matrix"], float))
    t_lower_to_bite = np.linalg.inv(np.asarray(lower_pair["matrix"], float))
    t_lower_to_upper = lower_to_upper_from_bite(t_upper_to_bite, t_lower_to_bite)
    rigid_contract = rigid_transform_report(t_lower_to_upper)
    lower_in_upper = lower.copy()
    lower_in_upper.apply_transform(t_lower_to_upper)
    penetration = jaw_penetration(lower_in_upper, clean_upper, profile)
    collision = surface_collision_diagnostic(lower_in_upper, clean_upper, profile)
    joint = joint_consistency_qc(upper_pair, lower_pair, collision=collision,
                                 penetration=penetration, rigid=rigid_contract, profile=profile)

    result.update(
        T_upper_to_bite=t_upper_to_bite, T_lower_to_bite=t_lower_to_bite,
        T_lower_to_upper=t_lower_to_upper, rigid_contract=rigid_contract,
        upper_match_rmse_mm=upper_pair["rmse_mm"], lower_match_rmse_mm=lower_pair["rmse_mm"],
        upper_inlier_ratio=upper_pair["inlier_ratio"], lower_inlier_ratio=lower_pair["inlier_ratio"],
        upper_overlap_area_mm2=upper_pair["overlap_area_mm2"], lower_overlap_area_mm2=lower_pair["overlap_area_mm2"],
        gross_penetration_area_mm2=penetration["gross_penetration_area_mm2"],
        gross_penetration_max_mm=penetration["gross_penetration_max_mm"],
        penetration=penetration, surface_collision=collision, joint_consistency=joint,
        transform_magnitude={"translation_mm": translation_mm(t_lower_to_upper),
                             "rotation_deg": rotation_deg(t_lower_to_upper)},
        per_side={"upper": dict(upper_pair),
                  "lower": dict(lower_pair)})

    status, reason, confidence = _verdict(result, profile)
    result.update(status=status, reason=reason, confidence=confidence)
    return result


def _verdict(result: dict, profile: BiteRegistrationProfile):
    """三档判定（定义冻结）。

        PASS   = upper PASS && lower PASS && rigid pose valid && no fatal gate
        REVIEW = upper/lower 都至少达到 REVIEW minimum && no fatal && 至少一侧未 PASS
        FAILED = 任一 side 低于 REVIEW minimum OR inconsistent bite evidence OR fatal collision/penetration
    """
    penetration = result.get("penetration") or {}
    if penetration.get("penetration_status") == PENETRATION_MEASURED and penetration.get("gross"):
        return STATUS_FAILED, REASON_GROSS_JAW_INTERSECTION, 0.0

    sides = result.get("per_side") or {}
    pairs = [sides.get("upper") or {}, sides.get("lower") or {}]
    if any(p.get("rmse_mm") is None or p.get("inlier_ratio") is None for p in pairs):
        return STATUS_FAILED, REASON_SIDE_UNAVAILABLE, 0.0

    def side_review_min(p):
        return p["rmse_mm"] <= profile.review_rmse_mm and p["inlier_ratio"] >= profile.review_inlier_ratio

    if not all(side_review_min(p) for p in pairs):
        return STATUS_FAILED, REASON_MATCH_BELOW_MINIMUM, 0.1

    if any((sides.get(side) or {}).get("insufficient_evidence") for side in ("upper", "lower")):
        # 某个自由度存在近连续等价位姿（弱约束）⇒ 不得 PASS
        return STATUS_REVIEW, REASON_INSUFFICIENT_REGISTRATION_EVIDENCE, 0.25
    if any((sides.get(side) or {}).get("ambiguous_solution") for side in ("upper", "lower")):
        # 多个高质量候选分数接近但位姿明显不同 —— 不得取第一名当 PASS
        return STATUS_REVIEW, REASON_SOLUTION_AMBIGUOUS, 0.3
    joint = result.get("joint_consistency") or {}
    joint_status = joint.get("status")
    rigid = result.get("rigid_contract") or {}
    if not rigid.get("rigid_pose_valid", False):
        return STATUS_FAILED, REASON_RIGID_CONTRACT_VIOLATED, 0.0
    if joint_status == "FAILED":
        # 两侧各自还行，但联合证据互相矛盾（覆盖率不足 / 刚性不成立 / 座位带不对）
        return STATUS_FAILED, REASON_INCONSISTENT_BITE_EVIDENCE, 0.15
    if joint_status == "PASS":
        worst_rmse = max(p["rmse_mm"] for p in pairs)
        worst_inlier = min(p["inlier_ratio"] for p in pairs)
        confidence = float(min(1.0, (worst_inlier / max(profile.pass_inlier_ratio, 1e-6)) * 0.5
                               + (1.0 - worst_rmse / max(profile.pass_rmse_mm, 1e-6)) * 0.5))
        return STATUS_PASS, None, max(0.0, min(1.0, confidence))
    return STATUS_REVIEW, REASON_LOCAL_MISMATCH + ":" + "+".join(joint.get("reasons") or ["JOINT_REVIEW"]), 0.5


def three_point_transform(source_points, target_points) -> np.ndarray:
    """三点辅助：由 ≥3 对对应点求刚体变换（Kabsch + SVD），供后续 ICP 细化。

    用户不需要自己调矩阵；这里只做 landmark rigid，**不当作 final**。
    """
    src = np.asarray(source_points, float)
    dst = np.asarray(target_points, float)
    if src.shape != dst.shape or src.shape[0] < 3:
        raise ValueError("至少需要 3 对对应点，且两侧点数一致")
    src_centre, dst_centre = src.mean(axis=0), dst.mean(axis=0)
    covariance = (src - src_centre).T @ (dst - dst_centre)
    u, _, vt = np.linalg.svd(covariance)
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0:
        vt[-1] *= -1
        rotation = vt.T @ u.T
    return _rigid(rotation, dst_centre - rotation @ src_centre)
