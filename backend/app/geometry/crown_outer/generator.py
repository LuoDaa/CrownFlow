"""CrownOuter v0 生成器：把 placement → correspondence → Laplacian 串起来。

顶层 API（方案 §12.1）::

    generate_crown_outer_v0(template, patient_margin, insertion_axis,
                            *, method="laplacian", config=None) -> CrownOuterResult

不做什么（方案 §2.2）：接触、咬合、厚度、Intaglio 合成、AI。
每一步 fail-loud，无静默 fallback。
"""
from __future__ import annotations

import numpy as np

from . import boundary as bd
from . import quality as qc
from .laplacian import DeformInputs, solve_laplacian
from .models import (
    STATUS_FAILED,
    STATUS_OK,
    STATUS_REVIEW_REQUIRED,
    CrownOuterConfig,
    CrownOuterResult,
)
from .patient_frame import PatientFrame, build_patient_frame
from .placement import patient_margin_perimeter, place_template
from .template import CrownTemplate


def generate_crown_outer_v0(
    template: CrownTemplate,
    patient_margin: np.ndarray,
    insertion_axis: np.ndarray,
    *,
    method: str = "laplacian",
    config: CrownOuterConfig | None = None,
    md_scale: float | None = None,
    md_body_keep: float | None = None,
    patient_frame: PatientFrame | None = None,
) -> CrownOuterResult:
    """...（H.1 原流程）

    ``md_scale``（**H.2**）：非 None 时把"近远中（模板局部 X）"这一轴
    的缩放从均匀缩放改成 ``scale·md_scale``，颊舌/咬合两轴保持 ``scale``。
    用于让解剖先验**按邻牙可用间隙定宽**，而不是按 margin 周长等比放大。

    ``patient_frame``（Crown 1.0）：调用方已经有人工/上游确认的近中、颊侧、就位方向时，
    必须复用该帧，避免重新从近似椭圆 Margin 的 PCA 猜 X/Y 导致 90°/180° 旋转二义。
    Margin 本身仍由下面的 hard Laplacian boundary constraint 逐点锁定。
    """
    if method != "laplacian":
        raise ValueError("v0 只实现 laplacian；arap 是条件升级项（方案 §9.4）")
    config = config or CrownOuterConfig()
    res = CrownOuterResult(status=STATUS_FAILED)

    try:
        margin = bd.ensure_closed_ring_order(patient_margin)
        perim = patient_margin_perimeter(margin)

        # ---- C: patient frame + similarity placement
        pf = patient_frame if patient_frame is not None else build_patient_frame(margin, insertion_axis)
        placement = place_template(template, pf, perim)
        if md_scale is not None:
            sv = placement.svec().copy()
            sv[0] = sv[0] * float(md_scale)      # 局部 X = 近远中
            placement.scale_vec = sv
        placed = (placement.apply_body_decoupled(
            template.vertices, template.margin_loop_indices,
            template.crown_height_mm, md_body_keep=md_body_keep)
            if config.body_scale_decoupling
            else placement.apply(template.vertices))
        res.placement_transform = placement.matrix()
        res.uniform_scale = round(placement.scale, 5)
        if md_scale is not None:
            res.mesh_qc["anisotropic_scale"] = [round(float(x), 5)
                                                for x in placement.svec()]

        # ---- D: 闭环对应（forward/reverse + cyclic shift）
        loop_pts = placed[template.margin_loop_indices]
        corr = bd.align_rings(loop_pts, margin, config.resample_n)
        res.boundary_shift = corr.shift
        res.boundary_reversed = corr.reversed_
        targets = bd.targets_for_loop_vertices(loop_pts, corr)

        # ---- E: Laplacian 编辑
        loop_idx = template.margin_loop_indices
        root_idx = np.flatnonzero(template.root_vertex_mask)
        crown_idx = np.flatnonzero(template.crown_vertex_mask)
        # 咬合弱锚：冠部 z 高分位（不含颈环顶点）
        z_thr = np.percentile(placed[crown_idx, 2],
                              config.occlusal_anchor_percentile)
        occ_idx = crown_idx[placed[crown_idx, 2] >= z_thr]
        deformed, solve_info = solve_laplacian(DeformInputs(
            vertices_placed=placed, faces=template.faces,
            margin_vertex_indices=loop_idx, margin_targets=targets,
            root_vertex_indices=root_idx, occlusal_vertex_indices=occ_idx,
            config=config))
        res.mesh_qc["solver"] = solve_info

        # ---- F: QC
        regions = qc.region_masks(placed, template.crown_vertex_mask,
                                  placed[loop_idx], config)
        res.displacement_by_region = qc.displacement_by_region(
            placed, deformed, regions)
        res.margin_error_median_mm = None
        fit = qc.margin_fit(deformed[loop_idx], margin)
        res.margin_error_median_mm = fit["symmetric"]["median"]
        res.margin_error_p95_mm = fit["symmetric"]["p95"]
        res.margin_error_max_mm = fit["symmetric"]["max"]
        res.mesh_qc["margin_fit"] = fit

        # 输出面 = 冠部 + 颈环（开放式外表面，底边即 Margin）：
        # 只排除**触碰根部**的面 —— 颈环顶点必须留在输出里，否则切口
        # 悬在半空（实测：按 crown_mask 选面会丢掉整个颈环圈，
        # 边界边变成 148 条不在颈环上的悬空边）。
        out_faces_mask = ~template.root_vertex_mask[template.faces].any(axis=1)
        out_faces = template.faces[out_faces_mask]
        mq = qc.mesh_quality(deformed, out_faces,
                             expected_boundary_count=len(loop_idx),
                             vertices_reference=placed)
        # 边界边必须**全部**落在颈环顶点上（否则切口不在 Margin 处）
        boundary_on_loop = _boundary_edges_on_loop(out_faces, loop_idx)
        mq["boundary_all_on_loop"] = boundary_on_loop
        if not boundary_on_loop:
            mq["boundary_consistent"] = False
        res.mesh_qc["mesh"] = mq

        status, notes = qc.decide_status(fit, res.displacement_by_region,
                                         mq, config)
        if solve_info["margin_constraint_residual_max_mm"] > 0.05:
            notes.append("margin 约束残差 %.4f 超过 0.05mm"
                         % solve_info["margin_constraint_residual_max_mm"])
            status = STATUS_REVIEW_REQUIRED
        res.notes = notes
        res.status = status

        # ---- 结果网格（trimesh 延迟导入：调用方没有 trimesh 也能跑纯计算）
        import trimesh
        res.mesh = trimesh.Trimesh(vertices=deformed, faces=out_faces,
                                   process=False)
        res.mesh_qc["n_output_vertices"] = int(len(deformed))
        res.mesh_qc["n_output_faces"] = int(len(out_faces))
        return res
    except Exception as exc:                     # noqa: BLE001  fail-loud 上抛前打标
        res.status = STATUS_FAILED
        res.notes = res.notes + [f"{type(exc).__name__}: {exc}"]
        raise


def _boundary_edges_on_loop(out_faces: np.ndarray, loop_idx: np.ndarray
                            ) -> bool:
    """开放面片的边界边（只出现一次的边）是否全部连接两个颈环顶点。"""
    from collections import Counter
    loop = set(int(i) for i in loop_idx)
    cnt: Counter = Counter()
    for a, b, c in out_faces:
        for u, v in ((a, b), (b, c), (c, a)):
            cnt[(min(u, v), max(u, v))] += 1
    boundary = [e for e, n in cnt.items() if n == 1]
    return bool(boundary) and all(u in loop and v in loop for u, v in boundary)
