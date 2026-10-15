"""Laplacian Mesh Editing v0.1（方案 §9，Sorkine 式微分坐标）。

**为什么是行替换而不是 LᵀL 正规方程**（本文件的第一版踩过的坑）：

正规方程 ``min ||L V′ − L V_placed||² + λ·约束`` 是双调和能量，**没有极大值
原理**，而且颈环顶点的 delta 行会把"边界顶点与邻居的微分关系"也当成强约束
——真实 FDI46 上，边界只动 1.5~3mm，自由内部却被拖出 11~20mm、39% 面翻转
（求解器残差 1e-8、目标函数确实更小 —— 数学上"正确"，几何上是灾难）。
行替换法（边界行改成恒等约束）是标准做法，实测 0% 翻转、内部位移衰减正常。

目标（对 V′ 的三个坐标分量分别求解）::

    L V′ = Δ,  其中 Δ = L V_placed
    颈环行替换为恒等：V′[loop] = margin_targets        （硬约束）
    根部行替换为恒等：V′[root] = V_placed[root]        （根不参与形变）
    咬合区弱锚行（超定，lsqr 求解）：α·V′[occ] ≈ α·V_placed[occ]

区域刚度策略（方案 §9.3）由"谁被硬约束、谁被弱锚"实现：
颈部硬约束（追随患者 Margin）→ 轴壁自由传播 → 咬合弱锚（保护解剖形态）。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .models import CrownOuterConfig


class LaplacianError(ValueError):
    pass


@dataclass
class DeformInputs:
    """一次形变的全部输入（由 generator 组装）。"""

    vertices_placed: np.ndarray       # (V,3) 摆位后的模板顶点
    faces: np.ndarray                 # (F,3)
    margin_vertex_indices: np.ndarray  # (L,) 有序牙颈环
    margin_targets: np.ndarray        # (L,3) 逐顶点目标
    root_vertex_indices: np.ndarray   # 根部锚点（可为空）
    occlusal_vertex_indices: np.ndarray
    config: CrownOuterConfig
    #: **H.2.1 proximal clearance 约束**（可选；None/空 = 与 H.1 逐位等价）。
    #: 由调用方按 **active set** 组装：只把"真的穿透/过近"的顶点放进来
    #: （单边不等式 → 只推不拉），目标 = q + n_smooth·clearance。
    proximal_vertex_indices: np.ndarray | None = None
    proximal_targets: np.ndarray | None = None
    proximal_weight: float | None = None
    #: **逐行权重**（长度 = proximal_vertex_indices）：1.0 = 全强度，
    #: 衰减值 = protection ring（把形变平滑拖进周围解剖）。
    proximal_row_weights: np.ndarray | None = None


def cotangent_laplacian(vertices: np.ndarray, faces: np.ndarray
                        ) -> sp.csr_matrix:
    """对称 cotangent 权重 Laplacian：L[i,j] = −w_ij（i≠j），L[i,i] = Σw。

    权重 w_ij = 0.5·(cot α + cot β)（α、β 为边 (i,j) 的对角）。
    钝角可能给负权重，clamp 到 ≥0（标准做法；负权会让系统失去凸性）。
    退化三角形（面积 ≈ 0）贡献 0。
    """
    v = vertices
    tri = np.asarray(faces, dtype=np.int64)
    n = len(v)
    rows, cols, vals = [], [], []
    for k in range(3):
        i, j, a = tri[:, k], tri[:, (k + 1) % 3], tri[:, (k + 2) % 3]
        e1 = v[i] - v[a]
        e2 = v[j] - v[a]
        dot = np.einsum("ij,ij->i", e1, e2)
        cross = np.cross(e1, e2)
        s = np.linalg.norm(cross, axis=1)
        cot = np.where(s > 1e-12, dot / s, 0.0)
        cot = np.clip(cot, 0.0, None) * 0.5
        rows += [i, j]
        cols += [j, i]
        vals += [cot, cot]
    w = sp.coo_matrix((np.concatenate(vals),
                       (np.concatenate(rows), np.concatenate(cols))),
                      shape=(n, n)).tocsr()
    w.sum_duplicates()
    deg = np.asarray(w.sum(axis=1)).ravel()
    return (sp.diags(deg) - w).tocsr()


def solve_laplacian(inp: DeformInputs) -> tuple[np.ndarray, dict]:
    """行替换 Laplacian 编辑。返回 (V′, info)。

    无弱锚行时走平方系统 LU（精确解）；有弱锚行走 lsqr（超定）。
    """
    v0 = np.asarray(inp.vertices_placed, dtype=np.float64)
    n = len(v0)
    if n == 0 or len(inp.faces) == 0:
        raise LaplacianError("空网格")
    if len(inp.margin_vertex_indices) < 8:
        raise LaplacianError("牙颈环顶点过少，无法建立边界约束")
    if len(inp.margin_targets) != len(inp.margin_vertex_indices):
        raise LaplacianError("margin 目标数与环顶点数不一致")
    loop = np.asarray(inp.margin_vertex_indices, dtype=np.int64)
    root = np.asarray(inp.root_vertex_indices, dtype=np.int64)
    occ = np.asarray(inp.occlusal_vertex_indices, dtype=np.int64)
    if np.intersect1d(loop, root).size:
        raise LaplacianError("牙颈环与根部锚点重叠 —— 冠/根切分有问题")

    l_mat = cotangent_laplacian(v0, inp.faces)
    delta = l_mat @ v0                                  # (V,3) 微分坐标

    # ---- 三段式组装（避免 LSQ 把硬约束残差摊给 L 行）：
    #   [W · I_hard]   = W · targets   硬约束（接触区/颈环 LOCK），权重大
    #   [L_free]       = delta_free    自由顶点的微分坐标保持
    #   [α · I_occ]    ≈ α · v0[occ]   咬合弱锚（soft）
    # 为什么硬行要加权：delta 是**旧边界**下的微分坐标，钉死新边界后
    # "自由 L 行 = delta" 子系统可能不相容；不加权的 LSQ 会把矛盾摊到
    # 硬约束行上（实测残差 0.12mm）。W 把硬残差压到 W⁻² 量级。
    cfg = inp.config
    #: 硬约束行权重。约束行很多时（I.1 精修实测 1290 个接触约束）lsqr 的
    #: 折中位会抬到 0.02mm，超过 0.01 阈值 —— 提高权重把残差压下去。
    #: 1e5 + lsqr(atol=1e-14, iter=200k) 曾把 **H.1 生成本身**搞坏
    #: （残差 1e90）—— 容差过紧会让 lsqr 跑满迭代后返回劣质解。
    #: 回到 1e4 / 1e-12 / 60000，这对接触约束规模已足够。
    weight_hard = float(getattr(cfg, "weight_hard_override", 0.0)) or 1.0e4
    alpha = cfg.weight_occlusal_anchor
    # 硬约束顶点**按顶点去重**：调用方可能把同一顶点同时放进 margin 与 root
    # （例如把颈环并进 LOCK 集合）—— 重复行会让矩阵奇异（实测 splu
    # "Factor is exactly singular"）。目标按"margin 优先、其余用当前位置"合并。
    tgt_by_v = {int(v): inp.margin_targets[k] for k, v in enumerate(loop)}
    hard = np.unique(np.concatenate([loop, root])) if len(root)         else np.unique(loop)
    hard_set = np.zeros(n, dtype=bool)
    hard_set[hard] = True
    free_rows = np.flatnonzero(~hard_set)
    hard_targets = np.asarray(
        [tgt_by_v.get(int(v), v0[v]) for v in hard], dtype=np.float64)

    eye = sp.identity(n, format="csr")
    blocks = [weight_hard * eye[hard], l_mat.tocsr()[free_rows]]
    # 注意 rhs 硬行必须同步乘 weight_hard（否则解 = target/W，残差巨大）
    rhs_blocks = [weight_hard * hard_targets, delta[free_rows]]
    weak = 0
    if len(occ) and alpha > 0:
        blocks.append(alpha * eye[occ])
        rhs_blocks.append(alpha * v0[occ])
        weak = len(occ)
    # ---- H.2.1：proximal clearance 约束（**新的一组 rows，不动上面的语义**）
    # 权重低于 margin（margin 最高优先级）；只对 active set 生效（单边不等式
    # 的近似：调用方只把"真穿透"的顶点送进来 → 只推不拉，不会把正常 proximal
    # anatomy 吸成邻牙反模）。
    prox = np.asarray(inp.proximal_vertex_indices, dtype=np.int64) \
        if inp.proximal_vertex_indices is not None \
        else np.zeros(0, dtype=np.int64)
    n_prox = 0
    if len(prox):
        if inp.proximal_targets is None \
                or len(inp.proximal_targets) != len(prox):
            raise LaplacianError("proximal 约束的顶点数与目标数不一致")
        wp = float(inp.proximal_weight if inp.proximal_weight is not None
                   else cfg.weight_proximal_clearance)
        tg_p_all = np.asarray(inp.proximal_targets, dtype=np.float64)
        w_all = (np.asarray(inp.proximal_row_weights, dtype=np.float64)
                 if inp.proximal_row_weights is not None else None)
        if not np.intersect1d(prox, hard).size == 0:
            # 与硬约束重叠时**硬约束赢**（margin 最高优先级），这里直接剔除
            keep = ~np.isin(prox, hard)
            prox = prox[keep]
            tg_p = tg_p_all[keep]
            if w_all is not None:
                w_all = w_all[keep]
        else:
            tg_p = tg_p_all
        if len(prox):
            # **逐行权重**（分级约束 / protection rings）：权重 1.0 = 全强度贴到
            # clearance envelope；衰减权重 = 把形变平滑拖进周围解剖 —— 活动边界
            # 不再有位移突变（实测二值 active set 会把 1mm 位移的应变全压在边界上，
            # α=1 时 67 片翻面，α=1/16 仍有真折叠）。
            if w_all is not None:
                blocks.append(sp.diags(wp * w_all) @ eye[prox])
                rhs_blocks.append((wp * w_all)[:, None] * tg_p)
            else:
                blocks.append(wp * eye[prox])
                rhs_blocks.append(wp * tg_p)
            weak += len(prox)
            n_prox = int(len(prox))
    a_mat = sp.vstack(blocks).tocsr().tocsc()
    rhs_all = np.vstack(rhs_blocks)

    if weak == 0:
        # 无弱锚行 → 方阵，直接 LU（硬约束精确满足）
        if a_mat.shape != (n, n):
            raise LaplacianError(
                "方阵路径要求 n×n，实际 %s（n=%d, |hard|=%d, |free|=%d）"
                % (a_mat.shape, n, len(hard), len(free_rows)))
        # 对角正则化：自由行的 Laplacian 子块在极端情形（孤立顶点/权全 0）
        # 会让方阵奇异（实测 splu "Factor is exactly singular"）。
        # 1e-10 相对硬行权重 1e5 无物理影响。
        solver = spla.splu(a_mat + sp.identity(n, format="csc") * 1e-10)
        v_new = np.column_stack(
            [solver.solve(np.ascontiguousarray(rhs_all[:, d])) for d in range(3)])
        solver_name = "splu"
    else:
        v_new = np.column_stack([
            spla.lsqr(a_mat, np.ascontiguousarray(rhs_all[:, d]),
                      atol=1e-12, btol=1e-12, iter_lim=60000)[0]
            for d in range(3)])
        solver_name = "lsqr"

    info = {
        "solver": solver_name,
        "n_vertices": n,
        "n_margin_constrained": int(len(loop)),
        "n_root_anchored": int(len(root)),
        "n_occlusal_anchored": int(weak) - int(locals().get("n_prox", 0)),
        "n_proximal_constrained": int(locals().get("n_prox", 0)),
        "weight_hard": weight_hard,
        "matrix_rows": int(a_mat.shape[0]),
    }
    if not np.isfinite(v_new).all():
        raise LaplacianError("Laplacian 求解出现非有限值（数值发散）")
    # 边界约束必须真的命中 —— 求解器自报成功不算（方案 §10.1）
    res = float(np.max(np.linalg.norm(
        v_new[loop] - inp.margin_targets, axis=1)))
    info["margin_constraint_residual_max_mm"] = round(res, 5)
    limit = float(getattr(cfg, "max_constraint_residual_mm", 0.01))
    if res > limit:
        raise LaplacianError(
            "边界约束残差 %.4fmm 超过上限 %.4fmm —— 求解失败" % (res, limit))
    return v_new, info



class LocalBandError(Exception):
    """局部带状求解的输入错误。"""


def solve_local_band(v0, faces, free_ids, fixed_ids, fixed_positions):
    """Contact v2：**局部带状** Laplacian 求解 —— 咬合/颈缘位移由构造精确为 0。

    Contact v1/I.1 的咬合拖动（实测 1.08mm）根源：全局系统里咬合只是弱锚
    （α=300 对硬约束 W=1e4 无抵抗力），接触区一动，"自由行 = 旧微分坐标"
    把整个曲面拖走。

    这里改为：**只有**接触带 + 少量过渡环是自由变量；其余全部顶点（咬合、
    颈缘、远场）是固定边界，进右端项 —— 它们不在未知数里，位移精确为 0。

        L[free,free] · x_free = delta[free] − L[free,fixed] · x_fixed

    其中 delta = L·v0（旧位形微分坐标），x_fixed = 固定顶点目标（接触区顶点
    在接触带内则取其 target，其余取原位）。方阵 → splu 精确解。

    ``fixed_positions`` 必须覆盖全部非 free 顶点（通常 = v0 的副本，
    再把接触区顶点覆写为 target）。
    """
    import numpy as np
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla

    v0 = np.asarray(v0, dtype=np.float64)
    n = len(v0)
    free = np.asarray(sorted(set(int(i) for i in free_ids)), dtype=np.int64)
    if len(free) == 0:
        return v0.copy(), {"n_free": 0}
    fixed = np.asarray(sorted(set(int(i) for i in fixed_ids)), dtype=np.int64)
    if len(free) + len(fixed) != n:
        raise LocalBandError(
            "free(%d) + fixed(%d) != V(%d) —— 两个集合必须恰好划分全部顶点"
            % (len(free), len(fixed), n))
    x_fixed = np.asarray(fixed_positions, dtype=np.float64)
    if x_fixed.shape != (n, 3):
        raise LocalBandError("fixed_positions 形状应为 (V,3)")

    l_mat = cotangent_laplacian(v0, faces)
    delta = l_mat @ v0
    l_ff = l_mat[free][:, free].tocsr()
    l_fx = l_mat[free][:, fixed].tocsr()
    rhs = delta[free] - l_fx @ x_fixed[fixed]
    # 对角正则：孤立/零权顶点会让 splu 奇异（I.1 期踩过）
    a = l_ff + sp.identity(len(free), format="csr") * 1e-10
    try:
        lu = spla.splu(a.tocsc())
        x_free = lu.solve(rhs)
    except Exception as exc:                       # noqa: BLE001
        raise LocalBandError("局部系统求解失败: %s" % exc) from exc
    if not np.all(np.isfinite(x_free)):
        raise LocalBandError("局部系统解出非有限值（网格退化？）")
    out = x_fixed.copy()
    out[free] = x_free
    return out, {"n_free": int(len(free)), "n_fixed": int(len(fixed))}
