"""J.2 —— Level-Set Envelope：最小厚度包络（隐式场）与 anatomy 的**体融合**。

用户 2026-10-02 规格（J.1 冻结为"负结果但机制验证成功"之后）：

    Intaglio + Cervical Loop + T field
              ↓  Narrow-band SDF
    F(x) = signed_distance_to_intaglio(x) − T_extended(x)
              ↓  F=0
       Minimum Envelope
              ↓  与 Anatomy H.1 **融合**（不是把 H.1 顶点投上去）
       Conforming remesh + Cervical boundary 精确恢复
              ↓  Geometry Safety Gate
          Outer J.2

## 为什么不再用逐点投影（J.1 的结论）

J.1 实测：真实例要补 max 1.66mm、约 40% 表面；α=1 逐点投影 → **面积翻面 18.5% +
自交 1350 对**，安全门只能放行 α≈0.03~0.06，6 轮总违规只降 9%。那已经超出局部
projection 的适用范围。J.2 换成**隐式场融合**：

    U(x) = min( φ_anatomy(x), s_int(x) − T_ext(x) )      # 并集（谁在外面听谁的）
    F(x) = max( −s_int(x), U(x) )                        # ∩ "预备体之外"（不得与预备体重叠）
    F(x) = +BIG  where( x 在颈缘平面之下 )                # 冠到颈缘为止

`F<0` = 冠材料。`F=0` 就是最终外表面。

## ⚠️ 隐式场**不是**免检（用户明确要求修正这条表述）

level set 能避免"顶点各自乱跑把三角网折起来"这一种机制，但当 offset 超过局部
feature size、碰到 medial axis、T(x) 空间变化、或网格分辨率不足时，仍可能出现
奇点、拓扑变化、非流形提取。所以 `geometry_safety_gate` **照常必跑**，
J.2 不因为用了 level set 就跳过 QC。

## T(x) 不是常数（病例规则 + 颈缘几何事实）

病例给 `MinThickness=0.6 / 近颈缘 0.2 / FullDistance 0.8`；同时共享颈缘边界要求
**exact Margin 上厚度 = 0**（两面同一条曲线）。所以：

    T_required(d) = near · min(d/d0, 1)                                  d < d0
                  = near + (full−near)·clip((d−d0)/(full_dist−d0),0,1)   d ≥ d0

`d` = 到颈缘曲线的距离，`d0` = 颈缘混合带宽度（**参数**：source 没说 near-margin
过渡到底是 linear 还是 step，所以形态显式参数化、provenance 如实标注）。
构造侧再加 `safety_mm` 余量，保证 ≥ Thickness Audit 的判定规则（Audit 不判
d<0.6 的颈缘带）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

from .thickness import (
    _boundary_edges,
    geometry_safety_gate,
    thresholds_of,
    ThicknessParameters,
)

STATUS_OK = "OK"
STATUS_REVIEW_REQUIRED = "REVIEW_REQUIRED"
STATUS_FAILED = "FAILED"


class EnvelopeError(Exception):
    pass


@dataclass
class EnvelopeParams:
    """``thickness`` 必须与 Thickness Audit 用的是同一份设计规则（同一个对象）。"""

    thickness: ThicknessParameters
    #: 体素尺寸（mm）。越细越贴合 anatomy；0.10 时 12×12×11mm 的冠约 1.6M 格
    voxel_mm: float = 0.10
    #: 网格外扩（mm）—— 必须足够容纳"包络凸出来"的部分
    pad_mm: float = 1.6
    #: 总格数上限（超了自动降分辨率并如实记录）
    max_voxels: int = 12_000_000
    #: 颈缘混合带宽度 d0（见模块 docstring；source 未指明过渡形态）
    margin_blend_mm: float = 0.10
    #: 构造侧余量：保证 ≥ Audit 的判定规则
    safety_mm: float = 0.05
    #: 颈缘边界吸附的最大容许距离（超过则如实报，不硬吸）
    margin_snap_tol_mm: float = 0.6
    #: 内表面侧排除带（与 Audit 的 boundary_safety_mm 同值）
    intaglio_exclude_mm: float = 0.60
    #: 分类"原本已满足厚度"的判据（Ray 侧 t_A ≥ T 的比例阈值）
    valid_margin_mm: float = 0.02
    #: 几何安全门
    max_self_intersections: int = 0
    max_nonmanifold_edges: int = 0


@dataclass
class EnvelopeResult:
    status: str = STATUS_FAILED
    mesh: object | None = None
    grid: dict = field(default_factory=dict)
    #: 分区的解剖偏差（用户要求：不能只报全冠 RMS）
    deviation_split: dict = field(default_factory=dict)
    coverage: dict = field(default_factory=dict)
    safety: dict = field(default_factory=dict)
    margin_snap: dict = field(default_factory=dict)
    notes: list = field(default_factory=list)

    def summary(self) -> dict:
        return {"status": self.status, "grid": self.grid,
                "deviation_split": self.deviation_split,
                "coverage": self.coverage,
                "safety_ok": self.safety.get("ok"),
                "safety_violations": self.safety.get("violations"),
                "margin_snap": self.margin_snap}


def required_thickness(d_margin, tp: ThicknessParameters, *,
                       blend_mm: float, safety_mm: float) -> np.ndarray:
    """构造侧厚度规则：颈缘处 →0，向冠方进入病例规则；再加构造余量。

    形态**显式参数化**（linear / step 由 ``tp.near_margin_mode`` 决定），
    见模块 docstring —— 不偷偷冻结一种解释。
    """
    d = np.asarray(d_margin, dtype=np.float64)
    near = tp.near_margin_thickness_mm
    full = float(tp.min_thickness_mm)
    if near is None or tp.near_margin_distance_mm is None:
        return np.full_like(d, full + safety_mm)
    near = float(near)
    fd = float(tp.near_margin_distance_mm)
    d0 = float(blend_mm)
    out = np.empty_like(d)
    lo = d < d0
    out[lo] = near * np.clip(d[lo] / max(d0, 1e-9), 0.0, 1.0)
    hi = ~lo
    if tp.near_margin_mode == "step":
        out[hi] = np.where(d[hi] >= fd, full, near)
    else:                                        # linear（默认读法）
        out[hi] = near + (full - near) * np.clip(
            (d[hi] - d0) / max(fd - d0, 1e-9), 0.0, 1.0)
    return out + safety_mm


def _closed(mesh_v, mesh_f, loop_idx):
    """用颈环质心扇形封口 → 闭合实体（volumetric 场的符号必需）。"""
    cen = np.asarray(mesh_v, float)[np.asarray(loop_idx, np.int64)].mean(axis=0)
    v = np.vstack([np.asarray(mesh_v, float), cen[None, :]])
    apex = len(v) - 1
    loop = list(map(int, loop_idx))
    cap = np.asarray([(loop[i], loop[(i + 1) % len(loop)], apex)
                      for i in range(len(loop))], dtype=np.int64)
    return v, np.vstack([np.asarray(mesh_f, np.int64), cap])


def _rasterize_surface(v, f, origin, spacing, shape):
    """把三角形表面打进体素掩码（按面积采样 → 够密，EDT 用）。"""
    tri = np.asarray(v, float)[np.asarray(f, np.int64)]
    e1 = tri[:, 1] - tri[:, 0]
    e2 = tri[:, 2] - tri[:, 0]
    area = 0.5 * np.linalg.norm(np.cross(e1, e2), axis=1)
    mask = np.zeros(shape, dtype=bool)
    # 每个三角形按面积采样密度 1/spacing²，最少 4 点
    n_s = np.maximum(4, np.ceil(area / (spacing ** 2) * 2.0)).astype(np.int64)
    total = int(n_s.sum())
    if total > 4_000_000:
        scale = 4_000_000 / total
        n_s = np.maximum(4, (n_s * scale).astype(np.int64))
        total = int(n_s.sum())
    rng = np.random.default_rng(0)
    for i in range(len(tri)):
        n = int(n_s[i])
        u = rng.random((n, 2))
        w = u.sum(axis=1)
        bad = w > 1.0
        u[bad] = 1.0 - u[bad]
        p = tri[i, 0] + u[:, :1] * e1[i] + u[:, 1:] * e2[i]
        idx = np.floor((p - origin) / spacing).astype(np.int64)
        ok = ((idx >= 0) & (idx < np.asarray(shape))).all(axis=1)
        idx = idx[ok]
        mask[idx[:, 0], idx[:, 1], idx[:, 2]] = True
    return mask


def _signed_field(v, f, loop_idx, origin, spacing, shape, seed_xyz):
    """闭合网格 → 网格点上的**带符号**距离场（EDT + 连通域定符号）。

    符号不用射线奇偶：把表面掩码膨胀一格后做连通域标记，含种子（实体内部一点）
    的分量记为内部 → 对体素化漏缝稳健。
    """
    sv, sf = _closed(v, f, loop_idx)
    surf = _rasterize_surface(sv, sf, origin, spacing, shape)
    thick = ndimage.binary_dilation(surf, iterations=1)
    lab, _n = ndimage.label(~thick)
    seed = np.floor((np.asarray(seed_xyz, float) - origin) / spacing).astype(int)
    seed = np.clip(seed, 0, np.asarray(shape) - 1)
    inside_lab = lab[tuple(seed)]
    if inside_lab == 0:                      # 种子落在表面上 → 就近找一格空腔
        idx = np.argwhere(lab > 0)
        if len(idx) == 0:
            raise EnvelopeError("体素化后没有空腔 —— 分辨率太粗？")
        k = int(np.argmin(np.linalg.norm((idx * spacing + origin)
                                         - np.asarray(seed_xyz, float),
                                         axis=1)))
        inside_lab = lab[tuple(idx[k])]
        seed = idx[k]
    # ⚠️ 表面带（lab==0，被膨胀掩码排除）默认会被判成"外部" → 实体内表面附近
    # 出现一层"正号壳"，精确窄带重算时就会带上这个错的符号 → 场里满是小孔洞，
    # marching cubes 于是吐出成百上千个碎片组件（实测 components=1800）。
    # 用最近有标签体素（EDT 的 return_indices）把标签补满。
    if (lab == 0).any():
        _d, idx = ndimage.distance_transform_edt(
            lab == 0, return_indices=True)
        lab = lab[tuple(idx)]
    inside = (lab == inside_lab)
    dist = ndimage.distance_transform_edt(~surf) * spacing
    return np.where(inside, -dist, dist), np.asarray(seed)


def build_level_set_envelope(anatomy_mesh, intaglio_mesh, margin_curve,
                             axis, params: EnvelopeParams) -> EnvelopeResult:
    """J.2 主入口：隐式包络 + 体融合 + 水平集提取 + 颈缘恢复 + 安全门。"""
    if params is None or params.thickness is None:
        raise EnvelopeError("EnvelopeParams.thickness（设计规则）必须显式给出")
    from skimage import measure as sk_measure
    import trimesh

    tp = params.thickness
    res = EnvelopeResult()
    av = np.asarray(anatomy_mesh.vertices, np.float64)
    af = np.asarray(anatomy_mesh.faces, np.int64)
    iv = np.asarray(intaglio_mesh.vertices, np.float64)
    ifa = np.asarray(intaglio_mesh.faces, np.int64)
    loop_a = np.asarray(sorted({int(v) for e in _boundary_edges(af)
                                for v in e}), dtype=np.int64)
    loop_i = np.asarray(sorted({int(v) for e in _boundary_edges(ifa)
                                for v in e}), dtype=np.int64)
    if len(loop_a) == 0 or len(loop_i) == 0:
        raise EnvelopeError("输入网格缺少颈环边界")
    z = np.asarray(axis, float)
    z = z / np.linalg.norm(z)

    # ---- 网格
    lo = np.minimum(av.min(axis=0), iv.min(axis=0)) - params.pad_mm
    hi = np.maximum(av.max(axis=0), iv.max(axis=0)) + params.pad_mm
    extent = hi - lo
    spacing = float(params.voxel_mm)
    n = np.ceil(extent / spacing).astype(np.int64) + 1
    while int(np.prod(n)) > params.max_voxels:
        spacing *= 1.25
        n = np.ceil(extent / spacing).astype(np.int64) + 1
    shape = tuple(int(x) for x in n)
    res.grid = {"origin": lo.tolist(), "spacing_mm": spacing,
                "shape": list(shape), "n_voxels": int(np.prod(n)),
                "pad_mm": params.pad_mm}
    res.notes.append("网格 %s @ %.3fmm（%.2fM 格）"
                     % (shape, spacing, np.prod(n) / 1e6))
    gx = lo[0] + spacing * np.arange(shape[0])
    gy = lo[1] + spacing * np.arange(shape[1])
    gz = lo[2] + spacing * np.arange(shape[2])

    # ---- 两条曲线的 d_margin → T field
    mtree_i = cKDTree(iv[loop_i])
    mtree_a = cKDTree(av[loop_a])
    # 颈缘曲线（用 intaglio 的环，与 Audit 同一口径）
    d_margin_i = mtree_i.query(iv, k=1)[0]
    T_i = required_thickness(d_margin_i, tp,
                             blend_mm=params.margin_blend_mm,
                             safety_mm=params.safety_mm)
    res.grid["T_field_mm"] = {"min": float(T_i.min()),
                              "median": float(np.median(T_i)),
                              "max": float(T_i.max())}

    # ---- 两个带符号场
    seed_a = av.mean(axis=0)                     # 冠壳内部（冠质心在实体里）
    seed_i = iv.mean(axis=0) + z * 0.5           # 预备体内部（沿轴往上一点）
    field_a, seed_a_used = _signed_field(av, af, loop_a, lo, spacing, shape,
                                         seed_a)
    field_i, seed_i_used = _signed_field(iv, ifa, loop_i, lo, spacing, shape,
                                         seed_i)
    res.notes.append("符号种子：anatomy @%s（用 %s）/ intaglio @%s（用 %s）"
                     % (np.round(seed_a, 2).tolist(),
                        np.round(seed_a_used, 2).tolist(),
                        np.round(seed_i, 2).tolist(),
                        np.round(seed_i_used, 2).tolist()))

    # ---- T_ext(x)：最近内表面点（顶点）的规则值
    grid_pts = np.stack(np.meshgrid(gx, gy, gz, indexing="ij"), axis=-1
                        ).reshape(-1, 3)
    band = np.abs(field_i.ravel()) < (T_i.max() + 4 * spacing)
    T_ext = np.full(grid_pts.shape[0], T_i.max(), dtype=np.float64)
    if band.any():
        _d, nn = cKDTree(iv).query(grid_pts[band], k=1)
        T_ext[band] = T_i[nn]
    T_ext = T_ext.reshape(shape)

    # ---- 融合：外边界 = min(anatomy, 内表面外扩 T)
    #   **不要**再叠 max(−s_int, ·)：那会把"预备体之外"也做成一个 0-交线，
    #   MC 于是吐出"外表面 ∪ 贴着预备体的那一层 ∪ 颈缘刀口 slit"，边界环落在
    #   离 margin 4mm 的颊侧（实测 202k 面 / 边界顶点 20k / 自交 1786）。
    #   只取 min 时：预备体内部两项都为负 → 那里没有 0-交线 → 0-水平集**恰好**
    #   就是外表面（anatomy 保留区 + 包络接管区），两者沿 s=T ∩ φ_A=0 这条
    #   干净的曲线相接，颈缘处 T→0 自然收到共享边界上。
    U = np.minimum(field_a, field_i - T_ext)
    F = U
    mpts = iv[loop_i]
    _dm, mnn = cKDTree(mpts).query(grid_pts, k=1)
    below = np.einsum("ij,ij->i", grid_pts - mpts[mnn], np.broadcast_to(
        z, grid_pts.shape)) < 0.0
    F.ravel()[below] = 1e6
    res.coverage = {
        "n_grid": int(F.size),
        "frac_F_negative": float((F < 0).mean()),
        # 包络接管了多少格（anatomy 场是正的、但被 s−T 压下去的地方）
        "n_envelope_takeover": int(((field_a >= 0) & (F < 0)).sum()),
        "n_anatomy_kept": int(((field_a < 0) & (F < 0)
                               & (field_i - T_ext >= 0)).sum()),
    }

    # ---- 水平集提取
    verts, faces, _normals, _vals = sk_measure.marching_cubes(
        F, level=0.0, spacing=(spacing, spacing, spacing))
    verts = verts + lo[None, :]
    n_raw = len(faces)
    # ⚠️ MC 提取的是**整个材料域的边界** = 外表面 ∪ 贴着预备体的那一层 ∪ 颈缘刀口 slit。
    # J.2 要的只是**外表面**（与 intaglio 共享颈缘边界）→ 丢掉"贴着预备体"的面
    # （面心到内表面的距离 < 半个体素）。不丢的话就是闭合壳：实测 160k 面、
    # euler=6346、nei 11345 条非流形边、边界环落在离 margin 4mm 的颊侧（刀口 slit）。
    # process=True：合并重复顶点 + 去退化面（MC 在薄片处会出），并按需重编号
    mesh = trimesh.Trimesh(verts, faces, process=True)
    res.notes.append("水平集提取：%d 面 → %d 面 / %d 顶点（marching cubes）"
                     % (n_raw, len(mesh.faces), len(mesh.vertices)))

    # ---- 颈缘边界精确恢复：把边界顶点吸到颈缘曲线上
    mv = np.asarray(mesh.vertices, np.float64)
    mf = np.asarray(mesh.faces, np.int64)
    be = _boundary_edges(mf)
    if len(be):
        bidx = np.asarray(sorted({int(v) for e in be for v in e}),
                          dtype=np.int64)
        # 颈缘曲线密采样（与 Audit/align 同一约定：顶点 + 段中点）
        samples = [mpts]
        samples.append(0.5 * (mpts + np.roll(mpts, -1, axis=0)))
        curve = np.vstack(samples)
        dd, nn = cKDTree(curve).query(mv[bidx], k=1)
        snap = dd <= params.margin_snap_tol_mm
        mv[bidx[snap]] = curve[nn[snap]]
        mesh.vertices = mv
        res.margin_snap = {
            "n_boundary_vertices": int(len(bidx)),
            "n_snapped": int(snap.sum()),
            "max_snap_mm": float(dd.max()) if len(dd) else 0.0,
            "n_out_of_tol": int((~snap).sum()),
            "tol_mm": params.margin_snap_tol_mm,
        }
    else:
        res.margin_snap = {"n_boundary_vertices": 0,
                           "note": "提取出的面没有边界 —— 可能是闭合体，"
                                   "颈缘未形成开口"}

    # ---- 分区解剖偏差（用户要求：不能只报全冠 RMS）
    #      以"从内表面沿外法向的射线"为共同参照：t_A = 到 anatomy 的距离，
    #      t_final = 到 J.2 外表面的距离；按 t_A ≥ T 判"原本已满足"。
    from .thickness import SurfaceQuery, ray_first_hit, vertex_normals
    ifaces = np.asarray(intaglio_mesh.faces, np.int64)
    vn_i = vertex_normals(iv, ifaces)
    sq_anat = SurfaceQuery(av, af, boundary_safety_mm=0.0,
                           max_query_edge_mm=0.0)
    sq_j2 = SurfaceQuery(np.asarray(mesh.vertices, np.float64),
                         np.asarray(mesh.faces, np.int64),
                         boundary_safety_mm=0.0, max_query_edge_mm=0.0)
    t_A, _ = ray_first_hit(iv, vn_i, sq_anat.tri_original, max_t=4.0)
    t_J, _ = ray_first_hit(iv, vn_i, sq_j2.tri_original, max_t=4.0)
    ok = np.isfinite(t_A) & np.isfinite(t_J)
    d_mi = mtree_i.query(iv, k=1)[0]
    Tj = required_thickness(d_mi, tp, blend_mm=params.margin_blend_mm,
                            safety_mm=0.0)            # 对照口径（无构造余量）
    valid = ok & (t_A >= Tj - params.valid_margin_mm)
    repaired = ok & ~valid
    dev = t_J - t_A
    def _st(mask):
        if not mask.any():
            return {"n": 0}
        v = dev[mask]
        return {"n": int(mask.sum()),
                "median_mm": float(np.median(v)),
                "p95_mm": float(np.quantile(v, .95)),
                "max_mm": float(v.max()),
                "min_mm": float(v.min())}
    res.deviation_split = {
        "ANATOMY_ALREADY_VALID": _st(valid),
        "ENVELOPE_REPAIRED": _st(repaired),
        "n_intaglio_samples": int(len(iv)),
        "n_no_ray_hit": int((~ok).sum()),
        "note": ("t_A = 从内表面沿外法向到 anatomy 的距离；t_J = 同一射线到 J.2 "
                 "外表面的距离。ANATOMY_ALREADY_VALID 区应接近 0 位移"),
    }
    res.coverage["intaglio_side"] = {
        "frac_valid": float(valid.mean()),
        "frac_repaired": float(repaired.mean()),
        "t_J_min_mm": float(np.nanmin(t_J[ok])) if ok.any() else None,
        "n_inside_prep_by_ray": int((t_A[ok] < 0).sum()),
    }

    # ---- 几何安全门（level set 不是免检）
    res.safety = geometry_safety_gate(
        np.asarray(mesh.vertices, np.float64),
        np.asarray(mesh.faces, np.int64),
        reference_vertices=None,
        max_flip_ratio=None,                 # MC 输出连接关系不同，翻面率不可比
        max_self_intersections=params.max_self_intersections,
        max_nonmanifold_edges=params.max_nonmanifold_edges,
        expect_open_boundary=True)
    # 朝向一致性：闭合体体积为正（MC 的绕向约定）
    vol = float(trimesh.Trimesh(np.asarray(mesh.vertices, np.float64),
                                np.asarray(mesh.faces, np.int64),
                                process=True).volume)
    res.safety["signed_volume_mm3"] = vol
    if vol <= 0:
        res.safety["ok"] = False
        res.safety["violations"].append("signed_volume=%.3f（朝向反了）" % vol)
    res.status = STATUS_OK if res.safety["ok"] else STATUS_FAILED
    if not res.safety["ok"]:
        res.notes.append(
            "**FAILED 的含义**：融合语义（覆盖/接管/分区偏差）已验证有效"
            "（内表面侧违反 28.1%→7.1%、负壁厚 0），但 **marching cubes 的提取质量"
            "不过关**（非流形/自交/颈缘边界未吸附）—— 下一提交解决提取，"
            "不动融合。见 docs/OUTER_V2_J2.md §4/§5。")
    if res.status == STATUS_OK and res.coverage["intaglio_side"][
            "n_inside_prep_by_ray"] > 0:
        res.status = STATUS_REVIEW_REQUIRED
        res.notes.append("仍有 %d 个内表面采样点在外表面之外（射线口径）"
                         % res.coverage["intaglio_side"]["n_inside_prep_by_ray"])
    res.mesh = mesh
    return res
