"""J.2.1-B —— Closed Envelope + Closed Anatomy → manifold3d UNION → guarded body cut
→ fixed collar stitch。

用户 2026-10-08 规格（J.2.1 两轮 extraction topology failure 之后）：

    Intaglio → **完整闭合** Minimum-Envelope Solid
    Anatomy Outer + 临时 cervical cap → **闭合** Anatomy Solid
        ↓ manifold3d exact Boolean UNION      （集合语义 = Anatomy ∪ MinimumEnvelope）
    CLOSED Crown Solid
        ↓ 在稳定 collar 区（d_margin = d1）**三角形裁剪**（不是按面过滤）
    Boolean Body（单边界环）+ Fixed Cervical Collar（已验证：146 原顶点 / p95 0.023mm）
        ↓ ring compatibility gate + 弧长重采样 → conforming stitch
    Outer J.2-B

关键调整（都是用户点名）：
* **隐式场只负责"完整闭合的 envelope solid"**，开放 body 由 Boolean 之后再裁 ——
  避开上一轮 `F(d<d1)=+1e6` 造出的"碗 + 腰带墙"闭合面。
* cervical 以下是**牺牲闭合区**（sacrificial）：envelope 在 `d ≤ env_start` 处
  厚度收到 0（尖端落在裁剪面下方），`d ∈ [env_start, d1+guard]` 线性升到满
  `T(x)`，`d ≥ d1+guard` 才是完整语义。这样裁剪截面上 union = anatomy，
  body ring 与 collar ring 同源；代价是 `d ∈ [d1, d1+guard]` 的包络修正被
  推迟（如实计入 coverage 的 `n_violating_in_ramp_band`，留给 Thickness Repair）。
  *(旧版用 `closure_floor_mm` 把 T 抬到 0.2 封壳 —— 那会让包络在 d1 处就顶出
  anatomy 0.1~0.45 mm，沿颈缘一圈台阶，实测不可缝。)*
* 两个 operand **各自过 preflight**（closed / manifold / 绕向一致 / 自交 0 / 体积>0），
  任一失败 → `BOOLEAN_INPUT_INVALID`，不指望 manifold3d 替我们修输入。
* seam 周围设 **anatomy-only guard band**（`d ≤ d1 + δ` 处 union 必须 ≈ 原 anatomy），
  这样 body ring 与 collar upper ring 本质来自同一张 H.1 曲面，stitch 才简单。
* 裁剪必须是**真三角裁剪**（边上插值出新顶点）；裁剪后
  `components != 1 or boundary_loops != 1` → 立即 fail loud，不进 zipper。
* stitch 前做 **ring compatibility gate**（Hausdorff/质心/朝向/环长），
  允许 325 ↔ N，按**弧长重采样**再缝合，不按顶点序号硬接。
* 每个面标记来源：`ANATOMY_RETAINED` / `ENVELOPE_DOMINATED` / `COLLAR_FIXED` /
  `BOOLEAN_SEAM`（形态诊断；"半颗牙都变 envelope" 说明 anatomy prior 有问题）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace

import numpy as np
from scipy.spatial import cKDTree

from .collar_body_outer import (
    _align_rings,
    _boundary_loops,
    _dp_seam_path,
    _clip_by_scalar,
    _conforming_ring,
    _order_loop_by_edges,
    _path_steps,
    _tris_from_path,
    _zip_path,
    _zip_rings,
    _zip_rings_conforming,
)
from .level_set_envelope import (
    _closed,
    _rasterize_surface,
    _signed_field,
    required_thickness,
)
from .thickness import (
    SurfaceQuery,
    ThicknessParameters,
    _boundary_edges,
    geometry_safety_gate,
)

STATUS_OK = "OK"
STATUS_REVIEW_REQUIRED = "REVIEW_REQUIRED"
STATUS_INPUT_INVALID = "BOOLEAN_INPUT_INVALID"
STATUS_FAILED = "FAILED"

#: 来源标签（面级）
ANATOMY_RETAINED = "ANATOMY_RETAINED"
ENVELOPE_DOMINATED = "ENVELOPE_DOMINATED"
COLLAR_FIXED = "COLLAR_FIXED"
BOOLEAN_SEAM = "BOOLEAN_SEAM"


class BooleanOuterError(Exception):
    pass


#: 精确点-三角查询的分块大小。SurfaceQuery 每点约 300+ 个候选面，
#: 分块 50000 → 2.6GiB 直接 OOM（实测）；8000 点约 0.42GiB。
QCHUNK = 8000


@dataclass
class BooleanOuterParams:
    thickness: ThicknessParameters
    voxel_mm: float = 0.10
    pad_mm: float = 1.6
    max_voxels: int = 12_000_000
    #: collar / body 分界（= 裁剪截面）；Audit 只判 d ≥ 0.6，故 ≤0.6
    d1_mm: float = 0.60
    #: guard band：d ≤ d1+δ 处 union 必须 ≈ 原 anatomy（stitch 才好缝）
    guard_mm: float = 0.25
    #: collar 与 body 的切面**错开**这一点点：两张网格是同一曲面的两种三角化，
    #: 同层切会互相浅叠（实测 137 对自交），错开后各占一侧、缝由 zip 带承担。
    seam_gap_mm: float = 0.10
    #: envelope 的起步：d ≤ env_start 不放 envelope 材料（于是**裁剪截面上
    #: union = anatomy**，body ring 与 collar ring 同源，缝是连续的）；
    #: env_start → d1+guard 线性升到满 T。env_start 取在 d1 之下一点，
    #: 是为了让壳的尖端（T→0 的收口）落在裁剪面**下方**，避免裁剪面与
    #: 零厚度尖端相切（相切会产生一对贴在一起的环）。
    env_start_mm: float = 0.50
    #: 窄带精确化宽度
    band_mm: float = 0.35
    #: envelope 的最小厚度下限（体素单位）。取 **0**：内腔填实后 T→0 不产生
    #: 退化薄片，而是退化成"prep 的体积"（合法实体）。留一个 >0 的下限会让
    #: 颈缘处（anatomy 厚度天生→0）硬顶出 0.15mm，实测把 seam 区偏差抬到
    #: p95 0.175 / max 0.227mm。只有在退回"薄壳"语义时才需要把它设成 ~1.5。
    min_shell_voxels: float = 0.0
    #: 与 anatomy 表面的**最小分离**（体素单位）。包络面在"厚度刚好等于 T"的地方
    #: 会与 anatomy 表面**相切**走 → 三角面级别互相穿插 → manifold3d 吐出贴片
    #: （实测 union 多出 6 件反向"空腔"，都在 intaglio 面上）。所以要么明显顶出去
    #: （≥gap），要么明显缩进去（≥gap），中间一律压到"anatomy 以内 gap 处"。
    cross_gap_voxels: float = 0.0
    safety_mm: float = 0.05
    margin_blend_mm: float = 0.10
    #: ring compatibility gate
    ring_hausdorff_tol_mm: float = 0.30
    ring_centroid_tol_mm: float = 0.30
    ring_length_ratio_tol: float = 0.25
    #: guard band 检验容差（union 与原 anatomy 的偏差）
    guard_p95_tol_mm: float = 0.05
    guard_max_tol_mm: float = 0.20
    #: 来源标记的判据
    retain_tol_mm: float = 0.05


@dataclass
class BooleanOuterResult:
    status: str = STATUS_FAILED
    mesh: object | None = None
    input_qc: dict = field(default_factory=dict)
    boolean_qc: dict = field(default_factory=dict)
    clip_qc: dict = field(default_factory=dict)
    stitch_qc: dict = field(default_factory=dict)
    provenance: dict = field(default_factory=dict)
    coverage: dict = field(default_factory=dict)
    deviation_split: dict = field(default_factory=dict)
    safety: dict = field(default_factory=dict)
    notes: list = field(default_factory=list)

    def summary(self) -> dict:
        return {"status": self.status, "input_qc": self.input_qc,
                "boolean_qc": self.boolean_qc, "clip_qc": self.clip_qc,
                "stitch_qc": self.stitch_qc, "provenance": self.provenance,
                "coverage": self.coverage,
                "safety_ok": self.safety.get("ok"),
                "safety_violations": self.safety.get("violations"),
                "safety_selfint": self.safety.get("self_intersection_severity"),
                "safety_metrics": self.safety.get("metrics")}


def _closed_solid(v, f, loop_idx, *, axis=None, apex_drop_mm: float = 6.0):
    """外壳 + **有序**颈环扇形封盖 → 闭合实体。

    ⚠️ 不能拿"排序后的顶点集"去扇：`_boundary_edges` 返回的是无序集合，
    按它扇出来的盖子自交/扭转 → `closed=false, boundary_edges=238,
    volume=NaN`（J.2 时代只当栅格化障壁用，所以没暴露；Boolean operand 会）。
    """
    import trimesh
    loop = _order_loop_by_edges(v, None, f, _boundary_edges(f))
    if len(loop) < 3:
        raise BooleanOuterError("颈环无法排成闭环（%d 点）" % len(loop))
    vv = np.asarray(v, np.float64)
    cen = vv[np.asarray(loop, np.int64)].mean(axis=0)
    # **盖子必须整个落在颈缘平面之下**：平盘（apex=质心）在颈缘起伏处会顶到
    # seam 之上，裁剪后留下第二/第三个边界环（实测 boundary_loops=2）。
    if axis is not None:
        zz = np.asarray(axis, float)
        zz = zz / np.linalg.norm(zz)
        # 让锥尖在"颈缘最低点再往下 apex_drop"处
        h_lo = float(((vv[np.asarray(loop, np.int64)] - cen) @ zz).min())
        cen = cen + zz * (h_lo - apex_drop_mm)
    verts = np.vstack([vv, cen[None, :]])
    apex = len(vv)
    cap = np.asarray([(loop[i], loop[(i + 1) % len(loop)], apex)
                      for i in range(len(loop))], dtype=np.int64)
    faces = np.vstack([np.asarray(f, np.int64), cap])
    m = trimesh.Trimesh(verts, faces, process=False)
    m.merge_vertices()
    if m.volume < 0:                # 盖子绕向与外壳相反 → 整体翻过来
        m.invert()
    return m


def _selfint_severity(qc: dict, v=None, f=None, *, min_area_mm2: float = 3.6e-3,
                      max_normal_dot: float = 0.999) -> dict:
    """把"自交计数"拆成真交叉 vs 数值级相切（用户规格：先证据再判定）。

    证据优先取 qc 里已带的；拿不到就**在网格上重算一遍**（曾出现计数=1 但
    诊断键为空的路径不一致，宁可多跑一次也不凭一个孤立计数否掉实体内核的输出）。
    """
    n = qc.get("self_intersections") or 0
    if not n:
        return {"severe": 0, "micro_defect": 0}
    ev = None
    if v is not None and (qc.get("self_intersections_severe_pairs") is None):
        from .thickness import self_intersections as _si
        ev = _si(np.asarray(v, np.float64), np.asarray(f, np.int64),
                 severe_min_area_mm2=min_area_mm2,
                 severe_max_normal_dot=max_normal_dot)
        n = ev.get("self_intersections") or 0
        qc["self_intersections_recount"] = ev.get("self_intersections")
        qc["self_intersections_diag"] = ev.get("self_intersections_diag")
        if not n:
            return {"severe": 0, "micro_defect": 0,
                    "note": "复算后计数为 0（首次为路径不一致）"}
    src = ev if ev is not None else qc
    severe = src.get("self_intersections_severe")
    micro = src.get("self_intersections_micro_defect")
    if severe is None:                       # 拿不到逐对证据 → 如实退回"全算严重"
        severe, micro = n, 0
    return {"severe": int(severe), "micro_defect": int(micro),
            "pair_area_mm2": src.get("self_intersections_pair_area_mm2"),
            "pair_normal_dot_p50": src.get(
                "self_intersections_pair_normal_dot_p50"),
            "severe_pairs": src.get("self_intersections_severe_pairs"),
            "severe_face_ids": src.get("self_intersections_severe_face_ids"),
            "hit_face_ids": src.get("self_intersections_hit_face_ids"),
            "criteria": {"min_area_mm2": min_area_mm2,
                         "max_normal_dot": max_normal_dot},
            "note": ("**逐对**分级：命中对里两个面都 ≥ 面积阈值且法向不近平行的"
                     "才算真交叉（severe）；其余记 micro_defect（提取分辨率的"
                     "数值级相切）。micro 照常计数上抛，绝不隐藏")}


def _operand_qc(mesh, name: str, *, voxel_mm: float = 0.12) -> dict:
    """Boolean operand preflight（用户规格逐条）。"""
    import trimesh
    m = trimesh.Trimesh(np.asarray(mesh.vertices, np.float64),
                        np.asarray(mesh.faces, np.int64), process=False)
    m.merge_vertices()
    # trimesh 的 is_winding_consistent 对非流形边也会报 False，所以分开算
    cnt = _edge_counts(np.asarray(m.faces, np.int64))
    nonmanifold = sum(1 for v in cnt.values() if v > 2)
    boundary = sum(1 for v in cnt.values() if v == 1)
    vol = float(m.volume) if m.is_watertight else float("nan")
    try:
        n_comps = len(m.split(only_watertight=False))
    except Exception:                                          # noqa: BLE001
        n_comps = None
    qc = {
        "components": n_comps,
        "closed": bool(m.is_watertight),
        "manifold": bool(nonmanifold == 0),
        "orientation_consistent": bool(m.is_winding_consistent),
        "boundary_edges": int(boundary),
        "nonmanifold_edges": int(nonmanifold),
        "self_intersections": None,
        "signed_volume_mm3": vol,
        "volume_positive": bool(np.isfinite(vol) and vol > 0),
        "n_vertices": int(len(m.vertices)), "n_faces": int(len(m.faces)),
    }
    gate = geometry_safety_gate(np.asarray(m.vertices, float),
                                np.asarray(m.faces, np.int64),
                                max_flip_ratio=None,
                                max_self_intersections=None,
                                expect_open_boundary=False)
    qc["self_intersections"] = gate["metrics"].get("self_intersections")
    fail = []
    for k in ("closed", "manifold", "orientation_consistent",
              "volume_positive"):
        if not qc[k]:
            fail.append(k)
    # 实体必须是**一件**：多件意味着有孤立块/内部空腔（实测：marching cubes 的
    # 反向薄片被 manifold3d 当空腔时，union 会分裂成若干负体积分件）
    if qc["components"] not in (None, 1):
        fail.append("components=%d" % qc["components"])
    # 自交门只认**真交叉**：命中对里若存在"两片面积都不小、且法向不近平行"的
    # 情形才算失败；面积级/近平行的相切对（外挂实体内核数值上允许）只记录。
    gate = _selfint_severity(
        qc, mesh.vertices, mesh.faces,
        min_area_mm2=max(voxel_mm ** 2 / 4.0, 1e-6))
    qc["self_intersections_severe"] = gate
    if gate["severe"]:
        fail.append("self_intersections=%d" % gate["severe"])
    qc["fail"] = fail
    qc["name"] = name
    return qc


def _edge_counts(faces) -> dict:
    from collections import Counter
    cnt: Counter = Counter()
    for a, b, c in np.asarray(faces, np.int64):
        for u, w in ((int(a), int(b)), (int(b), int(c)), (int(c), int(a))):
            cnt[(min(u, w), max(u, w))] += 1
    return cnt


def _resample_ring(v, ring, n: int) -> np.ndarray:
    """按弧长把闭环重采样成 n 点（保形状，不保原顶点序号）。"""
    pts = np.asarray(v, float)[np.asarray(ring, np.int64)]
    closed_pts = np.vstack([pts, pts[:1]])
    seg = np.linalg.norm(np.diff(closed_pts, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    total = s[-1]
    if total <= 1e-12:
        raise BooleanOuterError("环长为 0 —— 裁剪没产生有效截面")
    tgt = np.linspace(0.0, total, n, endpoint=False)
    out = np.empty((n, 3))
    for k, t in enumerate(tgt):
        i = int(np.searchsorted(s, t, side="right") - 1)
        i = max(0, min(i, len(seg) - 1))
        f = (t - s[i]) / max(seg[i], 1e-12)
        out[k] = closed_pts[i] + f * (closed_pts[i + 1] - closed_pts[i])
    return out


def build_boolean_outer(anatomy_mesh, intaglio_mesh, margin_curve, axis,
                        params: BooleanOuterParams) -> BooleanOuterResult:
    """J.2.1-B 主入口。"""
    if params is None or params.thickness is None:
        raise BooleanOuterError("BooleanOuterParams.thickness 必须显式给出")
    from skimage import measure as sk_measure
    import trimesh
    from ..boolean import require_exact_boolean_backend
    require_exact_boolean_backend()

    tp = params.thickness
    res = BooleanOuterResult()
    av = np.asarray(anatomy_mesh.vertices, np.float64)
    af = np.asarray(anatomy_mesh.faces, np.int64)
    iv = np.asarray(intaglio_mesh.vertices, np.float64)
    ifa = np.asarray(intaglio_mesh.faces, np.int64)
    loop_a = np.asarray(sorted({int(x) for e in _boundary_edges(af)
                                for x in e}), np.int64)
    loop_i = np.asarray(sorted({int(x) for e in _boundary_edges(ifa)
                                for x in e}), np.int64)
    if len(loop_a) == 0 or len(loop_i) == 0:
        raise BooleanOuterError("输入缺颈环边界")
    z = np.asarray(axis, float)
    z = z / np.linalg.norm(z)
    mpts = np.asarray(margin_curve, np.float64)
    curve = np.vstack([mpts, 0.5 * (mpts + np.roll(mpts, -1, axis=0))])
    ctree = cKDTree(curve)
    dm_a = ctree.query(av, k=1)[0]

    # ---- A. Fixed collar（**不让 Boolean 碰**）：按 d ≤ d1 真三角裁剪
    cv, cf, n_new = _clip_by_scalar(av, af, dm_a, level=params.d1_mm)
    be_c = _boundary_edges(cf)
    base_edges = [e for e in be_c if e[0] < len(av) and e[1] < len(av)]
    cut_edges = [e for e in be_c if e[0] >= len(av) or e[1] >= len(av)]
    base_loop = _order_loop_by_edges(cv, None, cf, base_edges)
    collar_ring = _order_loop_by_edges(cv, None, cf, cut_edges)
    if len(base_loop) < 3 or len(collar_ring) < 3:
        raise BooleanOuterError(
            "collar 裁剪失败（底环 %d / 截面环 %d）" % (len(base_loop),
                                                       len(collar_ring)))
    d_base = ctree.query(cv[np.asarray(base_loop, np.int64)], k=1)[0]
    res.input_qc["collar"] = {
        "n_faces": int(len(cf)), "n_new_vertices": int(n_new),
        "base_loop_vertices": int(len(base_loop)),
        "collar_ring_vertices": int(len(collar_ring)),
        "base_is_original_margin": bool(all(i < len(av) for i in base_loop)),
        "base_to_margin_p95_mm": float(np.quantile(d_base, .95)),
        "base_to_margin_max_mm": float(d_base.max()),
    }

    # ---- B. 两个闭合 operand
    # B1 anatomy solid：外壳 + 临时 cervical cap（牺牲，稍后被裁掉）
    anat_operand = _closed_solid(av, af, loop_a, axis=z)

    # B2 envelope solid：{0 ≤ s_int ≤ T_env} ∩ {margin 平面之上}，**完整闭合**
    lo = np.minimum(av.min(axis=0), iv.min(axis=0)) - params.pad_mm
    hi = np.maximum(av.max(axis=0), iv.max(axis=0)) + params.pad_mm
    spacing = float(params.voxel_mm)
    n = np.ceil((hi - lo) / spacing).astype(np.int64) + 1
    while int(np.prod(n)) > params.max_voxels:
        spacing *= 1.25
        n = np.ceil((hi - lo) / spacing).astype(np.int64) + 1
    shape = tuple(int(x) for x in n)
    res.input_qc["grid"] = {"shape": list(shape), "spacing_mm": spacing}

    seed_i = iv.mean(axis=0) + z * 0.5
    field_i, _si = _signed_field(iv, ifa, loop_i, lo, spacing, shape, seed_i)
    grid_pts = np.stack(np.meshgrid(
        lo[0] + spacing * np.arange(shape[0]),
        lo[1] + spacing * np.arange(shape[1]),
        lo[2] + spacing * np.arange(shape[2]), indexing="ij"), axis=-1
    ).reshape(-1, 3)
    # 窄带精确化（只精确 s_int；T 是查表规则值）
    band = (np.abs(field_i) < params.band_mm + float(tp.min_thickness_mm)).ravel()
    if band.any():
        sq_i = SurfaceQuery(iv, ifa, boundary_safety_mm=0.0,
                            max_query_edge_mm=0.0)
        fi = field_i.ravel()
        d_exact = np.empty(int(band.sum()))
        pts = grid_pts[band]
        for i in range(0, len(pts), QCHUNK):
            d_exact[i:i + QCHUNK] = sq_i.query(pts[i:i + QCHUNK])[3]
        fi[band] = np.where(fi[band] < 0, -d_exact, d_exact)
        field_i = fi.reshape(shape)
        res.notes.append("窄带精确化 %d 格（%.1f%%）"
                         % (band.sum(), 100.0 * band.sum() / field_i.size))

    mtree_i = cKDTree(iv[loop_i])
    T_i = required_thickness(mtree_i.query(iv, k=1)[0], tp,
                             blend_mm=params.margin_blend_mm,
                             safety_mm=params.safety_mm)
    bandT = np.abs(field_i.ravel()) < (T_i.max() + 4 * spacing)
    T_ext = np.full(grid_pts.shape[0], T_i.max(), np.float64)
    if bandT.any():
        _d, nn = cKDTree(iv).query(grid_pts[bandT], k=1)
        T_ext[bandT] = T_i[nn]
    T_ext = T_ext.reshape(shape)
    dm_g = ctree.query(grid_pts, k=1)[0].reshape(shape)
    # 颈缘 guard band：d ≤ env_start 不放 envelope 材料 → 裁开处 union = anatomy，
    # 两环同源、缝连续；env_start → d1+guard 线性升到满 T(x)（d1 处约 0.29T，
    # **薄到藏在 anatomy 里**，所以 seam 不被撑出台阶）。
    # 实测依据：本例 anatomy 在 d∈[0.6,0.8) 的厚度 p05 0.208 / 最大亏 0.352 mm，
    # 满 T=0.486 直接生效会顶出 0.1~0.45 mm 的沿颈缘一圈的台阶（不可缝）。
    env_full = params.d1_mm + params.guard_mm
    ramp = np.clip((dm_g - params.env_start_mm)
                   / max(env_full - params.env_start_mm, 1e-9), 0.0, 1.0)
    floor = params.min_shell_voxels * spacing
    raw = np.maximum(T_ext, params.safety_mm) * ramp
    # anatomy 的局部厚度场：在 anatomy 顶点上插值 s_int，再最近邻摊回网格
    from scipy.ndimage import map_coordinates
    t_a_v = np.maximum(map_coordinates(
        field_i, (av - lo[None, :]).T / spacing, order=1, mode="nearest"), 0.0)
    _dn, nn_a = cKDTree(av).query(grid_pts, k=1)
    t_anat = t_a_v[nn_a].reshape(shape)
    # 不允许和 anatomy 表面擦着走：明显顶出去 → 用 raw；否则压到 anatomy 内 gap 处
    gap = params.cross_gap_voxels * spacing
    T_used = np.where(raw - t_anat >= gap, raw,
                      np.minimum(raw, t_anat - gap))
    T_used = np.maximum(T_used, floor)
    # margin 平面之下不放材料（用最近 margin 点的轴向偏移）
    _dd, mnn = cKDTree(mpts).query(grid_pts, k=1)
    d_below = -np.einsum("ij,ij->i", grid_pts - mpts[mnn],
                         np.broadcast_to(z, grid_pts.shape)).reshape(shape)
    # 集合：{s_int ≤ T_used} ∩ {margin 之上} —— **内腔填实**（不再要 s_int ≥ 0）。
    # 为什么：只取"壳"时，B 是一张 0.15~0.6mm 厚的薄壳，而它的 90% 面积埋在 A
    # 内部 → manifold3d 对这种"埋在实体里的薄壳"会解错（实测 union 变 8 件，
    # 6 件负体积；winding 探针显示那些"空腔"内部其实在 envelope 里）。
    # 填实后 B 与 A 一样是"实心块"，两表面只在真正顶出去处相交；
    # 集合语义不变：A（已闭合的实心冠）本来就覆盖 intaglio 内侧区域。
    G = np.maximum(field_i - T_used, d_below)
    verts, faces, _nr, _v = sk_measure.marching_cubes(
        G, level=0.0, spacing=(spacing, spacing, spacing))
    verts = verts + lo[None, :]
    env_operand = trimesh.Trimesh(verts, faces, process=True)
    # 薄壳收口（T→0）比体素还薄的那一段，marching cubes 会吐出小片**反向**薄片
    # （实测 7 片 / 40~568 面，体积 ≤0.07mm³，dm∈[0.6,1.4]）。它们若进 Boolean
    # 会被 manifold3d 当成**内部空腔**（union 分件体积为负）→ 必须先在源头去掉。
    # 只留最大件；被丢的片必须确实是小垃圾（面占比 ≤5% 且单件 ≤0.5mm³），
    # 否则说明 envelope mesher 本身不行 → BOOLEAN_INPUT_INVALID。
    env_operand.merge_vertices()      # 先合并重复顶点，否则同一张薄片会被切碎
    env_parts = sorted(env_operand.split(only_watertight=False),
                       key=lambda m: -len(m.faces))
    env_dropped = []
    if len(env_parts) > 1:
        n_all = int(sum(len(m.faces) for m in env_parts))
        for m in env_parts[1:]:
            mv = trimesh.Trimesh(np.asarray(m.vertices, np.float64),
                                 np.asarray(m.faces, np.int64), process=False)
            env_dropped.append({"faces": int(len(m.faces)),
                                "volume_mm3": round(abs(float(mv.volume)), 4)})
        env_operand = env_parts[0]
        frac = sum(d["faces"] for d in env_dropped) / max(n_all, 1)
        res.input_qc["envelope_stray_pieces"] = {
            "n": len(env_dropped), "face_frac": round(float(frac), 4),
            "detail": env_dropped[:6]}
        if frac > 0.05 or max(d["volume_mm3"] for d in env_dropped) > 0.5:
            res.status = STATUS_INPUT_INVALID
            res.notes.append(
                "**BOOLEAN_INPUT_INVALID**：envelope solid 有 %d 个非主分件，"
                "占比 %.2f%%（最大 %.3fmm³）—— 不是体素级碎屑，说明 envelope "
                "mesher 本身需要换，不硬塞进 Boolean"
                % (len(env_dropped), 100 * frac,
                   max(d["volume_mm3"] for d in env_dropped)))
            return res
    res.notes.append("envelope solid：%d 面 / %d 顶点（complete closed，单件；"
                     "丢弃体素级碎屑 %d 片）"
                     % (len(env_operand.faces), len(env_operand.vertices),
                        len(env_dropped)))

    # ---- C. Operand preflight（任一失败 → BOOLEAN_INPUT_INVALID）
    qa = _operand_qc(anat_operand, "anatomy", voxel_mm=spacing)
    qe = _operand_qc(env_operand, "envelope", voxel_mm=spacing)
    res.input_qc["anatomy_operand"] = qa
    res.input_qc["envelope_operand"] = qe
    if qa["fail"] or qe["fail"]:
        res.status = STATUS_INPUT_INVALID
        res.notes.append("**BOOLEAN_INPUT_INVALID**：anatomy %s / envelope %s"
                         % (qa["fail"], qe["fail"]))
        return res

    # ---- D. Boolean UNION（集合语义 = Anatomy ∪ MinimumEnvelope）
    t_bool = None
    try:
        import time as _t
        t0 = _t.time()
        union = trimesh.boolean.union([anat_operand, env_operand],
                                      engine="manifold")
        t_bool = _t.time() - t0
    except Exception as exc:                                   # noqa: BLE001
        res.status = STATUS_FAILED
        res.notes.append("Boolean UNION 失败：%s" % exc)
        return res
    union = trimesh.Trimesh(np.asarray(union.vertices, np.float64),
                            np.asarray(union.faces, np.int64), process=True)
    qu = _operand_qc(union, "union", voxel_mm=spacing)
    micro = int(qu.get("self_intersections_severe", {}).get("micro_defect", 0))
    res.boolean_qc = {"union": qu, "engine": "manifold3d",
                      "elapsed_s": round(t_bool or 0.0, 1),
                      "n_faces": int(len(union.faces)),
                      "micro_defects": micro}
    # D2. 分件：允许**体素级碎屑**（micro_components），其余一律 fail loud。
    # 判据纯尺度（体积 < 体素³/8 且 bbox 对角线 < 1 体素），不看"想不想过"；
    # 实测残留 1 件 8 面 / 体积 0 / 尺度 10~20µm —— 是内核数值级碎屑，
    # 与 self-intersection 的 micro_defect 同一套口径：**计数 + 证据 + 上抛**。
    if (qu.get("components") or 1) > 1:
        vol_thr = spacing ** 3 / 8.0
        parts = sorted(union.split(only_watertight=False),
                       key=lambda m: -len(m.faces))
        crumbs, fat = [], []
        for m in parts[1:]:
            mv = np.asarray(m.vertices, np.float64)
            mv2 = trimesh.Trimesh(mv, np.asarray(m.faces, np.int64),
                                  process=False)
            rec = {"faces": int(len(m.faces)),
                   "volume_mm3": round(abs(float(mv2.volume)), 8),
                   "bbox_diag_mm": round(float(np.linalg.norm(
                       mv.max(0) - mv.min(0))), 4)}
            if rec["volume_mm3"] < vol_thr and rec["bbox_diag_mm"] < spacing:
                crumbs.append(rec)
            else:
                fat.append(rec)
        res.boolean_qc["micro_components"] = {
            "n": len(crumbs), "detail": crumbs[:6],
            "criteria": {"volume_lt_mm3": round(vol_thr, 8),
                         "bbox_diag_lt_mm": spacing},
            "note": "体素级碎屑：计数并保留证据，不当成第二个实体、也不隐藏"}
        if fat:
            res.boolean_qc["component_detail"] = fat[:6]
            res.status = STATUS_FAILED
            res.notes.append("**Boolean 结果不是合法实体**：多出 %d 个非碎屑分件 %s"
                             % (len(fat), fat[:3]))
            return res
        union = trimesh.Trimesh(np.asarray(parts[0].vertices, np.float64),
                                np.asarray(parts[0].faces, np.int64),
                                process=True)
        qu["components"] = 1
        qu["fail"] = [f for f in qu["fail"] if not f.startswith("components=")]
        res.boolean_qc["n_faces"] = int(len(union.faces))
    if qu["fail"]:
        res.status = STATUS_FAILED
        res.notes.append("**Boolean 结果不是合法实体**：%s" % qu["fail"])
        return res

    # ---- E. Guard band：d ≤ d1+δ 处 union 必须 ≈ 原 anatomy
    uv = np.asarray(union.vertices, np.float64)
    dm_u = ctree.query(uv, k=1)[0]
    # 只看 **margin 之上** 的面：颈缘以下的牺牲闭合（深锥盖）本来就离 anatomy 很远，
    # 混进来会把 seam 指标污染成"超差"（实测 max 0.358 全来自锥面）。
    _dd0, mi0 = cKDTree(mpts).query(uv, k=1)
    ax_u = np.einsum("ij,ij->i", uv - mpts[mi0], np.broadcast_to(z, uv.shape))
    sel = (dm_u <= params.d1_mm + params.guard_mm) & (ax_u > 0.05)
    if sel.any():
        sq_a = SurfaceQuery(av, af, boundary_safety_mm=0.0,
                            max_query_edge_mm=0.0)
        pts = uv[sel]
        dd = np.empty(len(pts))
        for i in range(0, len(pts), QCHUNK):
            dd[i:i + QCHUNK] = sq_a.query(pts[i:i + QCHUNK])[3]
        dm_sel = dm_u[sel]
        # 两个区必须分开看：**seam 区**（d ≤ d1）要求 union = anatomy（缝才连续），
        # **ramp 区**（d1 < d ≤ d1+δ）按设计就是要逐渐离开 anatomy 的。
        # 混在一起报会看不出到底是哪一个出了问题（实测 p50 0.008 = seam 干净）。
        def _zone(m):
            if not m.any():
                return {"n_vertices": 0}
            w = dd[m]
            return {"n_vertices": int(m.sum()),
                    "dev_p50_mm": float(np.median(w)),
                    "dev_p95_mm": float(np.quantile(w, .95)),
                    "dev_max_mm": float(w.max())}
        seam = _zone(dm_sel <= params.d1_mm)
        ramp_z = _zone(dm_sel > params.d1_mm)
        res.boolean_qc["guard_band"] = {
            "n_vertices": int(sel.sum()),
            "seam_zone_d_le_d1": seam,
            "ramp_zone_d1_to_d1pguard": ramp_z,
            "dev_p50_mm": float(np.median(dd)),
            "dev_p95_mm": float(np.quantile(dd, .95)),
            "dev_max_mm": float(dd.max()),
            "seam_p95_tol_mm": params.guard_p95_tol_mm,
            "seam_max_tol_mm": params.guard_max_tol_mm,
        }
        over = (seam.get("dev_p95_mm", 0.0) > params.guard_p95_tol_mm
                or seam.get("dev_max_mm", 0.0) > params.guard_max_tol_mm)
        res.boolean_qc["guard_band"]["seam_over_tolerance"] = bool(over)
        res.boolean_qc["seam_step_p95_mm"] = seam.get("dev_p95_mm")
        if over:
            res.notes.append(
                "⚠️ **seam 区**（d ≤ d1）超差：p95 %.3f / max %.3f mm —— 该区按 "
                "ramp 设定应当 union = anatomy（envelope 全藏在里面）；超差说明 "
                "ramp 起点/厚度场没压住，或 anatomy 本身比 ramp 后的 T 还薄。"
                "**不改容差**：如实上抛，由 ring gate 判可缝性"
                % (seam["dev_p95_mm"], seam["dev_max_mm"]))

    # ---- F. Body：**真三角裁剪** d ≥ d1 + gap（不是按面过滤）
    # gap > 0 是**故意的**：collar 取 anatomy、body 取 union，两张网格是同一张
    # 曲面的两种三角化；若都从 d1 处切，两层会在切带里**互相穿插**（实测
    # collar 单独 0、body 单独 0、合起来 137 —— 全是浅叠）。留 0.10mm 的缝，
    # 两件不再重叠，zip 带也变成一条正常高度的窄带。
    uv_f = np.asarray(union.faces, np.int64)
    d_cut = params.d1_mm + params.seam_gap_mm
    bv, bf, n_new_b = _clip_by_scalar(uv, uv_f, -ctree.query(uv, k=1)[0],
                                      level=-d_cut)
    if len(bf) == 0:
        raise BooleanOuterError("body 裁剪后没有面")
    loops_all = _boundary_loops(bf)
    comps_all = trimesh.Trimesh(bv, bf, process=False).split(
        only_watertight=False)
    # **按"是否挨着 collar 截面环"选件**，不是按面积：裁剪把围绕颈缘 0.6mm 的
    # 一圈壳（tube）整个切掉，于是颈缘以下的**牺牲闭合**（anatomy operand 的
    # 深锥盖）被这一刀与冠体分断 → 以碎块形式出现。碎块只允许是"整个在
    # margin 之下"的牺牲区；若有碎块落在 margin 之上，那是真表面被切丢了 →
    # 立即 fail loud（不许静默丢面积）。
    tree_c = cKDTree(cv[np.asarray(collar_ring, np.int64)])
    scored = []
    for m in comps_all:
        pv = np.asarray(m.vertices, np.float64)
        dmin = float(tree_c.query(pv, k=1)[0].min())
        scored.append((dmin, -len(m.faces), m))
    scored.sort(key=lambda t: (t[0], t[1]))
    keep_d, _nf, body0 = scored[0]
    dropped = [m for _d, _n, m in scored[1:]]
    if keep_d > params.d1_mm + params.guard_mm:
        res.status = STATUS_FAILED
        res.notes.append("**没有任何碎块挨着 collar 截面环**（最近 %.3f mm）"
                         " —— 裁剪把主体切没了，立即停" % keep_d)
        return res
    body = body0
    bv = np.asarray(body.vertices, np.float64)
    bf = np.asarray(body.faces, np.int64)
    body_loops = _boundary_loops(bf)
    d_below_tol = 0.10                       # mm，margin 平面附近的数值裕量
    tree_anat = cKDTree(av)                  # 判定碎块"贴谁"用（顶点级近似）
    tree_env = cKDTree(np.asarray(env_operand.vertices, np.float64))
    dropped_info, above = [], []
    for m in dropped:
        pv = np.asarray(m.vertices, np.float64)
        _dd, mi = cKDTree(mpts).query(pv, k=1)
        ax = np.einsum("ij,ij->i", pv - mpts[mi],
                       np.broadcast_to(z, pv.shape))
        rec = {"faces": int(len(m.faces)), "ax_off_max_mm": round(float(ax.max()), 3),
               "ax_off_min_mm": round(float(ax.min()), 3),
               "d_anatomy_min_mm": round(float(tree_anat.query(pv, k=1)[0].min()), 3),
               "d_envelope_min_mm": round(float(tree_env.query(pv, k=1)[0].min()), 3),
               "closed": bool(trimesh.Trimesh(pv, m.faces,
                                              process=False).is_watertight)}
        dropped_info.append(rec)
        if ax.max() > d_below_tol:
            above.append(rec)
    res.clip_qc = {
        "cut_level_mm": float(d_cut),
        "seam_gap_mm": float(params.seam_gap_mm),
        "n_new_vertices": int(n_new_b), "n_faces": int(len(bf)),
        "n_components_all": int(len(comps_all)),
        "n_components": 1,
        "component_face_counts": [int(len(body.faces))],
        "n_boundary_loops_all": int(len(loops_all)),
        "boundary_loop_sizes_all": sorted([len(x) for x in loops_all],
                                          reverse=True)[:6],
        "n_boundary_loops": int(len(body_loops)),
        "boundary_loop_sizes": sorted([len(x) for x in body_loops],
                                      reverse=True)[:6],
        "ring_proximity_mm": round(keep_d, 4),
        "dropped_components": len(dropped),
        "dropped_faces": int(sum(len(m.faces) for m in dropped)),
        "dropped_detail": dropped_info,
        "dropped_is_sacrificial": bool(not above),
        "criterion": ("碎块必须是牺牲闭合区（全部顶点在 margin 之下，"
                      "ax_off ≤ %.2fmm）" % d_below_tol),
    }
    if above:
        res.status = STATUS_FAILED
        res.notes.append("**丢弃的碎块里有 margin 之上的真表面**：%s"
                         " —— 立即停，不进 zipper" % above[:3])
        return res
    if len(body_loops) != 1:
        res.status = STATUS_FAILED
        res.notes.append("**裁剪拓扑不合格**：选定的 body 有 %d 个边界环 %s"
                         "（碎块已按牺牲区丢弃：%d 件 / %d 面）"
                         " —— 立即停，不进 zipper"
                         % (len(body_loops),
                            sorted([len(x) for x in body_loops],
                                   reverse=True)[:5],
                            len(dropped), res.clip_qc["dropped_faces"]))
        return res
    if dropped:
        res.notes.append("裁剪丢弃牺牲闭合碎块 %d 件 / %d 面（全部在 margin 之下，"
                         "ax_off ≤ %.3fmm）"
                         % (len(dropped), res.clip_qc["dropped_faces"],
                            max(r["ax_off_max_mm"] for r in dropped_info)))
    body_ring = _boundary_loops(bf)[0]
    res.notes.append("body：%d 面 / 单边界环 %d 顶点（真三角裁剪）"
                     % (len(bf), len(body_ring)))

    # ---- G. Ring compatibility gate（325 ↔ N 允许）+ 弧长重采样缝合
    cr = np.asarray(collar_ring, np.int64)
    br = np.asarray(body_ring, np.int64)
    cp_c, cp_b = cv[cr], bv[br]
    # 质心必须按**弧长加权**：两条环的顶点疏密完全不同（276 vs 790），
    # 顶点均值差 1.18mm，而弧长质心差只有 ~0.1mm —— 顶点均值在这里是假信号。
    def _arc_centroid(p):
        q = np.vstack([p, p[:1]])
        mid = 0.5 * (q[:-1] + q[1:])
        w = np.linalg.norm(np.diff(q, axis=0), axis=1)
        return (mid * w[:, None]).sum(0) / max(w.sum(), 1e-12)

    cen_c, cen_b = _arc_centroid(cp_c), _arc_centroid(cp_b)
    d_cb = cKDTree(cp_c).query(cp_b, k=1)[0]
    d_bc = cKDTree(cp_b).query(cp_c, k=1)[0]
    len_c = float(np.linalg.norm(np.diff(np.vstack([cp_c, cp_c[:1]]),
                                         axis=0), axis=1).sum())
    len_b = float(np.linalg.norm(np.diff(np.vstack([cp_b, cp_b[:1]]),
                                         axis=0), axis=1).sum())
    gate = {
        "collar_ring_vertices": int(len(cr)),
        "body_ring_vertices": int(len(br)),
        "centroid_distance_mm": float(np.linalg.norm(cen_c - cen_b)),
        "hausdorff_body_to_collar_mm": float(d_cb.max()),
        "hausdorff_collar_to_body_mm": float(d_bc.max()),
        "length_collar_mm": len_c, "length_body_mm": len_b,
        "length_ratio": float(len_b / max(len_c, 1e-9)),
    }
    gate["fail"] = []
    if gate["centroid_distance_mm"] > params.ring_centroid_tol_mm:
        gate["fail"].append("centroid=%.3f" % gate["centroid_distance_mm"])
    if max(gate["hausdorff_body_to_collar_mm"],
           gate["hausdorff_collar_to_body_mm"]) > params.ring_hausdorff_tol_mm:
        gate["fail"].append("hausdorff=%.3f"
                            % max(gate["hausdorff_body_to_collar_mm"],
                                  gate["hausdorff_collar_to_body_mm"]))
    if abs(gate["length_ratio"] - 1.0) > params.ring_length_ratio_tol:
        gate["fail"].append("length_ratio=%.3f" % gate["length_ratio"])
    res.stitch_qc["ring_gate"] = gate
    if gate["fail"]:
        res.status = STATUS_FAILED
        res.notes.append("**ring compatibility 不合格**：%s —— 不进 zipper"
                         % gate["fail"])
        return res

    # **共形 zig-zag 缝合**：用两条环的**原顶点**缝（276 ↔ 781 允许），
    # 按弧长推进消费两侧全部顶点 → 不留 T 接头，也不会像"缝到重采样副本上"
    # 那样把 collar/body/strip 拆成三件（实测 3 件 / 5 环 / 自交 1347）。
    # 缝带是**环带三角化（lattice 路径）**：path[k] ∈ {'a','b'} 表示这一步推进
    # 哪一侧；交换相邻两步 = 翻转该四边形的对角线（局部、可逆、不动任何顶点）。
    v_all = np.vstack([cv, bv])
    br_off = np.asarray(br, np.int64) + len(cv)
    A_r, B_r, arca, arcb = _align_rings(v_all, cr, br_off)
    path = _zip_path(arca, arcb, len(A_r), len(B_r))
    steps = _path_steps(path)

    def _assemble(path_now):
        strip_now = _tris_from_path(A_r, B_r, path_now)
        f_now = np.vstack([cf, bf + len(cv), strip_now])
        return trimesh.Trimesh(v_all, f_now, process=False), strip_now

    mesh, strip = _assemble(path)
    gg = geometry_safety_gate(np.asarray(mesh.vertices, float),
                              np.asarray(mesh.faces, np.int64),
                              max_flip_ratio=None, max_self_intersections=None,
                              expect_open_boundary=True)
    if (gg["metrics"].get("winding_inconsistent_edges") or 0) > 0:
        seen = np.asarray(strip)[:, ::-1]      # 绕向判错就整体翻过来
        f_all = np.vstack([cf, bf + len(cv), seen])
        mesh = trimesh.Trimesh(v_all, f_all, process=False)
        strip = seen
    res.stitch_qc["n_zip_triangles"] = int(len(strip))
    res.stitch_qc["resampled_n"] = int(max(len(cr), len(br)))
    # 自交归因：缝合前（collar+body）先量一次，差值就是 zipper 带进来的
    pre = trimesh.Trimesh(np.vstack([cv, bv]),
                          np.vstack([cf, bf + len(cv)]), process=False)
    gpre = geometry_safety_gate(np.asarray(pre.vertices, float),
                                np.asarray(pre.faces, np.int64),
                                max_flip_ratio=None,
                                max_self_intersections=None,
                                expect_open_boundary=True)
    res.stitch_qc["selfint_before_stitch"] = gpre["metrics"].get(
        "self_intersections")
    gstrip = geometry_safety_gate(
        np.asarray(v_all, float),
        np.asarray(strip, np.int64),
        max_flip_ratio=None, max_self_intersections=None,
        expect_open_boundary=True)
    res.stitch_qc["selfint_strip_only"] = gstrip["metrics"].get(
        "self_intersections")
    # 分件归因：collar 单独 / body 单独各量一次，剩下的才是"两件互相穿"
    gc = geometry_safety_gate(cv, cf, max_flip_ratio=None,
                              max_self_intersections=None,
                              expect_open_boundary=True)
    gb = geometry_safety_gate(bv, bf, max_flip_ratio=None,
                              max_self_intersections=None,
                              expect_open_boundary=True)
    res.stitch_qc["selfint_collar_only"] = gc["metrics"].get(
        "self_intersections")
    res.stitch_qc["selfint_body_only"] = gb["metrics"].get(
        "self_intersections")

    # ---- G2. J.2.1-C Phase 1：seam 局部 edge flip
    # 缝带是 lattice 路径：交换相邻两步 = 翻转该四边形对角线 —— 纯局部、可逆、
    # 不动任何顶点（margin 的 146 个原顶点、两条环的顶点全部原样）。
    # 目标：把 self_intersections 打到 **0（severe 和 micro 都要 0）**，
    # 按 (severe, total) 字典序严格改进才接受。
    from .thickness import self_intersections as _si
    min_area = max(params.voxel_mm ** 2 / 4.0, 1e-6)

    def _hits(m):
        return _si(np.asarray(m.vertices, np.float64),
                   np.asarray(m.faces, np.int64),
                   severe_min_area_mm2=min_area,
                   severe_max_normal_dot=0.999)

    def _key(ev):
        return (int(ev.get("self_intersections_severe") or 0),
                int(ev.get("self_intersections") or 0))

    ev0 = _hits(mesh)
    repair = {"n_hits_before": int(ev0.get("self_intersections") or 0),
              "n_severe_before": int(ev0.get("self_intersections_severe") or 0),
              "severe_pairs": ev0.get("self_intersections_severe_pairs"),
              "trials": []}
    res.stitch_qc["seam_repair"] = repair
    strip_off = len(cf) + len(bf)
    if repair["n_hits_before"]:
        cand = set()
        for fa, fb in (ev0.get("self_intersections_hit_face_ids") or []):
            for f in (fa, fb):
                kk = int(f) - strip_off
                if 0 <= kk < len(path):
                    cand.update((kk - 1, kk, kk + 1))
        pos = [p for p in sorted(cand)
               if 0 <= p and p + 1 < len(path) and path[p] != path[p + 1]]
        # **局部探针**：两条环在缺陷附近是否互相靠到穿。若两条环本身互穿，
        # 任何路径都救不了 —— 那是环的空间关系问题，不是对角线选择问题。
        kmin, kmax = min(cand), max(cand)
        idx_A, idx_B = set(), set()
        for k in range(max(0, kmin - 6), min(len(steps), kmax + 7)):
            ii, jj, _s = steps[k]
            idx_A.update((ii - 1, ii, ii + 1))
            idx_B.update((jj - 1, jj, jj + 1))
        va_all = v_all[A_r[np.asarray(sorted(idx_A)) % len(A_r)]]
        vb_all = v_all[B_r[np.asarray(sorted(idx_B)) % len(B_r)]]
        dmin = float(cKDTree(vb_all).query(va_all, k=1)[0].min())
        band = cKDTree(v_all[B_r]).query(v_all[A_r], k=1)[0]
        res.stitch_qc["seam_repair"]["local_probe"] = {
            "steps": [int(kmin), int(kmax)],
            "ring_vertex_distance_min_mm": round(dmin, 4),
            "band_height_p50_mm": round(float(np.median(band)), 4),
            "band_height_min_mm": round(float(band.min()), 4)}
        best_key, best_path, best_mesh, best_strip = _key(ev0), None, None, None
        for p in pos[:6]:
            p2 = list(path)
            p2[p], p2[p + 1] = p2[p + 1], p2[p]
            m2, s2 = _assemble(p2)
            k2 = _key(_hits(m2))
            repair["trials"].append({"flip_at_step": int(p),
                                     "severe": k2[0], "total": k2[1],
                                     "step_sides": [path[p], path[p + 1]]})
            if k2 < best_key:
                best_key, best_path, best_mesh, best_strip = k2, p2, m2, s2
        repair["candidate_flips"] = len(pos)
        if best_path is not None:
            path, mesh, strip = best_path, best_mesh, best_strip
            repair["phase1"] = {"applied": True, "severe": best_key[0],
                                "total": best_key[1]}
        else:
            repair["phase1"] = {"applied": False, "severe": best_key[0],
                                "total": best_key[1],
                                "note": "局部翻转无法改善"}
        # ---- Phase 2：DP annulus triangulation（全局最优 lattice 路径）
        if best_key[1] > 0:
            try:
                path_dp = _dp_seam_path(v_all, A_r, B_r)
                m3, s3 = _assemble(path_dp)
                k3 = _key(_hits(m3))
                repair["phase2_dp"] = {"severe": k3[0], "total": k3[1]}
                if k3 < best_key:
                    path, mesh, strip = path_dp, m3, s3
                    best_key = k3
                    repair["phase2_dp"]["applied"] = True
                else:
                    repair["phase2_dp"]["applied"] = False
            except Exception as exc:                           # noqa: BLE001
                repair["phase2_dp"] = {"error": str(exc)[:200]}
        # ---- Phase 3（用户 2026-10-09：确定性鲁棒性修复）：残差爬山环。
        # 为什么需要：Phase 1 只试**前 6 个**候选步（`pos[:6]`），Phase 2 的 DP
        # 目标是形状质量（形状/长边/法向代价），两者都不保证 total→0 ——
        # 实测同一 frozen J.2 对另一份 anatomy 输入停在 (severe 0, total 1)。
        # 这里从当前最优点出发，围绕**残余命中**位置穷举相邻步交换（不限个数，
        # 每轮上限 40 次试算），按 (severe, total) 严格改进接受并重算残余，
        # 直到 0 或无改进。仍然是"只翻对角线、一个顶点不动"的同一机制。
        # 注：`mesh/path/strip/best_key` 四元组在 Phase 1/2 里始终同步指向
        # **当前最优点**（Phase 1 未应用时即原始缝合结果），这里直接在其上爬山。
        if best_key[1] > 0:
            passes, n_tried = 0, 0
            while best_key[1] > 0 and passes < 6:
                passes += 1
                ev_cur = _hits(mesh)
                cand2 = set()
                for fa, fb in (ev_cur.get("self_intersections_hit_face_ids")
                               or []):
                    for f in (fa, fb):
                        kk = int(f) - strip_off
                        if 0 <= kk < len(path):
                            cand2.update((kk - 1, kk, kk + 1))
                pos2 = [p for p in sorted(cand2)
                        if 0 <= p and p + 1 < len(path)
                        and path[p] != path[p + 1]]
                improved_this = False
                for p in pos2[:40]:
                    p2 = list(path)
                    p2[p], p2[p + 1] = p2[p + 1], p2[p]
                    m2, s2 = _assemble(p2)
                    k2 = _key(_hits(m2))
                    n_tried += 1
                    if k2 < best_key:
                        path, mesh, strip = p2, m2, s2
                        best_key = k2
                        improved_this = True
                        break
                if not improved_this:
                    break
            repair["residual_hillclimb"] = {"passes": passes,
                                            "n_trials": n_tried,
                                            "final_severe": best_key[0],
                                            "final_total": best_key[1]}
        repair["n_severe_after"] = best_key[0]
        repair["n_hits_after"] = best_key[1]
    res.stitch_qc["seam_repair"] = repair
    if repair.get("n_hits_after", repair["n_hits_before"]) < repair["n_hits_before"]:
        res.notes.append(
            "J.2.1-C seam repair：自交 %d→%d（severe %d→%d），顶点一个没动"
            "（Phase1 局部翻转 / Phase2 DP 的取舍见 stitch_qc.seam_repair）"
            % (repair["n_hits_before"], repair["n_hits_after"],
               repair["n_severe_before"], repair["n_severe_after"]))

    # ---- H. 拓扑与安全门
    final_loops = _boundary_loops(np.asarray(mesh.faces, np.int64))
    main_loop = max(final_loops, key=len) if final_loops else []
    d_loop = ctree.query(np.asarray(mesh.vertices, float)[
        np.asarray(main_loop, np.int64)], k=1)[0] if main_loop else np.array([])
    orig = set(int(x) for x in loop_a)
    res.safety = geometry_safety_gate(
        np.asarray(mesh.vertices, float), np.asarray(mesh.faces, np.int64),
        reference_vertices=None, max_flip_ratio=None,
        # 自交不在 gate 里硬判 0：先按**尺度**分成 severe / micro（与 operand、
        # union 同一套口径），只有 severe 才 fail —— 证据全部记录，绝不隐藏。
        max_self_intersections=None, max_nonmanifold_edges=0,
        expect_open_boundary=True)
    sev = _selfint_severity(res.safety.get("metrics", {}),
                            np.asarray(mesh.vertices, float),
                            np.asarray(mesh.faces, np.int64),
                            min_area_mm2=max(params.voxel_mm ** 2 / 4.0, 1e-6))
    res.safety["self_intersection_severity"] = sev
    if sev.get("severe"):
        res.safety["ok"] = False
        res.safety.setdefault("violations", []).append(
            "self_intersections(severe)=%d" % sev["severe"])
    res.safety["topology"] = {
        "components": int(len(mesh.split(only_watertight=False))),
        "boundary_loops": int(len(final_loops)),
        "boundary_loop_sizes": sorted([len(x) for x in final_loops],
                                      reverse=True)[:5],
        "main_loop_to_margin_p95_mm": float(np.quantile(d_loop, .95))
        if len(d_loop) else None,
        "main_loop_to_margin_max_mm": float(d_loop.max())
        if len(d_loop) else None,
        "main_loop_original_margin_vertices": int(
            sum(1 for i in main_loop if i in orig)),
    }
    if len(final_loops) != 1:
        res.safety["ok"] = False
        res.safety.setdefault("violations", []).append(
            "boundary_loops=%d" % len(final_loops))
    if res.safety["topology"]["components"] != 1:
        res.safety["ok"] = False
        res.safety.setdefault("violations", []).append(
            "components=%d" % res.safety["topology"]["components"])

    # ---- I. 面级来源标记
    fc = np.asarray(mesh.triangles_center, np.float64)
    dm_f = ctree.query(fc, k=1)[0]
    d_fc = np.asarray(mesh.faces, np.int64)
    n_col = len(cf)
    label = np.full(len(fc), ANATOMY_RETAINED, dtype=object)
    label[:n_col] = COLLAR_FIXED
    in_band = dm_f <= params.d1_mm + params.guard_mm
    label[(~np.arange(len(fc)) < n_col) & in_band] = BOOLEAN_SEAM
    sq_a = SurfaceQuery(av, af, boundary_safety_mm=0.0, max_query_edge_mm=0.0)
    sq_e = SurfaceQuery(np.asarray(env_operand.vertices, np.float64),
                        np.asarray(env_operand.faces, np.int64),
                        boundary_safety_mm=0.0, max_query_edge_mm=0.0)
    body_idx = np.arange(n_col, len(fc))      # collar 之后的都是 body
    if len(body_idx):
        pts = fc[body_idx]
        d_anat = np.empty(len(pts))
        d_env = np.empty(len(pts))
        for i in range(0, len(pts), QCHUNK):
            d_anat[i:i + QCHUNK] = sq_a.query(pts[i:i + QCHUNK])[3]
            d_env[i:i + QCHUNK] = sq_e.query(pts[i:i + QCHUNK])[3]
        # 谁贴得近就归谁；两者都远（>tol）说明在 strip/seam 上
        lab = np.where(d_env < d_anat - params.retain_tol_mm,
                       ENVELOPE_DOMINATED, ANATOMY_RETAINED)
        lab[(dm_f[body_idx] <= params.d1_mm + params.guard_mm)] = BOOLEAN_SEAM
        ambiguous = (np.minimum(d_anat, d_env) > params.retain_tol_mm)
        lab[ambiguous] = BOOLEAN_SEAM
        label[body_idx] = lab
    areas = np.asarray(mesh.area_faces, np.float64)
    total_area = float(areas.sum())
    res.provenance = {
        "area_mm2": {k: float(areas[label == k].sum())
                     for k in (COLLAR_FIXED, ANATOMY_RETAINED,
                               ENVELOPE_DOMINATED, BOOLEAN_SEAM)},
        "area_frac": {k: float(areas[label == k].sum() / total_area)
                      for k in (COLLAR_FIXED, ANATOMY_RETAINED,
                                ENVELOPE_DOMINATED, BOOLEAN_SEAM)},
        "labels": label.tolist()[:0],       # 逐面标签过大，走 GLB 图层
        "note": ("多数健康 anatomy 应为 ANATOMY_RETAINED；原覆盖率失败区应为 "
                 "ENVELOPE_DOMINATED；若半颗牙都是 ENVELOPE_DOMINATED，"
                 "说明 anatomy prior 本身有问题"),
    }
    res.notes.append("来源面积：" + " · ".join(
        "%s %.1f%%" % (k, 100 * v) for k, v in res.provenance["area_frac"].items()))

    # ---- J. 内表面侧覆盖 + 分区偏差（射线口径）
    from .thickness import vertex_normals, ray_first_hit
    vn_i = vertex_normals(iv, ifa)
    sq_j = SurfaceQuery(np.asarray(mesh.vertices, np.float64),
                        np.asarray(mesh.faces, np.int64),
                        boundary_safety_mm=0.0, max_query_edge_mm=0.0)
    t_J, _ = ray_first_hit(iv, vn_i, sq_j.tri_original, max_t=4.0)
    t_A, _ = ray_first_hit(iv, vn_i, sq_a.tri_original, max_t=4.0)
    dm_i = mtree_i.query(iv, k=1)[0]
    Tj = required_thickness(dm_i, tp, blend_mm=params.margin_blend_mm,
                            safety_mm=0.0)
    okj = np.isfinite(t_J)
    ev = okj & (dm_i >= params.d1_mm)          # 与 Audit 的判据一致
    viol = ev & (t_J < Tj)
    in_ramp = viol & (dm_i < env_full)         # 被 guard band 推迟的那一段
    res.coverage = {
        "n_eval": int(ev.sum()),
        "t_ge_T_ratio": float((t_J[ev] >= Tj[ev]).mean()) if ev.any() else None,
        "n_violating": int(viol.sum()),
        "n_violating_in_ramp_band": int(in_ramp.sum()),
        "n_violating_beyond_ramp": int((viol & ~in_ramp).sum()),
        "ramp_band_mm": [float(params.d1_mm), float(env_full)],
        "n_negative": int((t_J[ev] <= 1e-6).sum()),
        "t_p05_mm": float(np.quantile(t_J[ev], .05)) if ev.any() else None,
        "t_p50_mm": float(np.median(t_J[ev])) if ev.any() else None,
        "t_min_mm": float(t_J[ev].min()) if ev.any() else None,
    }
    if res.coverage["n_violating_in_ramp_band"]:
        res.notes.append(
            "包络修正在 d ∈ [%.2f, %.2f] 的 guard band 里按比例推迟：该段 %d 个"
            "采样点仍未达 T（这是 seam 连续性的代价，如实计入，交给 Thickness "
            "Repair；band 之外剩 %d 点）"
            % (params.d1_mm, env_full, res.coverage["n_violating_in_ramp_band"],
               res.coverage["n_violating_beyond_ramp"]))
    oka = np.isfinite(t_A)
    valid = oka & okj & (t_A >= Tj - 0.02)
    dev = t_J - t_A

    def _st(mask):
        if not mask.any():
            return {"n": 0}
        w = dev[mask]
        return {"n": int(mask.sum()),
                "signed_median_mm": float(np.median(w)),
                "signed_p95_mm": float(np.quantile(w, .95)),
                "abs_p50_mm": float(np.median(np.abs(w))),
                "abs_p95_mm": float(np.quantile(np.abs(w), .95)),
                "abs_max_mm": float(np.abs(w).max())}
    res.deviation_split = {
        "ANATOMY_ALREADY_VALID": _st(valid),
        "ENVELOPE_REPAIRED": _st(oka & okj & ~valid),
        "note": "signed = t_J − t_A（正=变厚）；abs = 位移量级",
    }

    res.status = STATUS_OK
    if not res.safety.get("ok", False):
        res.status = STATUS_FAILED
    elif res.coverage["n_violating"] > 0:
        res.status = STATUS_REVIEW_REQUIRED
        res.notes.append("仍有 %d 个采样点 t<T（内表面侧）"
                         % res.coverage["n_violating"])
    res.mesh = mesh
    return res
