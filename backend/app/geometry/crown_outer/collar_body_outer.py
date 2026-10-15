"""J.2.1 —— Fixed Cervical Collar + Implicit Body + Conforming Stitch。

用户 2026-10-05 规格（J.2 冻结为"融合语义✓ / 提取质量✗"之后）：

    A. fixed cervical collar   ← 来自 H.1 anatomy，**margin 精确保留**
    B. implicit body           ← Anatomy ∪ Envelope 的 level set
    在高于 Margin 的稳定截面（d_margin = d1）做 conforming stitch

场按区域切换（**不让 T(x) 硬降到 margin 的 0 再要求 MC 解析无限变薄的楔**）：

    d_margin <= d0 :  F = φ_anatomy
    d0 < d < d1    :  F = blend(φ_anatomy, min(φ_anatomy, s_int − T))
    d_margin >= d1 :  F = min(φ_anatomy, s_int − T)

**为什么这样更安全**：margin 边界不再是"MC 随便产生一圈再吸附"（吸附只该做数值微调，
不该承担 4mm 级拓扑修复），而是**原 anatomy 的 boundary 顶点本身** —— 拓扑上就是
prescribed cervical loop。

其它纪律（都来自本轮用户规格）：
* 窄带内**所有参与 min() 的子场**（φ_anatomy、s_int）都用精确点-三角距离重算；
  只精确一个子场仍会让 min() 的交界抖。
* MC 出网格后**先做组件/拓扑审计**（连通分量、边界环、Euler、非流形、自交、绕向），
  不达标 **fail loud** —— 不让 remesh 把拓扑失败磨平。
* 本模块**不做自由 remesh**（怕 remesh 把表面往 envelope 里拉）；只做
  "去垃圾组件 + 绕向/拓扑修复 + 颈缘对应 + 受限缝合"。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

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
STATUS_FAILED = "FAILED"


class CollarBodyError(Exception):
    pass


@dataclass
class CollarBodyParams:
    thickness: ThicknessParameters
    voxel_mm: float = 0.10
    pad_mm: float = 1.6
    max_voxels: int = 12_000_000
    #: collar 区（F = φ_anatomy，完全不进 level set 重建）
    d0_mm: float = 0.30
    #: 过渡带上边界 = 缝合截面位置。**必须 ≤ Audit 的判定带起判点 0.6mm**：
    #  Audit 只判 d_margin ≥ 0.6 的采样点，而过渡带里包络只按比例生效，
    #  d1=1.0 时 0.6~1.0 段只被部分修正 → 覆盖指标反而退回 72.8%（实测）。
    d1_mm: float = 0.60
    #: T 规则的颈缘混合带（体内规则仍用它；coller 区不参与，所以不会造薄楔）
    margin_blend_mm: float = 0.10
    #: 构造侧余量（保证 ≥ Audit 判定规则）
    safety_mm: float = 0.05
    #: 窄带宽度：|·| < band 的子场用精确点-三角距离重算
    band_mm: float = 0.35
    #: 缝合时把 body 侧截面环吸到 collar 截面曲线上的容差
    stitch_snap_tol_mm: float = 0.60
    #: 组件/拓扑审计门
    require_single_component: bool = True
    require_single_boundary_loop: bool = True
    max_self_intersections: int = 0
    max_nonmanifold_edges: int = 0


@dataclass
class CollarBodyResult:
    status: str = STATUS_FAILED
    mesh: object | None = None
    collar: dict = field(default_factory=dict)
    body: dict = field(default_factory=dict)
    stitch: dict = field(default_factory=dict)
    topology: dict = field(default_factory=dict)
    deviation_split: dict = field(default_factory=dict)
    coverage: dict = field(default_factory=dict)
    safety: dict = field(default_factory=dict)
    notes: list = field(default_factory=list)

    def summary(self) -> dict:
        return {"status": self.status, "collar": self.collar, "body": self.body,
                "stitch": self.stitch, "topology": self.topology,
                "deviation_split": self.deviation_split,
                "coverage": self.coverage,
                "safety_ok": self.safety.get("ok"),
                "safety_violations": self.safety.get("violations")}


# ---------------------------------------------------------------- 基础工具

def _boundary_loops(faces) -> list[list[int]]:
    """边界边 → 有序闭环列表（每条边界边恰好属于一个环）。"""
    from collections import defaultdict
    cnt: dict = defaultdict(int)
    for a, b, c in np.asarray(faces, np.int64):
        for u, w in ((int(a), int(b)), (int(b), int(c)), (int(c), int(a))):
            cnt[(min(u, w), max(u, w))] += 1
    bedges = [e for e, n in cnt.items() if n == 1]
    adj: dict = defaultdict(list)
    for u, w in bedges:
        adj[u].append(w)
        adj[w].append(u)
    loops, seen = [], set()
    for start in sorted(adj):
        if start in seen:
            continue
        loop, prev, cur = [start], None, start
        seen.add(start)
        while True:
            nxt = [x for x in adj[cur] if x != prev and x not in seen]
            if not nxt:
                # 回到起点或断链
                if start in adj[cur] and len(loop) > 2:
                    break
                break
            prev, cur = cur, nxt[0]
            seen.add(cur)
            loop.append(cur)
        if len(loop) >= 3:
            loops.append(loop)
    return loops


def _clip_by_scalar(v, f, phi, level: float = 0.0, *, return_face_sources=False):
    """按顶点标量裁剪网格：保留 ``phi <= level`` 的一侧，边上插值出新顶点。

    返回 ``(verts, faces, n_added)``；与原网格顶点**共享**保留下来的部分
    （新顶点追加在后面），这样颈缘处的原顶点 id 原样保留 ——
    "margin boundary = 原 anatomy boundary" 这条才成立。
    """
    v = np.asarray(v, np.float64)
    f = np.asarray(f, np.int64)
    p = np.asarray(phi, np.float64)
    keep = p <= level
    out_faces = []
    sources=[]
    new_v = [tuple(x) for x in v]
    cache: dict = {}

    def crossing(a: int, b: int) -> int:
        if p[a]==level: return a
        if p[b]==level: return b
        key = (a, b) if a < b else (b, a)
        if key in cache:
            return cache[key]
        t = (level - p[key[0]]) / (p[key[1]] - p[key[0]])
        pt = v[key[0]] + t * (v[key[1]] - v[key[0]])
        idx = len(new_v)
        new_v.append(tuple(pt))
        cache[key] = idx
        return idx

    for source_id,(a, b, c) in enumerate(f):
        start=len(out_faces)
        a, b, c = int(a), int(b), int(c)
        ins = [keep[a], keep[b], keep[c]]
        n_in = int(sum(ins))
        if n_in == 0:
            continue
        if n_in == 3:
            out_faces.append((a, b, c))
            sources.append(source_id)
            continue
        verts = (a, b, c)
        if n_in == 1:                       # 一个顶点在内 → 一个三角形
            i = ins.index(True)
            p0, p1, p2 = verts[i], verts[(i + 1) % 3], verts[(i + 2) % 3]
            out_faces.append((p0, crossing(p0, p1), crossing(p2, p0)))
        else:                               # 两个顶点在内 → 四边形（p1,p2,m20,m01）
            i = ins.index(False)
            p0, p1, p2 = verts[i], verts[(i + 1) % 3], verts[(i + 2) % 3]
            m01, m20 = crossing(p0, p1), crossing(p2, p0)
            out_faces.append((p1, p2, m20))
            out_faces.append((p1, m20, m01))
            # ⚠️ 不允许再补 (m01, m20, p0)：那是三角形里**被裁掉**的部分。
            # 补上会让截面弦 (m01,m20) 被用两次（于是不算边界边），同时
            # 凭空造出 (p0,m01)/(m20,p0) 两条只被用一次的假边界边 ——
            # 边界环于是"绕道被裁顶点"走，裁剪后凭空多出环。
            # 实测（干净 icosphere + 平面裁剪，本该只有一圈纯插值点）：
            # 222 条边界边里 128 条 mixed（原顶点—插值点），环数虚高。
        sources.extend([source_id]*(len(out_faces)-start))
    retained=[i for i,face in enumerate(out_faces) if len(set(face))==3]
    result=(np.asarray(new_v, np.float64), np.asarray([out_faces[i] for i in retained], np.int64).reshape(-1,3),len(new_v)-len(v))
    return (*result,np.asarray(sources,np.int64)[retained]) if return_face_sources else result


def _order_loop_by_edges(v, loop_vertices, faces, ring_edges):
    """按 ring_edges（属于该环的边）把顶点排成闭环（几何无关，纯拓扑）。"""
    from collections import defaultdict
    adj: dict = defaultdict(list)
    for u, w in ring_edges:
        adj[int(u)].append(int(w))
        adj[int(w)].append(int(u))
    if not adj:
        return []
    start = min(adj)
    loop, prev, cur = [start], None, start
    while True:
        nxt = [x for x in adj[cur] if x != prev]
        if not nxt:
            break
        prev, cur = cur, nxt[0]
        if cur == start:
            break
        loop.append(cur)
        if len(loop) > len(adj) + 1:
            break
    return loop


def _split_edge(faces, u: int, w: int, nid: int):
    """在 faces 里找含**无向**边 (u,w) 的三角面，把顶点 nid 插进那条边。

    有向边取 tri 的绕向（x→y），插点后拆成 (x, nid, z) + (nid, y, z)，
    绕向保持不变 —— 这样边界折线几何不变、又不产生 T 型接头。
    """
    for fi, tri in enumerate(faces):
        for k in range(3):
            x, y, z = tri[k], tri[(k + 1) % 3], tri[(k + 2) % 3]
            if (x == u and y == w) or (x == w and y == u):
                faces[fi] = (x, nid, z)
                faces.append((nid, y, z))
                return faces
    raise CollarBodyError("边界边 (%d,%d) 找不到所属三角面" % (u, w))


def _conforming_ring(v, f, ring, n: int):
    """把边界环按弧长重采样成 n 点，并把新点**插入**网格边界边。

    返回 ``(v2, f2, ring2)``：``ring2`` 是 n 个顶点 id，按弧长均匀分布。
    每个插入点严格落在原边界线段上 → 边界折线几何不变；插点同时拆开相邻
    三角面 → 与网格是**共形**的（没有 T 型接头）。这正是用户要的
    "统一采样 → conforming zipper"：两侧环点数一致后，缝带是 1:1 的
    正常三角带，不再是长条折叠三角。
    """
    from collections import defaultdict
    v = np.asarray(v, np.float64)
    v_new = [tuple(float(x) for x in p) for p in v]
    faces = [tuple(int(x) for x in t) for t in np.asarray(f, np.int64)]
    ring = [int(x) for x in ring]
    pts = v[np.asarray(ring, np.int64)]
    q = np.vstack([pts, pts[:1]])
    seg = np.linalg.norm(np.diff(q, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    total = float(s[-1])
    if total <= 1e-12:
        raise CollarBodyError("环长为 0 —— 无法做共形重采样")
    tgt = np.linspace(0.0, total, n, endpoint=False)
    sidx = np.clip(np.searchsorted(s, tgt, side="right") - 1, 0, len(seg) - 1)
    frac = (tgt - s[sidx]) / np.maximum(seg[sidx], 1e-12)
    ring2, by_edge = [], defaultdict(list)
    for k in range(n):
        a = ring[int(sidx[k])]
        b = ring[(int(sidx[k]) + 1) % len(ring)]
        t = float(frac[k])
        if t < 1e-9:                       # 正好落在原顶点上 → 复用
            ring2.append(a)
            continue
        nid = len(v_new)
        v_new.append(tuple(v[a] + t * (v[b] - v[a])))
        ring2.append(nid)
        by_edge[(min(a, b), max(a, b))].append((t, a, b, nid))
    for (a, b), items in by_edge.items():
        cur = None
        for _t, ea, eb, nid in sorted(items):
            cur = ea if cur is None else cur
            faces = _split_edge(faces, cur, eb, nid)
            cur = nid
    return (np.asarray(v_new, np.float64), np.asarray(faces, np.int64),
            np.asarray(ring2, np.int64))


def _zip_rings_conforming(v, ring_a, ring_b, *, swap=False):
    """两条**等长**环之间 1:1 三角形带（共形缝合的第二步）。

    起点对齐（最近点对）+ 方向对齐（邻段点积），然后 i → i+1 连两片三角。
    """
    A = np.asarray(ring_a, np.int64)
    B = np.asarray(ring_b, np.int64)
    if len(A) != len(B) or len(A) < 3:
        raise CollarBodyError("共形缝合要求两环等长（%d vs %d）"
                              % (len(A), len(B)))
    va, vb = v[A], v[B]
    d = np.linalg.norm(va[:, None, :] - vb[None, :, :], axis=2)
    i0, j0 = np.unravel_index(int(np.argmin(d)), d.shape)
    A = np.roll(A, -i0)
    B = np.roll(B, -j0)
    va, vb = v[A], v[B]
    if np.dot(va[1] - va[0], vb[1] - vb[0]) + \
            np.dot(va[-1] - va[0], vb[-1] - vb[0]) < 0:
        B = B[::-1]
        vb = v[B]
    n = len(A)
    tris = []
    for i in range(n):
        j = (i + 1) % n
        tris.append((A[i], B[i], A[j]))
        tris.append((A[j], B[i], B[j]))
    tris = np.asarray(tris, np.int64)
    if swap:
        tris = tris[:, ::-1]
    return tris


def _align_rings(v, ring_a, ring_b):
    """对齐两条环：起点取最近点对，方向按邻段走向统一。返回 (A, B, arca, arcb)。"""
    A = np.asarray(ring_a, np.int64).copy()
    B = np.asarray(ring_b, np.int64).copy()
    if len(A) < 3 or len(B) < 3:
        raise CollarBodyError("缝合环太短：%d / %d" % (len(A), len(B)))
    va, vb = v[A], v[B]
    d = np.linalg.norm(va[:, None, :] - vb[None, :, :], axis=2)
    i0, j0 = np.unravel_index(int(np.argmin(d)), d.shape)
    A = np.roll(A, -i0)
    B = np.roll(B, -j0)
    va, vb = v[A], v[B]
    same_dir = np.dot(va[1] - va[0], vb[1] - vb[0]) + \
        np.dot(va[-1] - va[0], vb[-1] - vb[0])
    if same_dir < 0:
        B = B[::-1]
    arca = np.concatenate([[0.0], np.cumsum(np.linalg.norm(
        np.diff(np.vstack([va, va[:1]]), axis=0), axis=1))])
    vbb = v[B]
    arcb = np.concatenate([[0.0], np.cumsum(np.linalg.norm(
        np.diff(np.vstack([vbb, vbb[:1]]), axis=0), axis=1))])
    return A, B, arca, arcb


def _zip_path(arca, arcb, nA: int, nB: int) -> list:
    """greedy 弧长路径：长度 nA+nB 的 'a'/'b' 序列（每步推进一侧）。

    这是**环带三角化的 lattice 路径** —— 相邻两步交换 = 该四边形的对角线翻转。
    """
    path, i, j, guard = [], 0, 0, 0
    while (i < nA or j < nB) and guard < 4 * (nA + nB):
        guard += 1
        ma = arca[i + 1] if i + 1 <= nA else np.inf
        mb = arcb[j + 1] if j + 1 <= nB else np.inf
        if ma <= mb:
            path.append("a")
            i += 1
        else:
            path.append("b")
            j += 1
    return path


def _tris_from_path(A, B, path) -> np.ndarray:
    """按路径生成三角带（与 `_zip_rings` 同拓扑，但可任意改路径）。"""
    A = np.asarray(A, np.int64)
    B = np.asarray(B, np.int64)
    nA, nB = len(A), len(B)
    tris, i, j = [], 0, 0
    for s in path:
        if s == "a":
            tris.append((A[i % nA], A[(i + 1) % nA], B[j % nB]))
            i += 1
        else:
            tris.append((A[i % nA], B[(j + 1) % nB], B[j % nB]))
            j += 1
    return np.asarray(tris, np.int64)


def _path_steps(path) -> list:
    """每个三角形对应的 (i, j, side)，用于把 defect 面映射回 lattice 位置。"""
    out, i, j = [], 0, 0
    for s in path:
        out.append((i, j, s))
        if s == "a":
            i += 1
        else:
            j += 1
    return out


def _tri_cost_batch(P1, P2, P3, prev_n, band: float, w: dict):
    """一批三角形的代价 + 法向（DP 的转移代价；与上一片反号 = 折叠 → ∞）。"""
    e1, e2, e3 = P2 - P1, P3 - P1, P3 - P2
    cr = np.cross(e1, e2)
    twoA = np.linalg.norm(cr, axis=1)
    n = cr / np.maximum(twoA[:, None], 1e-18)
    l1 = np.linalg.norm(e1, axis=1)
    l2 = np.linalg.norm(e2, axis=1)
    l3 = np.linalg.norm(e3, axis=1)
    emax = np.maximum(np.maximum(l1, l2), l3)
    q = 2.0 * np.sqrt(3.0) * (0.5 * twoA) / np.maximum(
        l1 ** 2 + l2 ** 2 + l3 ** 2, 1e-18)
    cost = w["shape"] * (1.0 - np.clip(q, 0.0, 1.0))
    cost += w["long"] * np.maximum(0.0, emax - 2.0 * max(band, 1e-6)) ** 2 \
        / max(band, 1e-6) ** 2
    has_prev = np.linalg.norm(prev_n, axis=1) > 0.0
    d = np.einsum("ij,ij->i", n, prev_n)
    cost += np.where(has_prev, w["normal"] * (1.0 - np.clip(d, -1.0, 1.0)), 0.0)
    cost = np.where(has_prev & (d <= 0.0), 1e18, cost)   # 与上一片反号 = 折叠
    cost = np.where(twoA <= 1e-12, 1e18, cost)           # 退化
    return cost, n


def _dp_seam_path(v, A, B, *, band: float | None = None,
                  w: dict | None = None) -> list:
    """环带三角化的**全局最优 lattice 路径**（J.2.1-C Phase 2 的 DP zipper）。

    状态 ``(i, j, s)``：已推进 i 个 A 顶点、j 个 B 顶点，s ∈ {0,1} 表示上一小步
    推的是 A 还是 B。合法转移只有 `a`（i+1）与 `b`（j+1）→ **天然无 T-junction**，
    也天然覆盖 GT-free 的 276 ↔ 781。转移代价（用户规格）：

        形状(1 - 归一化质量) + 长边惩罚 + 法向不连续惩罚；**法向反号 → ∞**

    最后回溯 (0,0) → (nA,nB) 的最小代价路径。不新增、不移动任何顶点。
    """
    A = np.asarray(A, np.int64)
    B = np.asarray(B, np.int64)
    nA, nB = len(A), len(B)
    vv = np.asarray(v, np.float64)
    PA, PB = vv[A], vv[B]
    w = w or {"shape": 1.0, "long": 1.0, "normal": 8.0}
    if band is None:
        # 按弧长比例对齐两环，取中位间距当"缝带高度"的尺度
        idxB = np.clip((np.arange(nA) / max(nA, 1) * nB).astype(int),
                       0, nB - 1)
        band = max(float(np.median(np.linalg.norm(PA - PB[idxB], axis=1))),
                   1e-3)
    INF = 1e18
    cost = np.full((nA + 1, nB + 1, 2), INF)
    back = np.zeros((nA + 1, nB + 1, 2), np.int8)
    cost[0, 0, 0] = cost[0, 0, 1] = 0.0
    nrm = np.zeros((nA + 1, nB + 1, 2, 3))       # 到达该状态时"上一片"的法向
    for d in range(0, nA + nB):
        i_lo, i_hi = max(0, d - nB), min(d, nA)
        if i_lo > i_hi:
            continue
        ii = np.arange(i_lo, i_hi + 1)
        jj = d - ii
        for s in (0, 1):
            c0 = cost[ii, jj, s]
            if not np.isfinite(c0).any():
                continue
            # --- 转移到 (i+1, j)：推 A，出三角形 (A[i], A[i+1], B[j])
            okA = ii < nA
            if okA.any():
                i1, j1 = ii[okA], jj[okA]
                ia, ib = i1 % nA, (i1 + 1) % nA
                jb = j1 % nB
                tc, tn = _tri_cost_batch(PA[ia], PA[ib], PB[jb],
                                         nrm[i1, j1, s], band, w)
                nt = c0[okA] + tc
                upd = nt < cost[i1 + 1, j1, 0]
                tgt = (i1[upd] + 1, j1[upd])
                cost[tgt[0], tgt[1], 0] = nt[upd]
                back[tgt[0], tgt[1], 0] = s
                nrm[tgt[0], tgt[1], 0] = tn[upd]
            # --- 转移到 (i, j+1)：推 B，出三角形 (A[i], B[j+1], B[j])
            okB = jj < nB
            if okB.any():
                i1, j1 = ii[okB], jj[okB]
                jb, jb1 = j1 % nB, (j1 + 1) % nB
                tc, tn = _tri_cost_batch(PA[i1 % nA], PB[jb1], PB[jb],
                                         nrm[i1, j1, s], band, w)
                nt = c0[okB] + tc
                upd = nt < cost[i1, j1 + 1, 1]
                tgt = (i1[upd], j1[upd] + 1)
                cost[tgt[0], tgt[1], 1] = nt[upd]
                back[tgt[0], tgt[1], 1] = s
                nrm[tgt[0], tgt[1], 1] = tn[upd]
    # 回溯
    s = 0 if cost[nA, nB, 0] <= cost[nA, nB, 1] else 1
    if not np.isfinite(cost[nA, nB, s]):
        raise CollarBodyError("DP 缝合无可行路径（代价全 ∞）")
    i, j, path = nA, nB, []
    while i > 0 or j > 0:
        sp = int(back[i, j, s])
        if s == 0:
            path.append("a")
            i -= 1
        else:
            path.append("b")
            j -= 1
        s = sp
    path.reverse()
    return path


def _zip_rings(v, ring_a, ring_b, *, flip_hint=False):
    """两个闭环之间生成三角形带（conforming stitch）。

    两个环的顶点数不同也没问题：按弧长比例推进两侧指针。绕向由调用方用
    "绕向不一致边 = 0" 这条硬判据校验（不对就翻过来）。
    """
    A, B, arca, arcb = _align_rings(v, ring_a, ring_b)
    tris = _tris_from_path(A, B, _zip_path(arca, arcb, len(A), len(B)))
    if flip_hint:
        tris = tris[:, ::-1]
    return tris


# ---------------------------------------------------------------- 主入口

def build_collar_body_outer(anatomy_mesh, intaglio_mesh, margin_curve, axis,
                            params: CollarBodyParams) -> CollarBodyResult:
    """J.2.1 主入口。"""
    if params is None or params.thickness is None:
        raise CollarBodyError("CollarBodyParams.thickness 必须显式给出")
    from skimage import measure as sk_measure
    import trimesh

    tp = params.thickness
    res = CollarBodyResult()
    av = np.asarray(anatomy_mesh.vertices, np.float64)
    af = np.asarray(anatomy_mesh.faces, np.int64)
    iv = np.asarray(intaglio_mesh.vertices, np.float64)
    ifa = np.asarray(intaglio_mesh.faces, np.int64)
    loop_a = np.asarray(sorted({int(x) for e in _boundary_edges(af)
                                for x in e}), np.int64)
    loop_i = np.asarray(sorted({int(x) for e in _boundary_edges(ifa)
                                for x in e}), np.int64)
    if len(loop_a) == 0:
        raise CollarBodyError("anatomy 缺颈环边界")
    z = np.asarray(axis, float)
    z = z / np.linalg.norm(z)
    mpts = np.asarray(margin_curve, np.float64)      # prescribed cervical loop
    curve = np.vstack([mpts, 0.5 * (mpts + np.roll(mpts, -1, axis=0))])
    ctree = cKDTree(curve)
    d_margin_a = ctree.query(av, k=1)[0]             # anatomy 顶点的 d_margin

    # ---- A. collar：把 anatomy 按 d_margin <= d1 裁出来（margin 原顶点原样保留）
    cv, cf, n_new = _clip_by_scalar(av, af, d_margin_a, level=params.d1_mm)
    # collar 底环（= prescribed margin 那圈原顶点）与截面环
    be_c = _boundary_edges(cf)
    cut_edges = [e for e in be_c if e[0] >= len(av) or e[1] >= len(av)]
    base_edges = [e for e in be_c if e[0] < len(av) and e[1] < len(av)]
    base_loop = _order_loop_by_edges(cv, None, cf, base_edges)
    cut_loop = _order_loop_by_edges(cv, None, cf, cut_edges)
    res.collar = {
        "n_faces": int(len(cf)), "n_new_vertices": int(n_new),
        "n_base_loop_vertices": int(len(base_loop)),
        "n_cut_loop_vertices": int(len(cut_loop)),
        "base_is_original_margin": bool(
            len(base_loop) and all(i < len(av) for i in base_loop)),
    }
    if len(base_loop) == 0 or len(cut_loop) == 0:
        raise CollarBodyError("collar 裁切失败：底环 %d / 截面环 %d"
                              % (len(base_loop), len(cut_loop)))
    # 底环必须就是 prescribed margin（原顶点）
    d_base = ctree.query(cv[np.asarray(base_loop, np.int64)], k=1)[0] \
        if len(base_loop) else np.array([np.inf])
    res.collar["base_loop_to_margin_p95_mm"] = float(np.quantile(d_base, .95))
    res.collar["base_loop_to_margin_max_mm"] = float(d_base.max())

    # ---- B. 隐式体
    lo = np.minimum(av.min(axis=0), iv.min(axis=0)) - params.pad_mm
    hi = np.maximum(av.max(axis=0), iv.max(axis=0)) + params.pad_mm
    spacing = float(params.voxel_mm)
    n = np.ceil((hi - lo) / spacing).astype(np.int64) + 1
    while int(np.prod(n)) > params.max_voxels:
        spacing *= 1.25
        n = np.ceil((hi - lo) / spacing).astype(np.int64) + 1
    shape = tuple(int(x) for x in n)
    res.body["grid"] = {"shape": list(shape), "spacing_mm": spacing,
                        "n_voxels": int(np.prod(n))}

    seed_a = av.mean(axis=0)
    seed_i = iv.mean(axis=0) + z * 0.5
    field_a, _sa = _signed_field(av, af, loop_a, lo, spacing, shape, seed_a)
    field_i, _si = _signed_field(iv, ifa, loop_i, lo, spacing, shape, seed_i)

    # ---- 窄带精确化：min() 的两个子场都要精确（只精确一个照样抖）
    grid_pts = np.stack(np.meshgrid(
        lo[0] + spacing * np.arange(shape[0]),
        lo[1] + spacing * np.arange(shape[1]),
        lo[2] + spacing * np.arange(shape[2]), indexing="ij"), axis=-1
    ).reshape(-1, 3)
    band = (np.abs(field_a) < params.band_mm).ravel() | \
           (np.abs(field_i) < params.band_mm + float(tp.min_thickness_mm)).ravel()
    n_exact = 0
    if band.any():
        sq_a = SurfaceQuery(av, af, boundary_safety_mm=0.0,
                            max_query_edge_mm=0.0)
        sq_i = SurfaceQuery(iv, ifa, boundary_safety_mm=0.0,
                            max_query_edge_mm=0.0)
        pts = grid_pts[band]

        def _dist_chunked(sq, q, chunk=50000):
            """分块查询：一次 42 万点 × 264 候选 = 2.5GB，会 OOM（实测）。"""
            out = np.empty(len(q), np.float64)
            for i in range(0, len(q), chunk):
                out[i:i + chunk] = sq.query(q[i:i + chunk])[3]
            return out

        da = _dist_chunked(sq_a, pts)
        di = _dist_chunked(sq_i, pts)
        fa = field_a.ravel()
        fi = field_i.ravel()
        fa[band] = np.where(fa[band] < 0, -da, da)      # 符号沿用 EDT+连通域
        fi[band] = np.where(fi[band] < 0, -di, di)
        field_a = fa.reshape(shape)
        field_i = fi.reshape(shape)
        n_exact = int(band.sum())
    res.body["n_band_points_exact"] = n_exact
    res.notes.append("窄带精确化：%d 格（%.1f%%）用精确点-三角距离（φ_anatomy + s_int）"
                     % (n_exact, 100.0 * n_exact / float(np.prod(n))))

    # ---- T_ext / d_margin 场 / 分区融合
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
    d_m = ctree.query(grid_pts, k=1)[0].reshape(shape)

    body_F = np.minimum(field_a, field_i - T_ext)
    t_frac = np.clip((d_m - params.d0_mm)
                     / max(params.d1_mm - params.d0_mm, 1e-9), 0.0, 1.0)
    F = (1.0 - t_frac) * field_a + t_frac * body_F
    # **提取前掩码**（不是提取后按面心切）：让 level set 在腰带处干净收口成
    # **一个**环。上一版按面心过滤切出了两条以上闭合曲线 → body 裂成两块大块
    # （56245+29012 面）、"只留最大组件"丢掉 152.7mm² 真实表面。
    F = np.where(d_m < params.d1_mm, 1e6, F)
    # 注意：**不再**用"颈缘之下"硬掩码。collar 由 A 提供，body 只需按
    # d_margin ≥ d1 裁掉颈缘带；硬掩码会在颈缘附近切出碎面（实测碎成 1753 个
    # 组件、54 个边界环）。颈缘以下的场自然为正在（内表面止于颈缘）。
    res.body["region_frac"] = {
        "collar": float((d_m <= params.d0_mm).mean()),
        "transition": float(((d_m > params.d0_mm)
                             & (d_m < params.d1_mm)).mean()),
        "body": float((d_m >= params.d1_mm).mean())}

    verts, faces, _nrm, _val = sk_measure.marching_cubes(
        F, level=0.0, spacing=(spacing, spacing, spacing))
    verts = verts + lo[None, :]
    bv, bf = verts, faces
    bm = trimesh.Trimesh(bv, bf, process=True)
    # **去掉垃圾组件**（用户步骤：extract → remove garbage components → …）。
    # 选择判据 = "**是否含截面环**"（离 d1 腰带 ≤2 voxel 的顶点数），不再按面积 ——
    # 面积最大 ≠ 含截面环的那个（上一版就是这么丢掉 152.7mm² 真实表面的）。
    parts = bm.split(only_watertight=False)
    n_parts = len(parts)
    belt = []
    for p_ in parts:
        pv = np.asarray(p_.vertices, np.float64)
        dd = ctree.query(pv, k=1)[0]
        belt.append(int((dd <= params.d1_mm + 2 * spacing).sum()))
    keep_i = int(np.argmax(belt)) if max(belt) > 0         else int(np.argmax([float(p_.area) for p_ in parts]))
    areas = [float(p_.area) for p_ in parts]
    dropped = sorted(areas, reverse=True)[:3]
    n_dropped_faces = int(sum(len(p_.faces) for i, p_ in enumerate(parts)
                              if i != keep_i))
    bm = parts[keep_i]
    bv = np.asarray(bm.vertices, np.float64)
    bf = np.asarray(bm.faces, np.int64)
    res.body.update({"n_faces": int(len(bf)), "n_vertices": int(len(bv)),
                     "n_components_raw": int(n_parts),
                     "n_dropped_faces": n_dropped_faces,
                     "kept_component_belt_vertices": int(belt[keep_i]),
                     "top3_component_areas_mm2": [round(x, 2)
                                                  for x in dropped]})

    # ---- 缝合：body 截面环吸到 collar 截面曲线 → 三角形带
    be_b = _boundary_edges(bf)
    body_ring = _order_loop_by_edges(bv, None, bf, be_b)
    if len(body_ring) < 3:
        raise CollarBodyError("body 截面环缺失（%d 个顶点）" % len(body_ring))
    cut_pts = np.vstack([cv[np.asarray(cut_loop, np.int64)],
                         0.5 * (cv[np.asarray(cut_loop, np.int64)]
                                + np.roll(cv[np.asarray(cut_loop, np.int64)],
                                          -1, axis=0))])
    ctree_cut = cKDTree(cut_pts)
    dd, nn = ctree_cut.query(bv[np.asarray(body_ring, np.int64)], k=1)
    snap = dd <= params.stitch_snap_tol_mm
    bv = bv.copy()
    bv[np.asarray(body_ring, np.int64)[snap]] = cut_pts[nn[snap]]
    res.stitch = {"body_ring": int(len(body_ring)),
                  "collar_cut_ring": int(len(cut_loop)),
                  "n_snapped": int(snap.sum()),
                  "max_snap_mm": float(dd.max()) if len(dd) else 0.0,
                  "n_out_of_tol": int((~snap).sum())}

    strip = _zip_rings(bv, body_ring, cut_loop)
    # 合并：collar 顶点在前（索引不变 → margin 原顶点保持原 id）
    v_all = np.vstack([cv, bv])
    offset = len(cv)
    f_all = np.vstack([cf, bf + offset, strip + offset])
    mesh = trimesh.Trimesh(v_all, f_all, process=False)

    # ---- 组件 / 拓扑审计（先审再审，不达标 fail loud）
    comps = mesh.split(only_watertight=False)
    n_comp = len(comps)
    loops = _boundary_loops(np.asarray(mesh.faces, np.int64))
    res.topology = {
        "n_components": int(n_comp),
        "component_face_counts": sorted(
            [int(len(c.faces)) for c in comps], reverse=True)[:10],
        "n_boundary_loops": len(loops),
        "boundary_loop_sizes": sorted([len(x) for x in loops], reverse=True)[:10],
        "euler_number": int(mesh.euler_number),
    }
    # 期望：1 组件、1 边界环、且该环 = prescribed margin（原顶点）
    main_loop = max(loops, key=len) if loops else []
    if main_loop:
        mv = np.asarray(mesh.vertices, np.float64)
        d_loop = ctree.query(mv[np.asarray(main_loop, np.int64)], k=1)[0]
        res.topology["main_loop_to_margin_p95_mm"] = float(
            np.quantile(d_loop, .95))
        res.topology["main_loop_to_margin_max_mm"] = float(d_loop.max())
        orig = set(int(x) for x in loop_a)
        res.topology["main_loop_original_margin_vertices"] = int(
            sum(1 for i in main_loop if i in orig))
        res.topology["main_loop_vertex_count"] = int(len(main_loop))
    pre_fail = []
    if params.require_single_component and n_comp != 1:
        pre_fail.append("components=%d" % n_comp)
    if params.require_single_boundary_loop and len(loops) != 1:
        pre_fail.append("boundary_loops=%d" % len(loops))
    res.topology["fail_loud"] = pre_fail
    if pre_fail:
        res.notes.append("**拓扑审计未过**：%s —— 先报出来，不让 remesh 磨平"
                         % pre_fail)

    # ---- 几何安全门
    res.safety = geometry_safety_gate(
        np.asarray(mesh.vertices, np.float64),
        np.asarray(mesh.faces, np.int64),
        reference_vertices=None,
        max_flip_ratio=None,
        max_self_intersections=params.max_self_intersections,
        max_nonmanifold_edges=params.max_nonmanifold_edges,
        expect_open_boundary=True)
    if pre_fail:
        res.safety["ok"] = False
        res.safety.setdefault("violations", []).extend(pre_fail)

    # ---- 内表面侧覆盖 + 分区偏差（射线口径）
    from .thickness import vertex_normals, ray_first_hit
    vn_i = vertex_normals(iv, ifa)
    sq_j = SurfaceQuery(np.asarray(mesh.vertices, np.float64),
                        np.asarray(mesh.faces, np.int64),
                        boundary_safety_mm=0.0, max_query_edge_mm=0.0)
    sq_a = SurfaceQuery(av, af, boundary_safety_mm=0.0, max_query_edge_mm=0.0)
    t_J, _ = ray_first_hit(iv, vn_i, sq_j.tri_original, max_t=4.0)
    t_A, _ = ray_first_hit(iv, vn_i, sq_a.tri_original, max_t=4.0)
    dm_i = mtree_i.query(iv, k=1)[0]
    Tj = required_thickness(dm_i, tp, blend_mm=params.margin_blend_mm,
                            safety_mm=0.0)
    okj = np.isfinite(t_J)
    ev = okj & (dm_i >= 0.6)                       # 与 Audit 同口径
    res.coverage = {
        "n_eval": int(ev.sum()),
        "t_ge_T_ratio": float((t_J[ev] >= Tj[ev]).mean()) if ev.any() else None,
        "n_violating": int((t_J[ev] < Tj[ev]).sum()),
        "t_p05_mm": float(np.quantile(t_J[ev], .05)) if ev.any() else None,
        "t_p50_mm": float(np.median(t_J[ev])) if ev.any() else None,
        "n_negative": int((t_J[ev] <= 1e-6).sum()),
    }
    oka = np.isfinite(t_A)
    valid = oka & okj & (t_A >= Tj - 0.02)
    dev = t_J - t_A

    def _st(mask):
        if not mask.any():
            return {"n": 0}
        w = dev[mask]
        return {"n": int(mask.sum()),
                "median_mm": float(np.median(w)),
                "p90_mm": float(np.quantile(w, .90)),
                "p95_mm": float(np.quantile(w, .95)),
                "max_mm": float(w.max()),
                "abs_p95_mm": float(np.quantile(np.abs(w), .95))}
    res.deviation_split = {
        "ANATOMY_ALREADY_VALID": _st(valid),
        "ENVELOPE_REPAIRED": _st(oka & okj & ~valid),
        "note": "clean mesh 复测（拓扑审计%s）"
                % ("通过" if not pre_fail else "未过 —— 该统计需打折看"),
    }

    res.status = STATUS_OK
    for k in ("ok",):
        if not res.safety.get(k, False):
            res.status = STATUS_FAILED
    if res.status == STATUS_OK and res.coverage["n_violating"] > 0:
        res.status = STATUS_REVIEW_REQUIRED
        res.notes.append("仍有 %d 个采样点 t<T（内表面侧）"
                         % res.coverage["n_violating"])
    res.mesh = mesh
    return res
