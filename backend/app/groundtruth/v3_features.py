"""Localization v3 —— 中心凸起特征（三阶段 CLI，纯 stdlib）。

    python3 -m app.groundtruth.v3_features --stage features --out /tmp/gt_v3 [--limit N]
    python3 -m app.groundtruth.v3_features --stage analyze --out /tmp/gt_v3
    python3 -m app.groundtruth.v3_features --stage render  --out /tmp/gt_v3

## 背景（为什么 v2 被证伪）

v2 判据"Center 附近有表面（dist_min<=3.5mm 且邻域有顶点）"经人工 QA
（31 例分层样本）precision 仅 ~0.17：sectioned 插座的洞缘、完整解剖牙列、
模型底座、扫描伪影都能满足"附近有表面"。
有区分力的是**形态**：预备体在 → 中心是凸起（prominence>0）；
代型被抽走 → 中心是洞（prominence<0）；其它 → 接近 0 或形态不符。

## 坐标系（用户规格，正式特征不做 fallback）

    Origin = Center
    Z = AxisOcclusalOfToothModel
    X = AxisMesial
    Y = Z × X

局部轴缺失 → ``feature_status = MISSING_LOCAL_FRAME``，
**不**偷偷用 [0,0,1] 代替（渲染预览可以 fallback，特征不行）。

## 特征带（提取区域定义，不是生产阈值）

    中心   r < 1.5 mm
    肩台   1.5 <= r < 3.5 mm
    外环   4   <= r < 6 mm

## 三阶段

Stage A ``features``：单趟读取（SHA256 + 顶点网格一次完成），
    每个 (case × FDI × 被审计资产) 一行 -> ``features.csv`` +
    ``features_summary.json``。之后 Stage B/C 不再扫 STL。
Stage B ``analyze``：把第一轮 31 个人工标签按
    (case_uuid, fdi, candidate_class=FULL_ARCH_WITH_TARGET) 严格 join
    （QA1 的 manifest 没有 asset_id，多个候选资产时标记 multi_asset），
    打印 A/B/C/U 各组特征分布。**不训练任何分类器。**
Stage C ``render``：按特征分层选 ~55 个二轮 QA 候选
    （高置信 / 临界 / v2 假阳 / v2 可能假阴 / METADATA / 冲突），
    渲染俯视+侧视深度图 + 径向高度曲线，``manifest_v2.json`` 留空 human_label。
"""

from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import json
import math
import os
import struct
import sys
from urllib.parse import quote, unquote
import time
from collections import Counter

from ..importers.exocad.construction_info import parse_construction_case
from .human_labels import QA1_LABELS  # noqa: F401  (QA 投影用，见下方注释)

#: 审计 CSV（Stage A 的输入；由 arch_audit 产出）。
DEFAULT_AUDIT = "arch_asset_audit.csv"
DEFAULT_ROOTS = ("/data1/welo/dentistry/exo/all_data",
                 "/data1/welo/dentistry/exo/EXO")

#: 特征带（提取区域，非生产阈值）。
CENTER_R = 1.5
SHOULDER_R = (1.5, 3.5)
RING_R = (4.0, 6.0)
LOCAL_RADIUS = 7.0

#: 人工判定表的**唯一来源**在 ``app/groundtruth/human_labels.py``
#: （QA1+QA2 两轮都在那里，``models.LocalizationGTSet`` 也从那里读）。
#: 见文件头的 import。

FEATURE_COLUMNS = (
    "case_uuid", "fdi", "asset_id", "asset_path_local_only",
    "candidate_class", "classification_basis", "metadata_role",
    "feature_status", "frame_source",
    "center_x", "center_y", "center_z",
    "center_z_median", "center_z_p75", "center_z_p90",
    "ring_z_median", "ring_z_p75", "ring_z_p90",
    "prominence_median", "prominence_p75", "prominence_abs",
    "z_min", "z_max", "z_range", "rel_pos",
    "center_spread", "ring_spread",
    "shoulder_contrast", "shoulder_contrast_abs",
    "radial_coverage", "radial_asymmetry",
    "center_count", "ring_count", "shoulder_count",
    "target_surface_distance", "near_count_r4", "near_coverage",
    "profile_r", "profile_z",
)
FEATURE_COLUMNS


# ================================================================ 单趟读取

def read_stl_single_pass(path: str, *, max_points: int = 120_000,
                         cell: float = 1.0) -> dict:
    """一次打开 STL，同时得到 SHA256[:16] 与顶点格子。

    返回 ``{"ok", "sha256", "faces", "bytes", "buckets", "n_points"}``。
    采样步长与 arch_audit 相同量级（~120k 点），足够 7mm 邻域的统计。
    """
    out = {"ok": False, "sha256": None, "faces": 0, "bytes": 0,
           "buckets": {}, "n_points": 0}
    try:
        size = os.path.getsize(path)
    except OSError as exc:
        out["error"] = f"stat_failed:{exc}"
        return out
    out["bytes"] = size
    if size < 84:
        out["error"] = "too_small"
        return out
    try:
        with open(path, "rb") as f:
            head = f.read(84)
            if head[:5] == b"solid":
                out["error"] = "ascii_stl"
                return out
            n = struct.unpack("<I", head[80:84])[0]
            if 84 + n * 50 != size:
                n = max(0, (size - 84) // 50)
            h = hashlib.sha256()
            h.update(head)
            buf = f.read(n * 50)
            h.update(buf)
    except OSError as exc:
        out["error"] = f"read_failed:{exc}"
        return out
    out["sha256"] = h.hexdigest()[:16]
    out["faces"] = n
    stride = max(1, (n * 3) // max_points) if n * 3 > max_points else 1
    buckets: dict[tuple[int, int, int], list] = {}
    cnt = 0
    kept = 0
    for i in range(n):
        v = struct.unpack_from("<12f", buf, i * 50)
        for j in (3, 6, 9):
            cnt += 1
            if cnt % stride:
                continue
            p = (v[j], v[j + 1], v[j + 2])
            key = (int(math.floor(p[0] / cell)),
                   int(math.floor(p[1] / cell)),
                   int(math.floor(p[2] / cell)))
            b = buckets.get(key)
            if b is None:
                buckets[key] = [p]
            else:
                b.append(p)
            kept += 1
    out["buckets"] = buckets
    out["n_points"] = kept
    out["ok"] = kept > 0
    return out


def points_near(buckets: dict, center, radius: float, *,
                cell: float = 1.0) -> list:
    cx, cy, cz = center
    r = int(math.ceil(radius / cell))
    bx, by, bz = int(math.floor(cx / cell)), int(math.floor(cy / cell)), \
        int(math.floor(cz / cell))
    out = []
    r2 = radius * radius
    for dx in range(-r, r + 1):
        for dy in range(-r, r + 1):
            for dz in range(-r, r + 1):
                for q in buckets.get((bx + dx, by + dy, bz + dz), ()):
                    if ((q[0]-cx)**2 + (q[1]-cy)**2 + (q[2]-cz)**2) <= r2:
                        out.append(q)
    return out


# ================================================================ 局部系与特征

def tooth_local_frame(axis_occ, axis_mesial):
    """Z=咬合方向, X=近中方向, Y=Z×X。任一缺失返回 None。"""
    if not axis_occ or not axis_mesial:
        return None
    zn = math.sqrt(sum(x * x for x in axis_occ))
    if zn < 1e-9:
        return None
    z = [x / zn for x in axis_occ]
    xn = math.sqrt(sum(x * x for x in axis_mesial))
    if xn < 1e-9:
        return None
    x = [v / xn for v in axis_mesial]
    d = sum(x[i] * z[i] for i in range(3))
    x = [x[i] - d * z[i] for i in range(3)]
    xn = math.sqrt(sum(v * v for v in x))
    if xn < 1e-6:
        return None
    x = [v / xn for v in x]
    y = [z[1] * x[2] - z[2] * x[1],
         z[2] * x[0] - z[0] * x[2],
         z[0] * x[1] - z[1] * x[0]]
    return z, x, y


def _jacobi_local(a, *, sweeps=24, tol=1e-12):
    m = [r[:] for r in a]
    v = [[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]]
    for _ in range(sweeps):
        if m[0][1]**2 + m[0][2]**2 + m[1][2]**2 <= tol:
            break
        for p, q in ((0, 1), (0, 2), (1, 2)):
            if abs(m[p][q]) <= tol:
                continue
            th = (m[q][q]-m[p][p]) / (2*m[p][q])
            t = (1 if th >= 0 else -1) / (abs(th) + math.sqrt(th*th+1))
            c = 1/math.sqrt(t*t+1)
            s2 = t*c
            for k in range(3):
                mkp, mkq = m[k][p], m[k][q]
                m[k][p] = c*mkp - s2*mkq
                m[k][q] = s2*mkp + c*mkq
            for k in range(3):
                mpk, mqk = m[p][k], m[q][k]
                m[p][k] = c*mpk - s2*mqk
                m[q][k] = s2*mpk + c*mqk
            for k in range(3):
                vkp, vkq = v[k][p], v[k][q]
                v[k][p] = c*vkp - s2*vkq
                v[k][q] = s2*vkp + c*vkq
    pr = sorted(((m[i][i], [v[0][i], v[1][i], v[2][i]]) for i in range(3)),
                key=lambda x: -x[0])
    return [x[0] for x in pr], [x[1] for x in pr]


def margin_pca_frame(margin_points):
    """颈缘 PCA 局部帧：法向 = 最小特征值方向，X = 最大面内主轴，Y = Z×X。

    v3.1 特征对 z 翻转不变，所以 PCA 法向的符号无关紧要。
    用于 ci 缺 AxisOcclusalOfToothModel/AxisMesial 的病例（实测 40%）。
    """
    if not margin_points or len(margin_points) < 10:
        return None
    n = len(margin_points)
    c = [sum(p[i] for p in margin_points)/n for i in range(3)]
    cov = [[0.0]*3 for _ in range(3)]
    for p in margin_points:
        d = [p[i]-c[i] for i in range(3)]
        for i in range(3):
            for j in range(3):
                cov[i][j] += d[i]*d[j]
    _vals, vecs = _jacobi_local(cov)
    z = vecs[2]
    x = vecs[0]
    y = [z[1]*x[2]-z[2]*x[1], z[2]*x[0]-z[0]*x[2], z[0]*x[1]-z[1]*x[0]]
    return z, x, y


def resolve_frame(tooth, margin_points):
    """帧来源优先级（用户规格）：TOOTH_FRAME → MARGIN_PCA → UNAVAILABLE。"""
    frame = tooth_local_frame(tooth.axis_occlusal_of_tooth_model,
                              tooth.axis_mesial)
    if frame is not None:
        return frame, "TOOTH_FRAME"
    frame = margin_pca_frame(margin_points)
    if frame is not None:
        return frame, "MARGIN_PCA"
    return None, "UNAVAILABLE"


def _med(a):
    a = sorted(a)
    return a[len(a) // 2]


def _pct(a, f):
    a = sorted(a)
    return a[min(len(a) - 1, int(f * (len(a) - 1)))]


def local_features(buckets: dict, center, frame, pts=None) -> dict | None:
    """牙位局部系特征（用户规格：中心/肩台/外环带 + 扇区连续性/不对称）。"""
    if frame is None:
        return None
    z, x, y = frame
    if pts is None:
        pts = points_near(buckets, center, LOCAL_RADIUS)
    center_b, sh_b, ring_b = [], [], []
    sect: dict[int, dict[str, list]] = {}
    for p in pts:
        rel = [p[i] - center[i] for i in range(3)]
        zz = sum(rel[i] * z[i] for i in range(3))
        xx = sum(rel[i] * x[i] for i in range(3))
        yy = sum(rel[i] * y[i] for i in range(3))
        r = math.hypot(xx, yy)
        if r < CENTER_R:
            center_b.append(zz)
        if SHOULDER_R[0] <= r < SHOULDER_R[1]:
            sh_b.append(zz)
            si = int((math.atan2(yy, xx) + math.pi) / (math.pi / 6)) % 12
            sect.setdefault(si, {}).setdefault("sh", []).append(zz)
        if RING_R[0] <= r < RING_R[1]:
            ring_b.append(zz)
            si = int((math.atan2(yy, xx) + math.pi) / (math.pi / 6)) % 12
            sect.setdefault(si, {}).setdefault("ring", []).append(zz)
    if len(center_b) < 5 or len(ring_b) < 10:
        return None
    sect_prom = []
    for si in range(12):
        s = sect.get(si)
        if s and len(s.get("sh", [])) >= 3 and len(s.get("ring", [])) >= 3:
            sect_prom.append(_med(s["sh"]) - _med(s["ring"]))
    prom_m = _med(center_b) - _med(ring_b)
    prom_75 = _pct(center_b, .75) - _pct(ring_b, .75)
    mean_sp = sum(sect_prom) / len(sect_prom) if sect_prom else 0.0
    # ---- v3.1 符号无关特征：z 极值取 0~6mm 全部邻域点（对 z 翻转不变）
    all_z = []
    for p in pts:
        rel = [p[i] - center[i] for i in range(3)]
        all_z.append(sum(rel[k] * z[k] for k in range(3)))
    z_min, z_max = min(all_z), max(all_z)
    z_range = z_max - z_min
    c_med = _med(center_b)
    rel_pos = ((c_med - z_min) / z_range) if z_range > 1e-6 else None

    def spread(a):
        return (_pct(a, .90) - _pct(a, .10)) if len(a) >= 5 else None

    sh_c = (_med(sh_b) - _med(ring_b)) if sh_b else None
    return {
        "center_z_median": _med(center_b),
        "center_z_p75": _pct(center_b, .75),
        "center_z_p90": _pct(center_b, .90),
        "ring_z_median": _med(ring_b),
        "ring_z_p75": _pct(ring_b, .75),
        "ring_z_p90": _pct(ring_b, .90),
        "prominence_median": prom_m,
        "prominence_p75": prom_75,
        "prominence_abs": abs(prom_m),
        "z_min": z_min, "z_max": z_max, "z_range": z_range,
        "rel_pos": rel_pos,
        "center_spread": spread(center_b),
        "ring_spread": spread(ring_b),
        "shoulder_contrast": sh_c,
        "shoulder_contrast_abs": abs(sh_c) if sh_c is not None else None,
        #: 角度连续性（符号无关）：12 扇区里"肩台相对外环的偏移方向一致"的比例。
        #: 取 max(正, 负) —— 凸起（多数扇区正）与插座（多数扇区负）都得高分，
        #: 噪声/邻牙尖路过则接近 0.5。翻转 z 后 max(1-fp, 1-fn) = max(fp, fn)。
        "radial_coverage": (max(sum(1 for v in sect_prom if v > 0),
                                sum(1 for v in sect_prom if v < 0))
                            / len(sect_prom) if sect_prom else None),
        "radial_asymmetry": (math.sqrt(sum((v - mean_sp) ** 2
                                           for v in sect_prom) / len(sect_prom))
                             if len(sect_prom) >= 4 else None),
        "center_count": len(center_b),
        "ring_count": len(ring_b),
        "shoulder_count": len(sh_b),
    }


def radial_profile_xy(points, center, frame) -> tuple[list[float], list[float]]:
    z, x, y = frame
    bins: dict[int, list] = {}
    for p in points:
        rel = [p[i] - center[i] for i in range(3)]
        zz = sum(rel[i] * z[i] for i in range(3))
        xx = sum(rel[i] * x[i] for i in range(3))
        yy = sum(rel[i] * y[i] for i in range(3))
        r = math.hypot(xx, yy)
        if r < 7.0:
            bins.setdefault(int(r / 0.5), []).append(zz)
    rs, zs = [], []
    for k in sorted(bins):
        rs.append(round((k + 0.5) * 0.5, 2))
        zs.append(_med(bins[k]))
    return rs, zs


# ================================================================ v3.2 格搜索

def in_plane_basis(zdir):
    """格平面基（与渲染 depth_map 相同的构造，保证峰位标记可换算像素）。"""
    a = [0.0, 0.0, 1.0] if abs(zdir[2]) < 0.9 else [1.0, 0.0, 0.0]
    e1 = [zdir[1]*a[2]-zdir[2]*a[1], zdir[2]*a[0]-zdir[0]*a[2],
          zdir[0]*a[1]-zdir[1]*a[0]]
    n = math.sqrt(sum(v*v for v in e1)) or 1.0
    e1 = [v/n for v in e1]
    e2 = [zdir[1]*e1[2]-zdir[2]*e1[1], zdir[2]*e1[0]-zdir[0]*e1[2],
          zdir[0]*e1[1]-zdir[1]*e1[0]]
    return e1, e2


def point_to_polyline_dist(px, py, poly_xy):
    """2D 点到折线（闭合环）的最近距离。poly 为 [(x,y), ...]。"""
    if not poly_xy:
        return None
    best = float("inf")
    n = len(poly_xy)
    for i in range(n):
        ax, ay = poly_xy[i]
        bx, by = poly_xy[(i+1) % n]
        dx, dy = bx-ax, by-ay
        seg2 = dx*dx + dy*dy
        t = 0.0 if seg2 < 1e-12 else max(
            0.0, min(1.0, ((px-ax)*dx + (py-ay)*dy)/seg2))
        d = math.hypot(px - (ax + t*dx), py - (ay + t*dy))
        if d < best:
            best = d
    return best


def point_in_polygon(px, py, poly_xy):
    """射线法。poly 至少 3 点；返回 None 当不可判定。"""
    if len(poly_xy) < 3:
        return None
    inside = False
    j = len(poly_xy) - 1
    for i in range(len(poly_xy)):
        xi, yi = poly_xy[i]
        xj, yj = poly_xy[j]
        if (yi > py) != (yj > py):
            xint = (xj - xi) * (py - yi) / (yj - yi) + xi
            if px < xint:
                inside = not inside
        j = i
    return inside


def lattice_search(buckets, center, zdir, *, margin_points=None,
                   half_mm: float = 6.0, step_mm: float = 1.0,
                   nms_dist_mm: float = 3.0, top_k: int = 3,
                   points=None):
    """v3.2：子中心格搜索。

    以 Center 为原点、在 zdir 的面内 (e1,e2) 上铺 ±half_mm、step_mm 网格，
    每格算局部特征；按 ``prominence_abs × coverage`` 排序做非极大值抑制
    （峰间 ≥ nms_dist_mm），返回 top_k 峰。

    **采样中心 ≠ Prep 轴心的实证**（cf_0335f16/36，偏移 6mm、峰值 31.9×）：
    死盯 Center 的固定圆盘会错过预备体，故搜索而不死盯。

    每峰带与 Margin 的关系（用户规格第 2 点：邻牙尖检查）：
      * ``dist_to_margin_mm``：峰位到颈缘折线（面内投影）的最近距离
      * ``in_margin_polygon``：峰位是否落在颈缘闭合环内
    预备体峰应在 margin 环内/边缘；邻牙尖通常在环外或沿环远处。

    margin_points 缺失时两个 margin 字段为 None（不装作算过）。
    """
    frame = None
    # zdir 由调用方给出；面内基与渲染一致
    e1, e2 = in_plane_basis(zdir)
    poly = None
    if margin_points and len(margin_points) >= 10:
        poly = []
        for p in margin_points:
            rel = [p[i] - center[i] for i in range(3)]
            poly.append((sum(rel[k]*e1[k] for k in range(3)),
                         sum(rel[k]*e2[k] for k in range(3))))
    n_steps = int(round(half_mm / step_mm))
    reach = LOCAL_RADIUS + half_mm + step_mm
    pool = (points if points is not None
            else points_near(buckets, center, reach) if buckets is not None
            else [])
    cands = []
    for i in range(2*n_steps + 1):
        for j in range(2*n_steps + 1):
            du = (i - n_steps) * step_mm
            dv = (j - n_steps) * step_mm
            sub = [center[k] + du*e1[k] + dv*e2[k] for k in range(3)]
            r2 = LOCAL_RADIUS * LOCAL_RADIUS
            sub_pts = [q for q in pool
                       if (q[0]-sub[0])**2 + (q[1]-sub[1])**2
                       + (q[2]-sub[2])**2 <= r2]
            f = local_features(None, sub, (zdir, e1, e2), pts=sub_pts)
            if f is None:
                continue
            cov = f.get("radial_coverage")
            score = (f["prominence_abs"] or 0.0) * (cov or 0.0)
            cands.append({"du": du, "dv": dv, "score": score,
                          "prominence_abs": f["prominence_abs"],
                          "rel_pos": f.get("rel_pos"),
                          "shoulder_contrast_abs":
                              f.get("shoulder_contrast_abs"),
                          "radial_coverage": cov,
                          "center_spread": f.get("center_spread")})
    if not cands:
        return []
    cands.sort(key=lambda c: -c["score"])
    peaks = []
    for c in cands:
        if all(math.hypot(c["du"]-q["du"], c["dv"]-q["dv"]) >= nms_dist_mm
               for q in peaks):
            c["dist_to_margin_mm"] = point_to_polyline_dist(
                c["du"], c["dv"], poly) if poly is not None else None
            c["in_margin_polygon"] = (point_in_polygon(
                c["du"], c["dv"], poly) if poly is not None else None)
            peaks.append(c)
            if len(peaks) >= top_k:
                break
    return peaks


# ================================================================ Stage A

def stage_features(out: str, audit: str = DEFAULT_AUDIT,
                   roots=DEFAULT_ROOTS, limit: int = 0) -> dict:
    t0 = time.time()
    os.makedirs(out, exist_ok=True)
    rows = list(csv.DictReader(open(audit)))
    by_case: dict[str, list] = {}
    for r in rows:
        by_case.setdefault(r["source_case_id"], []).append(r)
    if limit:
        by_case = dict(sorted(by_case.items())[:limit])
    # 病例目录是**嵌套**的（实测 all_data/exo2/<id>、all_data/complete_cases/<id>、
    # EXO/CAD-Data/<id> 等），且同一 id 可能出现在多处 —— discover_cases 按目录
    # basename 聚合，同 id 的文件被合并。这里一次性递归建 cid -> [目录...] 映射。
    dirs_by_cid: dict[str, list] = {}
    for root in roots:
        for dp, _dn, _fn in os.walk(root):
            name = os.path.basename(dp)
            if name:
                dirs_by_cid.setdefault(name, []).append(dp)

    status_counts: Counter = Counter()
    out_rows: list[dict] = []
    n_assets = 0
    for ci_i, (cid, arows) in enumerate(sorted(by_case.items()), 1):
        case_uuid = arows[0]["case_uuid"]
        want = {r["asset_id"]: r for r in arows}
        # 病例目录（两个 root；递归，修 QA1 的 7 例"找不到资产"）
        stls = []
        for base in dirs_by_cid.get(cid, []):
            for dp, _dn, fn in os.walk(base):
                for f in fn:
                    if f.lower().endswith(".stl"):
                        p = os.path.join(dp, f)
                        try:
                            stls.append((os.path.getsize(p), p))
                        except OSError:
                            pass
        stls.sort(reverse=True)
        # 解析 ci 一次
        ci_files = []
        for base in dirs_by_cid.get(cid, []):
            for dp, _dn, fn in os.walk(base):
                for f in fn:
                    if f.endswith(".constructionInfo") or f.endswith(".modelInfo"):
                        ci_files.append(os.path.join(dp, f))
        case = None
        if ci_files:
            ci_files.sort(key=lambda p: (not p.endswith(".constructionInfo"),
                                         os.path.getsize(p)))
            try:
                case = parse_construction_case(ci_files[0])
            except Exception:
                case = None
        matched = set()
        for _sz, path in stls[:6]:
            got = read_stl_single_pass(path)
            n_assets += 1
            h16 = got["sha256"]
            if h16 not in want or h16 in matched:
                continue
            matched.add(h16)
            ar = want[h16]
            try:
                fdi = int(ar["target_fdi"])
            except (TypeError, ValueError):
                continue
            t = case.tooth(fdi) if case else None
            row = {
                "case_uuid": case_uuid, "fdi": fdi, "asset_id": h16,
                # 服务器本机路径仅用于 Stage C 回读；目录名含非 ASCII（EXO口扫），
                # percent-encode 保持 CSV 纯 ASCII，读时 unquote 还原。
                "asset_path_local_only": quote(path),
                "candidate_class": ar["candidate_class"],
                "classification_basis": ar["classification_basis"],
                "metadata_role": ar["metadata_role"],
                "target_surface_distance": ar["target_surface_distance"],
                "near_count_r4": ar.get("near_count_r4", ""),
                "near_coverage": ar.get("near_coverage", ""),
                "center_x": "", "center_y": "", "center_z": "",
            }
            if t is None or not t.center:
                row["feature_status"] = "NO_CENTER"
                row["frame_source"] = "UNAVAILABLE"
                status_counts["NO_CENTER"] += 1
                out_rows.append(row)
                continue
            row["center_x"], row["center_y"], row["center_z"] = \
                [round(float(v), 4) for v in t.center]
            frame, frame_src = resolve_frame(t, t.margin_points)
            if frame is None:
                row["feature_status"] = "FEATURE_UNAVAILABLE"
                row["frame_source"] = "UNAVAILABLE"
                status_counts["FEATURE_UNAVAILABLE"] += 1
                out_rows.append(row)
                continue
            row["frame_source"] = frame_src
            f = local_features(got["buckets"], tuple(t.center), frame)
            if f is None:
                row["feature_status"] = "NO_LOCAL_POINTS"
                row["frame_source"] = frame_src
                status_counts["NO_LOCAL_POINTS"] += 1
                out_rows.append(row)
                continue
            prof = radial_profile_xy(
                points_near(got["buckets"], tuple(t.center), LOCAL_RADIUS),
                tuple(t.center), frame)
            row.update({k: f[k] for k in f if k != "profile"})
            row["feature_status"] = "OK"
            row["profile_r"] = ";".join(f"{v:.2f}" for v in prof[0])
            row["profile_z"] = ";".join(f"{v:.3f}" for v in prof[1])
            status_counts["OK"] += 1
            out_rows.append(row)
        # 审计里有、但 top-6 里没匹配到的资产
        for h16, ar in want.items():
            if h16 in matched:
                continue
            status_counts["ASSET_NOT_REACHED"] += 1
        if ci_i % 50 == 0:
            print(f"  ...{ci_i}/{len(by_case)} 病例，资产 {n_assets}，"
                  f"行 {len(out_rows)}（{time.time()-t0:.0f}s）", flush=True)

    with open(os.path.join(out, "features.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(FEATURE_COLUMNS))
        w.writeheader()
        for r in out_rows:
            w.writerow({k: r.get(k, "") for k in FEATURE_COLUMNS})
    summary = {
        "n_audit_rows": len(rows), "n_cases": len(by_case),
        "n_assets_read": n_assets, "n_feature_rows": len(out_rows),
        "status_counts": dict(status_counts),
        "elapsed_s": round(time.time() - t0, 1),
    }
    json.dump(summary, open(os.path.join(out, "features_summary.json"), "w"),
              indent=1)
    print(json.dumps(summary, indent=1))
    return summary


# ================================================================ Stage B

def stage_analyze(out: str) -> dict:
    path = os.path.join(out, "features.csv")
    rows = list(csv.DictReader(open(path)))
    # QA1 join：(case_uuid, fdi) + candidate_class==WITH_TARGET（QA1 只从该类抽）。
    # 同键多资产时标 multi_asset，取 target_surface_distance 最小的行
    # （与 QA1 选样一致），并在 join_status 里如实标注。
    qa1_by_key: dict[tuple[str, int], str] = {}
    for (cu, fdi), v in QA1_LABELS.items():
        for cu_full in {r["case_uuid"] for r in rows}:
            if cu_full.startswith(cu):
                qa1_by_key[(cu_full, fdi)] = v
    joined, multi = [], 0
    for r in rows:
        key = (r["case_uuid"], int(r["fdi"]) if r["fdi"].isdigit() else -1)
        if key not in qa1_by_key:
            continue
        if r["candidate_class"] != "FULL_ARCH_WITH_TARGET":
            continue
        r = dict(r)
        same = [x for x in rows
                if (x["case_uuid"], x["fdi"]) == (r["case_uuid"], r["fdi"])
                and x["candidate_class"] == "FULL_ARCH_WITH_TARGET"]
        r["qa1"] = qa1_by_key[key]
        r["join_status"] = "multi_asset" if len(same) > 1 else "unique"
        multi += 1 if len(same) > 1 else 0
        joined.append(r)
    joined.sort(key=lambda r: float(r["target_surface_distance"] or 9e9))
    best: dict[tuple[str, int], dict] = {}
    for r in joined:
        best.setdefault((r["case_uuid"], int(r["fdi"])), r)
    final = sorted(best.values(), key=lambda r: r["qa1"])

    qpath = os.path.join(out, "qa1_features.csv")
    with open(qpath, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(FEATURE_COLUMNS) + ["qa1",
                                                                 "join_status"])
        w.writeheader()
        for r in final:
            w.writerow({k: r.get(k, "") for k in FEATURE_COLUMNS} |
                       {"qa1": r["qa1"], "join_status": r["join_status"]})

    lines = [f"join：{len(final)} 例（multi_asset {multi}）"]
    for grp in ("A", "B", "C", "U"):
        rs = [r for r in final if r["qa1"] == grp]
        lines.append(f"-- {grp}（{len(rs)} 例）--")
        for key in ("rel_pos", "prominence_abs", "prominence_median",
                    "center_spread", "ring_spread", "shoulder_contrast_abs",
                    "radial_coverage", "radial_asymmetry", "z_range"):
            vals = sorted(float(x[key]) for x in rs
                          if x.get(key) not in ("", None))
            if vals:
                lines.append(f"   {key:20s} min={vals[0]:+.3f} "
                             f"med={vals[len(vals)//2]:+.3f} max={vals[-1]:+.3f}")
            else:
                lines.append(f"   {key:20s} (无数据)")
        prom = sorted((float(r["prominence_median"]), r["case_uuid"][:10],
                       r["fdi"]) for r in rs if r.get("prominence_median"))
        lines.append("   prom 明细: " + ", ".join(f"{a:+.2f}({b}/{c})"
                                                for a, b, c in prom))
    text = "\n".join(lines)
    open(os.path.join(out, "qa1_projection.txt"), "w").write(text)
    print(text)
    return {"n_joined": len(final), "multi_asset": multi}


# ================================================================ Stage C

def stage_lattice(out: str, audit: str = DEFAULT_AUDIT,
                  roots=DEFAULT_ROOTS, limit: int = 0) -> dict:
    """v3.2: per (case,fdi,asset) sub-center lattice search -> lattice.csv."""
    t0 = time.time()
    fpath = os.path.join(out, "features.csv")
    frows = list(csv.DictReader(open(fpath)))
    todo = [r for r in frows
           if r["feature_status"] in ("OK", "NO_LOCAL_POINTS")]
    print("lattice rows:", len(todo), "/", len(frows), flush=True)
    cid_by_uuid = {}
    for root in roots:
        for dp, dn, _f in os.walk(root):
            for d in dn:
                k = "cf_" + hashlib.sha256(d.encode()).hexdigest()[:16]
                cid_by_uuid.setdefault(k, d)
    cols = ["case_uuid", "fdi", "asset_id", "feature_status",
            "candidate_class", "classification_basis", "metadata_role",
            "target_surface_distance",
            "peak1_score", "peak1_du", "peak1_dv", "peak1_prom",
            "peak1_rel_pos", "peak1_shoulder", "peak1_cov",
            "peak1_spread", "peak1_dist_margin", "peak1_in_poly",
            "peak2_score", "peak2_du", "peak2_dv", "peak2_prom",
            "peak3_score", "peak3_du", "peak3_dv", "peak3_prom"]
    lpath = os.path.join(out, "lattice.csv")
    done = set()
    if os.path.exists(lpath):
        for r in csv.DictReader(open(lpath)):
            done.add((r["case_uuid"], r["fdi"], r["asset_id"]))
    lf = open(lpath, "a", newline="")
    w = csv.DictWriter(lf, fieldnames=cols)
    if not done:
        w.writeheader()
    n = 0
    for r in todo:
        key = (r["case_uuid"], r["fdi"], r["asset_id"])
        if key in done:
            continue
        row = {k: r.get(k, "") for k in cols}
        cid = cid_by_uuid.get(r["case_uuid"])
        path = unquote(r["asset_path_local_only"])
        cifs = []
        if cid is not None:
            for root in roots:
                cifs += glob.glob(os.path.join(root, cid,
                                  "*.constructionInfo"))
                cifs += glob.glob(os.path.join(root, cid, "*.modelInfo"))
        case = None
        if cifs:
            cifs.sort(key=lambda q: not q.endswith(".constructionInfo"))
            try:
                case = parse_construction_case(cifs[0])
            except Exception:
                case = None
        t = case.tooth(int(r["fdi"])) if case else None
        if t is None or not t.center:
            row["peak1_score"] = "NO_CENTER"
            w.writerow(row)
            lf.flush()
            continue
        frame, src = resolve_frame(t, t.margin_points)
        if frame is None:
            row["peak1_score"] = "FEATURE_UNAVAILABLE"
            w.writerow(row)
            lf.flush()
            continue
        if not os.path.exists(path):
            row["peak1_score"] = "ASSET_MISSING"
            w.writerow(row)
            lf.flush()
            continue
        got = read_stl_single_pass(path)
        if not got["ok"]:
            row["peak1_score"] = "READ_FAILED"
            w.writerow(row)
            lf.flush()
            continue
        center = tuple(t.center)
        peaks = lattice_search(got["buckets"], center, frame[0],
                               margin_points=t.margin_points)
        for k, pk in enumerate(peaks[:3], 1):
            row["peak%d_score" % k] = round(pk["score"], 4)
            row["peak%d_du" % k] = pk["du"]
            row["peak%d_dv" % k] = pk["dv"]
            row["peak%d_prom" % k] = round(pk["prominence_abs"], 4)
        p1 = peaks[0] if peaks else None
        if p1 is not None:
            rp = p1.get("rel_pos")
            row["peak1_rel_pos"] = None if rp is None else round(rp, 4)
            sh = p1.get("shoulder_contrast_abs")
            row["peak1_shoulder"] = None if sh is None else round(sh, 4)
            cv = p1.get("radial_coverage")
            row["peak1_cov"] = None if cv is None else round(cv, 4)
            sp = p1.get("center_spread")
            row["peak1_spread"] = None if sp is None else round(sp, 4)
            dm = p1.get("dist_to_margin_mm")
            row["peak1_dist_margin"] = None if dm is None else round(dm, 3)
            row["peak1_in_poly"] = p1.get("in_margin_polygon")
        w.writerow(row)
        lf.flush()
        n += 1
        if n % 50 == 0:
            print("  ...%d rows (%.0fs)" % (n, time.time()-t0), flush=True)
    lf.close()
    summary = {"n_done": n, "elapsed_s": round(time.time()-t0, 1)}
    print(json.dumps(summary))
    return summary

def stage_render(out: str) -> dict:
    """按特征分层选二轮 QA 并渲染（俯视 | 侧视 | 径向曲线）。"""
    from . import qa_render as qr
    path = os.path.join(out, "features.csv")
    rows = [r for r in csv.DictReader(open(path))
            if r["feature_status"] == "OK"]
    for r in rows:
        r["asset_path_local_only"] = unquote(r["asset_path_local_only"])
    for r in rows:
        for k in ("prominence_median", "shoulder_contrast", "radial_coverage",
                  "radial_asymmetry"):
            try:
                r[k] = float(r[k]) if r[k] != "" else None
            except ValueError:
                r[k] = None

    def rk(r):
        return r["prominence_median"] or 0.0

    wt = sorted((r for r in rows
                 if r["candidate_class"] == "FULL_ARCH_WITH_TARGET"),
                key=rk, reverse=True)
    neg = sorted((r for r in rows
                  if r["candidate_class"] == "FULL_ARCH_TARGET_MISSING"),
                 key=rk, reverse=True)
    unk = sorted((r for r in rows if r["candidate_class"] not in
                  ("FULL_ARCH_WITH_TARGET", "FULL_ARCH_TARGET_MISSING")),
                 key=rk, reverse=True)
    sel: list[tuple[str, dict]] = []
    sel += [("high", r) for r in wt[:15]]
    mid0 = max(1, len(wt) // 2 - 5)
    sel += [("border", r) for r in wt[mid0:mid0 + 10]]
    sel += [("v2_fp_low", r) for r in wt[-10:]]
    sel += [("recall_neg", r) for r in neg[:10]]
    sel += [("meta", r) for r in wt
            if r["classification_basis"].startswith("METADATA")][:5]
    sel += [("unknown", r) for r in unk[:5]]
    seen: set = set()
    final = []
    for tag, r in sel:
        k = (r["case_uuid"], r["fdi"])
        if k not in seen:
            seen.add(k)
            final.append((tag, r))
    print(f"QA2 选样 {len(final)}", flush=True)

    os.makedirs(out, exist_ok=True)
    manifest2, tiles = [], []
    for tag, r in final:
        path_stl = r["asset_path_local_only"]
        if not os.path.exists(path_stl):
            continue
        center = (float(r["center_x"]), float(r["center_y"]),
                  float(r["center_z"]))
        got = read_stl_single_pass(path_stl, max_points=400_000)
        if not got["ok"]:
            continue
        # 渲染用三角形（邻域），方向 fallback 允许（仅预览）
        tris = _tris_near(path_stl, center, qr.WIN_MM * 1.6)
        if not tris:
            continue
        d = _view_dir_fallback(center, tris)
        top, _ = qr.depth_map(tris, center, d)
        e1, _e2 = qr.basis(d)
        side, _ = qr.depth_map(tris, center, e1)
        prof_r = [float(v) for v in r["profile_r"].split(";") if v]
        prof_z = [float(v) for v in r["profile_z"].split(";") if v]
        prof = _profile_tile(prof_r, prof_z)
        tiles.append((top, side, prof))
        manifest2.append({
            "idx": len(tiles) - 1, "tag": tag,
            "case_uuid": r["case_uuid"], "fdi": int(r["fdi"]),
            "asset_id": r["asset_id"], "v2_class": r["candidate_class"],
            "basis": r["classification_basis"],
            "v3_score": rk(r),
            "prominence": r["prominence_median"],
            "shoulder": r["shoulder_contrast"],
            "coverage": r["radial_coverage"],
            "asymmetry": r["radial_asymmetry"],
            "surf_dist": r["target_surface_distance"],
            "human_label": "",
        })
        print(f"  [{len(tiles)-1}] {tag:11s} {r['case_uuid'][:12]} "
              f"fdi={r['fdi']} prom={rk(r):+.2f}", flush=True)

    per = 5
    gw, gh = qr.GRID, 90
    for si in range(0, len(tiles), per):
        chunk = tiles[si:si + per]
        rows_img = []
        for top, side, prof in chunk:
            for pv in range(gw):
                rows_img.append(list(top[pv]) + [128]*8 + list(side[pv])
                                + [128]*8 + list(prof[pv % gh]
                                                 if pv < gh else [255]*gw))
            rows_img += [[30]*int(gw*3+16)]*6
        w = gw*3 + 16
        h = len(rows_img)
        for j in range(min(per, len(tiles)-si)):
            for k in range(si + j + 1):
                if k < w:
                    for rr in range(j*(gw+6), j*(gw+6)+4):
                        if rr < h:
                            rows_img[rr][k] = 255
        qr.write_png(os.path.join(out, f"sheet_{si//per}.png"), w, h, rows_img)
    json.dump(manifest2, open(os.path.join(out, "manifest_v2.json"), "w"),
              indent=1)
    print(f"QA2 渲染：{len(tiles)} 例，"
          f"{len(glob.glob(os.path.join(out, 'sheet_*.png')))} 张拼图")
    return {"n_selected": len(final), "n_rendered": len(tiles)}


def _tris_near(path, center, radius):
    import struct as _s
    size = os.path.getsize(path)
    with open(path, "rb") as f:
        head = f.read(84)
        if head[:5] == b"solid" or size < 84:
            return None
        n = _s.unpack("<I", head[80:84])[0]
        if 84 + n * 50 != size:
            n = max(0, (size - 84) // 50)
        buf = f.read(n * 50)
    r2 = radius * radius
    tris = []
    for i in range(n):
        v = _s.unpack_from("<12f", buf, i * 50)
        a = (v[3], v[4], v[5]); b = (v[6], v[7], v[8]); c = (v[9], v[10], v[11])
        for p in (a, b, c):
            if sum((p[k]-center[k])**2 for k in range(3)) <= r2:
                tris.append((a, b, c))
                break
    return tris


def _view_dir_fallback(center, tris):
    """渲染预览的视线方向 fallback（允许；正式特征不用它）：
    取邻域顶点 PCA 最小方差方向（法向近似），指向"顶点更多的一侧"。"""
    pts = [p for t in tris for p in t]
    n = len(pts)
    cx = sum(p[0] for p in pts)/n
    cy = sum(p[1] for p in pts)/n
    cz = sum(p[2] for p in pts)/n
    cov = [[0.0]*3 for _ in range(3)]
    for p in pts[:20000]:
        dx, dy, dz = p[0]-cx, p[1]-cy, p[2]-cz
        cov[0][0] += dx*dx; cov[0][1] += dx*dy; cov[0][2] += dx*dz
        cov[1][1] += dy*dy; cov[1][2] += dy*dz; cov[2][2] += dz*dz
    cov[1][0], cov[2][0], cov[2][1] = cov[0][1], cov[0][2], cov[1][2]
    vals, vecs = _jacobi_eigen_local(cov)
    d = vecs[-1]                      # 最小方差方向 = 法向
    # 指向点更多的一侧（让相机在"上面"）
    side = sum((p[0]-cx)*d[0] + (p[1]-cy)*d[1] + (p[2]-cz)*d[2]
               for p in pts[:5000])
    if side < 0:
        d = [-v for v in d]
    return d


def _jacobi_eigen_local(a, *, sweeps=24, tol=1e-12):
    m = [r[:] for r in a]
    v = [[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]]
    for _ in range(sweeps):
        off = m[0][1]**2 + m[0][2]**2 + m[1][2]**2
        if off <= tol:
            break
        for p, q in ((0, 1), (0, 2), (1, 2)):
            if abs(m[p][q]) <= tol:
                continue
            theta = (m[q][q]-m[p][p]) / (2*m[p][q])
            t = (1 if theta >= 0 else -1) / (abs(theta)
                                             + math.sqrt(theta*theta+1))
            c = 1/math.sqrt(t*t+1)
            s = t*c
            for k in range(3):
                mkp, mkq = m[k][p], m[k][q]
                m[k][p] = c*mkp - s*mkq
                m[k][q] = s*mkp + c*mkq
            for k in range(3):
                mpk, mqk = m[p][k], m[q][k]
                m[p][k] = c*mpk - s*mqk
                m[q][k] = s*mpk + c*mqk
            for k in range(3):
                vkp, vkq = v[k][p], v[k][q]
                v[k][p] = c*vkp - s*vkq
                v[k][q] = s*vkp + c*vkq
    pairs = sorted(((m[i][i], [v[0][i], v[1][i], v[2][i]]) for i in range(3)),
                   key=lambda x: -x[0])
    return [p[0] for p in pairs], [p[1] for p in pairs]


def _profile_tile(prof_r, prof_z, w=None, h=None):
    """径向高度曲线小图：z(r) 折线，0~7mm。"""
    gw = w or qr.GRID
    gh = h or 90
    img = [[255]*gw for _ in range(gh)]
    if not prof_r:
        return img
    zs = [v for v in prof_z]
    lo, hi = min(zs), max(zs)
    rng = (hi - lo) or 1.0
    rmax = max(prof_r) or 1.0
    # 轴
    for x in range(gw):
        img[gh-1][x] = 120
    for y in range(gh):
        img[y][0] = 120
    pts = [(int(r/rmax*(gw-8))+4, int((1-(z-lo)/rng)*(gh-14))+6)
           for r, z in zip(prof_r, zs)]
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        steps = max(abs(x1-x0), abs(y1-y0)) or 1
        for s in range(steps+1):
            xx = int(x0 + (x1-x0)*s/steps)
            yy = int(y0 + (y1-y0)*s/steps)
            if 0 <= yy < gh and 0 <= xx < gw:
                img[yy][xx] = 0
                if yy+1 < gh:
                    img[yy+1][xx] = 0
    return img


# ================================================================ CLI

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Localization v3 features")
    ap.add_argument("--stage", required=True,
                    choices=("features", "analyze", "lattice", "render"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--audit", default=DEFAULT_AUDIT)
    ap.add_argument("--root", action="append", default=None)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args(argv)
    roots = tuple(a.root) if a.root else DEFAULT_ROOTS
    if a.stage == "features":
        stage_features(a.out, audit=a.audit, roots=roots, limit=a.limit)
    elif a.stage == "lattice":
        stage_lattice(a.out, audit=a.audit, roots=roots, limit=a.limit)
    elif a.stage == "analyze":
        stage_analyze(a.out)
    else:
        stage_render(a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
