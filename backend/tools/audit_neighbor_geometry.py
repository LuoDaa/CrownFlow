#!/usr/bin/env python3
"""NeighborGeometryAvailability —— 邻牙几何可用性审计（提交 I 前置，纯 stdlib）。

目标（用户规格）：**不分割整颗 45/47**，只回答"每例能否可靠拿到近中/远中
邻接表面局部 patch"，并顺便用 Final Crown GT 反求技师实际做出的邻接几何。

流程（逐例）：
  1. 从冻结索引取 full_arch_asset_id，按内容哈希在病例目录找到牙弓 STL；
  2. 同坐标系证明：margin 点到牙弓顶点的最近距离分布（中位 / p95 / 超界数）；
  3. 患者系裁 ROI（尺寸由该例 margin 尺寸导出，不写死临床常量）：
     mesial 沿 axis_mesial 正向、distal 反向，u ∈ [margin边+0.8, +13mm]，
     v/w 由 margin 跨度与固定轴向窗导出；
  4. 每侧审计：点数 / 连通分量 / 角向覆盖 / 到 margin 最近距离 → 状态
     AVAILABLE / MISSING_TOOTH / SECTIONED_OR_HOLE / AMBIGUOUS；
  5. GT 反求：Final Crown ↔ 该侧 ROI 的最小距离 / 接触点数 / 接触面积估计 /
     接触质心；XML 设计参数（键含 Neighbor）一并列出。

判别"插座 vs 邻牙"的关键是**角向覆盖**：sectioned 底座的 46 插座壁紧贴
margin 圈且角向覆盖 ~360°；真邻牙只占一段弧且在 margin 边之外。
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import struct
import sys
import time
from collections import deque

sys.path.insert(0, "/tmp/gt")

from app.groundtruth.assets import content_hash                      # noqa: E402
from app.groundtruth.v3_features import read_stl_single_pass         # noqa: E402
from app.importers.exocad.construction_info import parse_construction_case  # noqa: E402

INDEX_CSV = "/tmp/gt_freeze_v1/frozen_tooth_index.csv"
CASE_ROOT = "/data/luoda/dentistry/crownflow_data/crown_outer_cases"
ROOTS = ("/data1/welo/dentistry/exo/all_data",
         "/data1/welo/dentistry/exo/EXO")

#: ROI 沿近远中向的进深：从 margin 边 +0.8mm 起，延伸 13mm。
ROI_DEPTH_MM = 13.0
ROI_EDGE_GAP_MM = 0.8
#: 轴向窗（相对 margin 平面）。
ROI_W_MIN, ROI_W_MAX = -6.0, 12.0
#: GT 接触判定距离（冠顶点 ↔ 邻牙顶点）。
CONTACT_MM = 0.3
#: 角向覆盖 ≥ 10/12 扇区 → 插座（sectioned base 的 46 洞壁）。
SOCKET_COVERAGE = 10 / 12
#: 最大连通分量少于该点数 → 不足以成为邻接表面。
MIN_COMPONENT_POINTS = 50


# ---------------------------------------------------------------- 基础工具

def unit(v):
    n = math.sqrt(sum(c * c for c in v)) or 1.0
    return tuple(c / n for c in v)


def cross(a, b):
    return (a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0])


def patient_frame(margin, insertion_axis, axis_mesial):
    """Origin=margin 质心；Z=就位轴；X=AxisMesial ⊥Z（缺失时闭式 PCA 兜底）。"""
    origin = tuple(sum(p[i] for p in margin) / len(margin) for i in range(3))
    z = unit(insertion_axis)
    x = None
    if axis_mesial is not None:
        d = sum(axis_mesial[i] * z[i] for i in range(3))
        m = tuple(axis_mesial[i] - d * z[i] for i in range(3))
        if sum(c * c for c in m) > 1e-9:
            x = unit(m)
    axis_source = "axis_mesial"
    if x is None:
        axis_source = "margin_pca"
        a = [0.0, 0.0, 1.0] if abs(z[2]) < 0.9 else [1.0, 0.0, 0.0]
        e1 = unit(cross(z, a))
        e2 = cross(z, e1)
        n = len(margin)
        su = sv = suu = svv = suv = 0.0
        for p in margin:
            rel = [p[i] - origin[i] for i in range(3)]
            u = sum(rel[i] * e1[i] for i in range(3))
            v = sum(rel[i] * e2[i] for i in range(3))
            su += u; sv += v; suu += u * u; svv += v * v; suv += u * v
        su /= n; sv /= n
        suu = suu / n - su * su; svv = svv / n - sv * sv; suv = suv / n - su * sv
        lam = (suu + svv) / 2 + math.sqrt(((suu - svv) / 2) ** 2 + suv ** 2)
        x = unit(tuple(e1[i] * (lam - svv) + e2[i] * suv for i in range(3)))
    y = cross(z, x)
    return origin, z, x, y, axis_source


def to_frame(p, origin, x, y, z):
    rel = [p[i] - origin[i] for i in range(3)]
    return (sum(rel[i] * x[i] for i in range(3)),
            sum(rel[i] * y[i] for i in range(3)),
            sum(rel[i] * z[i] for i in range(3)))


class PointGrid:
    """1mm 格子点云：半径查询 + 连通分量（occupied 格子 6-连通 BFS）。"""

    def __init__(self, points, cell=1.0):
        self.cell = cell
        self.buckets = {}
        for p in points:
            k = (int(math.floor(p[0] / cell)), int(math.floor(p[1] / cell)),
                 int(math.floor(p[2] / cell)))
            self.buckets.setdefault(k, []).append(p)

    def nearest(self, p, radius=8.0):
        r = int(math.ceil(radius / self.cell))
        bx, by, bz = (int(math.floor(p[i] / self.cell)) for i in range(3))
        best = math.inf
        r2 = radius * radius
        for dx in range(-r, r + 1):
            for dy in range(-r, r + 1):
                for dz in range(-r, r + 1):
                    for q in self.buckets.get((bx + dx, by + dy, bz + dz), ()):
                        d = ((q[0] - p[0]) ** 2 + (q[1] - p[1]) ** 2
                             + (q[2] - p[2]) ** 2)
                        if d < best and d <= r2:
                            best = d
        return math.sqrt(best) if best < math.inf else None

    def components(self, max_components=8):
        seen = set()
        comps = []
        for k in self.buckets:
            if k in seen:
                continue
            q = deque([k])
            seen.add(k)
            n = 0
            while q:
                c = q.popleft()
                n += len(self.buckets[c])
                for dx, dy, dz in ((1, 0, 0), (-1, 0, 0), (0, 1, 0),
                                   (0, -1, 0), (0, 0, 1), (0, 0, -1)):
                    nk = (c[0] + dx, c[1] + dy, c[2] + dz)
                    if nk in self.buckets and nk not in seen:
                        seen.add(nk)
                        q.append(nk)
            comps.append(n)
            if len(comps) >= max_components:
                break
        return sorted(comps, reverse=True)


def angular_coverage(uv_pairs):
    """侧壁点的角向覆盖（12 扇区里被占据的比例）。"""
    sect = set()
    for u, v in uv_pairs:
        if math.hypot(u, v) < 1.0:
            continue
        sect.add(int((math.atan2(v, u) + math.pi) / (math.pi / 6)) % 12)
    return len(sect) / 12.0


def read_triangles(path):
    raw = open(path, "rb").read()
    if raw[:5] == b"solid":
        raise ValueError("ASCII STL 不支持：%s" % path)
    n = struct.unpack_from("<I", raw, 80)[0]
    out = []
    for i in range(n):
        v = struct.unpack_from("<12f", raw, 84 + i * 50)
        out.append(((v[3], v[4], v[5]), (v[6], v[7], v[8]),
                    (v[9], v[10], v[11])))
    return out


def sample_stl_points(path, cap):
    raw = open(path, "rb").read()
    n = struct.unpack_from("<I", raw, 80)[0]
    stride = max(1, (n * 3) // cap)
    pts = []
    for i in range(0, n, stride):
        v = struct.unpack_from("<12f", raw, 84 + i * 50)
        for j in (3, 6, 9):
            pts.append((v[j], v[j + 1], v[j + 2]))
    return pts


def write_patch_stl(path, tris):
    with open(path, "wb") as fh:
        fh.write(b"patch".ljust(80, b"\x00"))
        fh.write(struct.pack("<I", len(tris)))
        for t in tris:
            fh.write(struct.pack("<3f", 0.0, 0.0, 0.0))
            for p in t:
                fh.write(struct.pack("<3f", *p))
            fh.write(struct.pack("<H", 0))


def _median(v):
    v = sorted(v)
    return round(v[len(v) // 2], 3) if v else None


def _p95(v):
    v = sorted(v)
    return round(v[min(len(v) - 1, int(0.95 * (len(v) - 1)))], 3) if v else None


# ---------------------------------------------------------------- 主流程

def find_case_dirs(case_uuid):
    """该病例的**全部**同名目录（语料里同一 basename 会出现在多个根：
    如 all_data/<id> 放设计产物、EXO/<id> 放扫描——只查第一个会漏掉牙弓）。"""
    want = case_uuid[3:]
    out = []
    for root in ROOTS:
        for dp, dn, _f in os.walk(root):
            for d in dn:
                if hashlib.sha256(d.encode()).hexdigest()[:16] == want:
                    out.append(os.path.join(dp, d))
    return out


def find_asset_by_hash(case_dirs, want_hash):
    if not want_hash:
        return None
    for case_dir in case_dirs:
        for dp, _dn, fns in os.walk(case_dir):
            for f in fns:
                if not f.lower().endswith(".stl"):
                    continue
                p = os.path.join(dp, f)
                if content_hash(p) == want_hash:
                    return p
    return None


def audit_side(roi_pts, margin, origin, x, y, z, sign, edge):
    """单侧审计：状态 + 几何量。`sign`=+1 mesial / −1 distal。"""
    st = {"point_count": len(roi_pts)}
    if len(roi_pts) < MIN_COMPONENT_POINTS:
        st.update({"status": "MISSING_TOOTH", "component_count": 0,
                   "angular_coverage": 0.0,
                   "min_distance_to_margin_mm": None, "u_range_mm": None})
        return st
    grid = PointGrid(roi_pts)
    comps = grid.components()
    largest = max(comps) if comps else 0
    uv = [(sign * to_frame(p, origin, x, y, z)[0],
           to_frame(p, origin, x, y, z)[1]) for p in roi_pts[::7]]
    cov = angular_coverage(uv)
    margin_grid = PointGrid(margin)
    dmin = math.inf
    for p in roi_pts[:: max(1, len(roi_pts) // 4000)]:
        d = margin_grid.nearest(p, 15.0)
        if d is not None and d < dmin:
            dmin = d
    u_vals = sorted(sign * to_frame(p, origin, x, y, z)[0] for p in roi_pts)
    u_lo, u_hi = round(u_vals[0], 2), round(u_vals[-1], 2)
    if largest < MIN_COMPONENT_POINTS:
        status = "MISSING_TOOTH"
    elif cov >= SOCKET_COVERAGE:
        status = "SECTIONED_OR_HOLE"
    elif dmin > 6.0:
        status = "AMBIGUOUS"
    else:
        status = "AVAILABLE"
    st.update({"status": status, "component_count": len(comps),
               "largest_component_points": largest,
               "angular_coverage": round(cov, 3),
               "min_distance_to_margin_mm": (round(dmin, 3)
                                             if dmin < math.inf else None),
               "u_range_mm": [u_lo, u_hi]})
    return st


def gt_contact(roi_pts, crown_pts):
    """GT 冠 ↔ 该侧邻接 ROI：最小距离 / 接触点数 / 面积估计 / 质心（患者系）。"""
    if not roi_pts or not crown_pts:
        return None
    roi_grid = PointGrid(roi_pts)
    dmin = math.inf
    contact = []
    for p in crown_pts:
        d = roi_grid.nearest(p, 2.0)
        if d is None:
            continue
        if d < dmin:
            dmin = d
        if d <= CONTACT_MM:
            contact.append(p)
    out = {"min_distance_mm": round(dmin, 3) if dmin < math.inf else None,
           "contact_point_count": len(contact)}
    if contact:
        # 面积估计：接触点数 × 冠采样点平均间距²
        spacing = 0.25                    # 40k 采样点在 ~200mm² 冠面上 ≈ 0.22mm
        out["contact_area_mm2_est"] = round(len(contact) * spacing * spacing, 2)
        cent = [sum(p[i] for p in contact) / len(contact) for i in range(3)]
        out["contact_centroid_world"] = [round(v, 2) for v in cent]
    return out


def audit_case(case_uuid, index_rows):
    # 冻结索引的 arch 绑定不可靠（大量 SINGLE_TOOTH 病例没绑）；
    # row=None 不早退，走"最大 STL 当牙弓候选 + 同帧证明裁决"的回退。
    row = index_rows.get(case_uuid)
    case_dirs = find_case_dirs(case_uuid)
    if not case_dirs:
        return {"case_uuid": case_uuid, "error": "case_dir_not_found"}
    meta = json.load(open(os.path.join(CASE_ROOT, case_uuid, "case.json"),
                          encoding="ascii"))
    margin = [tuple(map(float, p)) for p in meta["margin_points"]]
    origin, z, x, y, axis_source = patient_frame(margin, meta["insertion_axis"],
                                                 meta.get("axis_mesial"))
    m_f = [to_frame(p, origin, x, y, z) for p in margin]
    edge_mes = max(u for u, _v, _w in m_f) + ROI_EDGE_GAP_MM
    edge_dis = max(-u for u, _v, _w in m_f) + ROI_EDGE_GAP_MM
    v_half = max(abs(v) for _u, v, _w in m_f) * 0.75 + 3.0

    arch_hash = row.get("full_arch_asset_id", "") if row else ""
    arch_hash = arch_hash[4:] if arch_hash.startswith("sha:") else arch_hash
    arch_path = find_asset_by_hash(case_dirs, arch_hash)
    arch_source = "index_asset_id"
    if arch_path is None:
        # 索引没绑定牙弓（indexer 的绑定不可靠）→ 回退：所有病例目录里
        # 最大的 STL 当牙弓候选（牙弓 10~22MB，冠/代型 ≤4MB），
        # 选择是否正确由下面的同帧证明裁决。
        cands = []
        seen = set()
        for cd in case_dirs:
            for dp, _dn, fns in os.walk(cd):
                for f in fns:
                    if not f.lower().endswith(".stl"):
                        continue
                    p = os.path.join(dp, f)
                    if p in seen:
                        continue
                    seen.add(p)
                    sz = os.path.getsize(p)
                    if sz > 5_000_000:
                        cands.append((sz, p))
        if not cands:
            return {"case_uuid": case_uuid, "error": "no_arch_candidate",
                    "n_case_dirs": len(case_dirs)}
        cands.sort(reverse=True)
        arch_path = cands[0][1]
        arch_source = "largest_stl_fallback"
    got = read_stl_single_pass(arch_path, max_points=400_000, cell=1.0)
    if not got["ok"]:
        return {"case_uuid": case_uuid, "error": "arch_read_failed"}
    pts = []
    for bucket in got["buckets"].values():
        pts.extend(bucket)
    uvw = [to_frame(p, origin, x, y, z) for p in pts]

    def in_roi(f, sign, edge):
        u, v, w = f
        return (sign * u >= edge and sign * u <= edge + ROI_DEPTH_MM
                and abs(v) <= v_half and ROI_W_MIN <= w <= ROI_W_MAX)

    res = {"case_uuid": case_uuid,
           "arch_asset_id": arch_hash,
           "arch_path_basename": os.path.basename(arch_path)[:60],
           "arch_source": arch_source,
           "arch_points_sampled": len(pts),
           "n_margin_points": len(margin),
           "frame_axis_source": axis_source,
           "same_frame": None,          # 下面按 margin→arch 距离填
           "margin_to_arch_mm": None,
           "center_to_arch_mm": None,
           }
    margin_grid = PointGrid(margin)
    d_list = []
    for p in margin[:: max(1, len(margin) // 120)]:
        d = margin_grid_nearest(got["buckets"], p)
        d_list.append(d)
    ok_d = [d for d in d_list if d is not None]
    res["margin_to_arch_mm"] = {"median": _median(ok_d),
                                "p95": _p95(ok_d),
                                "n_beyond_8mm": len(d_list) - len(ok_d)}
    if meta.get("center"):
        res["center_to_arch_mm"] = margin_grid_nearest(
            got["buckets"], tuple(meta["center"]))
    # GT 冠 → 牙弓：冠与 margin 同一坐标系，且物理上坐在牙弓里。
    # 注意不能用 median（冠大部分悬浮在牙弓上方，分段模型尤其如此）——
    # 用"冠采样点中距牙弓 < 0.8mm 的比例"：颈缘区坐在插座缘/邻牙上的
    # 比例在真同帧模型里 ≥2%。
    crown_path0 = os.path.join(CASE_ROOT, case_uuid, "final_crown.stl")
    crown_frac_near = None
    if os.path.isfile(crown_path0):
        cpts = sample_stl_points(crown_path0, 8000)
        near = 0
        tot = 0
        for p in cpts[::10]:
            tot += 1
            d = margin_grid_nearest(got["buckets"], p, radius=3.0)
            if d is not None and d < 0.8:
                near += 1
        crown_frac_near = round(near / tot, 4) if tot else None
    res["crown_fraction_near_arch"] = crown_frac_near
    res["same_frame"] = bool(
        (ok_d and _median(ok_d) < 1.0)
        or (crown_frac_near is not None and crown_frac_near >= 0.02))

    # XML 设计参数（键含 Neighbor）
    xml_targets = {}
    ci_files = []
    for case_dir in case_dirs:
        for dp, _dn, fns in os.walk(case_dir):
            for f in fns:
                if f.endswith(".constructionInfo") or f.endswith(".modelInfo"):
                    ci_files.append(os.path.join(dp, f))
    ci_files.sort(key=lambda p: not p.endswith(".constructionInfo"))
    if ci_files:
        try:
            case = parse_construction_case(ci_files[0])
            for name, cell in case.parameters.items():
                if "neighbor" in name.lower():
                    xml_targets[name] = cell.value
        except Exception as exc:                 # noqa: BLE001
            xml_targets["_error"] = str(exc)[:80]
    res["xml_neighbor_parameters"] = xml_targets

    crown_path = os.path.join(CASE_ROOT, case_uuid, "final_crown.stl")
    crown_pts = sample_stl_points(crown_path, 40000) \
        if os.path.isfile(crown_path) else None

    for side, sign, edge in (("mesial", 1.0, edge_mes),
                             ("distal", -1.0, edge_dis)):
        roi = [p for f, p in zip(uvw, pts) if in_roi(f, sign, edge)]
        st = audit_side(roi, margin, origin, x, y, z, sign, edge)
        if not res["same_frame"] and st["status"] not in ("MISSING_TOOTH",):
            st["status"] = "OUT_OF_FRAME"
        if crown_pts is not None and st["status"] == "AVAILABLE":
            st["gt_contact"] = gt_contact(roi, crown_pts)
        else:
            st["gt_contact"] = None
        res[side] = st

    # AVAILABLE 侧导出 patch STL（原网格三角形全部顶点落在 ROI 内）
    try:
        tris = read_triangles(arch_path)
        for side, sign, edge in (("mesial", 1.0, edge_mes),
                                 ("distal", -1.0, edge_dis)):
            if res[side]["status"] != "AVAILABLE":
                continue
            keep = []
            for t in tris:
                fs = [to_frame(p, origin, x, y, z) for p in t]
                if all(in_roi(f, sign, edge) for f in fs):
                    keep.append(t)
            if keep:
                write_patch_stl(os.path.join(CASE_ROOT, case_uuid,
                                             f"{side}_neighbor_patch.stl"),
                                keep)
                res[side]["patch_triangles"] = len(keep)
    except Exception as exc:                         # noqa: BLE001
        res["patch_error"] = str(exc)[:120]
    return res


def margin_grid_nearest(buckets, p, radius=8.0):
    r = int(math.ceil(radius))
    bx, by, bz = (int(math.floor(p[i])) for i in range(3))
    best = None
    r2 = radius * radius
    for dx in range(-r, r + 1):
        for dy in range(-r, r + 1):
            for dz in range(-r, r + 1):
                for q in buckets.get((bx + dx, by + dy, bz + dz), ()):
                    d = ((q[0] - p[0]) ** 2 + (q[1] - p[1]) ** 2
                         + (q[2] - p[2]) ** 2)
                    if d <= r2 and (best is None or d < best):
                        best = d
    return math.sqrt(best) if best is not None else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/tmp/neighbor_audit")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    index_rows = {}
    for r in csv.DictReader(open(INDEX_CSV, encoding="ascii")):
        if r["fdi"] == "46" and r["has_full_arch_candidate"] == "True":
            index_rows[r["case_uuid"]] = r
    cases = sorted(d for d in os.listdir(CASE_ROOT) if d.startswith("cf_"))
    print("病例 %d，索引命中 %d" % (len(cases), len(index_rows)), flush=True)
    rows = []
    for i, cid in enumerate(cases):
        t0 = time.time()
        try:
            res = audit_case(cid, index_rows)
        except Exception as exc:                     # noqa: BLE001
            res = {"case_uuid": cid,
                   "error": f"{type(exc).__name__}: {exc}"}
        res["elapsed_s"] = round(time.time() - t0, 1)
        rows.append(res)
        brief = {k: res.get(k) for k in
                 ("error", "same_frame", "mesial", "distal")}
        print("[%2d] %s %s (%.0fs)" % (i, cid[:14],
                                       json.dumps(brief, ensure_ascii=False),
                                       res["elapsed_s"]), flush=True)
        json.dump(res, open(os.path.join(a.out, cid + "_audit.json"), "w"),
                  indent=1)
    json.dump(rows, open(os.path.join(a.out, "neighbor_audit_details.json"),
                         "w"), indent=1)
    print("写出", a.out)


if __name__ == "__main__":
    main()
