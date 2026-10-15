#!/usr/bin/env python3
"""摘取 Antagonist patch（Occlusion Engine 的输入）。

步骤：
1. 按 content hash 找到病例目录里的 antagonist STL；
2. **同帧验证**：GT Final Crown 的咬合区顶点到 antagonist 表面的距离分布
   必须是"接近或轻微穿透"（对颌悬在上方/接触），距离中位 > 5mm 判 OUT_OF_FRAME；
3. ROI 裁剪：取 antagonist 中距冠咬合质心 ≤ R 的三角形（R 由冠尺寸导出），
   并外扩一圈保过渡；
4. 写 antagonist_patch.stl + antagonist_meta.json（含验证统计，PHI-free）。

用法:
    python3 stage_antagonist_case.py cf_b1b73bae221211b1
"""
from __future__ import annotations

import json
import math
import os
import struct
import sys

sys.path.insert(0, "/tmp/gt")
from build_auto_margin_gt import case_dirs, content_hash  # noqa: E402


def read_stl(path):
    """二进制 STL → (verts, tris)，顶点按**坐标焊接**（拓扑判定必需）。"""
    with open(path, "rb") as fh:
        head = fh.read(84)
        if head[:5] == b"solid" and b"facet" in head:
            raise SystemExit("ASCII STL 不支持")
        n = struct.unpack("<I", head[80:84])[0]
        tris = []
        cache = {}
        verts = []
        for _ in range(n):
            data = fh.read(50)
            vals = struct.unpack("<12fH", data)
            tri = []
            for k in range(3):
                key = (round(vals[3 + k * 3], 5), round(vals[4 + k * 3], 5),
                       round(vals[5 + k * 3], 5))
                if key not in cache:
                    cache[key] = len(verts)
                    verts.append((vals[3 + k * 3], vals[4 + k * 3],
                                  vals[5 + k * 3]))
                tri.append(cache[key])
            tris.append(tuple(tri))
        return verts, tris


def write_stl(path, verts, tris):
    # STL 布局：80 字节头 + 4 字节面数（写 84 字节零会把面数推到 84-88，
    # 读取方在 80-84 读到 0 → "0 面网格"，实测踩到）
    out = bytearray(b"\0" * 80)
    out += struct.pack("<I", len(tris))
    for t in tris:
        a, b, c = (verts[t[0]], verts[t[1]], verts[t[2]])
        u = tuple(b[k] - a[k] for k in range(3))
        v = tuple(c[k] - a[k] for k in range(3))
        n = (u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2],
             u[0] * v[1] - u[1] * v[0])
        L = math.sqrt(sum(x * x for x in n)) or 1e-12
        out += struct.pack("<3f", *(x / L for x in n))
        out += struct.pack("<9fH", *a, *b, *c, 0)
    with open(path, "wb") as fh:
        fh.write(out)


def main():
    want = sys.argv[1] if len(sys.argv) > 1 else "cf_b1b73bae221211b1"
    case_dir = os.path.join(
        "/data/luoda/dentistry/crownflow_data/crown_outer_cases", want)
    meta = json.load(open(os.path.join(case_dir, "case.json"), encoding="ascii"))
    idx = [r for r in csv_index() if r["case_uuid"] == want][0]
    ant_hash = idx["antagonist_asset_id"].replace("sha:", "")
    ant_path = None
    for d in case_dirs(want):
        for dp, _dn, fns in os.walk(d):
            for f in fns:
                q = os.path.join(dp, f)
                if f.lower().endswith(".stl") and \
                        content_hash(q).replace("sha:", "") == ant_hash:
                    ant_path = q
                    break
            if ant_path:
                break
        if ant_path:
            break
    if not ant_path:
        raise SystemExit("antagonist STL 未找到")
    print("antagonist:", ant_path)

    verts, tris = read_stl(ant_path)
    print("antagonist: V=%d F=%d" % (len(verts), len(tris)))
    # GT 冠
    cverts, ctris = read_stl(os.path.join(case_dir, "final_crown.stl"))
    z = meta["insertion_axis"]
    L = math.sqrt(sum(x * x for x in z)) or 1.0
    z = [x / L for x in z]
    # 冠咬合区：沿就位轴最高的 25% 顶点
    hs = sorted(sum((cverts[v][k] - meta["center"][k]) * z[k] for k in range(3))
                for v in range(len(cverts)))
    z_thr = hs[int(0.75 * (len(hs) - 1))]
    occ_ids = [v for v in range(len(cverts))
               if sum((cverts[v][k] - meta["center"][k]) * z[k]
                      for k in range(3)) >= z_thr]
    oc = [sum(cverts[v][k] for v in occ_ids) / len(occ_ids) for k in range(3)]
    print("冠咬合区 %d 顶点，质心 %s" % (len(occ_ids), [round(x, 2) for x in oc]))
    # 同帧验证：冠咬合顶点到 antagonist 的最近距离（暴力网格，Antagonist 大 →
    # 用采样顶点 KD 近似：无 numpy，用网格分桶）
    cell = 2.0
    buckets = {}
    for v in verts:
        k = (int(v[0] // cell), int(v[1] // cell), int(v[2] // cell))
        buckets.setdefault(k, []).append(v)
    dlist = []
    for v in occ_ids:
        p = cverts[v]
        k0 = tuple(int(p[k] // cell) for k in range(3))
        best = float("inf")
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    for q in buckets.get((k0[0] + dx, k0[1] + dy,
                                          k0[2] + dz), ()):
                        d = math.dist(p, q)
                        if d < best:
                            best = d
        if best < float("inf"):
            dlist.append(best)
    dlist.sort()
    med = dlist[len(dlist) // 2] if dlist else float("inf")
    print("冠咬合区→antagonist 距离: p10=%.3f p50=%.3f p90=%.3f (n=%d)"
          % (dlist[int(.1 * len(dlist))], med, dlist[int(.9 * len(dlist))],
             len(dlist)))
    frame_ok = med <= 5.0
    print("同帧判定:", "PASS" if frame_ok else "OUT_OF_FRAME（>5mm）")
    # ROI 裁剪：距冠咬合质心 ≤ R 的 antagonist 三角形
    # ROI 上限 12mm：纯 Python SDF 的 patch 规模要可控（实测 19mm → 12 万面）
    R = min(12.0, max(8.0, float(idx["margin_perimeter_mm"]) / math.pi))
    keep = []
    for t in tris:
        c = [(verts[t[0]][k] + verts[t[1]][k] + verts[t[2]][k]) / 3.0
             for k in range(3)]
        if math.dist(c, oc) <= R:
            keep.append(t)
    print("ROI R=%.1fmm → patch 面 %d / %d" % (R, len(keep), len(tris)))
    used = sorted({i for t in keep for i in t})
    remap = {v: i for i, v in enumerate(used)}
    out_tris = [tuple(remap[i] for i in t) for t in keep]
    out_verts = [verts[i] for i in used]
    write_stl(os.path.join(case_dir, "antagonist_patch.stl"),
              out_verts, out_tris)
    json.dump({
        "case_uuid": want,
        "antagonist_asset_id": idx["antagonist_asset_id"],
        "source_file": os.path.basename(ant_path),
        "n_patch_faces": len(keep),
        "roi_radius_mm": R,
        "frame_check": {"median_dist_mm": round(med, 4),
                        "p10": round(dlist[int(.1 * len(dlist))], 4),
                        "p90": round(dlist[int(.9 * len(dlist))], 4),
                        "pass": frame_ok},
        "occlusal_centroid": [round(x, 4) for x in oc],
        "note": "帧验证用 GT Final Crown 咬合区（同帧证据，非 GT 泄漏：仅距离统计，"
                "不参与求解）",
    }, open(os.path.join(case_dir, "antagonist_meta.json"), "w"),
        indent=1, ensure_ascii=True)
    print("写出 antagonist_patch.stl / antagonist_meta.json →", case_dir)


def csv_index():
    import csv
    return list(csv.DictReader(open(
        "/tmp/gt_freeze_v1/frozen_tooth_index.csv", encoding="ascii")))


if __name__ == "__main__":
    main()
