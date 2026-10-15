#!/usr/bin/env python3
"""邻牙 patch 区域生长（修 I.1 的卡点，纯 stdlib，服务器跑）。

## 为什么需要它

审计脚本按**盒子 ROI**裁 patch（``u ∈ [margin边+0.8, +13]``），内边界正好
切在接触区起始处，于是冠的最近点几乎全落在 patch 的**内切割边缘**上：
实测 mesial patch 555 条边界边、2621 个查询点里 2620 个落在边界安全带内，
I.1 的边界歧义剔除把所有约束都剔掉了（活跃集只剩 1 个顶点）。

## 做法

从**离冠最近的那个牙弓三角形**出发做面邻接区域生长，只收"面心到种子
距离 ≤ R"的三角形 —— 得到一块以接触区为中心、接触区在内部的圆盘状 patch，
边界自然远离接触区。

    python3 grow_neighbor_patch.py --case cf_xxx --fdi 46 --side mesial \
        --radius 9 --out /data/.../crown_outer_cases
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import struct
import sys
from collections import deque

sys.path.insert(0, "/tmp/gt")

from app.groundtruth.assets import content_hash                      # noqa: E402

INDEX_CSV = "/tmp/gt_freeze_v1/frozen_tooth_index.csv"
CASE_ROOT = "/data/luoda/dentistry/crownflow_data/crown_outer_cases"
ROOTS = ("/data1/welo/dentistry/exo/all_data",
         "/data1/welo/dentistry/exo/EXO")


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


def write_patch(path, tris):
    with open(path, "wb") as fh:
        fh.write(b"patch".ljust(80, b"\x00"))
        fh.write(struct.pack("<I", len(tris)))
        for t in tris:
            fh.write(struct.pack("<3f", 0.0, 0.0, 0.0))
            for p in t:
                fh.write(struct.pack("<3f", *p))
            fh.write(struct.pack("<H", 0))


def find_case_dirs(uuid):
    want = uuid[3:]
    out = []
    for root in ROOTS:
        for dp, dn, _f in os.walk(root):
            for d in dn:
                if hashlib.sha256(d.encode()).hexdigest()[:16] == want:
                    out.append(os.path.join(dp, d))
    return out


def find_by_hash(dirs, want_hash):
    for d in dirs:
        for dp, _dn, fns in os.walk(d):
            for f in fns:
                if not f.lower().endswith(".stl"):
                    continue
                p = os.path.join(dp, f)
                if content_hash(p) == want_hash:
                    return p
    return None


def face_adjacency(faces):
    from collections import defaultdict
    ef = defaultdict(list)
    for i, (a, b, c) in enumerate(faces):
        for u, v in ((a, b), (b, c), (c, a)):
            ef[(min(u, v), max(u, v))].append(i)
    adj = defaultdict(set)
    for fl in ef.values():
        for i in fl:
            for j in fl:
                if i != j:
                    adj[i].add(j)
    return adj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True)
    ap.add_argument("--fdi", type=int, default=46)
    ap.add_argument("--side", required=True, choices=["mesial", "distal"])
    ap.add_argument("--radius", type=float, default=9.0,
                    help="patch 半径（面心到种子点的上限，mm）")
    ap.add_argument("--out", default=CASE_ROOT)
    a = ap.parse_args()

    row = None
    for r in csv.DictReader(open(INDEX_CSV, encoding="ascii")):
        if r["case_uuid"] == a.case and int(r["fdi"]) == a.fdi:
            row = r
            break
    dirs = find_case_dirs(a.case)
    if not dirs:
        raise SystemExit("找不到病例目录")
    arch_hash = (row or {}).get("full_arch_asset_id", "")
    arch_hash = arch_hash[4:] if arch_hash.startswith("sha:") else arch_hash
    arch = find_by_hash(dirs, arch_hash)
    if arch is None:
        cands = []
        for d in dirs:
            for dp, _dn, fns in os.walk(d):
                for f in fns:
                    if f.lower().endswith(".stl"):
                        p = os.path.join(dp, f)
                        if os.path.getsize(p) > 5_000_000:
                            cands.append((os.path.getsize(p), p))
        if not cands:
            raise SystemExit("找不到牙弓 STL")
        arch = sorted(cands)[-1][1]
    print("牙弓:", os.path.basename(arch)[:40])

    meta = json.load(open(os.path.join(a.out, a.case, "case.json"),
                          encoding="ascii"))
    margin = [tuple(map(float, p)) for p in meta["margin_points"]]
    origin = tuple(sum(p[i] for p in margin) / len(margin) for i in range(3))
    z = list(meta["insertion_axis"])
    nz = sum(c * c for c in z) ** 0.5
    z = [c / nz for c in z]
    m = list(meta["axis_mesial"])
    d = sum(m[i] * z[i] for i in range(3))
    m = [m[i] - d * z[i] for i in range(3)]
    nm = sum(c * c for c in m) ** 0.5
    x = [c / nm for c in m]
    sign = 1.0 if a.side == "mesial" else -1.0

    # 种子：牙弓上离"该侧 margin 最外点"最近的三角形
    def proj_u(p):
        return sum((p[i] - origin[i]) * x[i] for i in range(3))
    u_max = max(sign * proj_u(p) for p in margin)
    target_u = u_max + 1.5          # 稍微越过 margin 边，落进邻牙
    seed_pt = tuple(origin[i] + sign * target_u * x[i] + 0.0 * z[i]
                    for i in range(3))

    tris = read_triangles(arch)
    print("牙弓三角形:", len(tris))
    cents = []
    best, bi = None, -1
    for i, t in enumerate(tris):
        c = tuple(sum(v[k] for v in t) / 3.0 for k in range(3))
        cents.append(c)
        dd = sum((c[k] - seed_pt[k]) ** 2 for k in range(3))
        if best is None or dd < best:
            best, bi = dd, i
    print("种子面 %d，距目标点 %.2fmm" % (bi, best ** 0.5))

    adj = face_adjacency(tris)
    keep, seen = set(), {bi}
    q = deque([bi])
    r2 = a.radius * a.radius
    while q:
        f = q.popleft()
        c = cents[f]
        if sum((c[k] - cents[bi][k]) ** 2 for k in range(3)) > r2:
            continue
        keep.add(f)
        for g in adj.get(f, ()):
            if g not in seen:
                seen.add(g)
                q.append(g)
    print("生长得到 %d 个三角形（半径 %.1fmm）" % (len(keep), a.radius))
    out = os.path.join(a.out, a.case, "%s_neighbor_patch.stl" % a.side)
    write_patch(out, [tris[i] for i in sorted(keep)])
    print("写出", os.path.basename(out), os.path.getsize(out), "字节")


if __name__ == "__main__":
    main()
