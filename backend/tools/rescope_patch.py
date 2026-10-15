#!/usr/bin/env python3
"""邻牙/对颌 patch **作用域正确的重新提取**（纯 stdlib，可本地/服务器跑）。

用户 2026-10-09 锁定：patch 输入层是当前第一优先级 ——
把"手工大半径 grow"升级成**作用域正确**的提取，并配前置 Gate。

## 规则（与 `app/geometry/crown_outer/patch_scope.py` 同参同义）

逐面判（候选 = 源牙弓截面/整段牙弓，参考 = **代型/内表面**，永远可得，不依赖 GT）：

1. `d_min ≤ dist(面心, 参考表面) ≤ d_max`    —— 近接壁就在代型外 0.05–4mm
2. `(面心 − 颈缘质心)·ẑ ≥ margin_above`       —— 颈缘/牙龈/牙乳头以下不要
3. `sign·u ≥ sign·u_ref_extreme − 0.6`        —— 本侧近接壁；内边界留 0.6mm
4. `dot(n̂, axis_pt − 面心) > facing_cos`      —— 只留朝着目标牙的面
5. 面心在**冠体内部**的丢掉（`--crown` 给了才做）—— 目标牙自身/临时冠

保留面做**邻接连通生长**取最大连通块（去碎块）。

## 用法

    # 邻牙（mesial/distal）——候选可给多个 STL（A组各段）
    python3 rescope_patch.py --side mesial --out patch.stl \
        --candidate "…-A组 (1).stl" "…-A组 (2).stl" --prep prep_design_frame.stl \
        --case-json case.json [--crown final_crown.stl]

    # 对颌（antagonist）——参考改冠体咬合面，朝向改为"朝咬合中心"
    python3 rescope_patch.py --side antagonist --out antagon.stl \
        --candidate "…B组 (2).stl" --crown final_crown.stl --case-json case.json

验收（本地）：`tools/audit_neighbor_patch_scope.py` / `patch_scope.verify_patch`
—— GT 冠在该 patch 上穿透 p95 ≤ 0.2mm 且近接大陆覆盖 ≥ 20 点，否则
`PATCH_SCOPE_CONTAMINATED` / `PATCH_TRUNCATED`，不进链。
"""
from __future__ import annotations

import argparse
import json
import math
import struct
import sys
from collections import defaultdict, deque


def read_stl(path):
    """二进制 STL → [(p0, p1, p2), ...]（保持原始三角形，不焊接）。"""
    with open(path, "rb") as fh:
        head = fh.read(84)
        if head[:5] == b"solid" and b"facet" in head[:200]:
            raise SystemExit("ASCII STL 不支持：%s" % path)
        n = struct.unpack("<I", head[80:84])[0]
        out = []
        for _ in range(n):
            v = struct.unpack("<12fH", fh.read(50))
            out.append(((v[3], v[4], v[5]), (v[6], v[7], v[8]),
                        (v[9], v[10], v[11])))
    return out


def write_stl(path, tris):
    with open(path, "wb") as fh:
        fh.write(b"patch".ljust(80, b"\x00"))
        fh.write(struct.pack("<I", len(tris)))
        for t in tris:
            fh.write(struct.pack("<3f", 0.0, 0.0, 0.0))
            for p in t:
                fh.write(struct.pack("<3f", *p))
            fh.write(struct.pack("<H", 0))


def sub(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def norm(a):
    n = math.sqrt(dot(a, a))
    return (a[0] / n, a[1] / n, a[2] / n) if n > 1e-12 else (0.0, 0.0, 0.0)


def cross(a, b):
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0])


class Grid:
    """顶点均匀格点最近距离（纯 stdlib）。

    格边长取等于距离带上限 ⇒ 只需查 27 个邻格；参考网格顶点采样足够密
    （实测代型 ~8k 顶点/十几 mm²，点距 ~0.2mm），"格内无点 ⇒ 距离 > 带" 
    对**带通滤波器**足够（最终由 GT control 验收，不靠这里定精度）。
    """

    def __init__(self, pts, cell):
        self.cell = cell
        self.g = defaultdict(list)
        for i, p in enumerate(pts):
            self.g[self._key(p)].append(i)
        self.pts = pts

    def _key(self, p):
        c = self.cell
        return (int(math.floor(p[0] / c)), int(math.floor(p[1] / c)),
                int(math.floor(p[2] / c)))

    def nearest(self, p, _max_r=None):
        c = self.cell
        k0 = self._key(p)
        best = None
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    for i in self.g.get((k0[0] + dx, k0[1] + dy, k0[2] + dz), ()):
                        d2 = dot(sub(self.pts[i], p), sub(self.pts[i], p))
                        if best is None or d2 < best:
                            best = d2
        return math.sqrt(best) if best is not None else float("inf")


def inside_mesh(p, tris):
    """射线法（Möller–Trumbore，+x 射线）判点是否在闭合网格内。"""
    cnt = 0
    for a, b, c in tris:
        e1 = sub(b, a)
        e2 = sub(c, a)
        h = cross((1.0, 0.0, 0.0), e2)
        det = dot(e1, h)
        if abs(det) < 1e-12:
            continue
        inv = 1.0 / det
        s = sub(p, a)
        u = dot(s, h) * inv
        if u < 0.0 or u > 1.0:
            continue
        q = cross(s, e1)
        v = dot((1.0, 0.0, 0.0), q) * inv
        if v < 0.0 or u + v > 1.0:
            continue
        t = dot(e2, q) * inv
        if t > 1e-9:
            cnt += 1
    return cnt % 2 == 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--side", required=True,
                    choices=["mesial", "distal", "antagonist"])
    ap.add_argument("--candidate", nargs="+", required=True,
                    help="候选 STL（源牙弓/对颌的原始段，可多个）")
    ap.add_argument("--out", required=True)
    ap.add_argument("--prep", default=None,
                    help="设计系代型/内表面 STL（邻牙模式必需）")
    ap.add_argument("--crown", default=None,
                    help="冠体 STL（冠体内排除；对颌模式的参考）")
    ap.add_argument("--case-json", required=True,
                    help="case.json（margin_points / insertion_axis）")
    ap.add_argument("--d-min", type=float, default=0.05)
    ap.add_argument("--d-max", type=float, default=4.0)
    ap.add_argument("--margin-above", type=float, default=0.5)
    ap.add_argument("--side-margin", type=float, default=-0.6,
                    help="内边界相对参考本侧极值的偏移（负=往内留）")
    ap.add_argument("--facing-cos", type=float, default=0.10)
    ap.add_argument("--antagonist-reach", type=float, default=6.0,
                    help="对颌模式：到冠体的距离上限")
    ap.add_argument("--stats-json", default=None)
    a = ap.parse_args()

    meta = json.load(open(a.case_json, encoding="ascii"))
    mg = [tuple(map(float, p)) for p in meta["margin_points"]]
    o = tuple(sum(p[i] for p in mg) / len(mg) for i in range(3))
    z = norm(tuple(map(float, meta["insertion_axis"])))
    m = norm(tuple(map(float, meta["axis_mesial"])))
    d = dot(m, z)
    x = norm(tuple(m[i] - d * z[i] for i in range(3)))

    tris = []
    for p in a.candidate:
        t = read_stl(p)
        print("候选 %s: %d 面" % (p.split("/")[-1][:44], len(t)))
        tris.extend(t)
    cen = [tuple(sum(v[k] for v in t) / 3.0 for k in range(3)) for t in tris]
    nrm = []
    for t in tris:
        n = cross(sub(t[1], t[0]), sub(t[2], t[0]))
        nrm.append(norm(n))

    ref = None
    if a.side != "antagonist":
        if not a.prep:
            raise SystemExit("邻牙模式必须给 --prep（设计系代型）")
        rv = [tuple(v) for t in read_stl(a.prep) for v in (t[0], t[1], t[2])]
        ref = Grid(rv, max(a.d_max, 4.0))     # 格边长=带通上限
        u_ref = [sum((p[i] - o[i]) * x[i] for i in range(3)) for p in rv]
        u_ext = max(u_ref) if a.side == "mesial" else min(u_ref)
    else:
        if not a.crown:
            raise SystemExit("对颌模式必须给 --crown（参考冠体）")
        cvt = read_stl(a.crown)
        rv = [tuple(v) for t in cvt for v in (t[0], t[1], t[2])]
        ref = Grid(rv, max(a.antagonist_reach, 4.0))
        occ_c = tuple(sum(p[k] for p in rv) / len(rv) for k in range(3))
        # 更稳的咬合中心：取冠顶点里 z 最高的 15% 的重心
        ss = sorted(rv, key=lambda p: dot(p, z))
        top = ss[int(len(ss) * 0.85):]
        occ_c = tuple(sum(p[k] for p in top) / len(top) for k in range(3))
        crown_tris = cvt

    crown_tris = read_stl(a.crown) if (a.crown and a.side != "antagonist") else None
    sign = 1.0 if a.side == "mesial" else -1.0
    keep = []
    stat = defaultdict(int)
    for i, c in enumerate(cen):
        stat["n_candidate"] += 1
        dd = ref.nearest(c)
        if a.side == "antagonist":
            if dd > a.antagonist_reach or dd < a.d_min:
                stat["dist_band"] += 1
                continue
            # 朝向：朝咬合中心（对颌悬在冠上方，其咬合面朝向冠体）
            to_occ = sub(occ_c, c)
            if dot(nrm[i], norm(to_occ)) <= a.facing_cos:
                stat["not_facing"] += 1
                continue
            if crown_tris and inside_mesh(c, crown_tris):
                stat["inside_crown"] += 1
                continue
            keep.append(i)
            continue
        if dd > a.d_max or dd < a.d_min:
            stat["dist_band"] += 1
            continue
        zz = sum((c[k] - o[k]) * z[k] for k in range(3))
        if zz < a.margin_above:
            stat["below_margin"] += 1
            continue
        u = sum((c[k] - o[k]) * x[k] for k in range(3))
        if sign * u < sign * u_ext + a.side_margin:
            stat["wrong_side"] += 1
            continue
        # 朝向：朝就位轴（过 o、方向 z）
        rel = sub(c, o)
        t_ax = sum(rel[k] * z[k] for k in range(3))
        axis_pt = tuple(o[k] + t_ax * z[k] for k in range(3))
        if dot(nrm[i], norm(sub(axis_pt, c))) <= a.facing_cos:
            stat["not_facing"] += 1
            continue
        if crown_tris and inside_mesh(c, crown_tris):
            stat["inside_crown"] += 1
            continue
        keep.append(i)

    # 邻接连通生长：按焊接顶点取最大连通块（去碎块）
    if keep:
        keep_set = set(keep)

        def key(p):
            return (round(p[0], 4), round(p[1], 4), round(p[2], 4))

        faces_by_v = defaultdict(set)
        for i in keep:
            for p in tris[i]:
                faces_by_v[key(p)].add(i)
        seen = set()
        best_comp = []
        for i0 in keep:
            if i0 in seen:
                continue
            comp, q = [], deque([i0])
            seen.add(i0)
            while q:
                f = q.popleft()
                comp.append(f)
                for p in tris[f]:
                    for j in faces_by_v[key(p)]:
                        if j not in seen:
                            seen.add(j)
                            q.append(j)
            if len(comp) > len(best_comp):
                best_comp = comp
        keep = sorted(best_comp)
    stat["n_kept"] = len(keep)
    print("保留 %d 面 | 排除：%s" % (len(keep), dict(stat)))
    if len(keep) < 50:
        print("⚠️ 保留面过少（<50）—— 规则过严或候选里没有该侧近接壁")
    write_stl(a.out, [tris[i] for i in keep])
    print("写出 %s" % a.out)
    if a.stats_json:
        json.dump({"side": a.side, "n_candidate": len(tris), **stat},
                  open(a.stats_json, "w", encoding="utf-8"), indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
