#!/usr/bin/env python3
"""Localization QA 渲染：给 FULL_ARCH_WITH_TARGET 候选生成 Center 邻域深度图。

输出：/tmp/qa/sheet_*.png（拼图）+ /tmp/qa/manifest.json（索引 -> 特征）。
人工判定 A=Prep在 / B=只剩洞 / C=其它，随后与分类器对比算混淆矩阵。
纯 stdlib。文件名只用 case_uuid，不写患者信息。
"""
import csv
import glob
import hashlib
import json
import math
import os
import struct
import sys
import zlib

from ..importers.exocad.construction_info import parse_construction_case

#: 默认输入输出。真实数据路径由 --audit/--roots/--out 传入（见 main）。
AUDIT = "arch_asset_audit.csv"
OUT = "qa"
_ROOTS = ["/data1/welo/dentistry/exo/all_data",
          "/data1/welo/dentistry/exo/EXO"]
WIN_MM = 12.0
GRID = 200


# ---------------- PNG ----------------
def write_png(path, w, h, rows):
    raw = b"".join(b"\x00" + bytes(r) for r in rows)
    def chunk(tag, data):
        return (struct.pack(">I", len(data)) + tag + data
                + struct.pack(">I", zlib.crc32(tag + data) & 0xffffffff))
    png = (b"\x89PNG\r\n\x1a\n"
           + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 0, 0, 0, 0))
           + chunk(b"IDAT", zlib.compress(raw, 6))
           + chunk(b"IEND", b""))
    open(path, "wb").write(png)


# ---------------- 几何 ----------------
def basis(d):
    a = [0.0, 0.0, 1.0] if abs(d[2]) < 0.9 else [1.0, 0.0, 0.0]
    e1 = [d[1]*a[2]-d[2]*a[1], d[2]*a[0]-d[0]*a[2], d[0]*a[1]-d[1]*a[0]]
    n = math.sqrt(sum(x*x for x in e1))
    e1 = [x/n for x in e1]
    e2 = [d[1]*e1[2]-d[2]*e1[1], d[2]*e1[0]-d[0]*e1[2], d[0]*e1[1]-d[1]*e1[0]]
    return e1, e2


def depth_map(tris, center, view_dir, *, win=WIN_MM, grid=GRID):
    """正交深度图：沿 view_dir 看，取最先命中的表面（depth 最大）。
    亮 = 离相机近。返回 rows。"""
    e1, e2 = basis(view_dir)
    zbuf = [[None]*grid for _ in range(grid)]
    scale = (2*win)/grid
    for tri in tris:
        # 视口剔除
        pts = []
        skip = False
        for p in tri:
            rel = [p[i]-center[i] for i in range(3)]
            du, dv = rel[0]*e1[0]+rel[1]*e1[1]+rel[2]*e1[2], \
                     rel[0]*e2[0]+rel[1]*e2[1]+rel[2]*e2[2]
            dd = rel[0]*view_dir[0]+rel[1]*view_dir[1]+rel[2]*view_dir[2]
            if abs(du) > win or abs(dv) > win:
                skip = True
                break
            pts.append((du, dv, dd))
        if skip:
            continue
        # 背面剔除不做（双面可见）
        (x0, y0, z0), (x1, y1, z1), (x2, y2, z2) = pts
        minu = max(0, int((min(x0, x1, x2)+win)/scale))
        maxu = min(grid-1, int((max(x0, x1, x2)+win)/scale))
        minv = max(0, int((min(y0, y1, y2)+win)/scale))
        maxv = min(grid-1, int((max(y0, y1, y2)+win)/scale))
        det = (x1-x0)*(y2-y0)-(x2-x0)*(y1-y0)
        if abs(det) < 1e-12:
            continue
        for pv in range(minv, maxv+1):
            v = -win + (pv+0.5)*scale
            for pu in range(minu, maxu+1):
                u = -win + (pu+0.5)*scale
                w0 = ((x1-u)*(y2-v)-(x2-u)*(y1-v))/det
                w1 = ((x2-u)*(y0-v)-(x0-u)*(y2-v))/det
                w2 = 1.0-w0-w1
                if w0 < -1e-9 or w1 < -1e-9 or w2 < -1e-9:
                    continue
                dd = w0*z0 + w1*z1 + w2*z2
                cur = zbuf[pv][pu]
                if cur is None or dd > cur:
                    zbuf[pv][pu] = dd
    # 深度 -> 灰度（亮=近）。缺表面 = 0（黑）
    ds = [z for row in zbuf for z in row if z is not None]
    if not ds:
        return [[0]*grid for _ in range(grid)], None
    lo, hi = min(ds), max(ds)
    rng = (hi-lo) or 1.0
    rows = []
    for pv in range(grid):
        row = []
        for pu in range(grid):
            z = zbuf[pv][pu]
            if z is None:
                row.append(18)                     # 无表面：深灰黑
            else:
                row.append(int(40 + 200*(z-lo)/rng))
        rows.append(row)
    # 中心十字（视口中心 = Center 投影）
    c = grid//2
    for k in range(-6, 7):
        for rr, cc in ((c, c+k), (c+k, c)):
            if 0 <= rr < grid and 0 <= cc < grid:
                rows[rr][cc] = 255 if abs(k) > 1 else 0
    return rows, (lo, hi)


def stl_tris_near(path, center, radius):
    import struct as _s
    size = os.path.getsize(path)
    with open(path, "rb") as f:
        head = f.read(84)
        if head[:5] == b"solid" or size < 84:
            return None
        n = struct.unpack("<I", head[80:84])[0]
        if 84 + n*50 != size:
            n = max(0, (size-84)//50)
        buf = f.read(n*50)
    r2 = radius*radius
    tris = []
    for i in range(n):
        v = _s.unpack_from("<12f", buf, i*50)
        a = (v[3], v[4], v[5]); b = (v[6], v[7], v[8]); c = (v[9], v[10], v[11])
        for p in (a, b, c):
            d = sum((p[k]-center[k])**2 for k in range(3))
            if d <= r2:
                tris.append((a, b, c))
                break
    return tris


def content_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(1 << 20)
            if not b:
                break
            h.update(b)
    return h.hexdigest()[:16]


def main():
    os.makedirs(OUT, exist_ok=True)
    rows = [r for r in csv.DictReader(open(AUDIT))
            if r["candidate_class"] == "FULL_ARCH_WITH_TARGET"]
    # 每个候选先标 stratum
    def bucket(v, edges):
        try:
            x = float(v)
        except (TypeError, ValueError):
            return "na"
        for i, e in enumerate(edges):
            if x < e:
                return f"{edges[i-1] if i else 0}-{e}"
        return f">={edges[-1]}"
    for r in rows:
        r["sb"] = bucket(r["target_surface_distance"], [1, 2, 3])
        r["grp"] = "META" if r["classification_basis"].startswith("METADATA") \
            else "GEO"
    # 分层抽样：GEO 按 (sb) 每层取 8；META 全取（7）；每层内按 case_uuid 排序取前 N（可复现）
    picked = {}
    for r in sorted(rows, key=lambda r: (r["grp"], r["sb"], r["case_uuid"])):
        key = (r["grp"], r["sb"])
        cnt = sum(1 for x in picked.values() if x["grp"] == r["grp"]
                  and x["sb"] == r["sb"])
        limit = 8 if r["grp"] == "GEO" else 10
        if cnt >= limit:
            continue
        k = f"{r['case_uuid']}:{r['target_fdi']}"
        if k in picked:
            continue
        picked[k] = r
    sel = list(picked.values())
    print(f"抽样 {len(sel)} 个候选", flush=True)

    manifest = []
    tiles = []
    for idx, r in enumerate(sel):
        cid = r["source_case_id"]
        # 校验 case_uuid 与 cid 对应（防呆）
        want = "cf_" + hashlib.sha256(cid.encode()).hexdigest()[:16]
        if want != r["case_uuid"]:
            print(f"  skip(哈希不匹配) {cid}", flush=True)
            continue
        # 找资产文件
        stls = []
        for root in ROOTS:
            stls += glob.glob(os.path.join(root, cid, "*.stl"))
        path = None
        for p in stls:
            try:
                if content_hash(p) == r["asset_id"]:
                    path = p
                    break
            except OSError:
                continue
        if path is None:
            print(f"  skip(找不到资产) {cid}", flush=True)
            continue
        # Center / insertion axis
        ci_files = (glob.glob(os.path.join(ROOTS[0], cid, "*.constructionInfo"))
                    + glob.glob(os.path.join(ROOTS[1], cid, "*.constructionInfo"))
                    + glob.glob(os.path.join(ROOTS[0], cid, "*.modelInfo")))
        center = axis = None
        if ci_files:
            try:
                case = parse_construction_case(ci_files[0])
                t = case.tooth(int(r["target_fdi"]))
                if t is not None:
                    center = tuple(t.center) if t.center else None
                    axis = t.insertion_axis
            except Exception:
                pass
        if center is None:
            print(f"  skip(无 Center) {cid}", flush=True)
            continue
        tris = stl_tris_near(path, center, WIN_MM*1.6)
        if not tris:
            print(f"  skip(邻域无三角) {cid}", flush=True)
            continue
        d = axis if axis and norm(axis) > 0.5 else [0.0, 0.0, 1.0]
        n = math.sqrt(sum(x*x for x in d))
        d = [x/n for x in d]
        top, _ = depth_map(tris, center, d)
        e1, _e2 = basis(d)
        side, _ = depth_map(tris, center, e1)
        tiles.append((top, side))
        manifest.append({
            "idx": len(tiles)-1, "case_uuid": r["case_uuid"],
            "fdi": int(r["target_fdi"]), "grp": r["grp"], "sb": r["sb"],
            "basis": r["classification_basis"], "role": r["metadata_role"],
            "surf_dist": r["target_surface_distance"],
            "near_min": r["near_dist_min"], "near_med": r["near_dist_median"],
            "cnt_r4": r["near_count_r4"], "coverage": r["near_coverage"],
        })
        print(f"  [{len(tiles)-1}] {r['case_uuid'][:12]} fdi={r['target_fdi']} "
              f"{r['grp']}/{r['sb']} dist={r['target_surface_distance']}", flush=True)

    # 拼图：每行一个候选（top | side），每 6 行一张
    per = 6
    for si in range(0, len(tiles), per):
        chunk = tiles[si:si+per]
        rows_img = []
        for top, side in chunk:
            for pv in range(GRID):
                rows_img.append(list(top[pv]) + [128]*8 + list(side[pv]))
            rows_img += [[30]*int(GRID*2+8)]*6          # 行分隔
        w = GRID*2+8
        h = len(rows_img)
        # 顶部画索引条：长度 = 全局序号（像素）
        idx0 = si
        for j in range(min(per, len(tiles)-si)):
            for k in range(idx0+j + 1):
                if k < w:
                    for rr in range(j*(GRID+6), j*(GRID+6)+4):
                        if rr < h:
                            rows_img[rr][k] = 255
        write_png(f"{OUT}/sheet_{si//per}.png", w, h, rows_img)
    json.dump(manifest, open(f"{OUT}/manifest.json", "w"), indent=1)
    print(f"\n完成：{len(tiles)} 个候选，{len(glob.glob(OUT + '/sheet_*.png'))} 张拼图")


def norm(v):
    return math.sqrt(sum(x*x for x in v))


if __name__ == "__main__":
    main()
