#!/usr/bin/env python3
"""M0 标签可视化 —— 随机抽 N 例渲染 face-level 标签（用户要求：先看这个）。

**这是 M0 验收的关键一步**：标签错了，后面再强的模型也没意义。

每例渲两个视角（沿 InsertionAxis 的俯视 = 冠方视角；一个横向侧视），
按面着色：Class1（冠方）红、Class0（颈方）蓝、切割带（未标）黄，
**再把 Margin 曲线画成绿色** —— 绿线必须正好落在红蓝交界上。

    python3 render_auto_margin_qa.py --out /tmp/automargin --n 24 \
        --sheet 6 --seed 7
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import struct
import sys
import zlib

sys.path.insert(0, "/tmp/gt")

from build_auto_margin_gt import (case_dirs, find_by_hash, find_ci,  # noqa: E402
                                 read_stl, strip_sha, unit, _hash_to_path)
from app.importers.exocad.construction_info import parse_construction_case  # noqa: E402
from app.groundtruth.indexer import fdis_in_name                        # noqa: E402

GRID = 260
WIN_MM = 14.0          # 视口半宽
COL_CLASS1 = (215, 70, 70)      # 冠方 红
COL_CLASS0 = (60, 90, 200)      # 颈方 蓝
COL_CUT = (240, 215, 80)        # 切割带 黄
COL_MARGIN = (40, 220, 60)      # Margin 曲线 绿
COL_EMPTY = (18, 18, 18)


def basis(d):
    a = (0.0, 0.0, 1.0) if abs(d[2]) < 0.9 else (1.0, 0.0, 0.0)
    e1 = (d[1]*a[2]-d[2]*a[1], d[2]*a[0]-d[0]*a[2], d[0]*a[1]-d[1]*a[0])
    n = math.sqrt(sum(x*x for x in e1)) or 1.0
    e1 = tuple(x/n for x in e1)
    e2 = (d[1]*e1[2]-d[2]*e1[1], d[2]*e1[0]-d[0]*e1[2], d[0]*e1[1]-d[1]*e1[0])
    return e1, e2


def render(tris, labels, margin, view_dir, center):
    """z-buffer 上色（沿 view_dir 看，取最近面），返回 GRID×GRID 的 RGB。"""
    e1, e2 = basis(view_dir)
    zbuf = [[None]*GRID for _ in range(GRID)]
    cbuf = [[COL_EMPTY]*GRID for _ in range(GRID)]
    scale = (2*WIN_MM)/GRID
    for fi, t in enumerate(tris):
        pts = []
        skip = False
        for p in t:
            r = (p[0]-center[0], p[1]-center[1], p[2]-center[2])
            u = r[0]*e1[0]+r[1]*e1[1]+r[2]*e1[2]
            v = r[0]*e2[0]+r[1]*e2[1]+r[2]*e2[2]
            d = r[0]*view_dir[0]+r[1]*view_dir[1]+r[2]*view_dir[2]
            if abs(u) > WIN_MM or abs(v) > WIN_MM:
                skip = True
                break
            pts.append((u, v, d))
        if skip:
            continue
        lab = labels[fi]
        col = (COL_CLASS1 if lab == 1 else
               (COL_CLASS0 if lab == 0 else COL_CUT))
        (x0, y0, z0), (x1, y1, z1), (x2, y2, z2) = pts
        minu = max(0, int((min(x0, x1, x2)+WIN_MM)/scale))
        maxu = min(GRID-1, int((max(x0, x1, x2)+WIN_MM)/scale))
        minv = max(0, int((min(y0, y1, y2)+WIN_MM)/scale))
        maxv = min(GRID-1, int((max(y0, y1, y2)+WIN_MM)/scale))
        det = (x1-x0)*(y2-y0)-(x2-x0)*(y1-y0)
        if abs(det) < 1e-12:
            continue
        for pv in range(minv, maxv+1):
            v = -WIN_MM + (pv+0.5)*scale
            for pu in range(minu, maxu+1):
                u = -WIN_MM + (pu+0.5)*scale
                w0 = ((x1-u)*(y2-v)-(x2-u)*(y1-v))/det
                w1 = ((x2-u)*(y0-v)-(x0-u)*(y2-v))/det
                w2 = 1.0-w0-w1
                if w0 < -1e-9 or w1 < -1e-9 or w2 < -1e-9:
                    continue
                dd = w0*z0 + w1*z1 + w2*z2
                if zbuf[pv][pu] is None or dd > zbuf[pv][pu]:
                    zbuf[pv][pu] = dd
                    cbuf[pv][pu] = col
    # Margin 曲线覆盖画（绿点）—— 必须落在红蓝交界
    for p in margin:
        r = (p[0]-center[0], p[1]-center[1], p[2]-center[2])
        u = r[0]*e1[0]+r[1]*e1[1]+r[2]*e1[2]
        v = r[0]*e2[0]+r[1]*e2[1]+r[2]*e2[2]
        pu = int((u+WIN_MM)/scale)
        pv = int((v+WIN_MM)/scale)
        for a in (-1, 0, 1):
            for b in (-1, 0, 1):
                yy, xx = pv+a, pu+b
                if 0 <= yy < GRID and 0 <= xx < GRID:
                    cbuf[yy][xx] = COL_MARGIN
    return cbuf


def write_png_rgb(path, w, h, rows):
    raw = bytearray()
    for row in rows:
        raw.append(0)
        for px in row:
            raw += bytes(px)
    def chunk(tag, data):
        c = struct.pack(">I", len(data)) + tag + data
        return c + struct.pack(">I", zlib.crc32(tag + data) & 0xffffffff)
    png = b"\x89PNG\r\n\x1a\n"
    png += chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
    png += chunk(b"IDAT", zlib.compress(bytes(raw), 6))
    png += chunk(b"IEND", b"")
    open(path, "wb").write(png)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--sheet", type=int, default=6)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--die-selection", action="append", default=None,
                    help="几何选代型 CSV（与 build 用同一份，保证网格一致）")
    a = ap.parse_args()
    # 几何选代型表：与 build 同一份，保证"标签在哪颗网格上"一致
    die_selection = {}
    for path in (a.die_selection or []):
        for r in csv.DictReader(open(path, encoding="ascii")):
            if r.get("selection") == "MARGIN_ON_DIE" and r.get("die_asset_id"):
                die_selection[(r["case_uuid"], r["fdi"])] = r["die_asset_id"]
    print("几何选代型表: %d 行" % len(die_selection))
    ok = [r for r in csv.DictReader(
        open(os.path.join(a.out, "auto_margin_gt.csv"), encoding="ascii"))
        if r["qc_status"] == "OK"]
    if not ok:
        raise SystemExit("没有 OK 行")
    rng = random.Random(a.seed)
    pick = rng.sample(ok, min(a.n, len(ok)))
    print("抽 %d / %d 例（seed=%d）" % (len(pick), len(ok), a.seed))

    img_dir = os.path.join(a.out, "qa_render")
    os.makedirs(img_dir, exist_ok=True)
    tiles = []
    for k, r in enumerate(pick):
        try:
            dirs = case_dirs(r["case_uuid"])
            ci = find_ci(dirs)
            case = parse_construction_case(ci[0])
            tooth = case.tooth(int(r["fdi"]))
            axis = unit(tuple(float(v) for v in tooth.insertion_axis))
            margin = [(float(p[0]), float(p[1]), float(p[2]))
                      for p in tooth.margin_points]
            # 用病例目录里的代型（与 build 同一路径）
            idx_row = None
            for x in csv.DictReader(open(
                    "/tmp/gt_freeze_v1/frozen_tooth_index.csv",
                    encoding="ascii")):
                if x["case_uuid"] == r["case_uuid"] and x["fdi"] == r["fdi"]:
                    idx_row = x
                    break
            # 网格解析必须**跟构建用同一个资产**：证据模式（标签来自
            # *modeldie* 真代型）下索引里的 modeldie_asset_id 是回退挑中的
            # 冠壳，用它会"标签在代型、网格画冠壳"。所以这里优先按 fdi
            # 匹配目录里的 *modeldie* 文件，匹配不到才回退索引资产。
            # 网格解析必须与构建**逐字一致**。构建在用几何选代型
            # （--die-selection，按内容哈希），这里也走同一条路；没有选择表时
            # 才退回文件名证据 / 索引资产（旧行为）。
            die = None
            h = die_selection.get((r["case_uuid"], r["fdi"]))
            if h:
                die = _hash_to_path(dirs).get(h)
            if die is None and r.get("asset_kind") == "DIE_LIKE":
                want = int(r["fdi"])
                cand = []
                for d in dirs:
                    try:
                        cand += [os.path.join(d, f) for f in os.listdir(d)
                                 if f.lower().endswith(".stl")
                                 and "modeldie" in f.lower()]
                    except OSError:
                        pass
                for f in cand:
                    try:
                        if want in fdis_in_name(f):
                            die = f
                            break
                    except Exception:                  # noqa: BLE001
                        continue
            if die is None:
                die = find_by_hash(dirs, strip_sha(idx_row["modeldie_asset_id"]))
            if die is None:
                print("  [%2d] %s fdi=%-3s 网格解析失败，跳过"
                      % (k, r["case_uuid"][:14], r["fdi"]), flush=True)
                continue
            tris = read_stl(die)
            lab_path = os.path.join(a.out, "labels",
                                    r["prep_mesh_ref"])
            # prep_mesh_ref 存的是 labels bin 名（见 _build_one）
            lb = open(os.path.join(a.out, "labels", r["prep_mesh_ref"]),
                      "rb").read()
            labels = list(lb)
            center = tuple(sum(sum(vv[i] for vv in t) / 3.0
                               for t in tris) / len(tris) for i in range(3))
            top = render(tris, labels, margin, axis, center)
            e1, _e2 = basis(axis)
            side = render(tris, labels, margin, e1, center)
            tiles.append((top, side, r))
            print("  [%2d] %s fdi=%-3s" % (k, r["case_uuid"][:14], r["fdi"]),
                  flush=True)
        except Exception as exc:                     # noqa: BLE001
            print("  [%2d] %s 渲染失败 %s" % (k, r["case_uuid"][:14], exc),
                  flush=True)
    # 拼图：每张 sheet 放 per 例，两个视角并排
    per = a.sheet
    for si in range(0, len(tiles), per):
        chunk = tiles[si:si+per]
        gap = 6
        W = GRID*2 + gap
        H = (GRID + gap) * len(chunk)
        img = [[COL_EMPTY]*W for _ in range(H)]
        for j, (top, side, _r) in enumerate(chunk):
            y0 = j*(GRID+gap)
            for yy in range(GRID):
                row = img[y0+yy]
                trow = top[GRID-1-yy]
                srow = side[GRID-1-yy]
                for xx in range(GRID):
                    row[xx] = trow[xx]
                    row[GRID+gap+xx] = srow[xx]
        write_png_rgb(os.path.join(img_dir, "sheet_%d.png" % (si//per)),
                      W, H, img)
    print("写出", img_dir)


if __name__ == "__main__":
    main()
