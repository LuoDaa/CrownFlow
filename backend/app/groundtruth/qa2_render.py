"""QA2 渲染：按 v3.2 格搜索结果分层抽 55 例，峰位 + 目标牙 margin 轮廓标在俯视图上。

    python3 -m app.groundtruth.qa2_render \
        --lattice /tmp/gt_v31/lattice.csv \
        --features /tmp/gt_v31/features.csv \
        --root /data1/welo/dentistry/exo/all_data \
        --root /data1/welo/dentistry/exo/EXO \
        --out /tmp/qa2

产出 ``sheet_*.png``（每张 5 例，三栏：俯视 / 侧视 / 径向高度曲线）与
``manifest_v2.json``（``human_label`` 留空，由人填）。

**为什么俯视图上要叠 margin 轮廓**：v3.2 的判据是"最高峰落在目标牙自己的
margin 轮廓内"。轮廓与几何用同一个 ``in_plane_basis(zdir)`` 投影，所以
"峰是否落在轮廓内"可以纯目视核对，不必相信判据实现。这一条是 QA2 的
关键读数，结论见 ``docs/LOCALIZATION_QA2.md``。

分层刻意**过采样工作点** ``in_poly``（v3.2 唯一会判 Prep 的那批）。两个
分层在语料里本身近乎为空（``v2pos_weak`` 0 例、``miss_strong`` 1 例），
所以实际出 55 例而不是按容量算的 68 例 —— 这是数据结构，不是抽样失误。
"""
from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import json
import math
import os
from urllib.parse import unquote

from . import qa_render as qr
from .v3_features import (
    DEFAULT_ROOTS,
    in_plane_basis,
    resolve_frame,
)
from ..importers.exocad.construction_info import parse_construction_case

#: 上限 55 = 11 张拼图（用户规格 50~60）。
CAP = 55
PER_SHEET = 5

#: 分层容量：(标签, 说明)。实际样本数受语料可用量限制。
STRATA = (
    ("A_high", 10),
    ("in_poly", 14),
    ("border", 14),
    ("mid", 12),
    ("v2pos_weak", 6),
    ("miss_strong", 10),
    ("unk_strong", 14),
    ("meta", 5),
)


def _case_dir_map(roots) -> dict[str, str]:
    """case_uuid -> 病例目录名。uuid 是目录 basename 的 sha256[:16]。"""
    out: dict[str, str] = {}
    for root in roots:
        for dp, dn, _f in os.walk(root):
            for d in dn:
                k = "cf_" + hashlib.sha256(d.encode()).hexdigest()[:16]
                out.setdefault(k, d)
    return out


def select(lat_rows, feat_by_key) -> list[tuple[str, dict, dict]]:
    """分层选样（去重按 (case_uuid, fdi)），返回 [(tag, lattice_row, feat_row)]。"""
    joined = []
    for r in lat_rows:
        f = feat_by_key.get((r["case_uuid"], r["fdi"], r["asset_id"]))
        if f is not None:
            joined.append((r, f))

    def cls(x):
        return x[0]["candidate_class"]

    wt = [x for x in joined if cls(x) == "FULL_ARCH_WITH_TARGET"]
    tm = [x for x in joined if cls(x) == "FULL_ARCH_TARGET_MISSING"]
    unk = [x for x in joined if cls(x) not in ("FULL_ARCH_WITH_TARGET",
                                               "FULL_ARCH_TARGET_MISSING")]
    for group in (wt, tm, unk):
        group.sort(key=lambda x: -x[0]["score"])

    in_poly = [x for x in wt if x[0].get("peak1_in_poly") == "True"]
    buckets = {
        "A_high": wt[:10],
        "in_poly": in_poly[:14],
        "border": wt[12:26],
        "mid": wt[30:42],
        "v2pos_weak": [x for x in wt if x[0]["prom"] < 1.0][:6],
        "miss_strong": tm[:10],
        "unk_strong": unk[:14],
        "meta": [x for x in wt
                 if x[0]["classification_basis"].startswith("METADATA")][:5],
    }
    sel = []
    for tag, _cap in STRATA:
        sel += [(tag, x) for x in buckets.get(tag, [])]

    seen, final = set(), []
    for tag, (lr, fr) in sel:
        k = (lr["case_uuid"], lr["fdi"])
        if k in seen:
            continue
        seen.add(k)
        final.append((tag, lr, fr))
    return final[:CAP]


def profile_tile(pr, pz, grid=200, h=200):
    """径向高度曲线：横轴 r（带 r=4/6mm 参考线），纵轴 z。"""
    img = [[255] * grid for _ in range(h)]
    if not pr:
        return img
    lo, hi = min(pz), max(pz)
    rng = (hi - lo) or 1.0
    rmax = max(pr) or 1.0
    for x in range(grid):
        img[grid-1][x] = 120
    for y in range(h):
        img[y][0] = 120
    pts = [(int(r/rmax*(grid-8))+4, int((1-(z-lo)/rng)*(h-14))+6)
           for r, z in zip(pr, pz)]
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        steps = max(abs(x1-x0), abs(y1-y0)) or 1
        for s in range(steps+1):
            xx = int(x0 + (x1-x0)*s/steps)
            yy = int(y0 + (y1-y0)*s/steps)
            if 0 <= yy < h and 0 <= xx < grid:
                img[yy][xx] = 0
                if yy+1 < h:
                    img[yy+1][xx] = 0
    for rr in (4.0, 6.0):
        x = int(rr/rmax*(grid-8))+4
        if 0 <= x < grid:
            for y in range(h):
                if y % 4 < 2:
                    img[y][x] = 200
    return img


def _mark_polygon(top, margin_points, center, e1, e2, grid, win):
    """把目标牙 margin 轮廓以点线画在俯视图上（每 5 像素画 3）。"""
    if len(margin_points) < 3:
        return 0
    poly = []
    for p in margin_points:
        rel = [p[i] - center[i] for i in range(3)]
        u = sum(rel[i] * e1[i] for i in range(3))
        v = sum(rel[i] * e2[i] for i in range(3))
        poly.append((int((u + win) / (2*win) * (grid-1)),
                     int((v + win) / (2*win) * (grid-1))))
    for (x0, y0), (x1, y1) in zip(poly, poly[1:] + poly[:1]):
        steps = max(abs(x1-x0), abs(y1-y0)) or 1
        for s in range(steps + 1):
            xx = int(x0 + (x1-x0)*s/steps)
            yy = int(y0 + (y1-y0)*s/steps)
            if (xx + yy) % 5 < 3 and 0 <= yy < grid and 0 <= xx < grid:
                top[yy][xx] = 255
    return len(margin_points)


def _mark_peaks(top, lr, grid, win):
    """峰位方块标记（1=亮，2/3 依次变暗）。"""
    for k in (1, 2, 3):
        du = lr.get("peak%d_du" % k)
        dv = lr.get("peak%d_dv" % k)
        if du in ("", None) or dv in ("", None):
            continue
        du, dv = float(du), float(dv)
        px = int((du + win) / (2*win) * (grid-1))
        py = int((dv + win) / (2*win) * (grid-1))
        color = 255 if k == 1 else (170 if k == 2 else 120)
        for a in range(-4, 5):
            for b in range(-4, 5):
                if max(abs(a), abs(b)) in (3, 4) and \
                        0 <= py+a < grid and 0 <= px+b < grid:
                    top[py+a][px+b] = color


def render(lattice: str, features: str, roots, out: str) -> dict:
    os.makedirs(out, exist_ok=True)
    feat = {}
    for r in csv.DictReader(open(features, encoding="utf-8")):
        if r["feature_status"] == "OK":
            feat[(r["case_uuid"], r["fdi"], r["asset_id"])] = r

    lat = []
    for r in csv.DictReader(open(lattice, encoding="utf-8")):
        s = r.get("peak1_score", "")
        if s in ("", "NO_CENTER", "FEATURE_UNAVAILABLE", "ASSET_MISSING",
                 "READ_FAILED"):
            continue
        r["score"] = float(s)
        r["prom"] = float(r.get("peak1_prom") or 0)
        r["cov"] = float(r.get("peak1_cov") or 0)
        lat.append(r)

    final = select(lat, feat)
    cid_by_uuid = _case_dir_map(roots)
    print("选样 %d 例（容量 %d；v2pos_weak 可得 %d，miss 可得 %d）" % (
        len(final), sum(c for _t, c in STRATA),
        len([r for r in lat if r["candidate_class"] == "FULL_ARCH_WITH_TARGET"
             and r["prom"] < 1.0]),
        len([r for r in lat
             if r["candidate_class"] == "FULL_ARCH_TARGET_MISSING"])), flush=True)

    manifest, tiles = [], []
    skipped = {"no_case_dir": 0, "no_ci": 0, "no_center": 0,
               "no_frame": 0, "no_tris": 0}
    for tag, lr, fr in final:
        cid = cid_by_uuid.get(lr["case_uuid"])
        path = unquote(fr["asset_path_local_only"])
        fdi = int(lr["fdi"])
        if cid is None or not os.path.exists(path):
            skipped["no_case_dir"] += 1
            print("  skip", lr["case_uuid"][:12], flush=True)
            continue
        cifs = []
        for root in roots:
            cifs += glob.glob(os.path.join(root, cid, "*.constructionInfo"))
            cifs += glob.glob(os.path.join(root, cid, "*.modelInfo"))
        if not cifs:
            skipped["no_ci"] += 1
            print("  skip(no ci)", lr["case_uuid"][:12], flush=True)
            continue
        cifs.sort(key=lambda q: not q.endswith(".constructionInfo"))
        try:
            case = parse_construction_case(cifs[0])
        except Exception:
            skipped["no_ci"] += 1
            continue
        t = case.tooth(fdi)
        if t is None or not t.center:
            skipped["no_center"] += 1
            continue
        center = tuple(t.center)
        frame, _src = resolve_frame(t, t.margin_points)
        if frame is None:
            skipped["no_frame"] += 1
            continue
        zdir = frame[0]
        try:
            tris = qr.stl_tris_near(path, center, qr.WIN_MM * 1.6)
        except Exception:
            skipped["no_tris"] += 1
            continue
        if not tris:
            skipped["no_tris"] += 1
            continue

        top, _ = qr.depth_map(tris, center, zdir)
        e1, e2 = in_plane_basis(zdir)      # == depth_map 内部用的 basis(zdir)
        side, _ = qr.depth_map(tris, center, e1)
        n_mp = _mark_polygon(top, list(t.margin_points or []), center,
                             e1, e2, qr.GRID, qr.WIN_MM)
        _mark_peaks(top, lr, qr.GRID, qr.WIN_MM)
        pr = [float(v) for v in (fr["profile_r"] or "").split(";") if v]
        pz = [float(v) for v in (fr["profile_z"] or "").split(";") if v]
        tiles.append((top, side, profile_tile(pr, pz)))
        manifest.append({
            "idx": len(tiles) - 1, "tag": tag,
            "case_uuid": lr["case_uuid"], "fdi": fdi,
            "v2_class": lr["candidate_class"],
            "basis": lr["classification_basis"],
            "v3_2_score": lr["score"], "peak1_prom": lr["prom"],
            "peak1_cov": lr["cov"],
            "peak1_rel_pos": lr.get("peak1_rel_pos"),
            "peak1_dist_margin": lr.get("peak1_dist_margin"),
            "peak1_in_poly": lr.get("peak1_in_poly"),
            "surf_dist": lr["target_surface_distance"],
            "n_margin_pts": n_mp,
            "human_label": "",
        })
        print("  [%d] %-11s %s fdi=%-2d prom=%.2f cov=%.2f d_marg=%s" % (
            len(tiles) - 1, tag, lr["case_uuid"][:12], fdi, lr["prom"],
            lr["cov"], lr.get("peak1_dist_margin")), flush=True)

    _write_sheets(tiles, out)
    json.dump(manifest, open(os.path.join(out, "manifest_v2.json"), "w"),
              indent=1)
    summary = {"n_selected": len(final), "n_rendered": len(tiles),
               "n_sheets": len(glob.glob(os.path.join(out, "sheet_*.png"))),
               "skipped": skipped}
    json.dump(summary, open(os.path.join(out, "summary_v2.json"), "w"), indent=1)
    print(json.dumps(summary, indent=1))
    return summary


def _write_sheets(tiles, out):
    """每 PER_SHEET 例拼一张：三栏（俯视 / 侧视 / 径向曲线）+ 行号标记。"""
    total_w = qr.GRID * 3 + 16
    for si in range(0, len(tiles), PER_SHEET):
        chunk = tiles[si:si + PER_SHEET]
        rows_img = []
        for top, side, prof in chunk:
            for pv in range(qr.GRID):
                rows_img.append(list(top[pv]) + [128]*8 + list(side[pv])
                                + [128]*8 + list(prof[pv]))
            rows_img += [[30] * total_w] * 6
        h = len(rows_img)
        for j in range(len(chunk)):
            for k in range(si + j + 1):
                if k < total_w:
                    for rr in range(j * (qr.GRID + 6), j * (qr.GRID + 6) + 4):
                        if rr < h:
                            rows_img[rr][k] = 255
        qr.write_png(os.path.join(out, "sheet_%d.png" % (si // PER_SHEET)),
                     total_w, h, rows_img)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="QA2 分层渲染（v3.2 判据目视核对）")
    ap.add_argument("--lattice", required=True)
    ap.add_argument("--features", required=True)
    ap.add_argument("--root", action="append", default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    render(a.lattice, a.features, a.root or list(DEFAULT_ROOTS), a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
