"""三视图渲染：placed template / deformed outer / GT outer（方案 §16 第 10 歘）。

复用 app/groundtruth/qa_render 的 stdlib z-buffer。每 meshes 两视图
（咬合向俯视 + 侧视），6 块拼一张 PNG 供人工确认。
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.groundtruth import qa_render as qr  # noqa: E402

CASE = r"D:\code\牙冠自动生成\数据\crown_outer_results\cf_d310209b58909a45"
CASE_IN = r"D:\code\牙冠自动生成\数据\crown_outer_cases\cf_d310209b58909a45"


def main():
    import trimesh
    meta = json.load(open(os.path.join(CASE_IN, "case.json"), encoding="ascii"))
    center = np.asarray(meta["margin_points"]).mean(axis=0)
    axis = np.asarray(meta["insertion_axis"])
    axis = axis / np.linalg.norm(axis)

    deformed = trimesh.load(os.path.join(CASE, "crown_outer_v0.stl"),
                            force="mesh")
    gt = trimesh.load(os.path.join(CASE, "gt_outer.stl"), force="mesh")

    # placed template（重算：模板 → 相似摆位）
    os.environ.setdefault("CROWNFLOW_TOOTH_LIBRARY_ROOT",
                          r"D:\code\牙冠自动生成\数据\牙齿数据库")
    from app.geometry.crown_outer import (build_patient_frame,  # noqa: E402
                                          load_crown_template, place_template,
                                          patient_margin_perimeter)
    from app.geometry.crown_outer import ensure_closed_ring_order  # noqa: E402
    from app.importers.tooth_library import discover_libraries  # noqa: E402
    mp = ensure_closed_ring_order(np.asarray(meta["margin_points"]))
    lib = {l.name: l for l in discover_libraries()}["炼金王"]
    t = load_crown_template(lib, 46)
    pf = build_patient_frame(mp, axis)
    pl = place_template(t, pf, patient_margin_perimeter(mp))
    placed = pl.apply(t.vertices)
    crown_faces = t.faces[~t.root_vertex_mask[t.faces].any(axis=1)]

    r = max(placed.max(0) - placed.min(0)) * 0.75
    meshes = [
        ("placed", (placed, crown_faces)),
        ("deformed", (np.asarray(deformed.vertices),
                      np.asarray(deformed.faces))),
        ("gt_outer", (np.asarray(gt.vertices), np.asarray(gt.faces))),
    ]
    GRID = qr.GRID
    tiles = []
    for name, (verts, faces) in meshes:
        tris = verts[faces]
        top, _ = qr.depth_map(tris, center, axis)         # 咬合向
        side, _ = qr.depth_map(tris, center, _side_dir(axis))  # 侧向
        tiles.append((name + " top", top))
        tiles.append((name + " side", side))
        print(name, "tris", len(tris))

    # 拼图：2 列（top/side）× 3 行（placed/deformed/gt）
    gap = 8
    W = GRID * 2 + gap
    H = (GRID + 14) * 3
    img = [[30] * W for _ in range(H)]
    for i, (name, tile) in enumerate(tiles):
        row, col = divmod(i, 2)
        y0 = row * (GRID + 14)
        x0 = col * (GRID + gap)
        for yy in range(GRID):
            for xx in range(GRID):
                img[y0 + 14 + yy][x0 + xx] = tile[GRID - 1 - yy][xx]
        for k, ch in enumerate(name.encode("ascii")):
            for px in range(4):
                for py in range(6):
                    if ch >> (px * 6 + py) & 1 and 5 + py < 12:
                        img[y0 + 5 + py][x0 + 5 + k * 6 + px] = 255
    out = os.path.join(CASE, "renders")
    os.makedirs(out, exist_ok=True)
    p = os.path.join(out, "three_views.png")
    qr.write_png(p, W, H, img)
    print("写出", p)


def _side_dir(axis):
    e = np.array([0.0, 0.0, 1.0]) if abs(axis[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    e1 = np.cross(axis, e)
    return e1 / np.linalg.norm(e1)


if __name__ == "__main__":
    main()
