"""把 debug 网格（红=自交面 / 黄=邻近 / 绿=正常）渲染成 PNG。

颜色信息从同名 `.json` 的 `red_faces` 拿（GLB 往返会丢失逐面色），
soup 的面序与失败当次的 result 面序一致，所以面 id 直接可用。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import trimesh

RED = np.array([0.90, 0.24, 0.24, 1.0])
YELLOW = np.array([0.96, 0.78, 0.24, 1.0])
GREEN = np.array([0.35, 0.75, 0.47, 1.0])


def load_coloured(glb_path: Path, json_path: Path):
    mesh = trimesh.load(glb_path, force="mesh", process=False)
    if hasattr(mesh, "geometry"):
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
    faces = np.asarray(mesh.faces)
    report = json.loads(json_path.read_text(encoding="utf-8"))
    red_ids = np.asarray(report.get("red_faces") or [], np.int64)
    colours = np.tile(GREEN, (len(faces), 1))
    if len(red_ids):
        touched = np.zeros(len(mesh.vertices), bool)
        touched[np.unique(faces[red_ids].ravel())] = True
        neighbour = np.any(touched[faces], axis=1) & ~np.isin(np.arange(len(faces)), red_ids)
        colours[neighbour] = YELLOW
        colours[red_ids] = RED
    return mesh, colours, red_ids, report


def render(mesh, colours, out_path, *, title, focus=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    tri_all = np.asarray(mesh.triangles)
    keep = np.ones(len(tri_all), bool)
    if focus is not None:
        centre_all = tri_all.mean(axis=1)
        keep = np.all((centre_all >= focus[0]) & (centre_all <= focus[1]), axis=1)
    tri = tri_all[keep]
    face_colours = colours[keep]
    if not len(tri):
        print("no faces in focus box")
        return None
    fig = plt.figure(figsize=(11, 5.4), dpi=110)
    for index, (elev, azim, label) in enumerate(((22, -60, "view A"), (22, 35, "view B"))):
        ax = fig.add_subplot(1, 2, index + 1, projection="3d")
        ax.add_collection3d(Poly3DCollection(tri, facecolors=face_colours, edgecolors="none", linewidths=0))
        centre = tri.reshape(-1, 3).mean(axis=0)
        span = float(np.ptp(tri.reshape(-1, 3), axis=0).max()) / 2 * 1.02
        for setter, value in ((ax.set_xlim, centre[0]), (ax.set_ylim, centre[1]), (ax.set_zlim, centre[2])):
            setter(value - span, value + span)
        ax.view_init(elev=elev, azim=azim)
        ax.set_axis_off()
        ax.set_title(label, fontsize=8)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out_path, "faces", len(tri))
    return str(out_path)


def main():
    glb = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("debug_out/cervical_stitch_00272081.glb")
    js = glb.with_suffix(".json")
    mesh, colours, red_ids, report = load_coloured(glb, js)
    print("faces", len(mesh.faces), "red", len(red_ids))
    out = glb.parent
    render(mesh, colours, out / "overview.png",
           title="Cervical stitch: red = self-intersecting faces, yellow = neighbours, green = ok")
    if len(red_ids):
        centre = np.asarray(mesh.triangles)[red_ids].reshape(-1, 3).mean(axis=0)
        render(mesh, colours, out / "self_intersection_zoom.png",
               title="Self-intersection zoom (5 sliver pairs)", focus=(centre - 5.0, centre + 5.0))
    # 只画红色面 + 黄色面，更清楚
    subset = np.isin(np.arange(len(mesh.faces)), red_ids) | np.all(colours == YELLOW, axis=1)
    if subset.any():
        scene = mesh.copy()
        scene.update_faces(subset)
        render(scene, colours[subset], out / "red_faces_only.png",
               title="Only red (self-intersecting) + yellow (neighbouring) faces",
               focus=(np.asarray(mesh.triangles)[red_ids].reshape(-1, 3).mean(axis=0) - 8.0,
                  np.asarray(mesh.triangles)[red_ids].reshape(-1, 3).mean(axis=0) + 8.0) if len(red_ids) else None)
    print("report summary:", json.dumps(report.get("entries", [{}])[0].get("category", "n/a")))


if __name__ == "__main__":
    main()
