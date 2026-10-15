"""牙库（``牙齿数据库`` 的 ``.eoff``）查看/导出工具（纯 stdlib，无 numpy/trimesh）。

用途：把这个专业牙库数据变成**任何三维查看器都能打开的 STL / OBJ**，
以及直接看二进制格式内部结构（段名、逐顶点标签位、牙颈环）。

子命令::

    python tools/inspect_tooth_library.py list                 # 全库清点（三库 × 两颌 × 8 形态）
    python tools/inspect_tooth_library.py dump <x.eoff>        # 二进制结构 + 标签位 + 牙颈环
    python tools/inspect_tooth_library.py stl  46              # FDI 模板 -> STL
    python tools/inspect_tooth_library.py arch lowerjaw        # 整颌 16 牙装配 -> STL（看镜像/位姿）
    python tools/inspect_tooth_library.py margin 46            # 牙颈环 -> OBJ 折线（在查看器里当曲线看）

常用选项：``--lib 炼金王``（默认取第一个可用库）、``--out <目录>``（默认 data/tooth_library_viz）、
``--frame file|local|arch``（stl 子命令：原文件坐标 / 牙位局部系 / 装配到牙弓）。
"""
from __future__ import annotations

import argparse
import json
import math
import os
import struct
import sys
from collections import Counter

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from app.importers.tooth_library import (  # noqa: E402
    ENV_LIBRARY_ROOT,
    discover_libraries,
    find_margin_loop,
    jaw_for_fdi,
    label_bits_present,
    parse_eoff,
    sections_cover_file,
    tooth_index_for_fdi,
)


def _libs():
    libs = discover_libraries()
    if not libs:
        raise SystemExit(
            "找不到牙库：请设置 %s 指向 牙齿数据库 目录（当前=%r）"
            % (ENV_LIBRARY_ROOT, os.environ.get(ENV_LIBRARY_ROOT)))
    return libs


def _pick(libs, name):
    if name is None:
        return libs[0]
    for l in libs:
        if l.name == name:
            return l
    raise SystemExit("没有这个库：%s（可用：%s）" % (name, ", ".join(l.name for l in libs)))


def _default_out():
    out = os.path.join(REPO, "data", "tooth_library_viz")
    os.makedirs(out, exist_ok=True)
    return out


def _write_binary_stl(path, vertices, faces, name="tooth"):
    with open(path, "wb") as f:
        f.write(name.encode("ascii", "replace")[:80].ljust(80, b"\0"))
        f.write(struct.pack("<I", len(faces)))
        for a, b, c in faces:
            va, vb, vc = vertices[a], vertices[b], vertices[c]
            ux, uy, uz = (vb[0] - va[0], vb[1] - va[1], vb[2] - va[2])
            vx, vy, vz = (vc[0] - va[0], vc[1] - va[1], vc[2] - va[2])
            nx, ny, nz = (uy * vz - uz * vy, uz * vx - ux * vz, ux * vy - uy * vx)
            n = math.sqrt(nx * nx + ny * ny + nz * nz)
            if n > 1e-20:
                nx, ny, nz = nx / n, ny / n, nz / n
            else:
                nx = ny = nz = 0.0
            f.write(struct.pack("<12fH", nx, ny, nz, *va, *vb, *vc, 0))


def _write_obj_polyline(path, pts, closed=True):
    with open(path, "w", encoding="utf-8") as f:
        f.write("# margin (cervical) loop, mm\n")
        for p in pts:
            f.write("v %.6f %.6f %.6f\n" % p)
        idx = list(range(1, len(pts) + 1))
        if closed and len(idx) > 2:
            idx.append(1)
        f.write("l " + " ".join(str(i) for i in idx) + "\n")


def _perimeter(vs, idx, closed=True):
    L = 0.0
    m = len(idx)
    rng = range(m) if closed else range(m - 1)
    for i in rng:
        a, b = vs[idx[i]], vs[idx[(i + 1) % m]]
        L += math.dist(a, b)
    return L


def cmd_list(args):
    for lib in _libs():
        print("=" * 78)
        print("牙库：%s" % lib.name)
        for jaw in ("upperjaw", "lowerjaw"):
            bow = lib.bow(jaw)
            print("  %s：XML 覆盖 %d 个 FDI 槽位 %s" % (jaw, len(bow), sorted(bow)))
            for n in range(1, 9):
                p = os.path.join(lib.teeth_dir, jaw, "%d.eoff" % n)
                if not os.path.isfile(p):
                    continue
                m = parse_eoff(p)
                (lo, hi) = m.bbox()
                ex = sorted((hi[i] - lo[i] for i in range(3)), reverse=True)
                r = find_margin_loop(m)
                loop = ("牙颈环 %d 点 / %.1f mm / 位 0x%X" % (r[0].n_vertices, r[0].perimeter_mm, r[0].bit)
                        if r else "无标签环")
                print("    n=%d  V=%-6d F=%-6d  尺寸 %5.2f×%5.2f×%5.2f mm  %s"
                      % (n, m.n_vertices, m.n_faces, ex[0], ex[1], ex[2], loop))
        kinds = Counter()
        for jaw in ("upperjaw", "lowerjaw"):
            d = os.path.join(lib.teeth_dir, jaw)
            for f in os.listdir(d):
                if f.endswith(".eoff") and "-" in f:
                    kinds[f.split("-", 1)[1].rsplit(".", 1)[0]] += 1
        if kinds:
            print("  变体（非完整冠）：%s" % dict(kinds))
    print("=" * 78)
    print("打开方式：把导出的 .stl 直接拖进 MeshLab / Blender / Windows「3D 查看器」；")
    print(".obj（牙颈环折线）建议用 MeshLab（显示 Edges）。")


def cmd_dump(args):
    path = args.path
    m = parse_eoff(path)
    (lo, hi) = m.bbox()
    ex = [hi[i] - lo[i] for i in range(3)]
    print("文件        : %s" % path)
    print("魔数/头     : OFF BINARY, V=%d F=%d（声明 E=%d = 3F/2）"
          % (m.declared_vertex_count, m.declared_face_count, m.declared_edge_count))
    print("顶点/面     : %d / %d   内容哈希 %s" % (m.n_vertices, m.n_faces, m.content_hash[:16]))
    print("包围盒      : %s .. %s   尺寸 %.2f×%.2f×%.2f mm"
          % (tuple(round(v, 3) for v in lo), tuple(round(v, 3) for v in hi), ex[0], ex[1], ex[2]))
    print("段（extension，25 字节定长名）:")
    for s in m.sections:
        print("    %-28s payload=%s 偏移=%d" % (s.name, "无" if s.payload is None else "%d 字节/%d uint32"
                                                % (len(s.payload), s.n_uint32), s.offset))
    print("段覆盖文件  : %s" % sections_cover_file(path))
    be = m.boundary_edges()
    print("边界边      : %d（>0 表示网格有开放边界，可能是牙颈环、也可能是破洞）" % len(be))
    if m.labels is None:
        print("逐顶点标签  : 无")
    else:
        bits = label_bits_present(m)
        cnt = Counter()
        for lb in m.labels:
            for b in range(64):
                if lb >> b & 1:
                    cnt[1 << b] += 1
        print("逐顶点标签  : %d 位在用" % len(bits))
        for b in sorted(cnt):
            print("    位 0x%-10X 命中 %d 顶点" % (b, cnt[b]))
        r = find_margin_loop(m)
        if r:
            c, _bits = r
            print("牙颈环      : 位 0x%X，%d 点，周长 %.2f mm，z 中位 %.2f"
                  % (c.bit, c.n_vertices, c.perimeter_mm, c.z_median))
        else:
            print("牙颈环      : 未找到（不要硬猜位号）")


def cmd_stl(args):
    lib = _pick(_libs(), args.lib)
    fdi = args.fdi
    jaw = jaw_for_fdi(fdi)
    mesh = lib.template_mesh(fdi)
    out = args.out or _default_out()

    if args.frame == "file":
        verts, tag = list(mesh.vertices), "file"
    elif args.frame == "local":
        _e, _m, pts = lib.local_points(fdi)
        verts, tag = pts, "local"
    else:  # arch
        org, z, x, y = lib.bow_entry(fdi).frame()
        verts = [tuple(org[i] + v[0] * x[i] + v[1] * y[i] + v[2] * z[i] for i in range(3))
                 for v in mesh.vertices]
        tag = "arch"
    p = os.path.join(out, "%s_FDI%d_%s.stl" % (lib.name.replace("/", "_"), fdi, tag))
    _write_binary_stl(p, verts, mesh.faces, name="FDI%d %s" % (fdi, lib.name))
    r = find_margin_loop(mesh)
    print("已导出 %s（%d 顶点 / %d 面，坐标系=%s）" % (p, mesh.n_vertices, mesh.n_faces, tag))
    if r:
        print("  牙颈环：位 0x%X，%d 点，周长 %.2f mm" % (r[0].bit, r[0].n_vertices, r[0].perimeter_mm))


def cmd_arch(args):
    lib = _pick(_libs(), args.lib)
    jaw = args.jaw
    bow = lib.bow(jaw)
    out = args.out or _default_out()
    verts, faces = [], []
    split_dir = None
    if args.split_dir:
        split_dir = os.path.join(out, "split_%s_%s" % (lib.name.replace("/", "_"), jaw))
        os.makedirs(split_dir, exist_ok=True)
    for fdi in sorted(bow):
        m = lib.template_mesh(fdi)
        org, z, x, y = bow[fdi].frame()
        pl = [tuple(org[i] + v[0] * x[i] + v[1] * y[i] + v[2] * z[i] for i in range(3))
              for v in m.vertices]
        off = len(verts)
        verts.extend(pl)
        faces.extend((a + off, b + off, c + off) for a, b, c in m.faces)
        if split_dir:
            _write_binary_stl(os.path.join(split_dir, "FDI%d.stl" % fdi), pl, m.faces, "FDI%d" % fdi)
    p = os.path.join(out, "%s_%s_arch.stl" % (lib.name.replace("/", "_"), jaw))
    _write_binary_stl(p, verts, faces, name="%s %s" % (lib.name, jaw))
    print("已导出整颌装配 %s（%d 颗牙 / %d 顶点 / %d 面）" % (p, len(bow), len(verts), len(faces)))
    if split_dir:
        print("  每颗牙单文件：%s" % split_dir)
    print("提示：这是牙库 XML 位姿装配出来的牙弓——左右镜像与 FDI 映射是否正确，肉眼一看便知。")


def cmd_margin(args):
    lib = _pick(_libs(), args.lib)
    mesh = lib.template_mesh(args.fdi)
    r = find_margin_loop(mesh)
    if not r:
        raise SystemExit("FDI %d 在该库没有可恢复的牙颈环（不要硬猜）" % args.fdi)
    cand, bits = r
    idx = [i for i, lb in enumerate(mesh.labels) if lb & cand.bit]
    # 按网格邻接把环排成顺序折线
    adj = mesh.vertex_adjacency()
    order = [idx[0]]
    seen = {idx[0]}
    while True:
        nxt = [v for v in adj.get(order[-1], ()) if v in set(idx) and v not in seen]
        if not nxt:
            break
        order.append(nxt[0])
        seen.add(nxt[0])
    out = args.out or _default_out()
    p = os.path.join(out, "%s_FDI%d_margin.obj" % (lib.name.replace("/", "_"), args.fdi))
    _write_obj_polyline(p, [mesh.vertices[i] for i in order])
    print("已导出牙颈环折线 %s（%d 点，周长 %.2f mm，位 0x%X）"
          % (p, len(order), _perimeter(mesh.vertices, order), cand.bit))
    print("在 MeshLab 里：File→Import，图层里勾 Edges，就能看到这条颈缘曲线。")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("list", help="全库清点")
    p.set_defaults(fn=cmd_list)

    p = sub.add_parser("dump", help="看 .eoff 二进制结构/标签位/牙颈环")
    p.add_argument("path")
    p.set_defaults(fn=cmd_dump)

    p = sub.add_parser("stl", help="FDI 模板导出 STL")
    p.add_argument("fdi", type=int)
    p.add_argument("--lib", default=None)
    p.add_argument("--frame", default="arch", choices=["file", "local", "arch"])
    p.add_argument("--out", default=None)
    p.set_defaults(fn=cmd_stl)

    p = sub.add_parser("arch", help="整颌装配导出 STL（看镜像/位姿）")
    p.add_argument("jaw", choices=["upperjaw", "lowerjaw"])
    p.add_argument("--lib", default=None)
    p.add_argument("--out", default=None)
    p.add_argument("--split-dir", action="store_true", help="同时每颗牙单独一个 STL")
    p.set_defaults(fn=cmd_arch)

    p = sub.add_parser("margin", help="牙颈环导出 OBJ 折线")
    p.add_argument("fdi", type=int)
    p.add_argument("--lib", default=None)
    p.add_argument("--out", default=None)
    p.set_defaults(fn=cmd_margin)

    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
