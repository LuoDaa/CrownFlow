#!/usr/bin/env python3
"""把代型（prep）对齐到设计坐标系，并切出 Margin 以上的预备体表面。

## 坐标系链条（每一步都有几何证据，不假设）

```
代型 STL            ← 与 .modelInfo 同帧（margin 精确投影 p95=0，AutoMargin 已证）
.modelInfo margin   ← MODEL 坐标系的同一条颈线
case.json margin    ← constructionInfo margin（设计坐标系，== 冠导出坐标系）
```

两套 margin 是**同一条曲线的两次采样**（同顺序、差一个环移点，见 ``--report``），
所以用 Kabsch 求刚性变换 ``(R, t): MODEL → 设计系``；逐点残差就是两个采样之间
的差异，**不引入 GT**。变换后的对账：

* ``case.json margin → 代型表面`` 应 ≈ 0（同一颈线）；
* （仅验收）``GT 冠内表面 → 代型`` 应 ≈ CementGapThickness 量级 —— 证明
  "intaglio = 代型 + cement gap" 这一物理模型，对齐本身不用 GT。

## 预备体表面切法

代型是「预备体 + 代型基座」的实体。以 margin 折线为界：
顶点到 margin 的**轴向偏移** ``s = (v − m_near)·axis``，取 ``s > −eps`` 的连通
分量（含顶点最上方那个），面取"三点全在分量内"。这是拓扑切分，不假设颈线是平的。

用法::

    python3 align_prep_to_design.py \
        --case-dir <...>/crown_outer_cases/cf_b1b73bae221211b1 \
        --die /path/to/die.stl [--report]
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class PrepAlignError(Exception):
    pass


def kabsch(p: np.ndarray, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """求 (R, t) 使 R·p + t ≈ q（最小二乘，含反射修正）。"""
    pc, qc = p.mean(0), q.mean(0)
    h = (p - pc).T @ (q - qc)
    u, _s, vt = np.linalg.svd(h)
    d = np.diag([1.0, 1.0, float(np.sign(np.linalg.det(vt.T @ u.T)))])
    r = vt.T @ d @ u.T
    return r, qc - r @ pc


def fit_rigid_by_margin(mi: np.ndarray, cj: np.ndarray) -> dict:
    """两套同序 margin 点列的刚性对齐（搜索环移，取残差最小者）。"""
    if len(mi) != len(cj):
        raise PrepAlignError("两套 margin 点数不同：%d vs %d" % (len(mi), len(cj)))
    best = None
    for shift in range(len(cj)):
        q = np.roll(cj, shift, axis=0)
        r, t = kabsch(mi, q)
        res = (r @ mi.T).T + t - q
        rms = float(np.sqrt((res ** 2).sum(1).mean()))
        if best is None or rms < best["rms_mm"]:
            best = {"rms_mm": rms, "shift": shift, "R": r, "t": t,
                    "max_mm": float(np.abs(res).max())}
    return best


def prep_region(vertices: np.ndarray, faces: np.ndarray, margin: np.ndarray,
                axis: np.ndarray, *, eps_mm: float = 0.02) -> dict:
    """切出 margin 以上的预备体表面（顶点掩码 + 面掩码）。"""
    from scipy.spatial import cKDTree
    tree = cKDTree(margin)
    d_m, idx = tree.query(vertices, k=1)
    near = margin[idx]
    s = np.einsum("ij,j->i", vertices - near, axis)

    above = s > -eps_mm
    if not above.any():
        raise PrepAlignError("没有顶点在 margin 之上 —— 轴或点列有误")
    # 顶点邻接
    adj = {}
    for a, b, c in faces:
        for u, v in ((int(a), int(b)), (int(b), int(c)), (int(c), int(a))):
            adj.setdefault(u, set()).add(v)
            adj.setdefault(v, set()).add(u)
    apex = int(np.argmax(vertices @ axis))
    if not above[apex]:
        raise PrepAlignError("最上方顶点不在 margin 之上 —— 轴方向应为咬合向")
    seen, stack = {apex}, [apex]
    while stack:
        u = stack.pop()
        for v in adj.get(u, ()):
            if above[v] and v not in seen:
                seen.add(v)
                stack.append(v)
    vmask = np.zeros(len(vertices), dtype=bool)
    vmask[sorted(seen)] = True
    fmask = vmask[faces].all(axis=1)
    return {"vertex_mask": vmask, "face_mask": fmask,
            "d_margin_mm": d_m, "axial_offset_mm": s}


def apply_transform(mesh, r: np.ndarray, t: np.ndarray):
    out = mesh.copy()
    out.vertices = (r @ np.asarray(mesh.vertices, float).T).T + t
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case-dir", required=True)
    ap.add_argument("--die", required=True)
    ap.add_argument("--out-die", default="die_prep_frame.stl",
                    help="写出对齐后的整只代型（默认名）")
    ap.add_argument("--eps-mm", type=float, default=0.02)
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()

    import trimesh
    case = json.load(open(os.path.join(a.case_dir, "case.json"),
                          encoding="ascii"))
    ref = json.load(open(os.path.join(a.case_dir, "die_ref.json"),
                         encoding="ascii"))
    margin_cj = np.asarray(case["margin_points"], dtype=np.float64)
    margin_mi = np.asarray(ref["modelinfo_margin_points"], dtype=np.float64)
    axis = np.asarray(case["insertion_axis"], dtype=np.float64)
    axis = axis / np.linalg.norm(axis)

    fit = fit_rigid_by_margin(margin_mi, margin_cj)
    r, t = fit["R"], fit["t"]
    if fit["rms_mm"] > 0.10:
        raise PrepAlignError(
            "margin 刚性拟合残差 %.4fmm > 0.10mm —— 两帧不是刚性关系，不猜"
            % fit["rms_mm"])

    die = trimesh.load(a.die, force="mesh")
    if len(die.faces) == 0:
        raise PrepAlignError("代型网格为空")
    die_x = apply_transform(die, r, t)

    d_margin = trimesh.proximity.closest_point(die_x, margin_cj)[1]
    reg = prep_region(np.asarray(die_x.vertices, float),
                      np.asarray(die_x.faces, np.int64), margin_cj, axis,
                      eps_mm=a.eps_mm)
    fmask, vmask = reg["face_mask"], reg["vertex_mask"]
    if fmask.sum() < 100:
        raise PrepAlignError("预备体面数 %d 太少 —— 切分失败" % fmask.sum())

    # trimesh 5.x 的 submesh 收的是「子网格索引序列」——必须包一层 list，
    # 传平铺 int 列表会把每个 int 当成一个 index array（len() of unsized）。
    prep = die_x.submesh([np.flatnonzero(fmask)], append=True, repair=False)
    areas = prep.area_faces
    info = {
        "case_uuid": ref.get("case_uuid"),
        "fdi": int(ref["fdi"]),
        "die_asset_sha256": ref["die"]["sha256"],
        "die_part_type": ref["die"]["part_type"],
        "rigid_fit": {"shift": int(fit["shift"]), "rms_mm": fit["rms_mm"],
                      "max_mm": fit["max_mm"],
                      "R": np.asarray(r).tolist(), "t": np.asarray(t).tolist()},
        "margin_to_die_mm": {"p50": float(np.median(d_margin)),
                             "p95": float(np.quantile(d_margin, .95)),
                             "max": float(d_margin.max())},
        "prep_region": {"n_vertices": int(vmask.sum()),
                        "n_faces": int(fmask.sum()),
                        "area_mm2": float(areas.sum()),
                        "face_area_p50_mm2": float(np.median(areas))},
        "design_parameters": ref.get("design_parameters", {}),
        "note": ("代型切分只到 margin 以上；厚度审计在预备体顶点上采样。"
                 "R/t 只用两套 margin（设计输入），不用 GT。"),
    }

    # ---- 验收用同帧校验（仅距离统计，不参与求解）----
    gt_path = os.path.join(a.case_dir, "final_crown.stl")
    if os.path.exists(gt_path):
        from app.geometry.crown_outer.gt_compare import extract_outer_surface
        gt = trimesh.load(gt_path, force="mesh")
        outer_f = extract_outer_surface(gt, axis, margin_cj.mean(0))
        m = np.zeros(len(gt.faces), dtype=bool)
        m[outer_f] = True
        ip = gt.triangles_center[np.flatnonzero(~m)]
        d_gt = trimesh.proximity.closest_point(die_x, ip)[1]
        info["gt_intaglio_to_die_mm"] = {
            "p05": float(np.quantile(d_gt, .05)),
            "p50": float(np.quantile(d_gt, .50)),
            "p95": float(np.quantile(d_gt, .95)),
            "n": int(len(ip)),
            "audit_only": True,
        }
        info["gt_intaglio_note"] = (
            "仅同帧校验：GT 冠内表面应落在 CementGapThickness(%s mm) 量级；"
            "对齐本身不用 GT。" % ref.get("design_parameters", {})
            .get("CementGapThickness"))

    die_x.export(os.path.join(a.case_dir, a.out_die))
    prep.export(os.path.join(a.case_dir, "prep.stl"))
    with open(os.path.join(a.case_dir, "prep_meta.json"), "w",
              encoding="ascii") as fh:
        json.dump(info, fh, indent=1)
    print(json.dumps(info, indent=1, ensure_ascii=False))
    if a.report:
        print("\n代型 bounds(设计系):",
              np.round(die_x.bounds, 2).tolist())
        print("预备体 bounds:", np.round(prep.bounds, 2).tolist())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
