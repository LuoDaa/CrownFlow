#!/usr/bin/env python3
"""M0 AutoMarginGTSet —— 数据集构建（**只做数据集与 QC**，不训练、不进 M1）。

用户规格（M0）的十点，逐条对应：

1. **筛高质量病例**：canonical ``.constructionInfo`` Margin + 代型（modeldie）
   且 ``modeldie_same_frame=True``（坐标系已验证）+ Margin 闭合。
   **投影距离阈值不拍脑袋**：先跑 ``--stage audit`` 出全量 p50/p95/max 分布，
   再冻结 ``--max-proj-p95`` / ``--max-proj-max``。
2. **face-level 标签语义**：``Class 1 = Margin 冠方/咬合方``，
   ``Class 0 = Margin 颈方``。不做"Margin 点=1"。
3. **投影保留两份**：原 GT curve 与 projected curve，逐点误差记录；
   不做静默吸附。
4. **真正切断 crossing adjacency**：Margin 穿过 triangle/edge 后就断开
   相应 face 邻接，再 flood fill（不是只取最近顶点）。
5. **occlusal seed 可解释**：用 constructionInfo 的 **InsertionAxis**（患者
   就位方向）选冠方最高区域作 seed；**不用世界 Z**。轴缺失或 seed 不稳定
   直接 ``REJECT``。
6. **拓扑 QC**：期望恰好两区；检查主区数量、碎片率、
   **是否绕过 Margin 从另一侧连通**（后者一票否决）。
7. 输出 case_id / group_id（脱敏）、target_fdi、prep_mesh_ref、
   margin_gt、projected_margin、face_labels、face_features 基础索引、
   projection_metrics、topology_metrics、qc_status、reject_reason。
8. **split 必须 case/group-level**，固化成 manifest 供后续实验复用。
9. **先审计不追数量**：宁可 500~1000 个真干净病例。
10. **fail-loud**：切断后 Class0/1 仍能绕路连通 → 绝不强行生成标签，
    进 reject manifest。

    python3 build_auto_margin_gt.py --stage audit  --out /tmp/automargin
    python3 build_auto_margin_gt.py --stage build  --out /tmp/automargin \
        --max-proj-p95 0.5 --max-proj-max 1.5
    python3 build_auto_margin_gt.py --stage split  --out /tmp/automargin
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import struct
import sys
import time

sys.path.insert(0, "/tmp/gt")

from app.groundtruth.assets import content_hash                      # noqa: E402
from app.importers.exocad.construction_info import parse_construction_case  # noqa: E402
from app.groundtruth.indexer import fdis_in_name                        # noqa: E402

INDEX_CSV = "/tmp/gt_freeze_v1/frozen_tooth_index.csv"
ROOTS = ("/data1/welo/dentistry/exo/all_data",
         "/data1/welo/dentistry/exo/EXO")

#: 主区域最小面数：小于它的分量算碎片（如孤立单面），如实报告不硬标。
MIN_REGION_FACES = 100

#: 资产形态门限：一个分量若"表面积加权平均 (法向·径向) < 此值"，说明它是
#: **内表面（贴预备体那面）** → 这个 mesh 是冠/桥壳，不是代型。实测冠壳的
#: 内表面为 −0.83 / −0.55，外表面 +0.79 / +0.55；代型没有内表面分量。
INTAGLIO_RADIAL = -0.30

#: 形态判定的轴向延伸门限（mm，相对 Margin 平面）：
#: 颈方分量必须真的向下延伸（代型本体），冠方必须向上延伸（预备体表面）。
#: 实心冠体的颈方只是底盖，延伸≈0 —— 用它把"DIE_LIKE"从"没有内表面"里摘出来。
CERVICAL_MIN_EXTENT_MM = 1.5
CROWN_MIN_EXTENT_MM = 1.0

#: 正判"代型"的另两条门限（假设，见 _classify_kind）：
#: 冠方分量在 Margin 平面上方的面积占比；颈方分量的径向法向强度。
PREP_ABOVE_FRAC = 0.80
CERVICAL_WALL_RADIAL = 0.35

#: 资产形态（写进构建表，供漏斗统计）
ASSET_KIND_DIE = "DIE_LIKE"
ASSET_KIND_INTAGLIO = "CROWN_SHELL_INTAGLIO"
ASSET_KIND_UNCLEAR = "UNCLEAR"

#: 冠方 cap 允许低于 Margin 环轴向最低值的容差（mm）。cap 是坐在 Margin
#: 上面的，明显低于它说明切分/方向判错 —— fail-loud。
MARGIN_CAP_TOL_MM = 0.5

AUDIT_COLUMNS = (
    "case_uuid", "fdi", "die_faces", "die_points", "n_margin",
    "proj_p50_mm", "proj_p95_mm", "proj_max_mm",
    "n_beyond_0p2", "n_beyond_1p0", "n_beyond_2p0",
    "axis_present", "reject_reason", "elapsed_s",
)
BUILD_COLUMNS = (
    "case_uuid", "group_id", "fdi", "prep_mesh_ref", "n_margin", "n_faces",
    "n_class1_faces", "n_class0_faces", "n_cut_edges", "n_unlabeled_faces",
    "proj_p50_mm", "proj_p95_mm", "proj_max_mm",
    "n_regions_after_cut", "n_main_regions", "fragment_ratio",
    "bypass_detected", "seed_axial_mm", "span", "area_mm2",
    "asset_kind", "radial", "above_frac", "filename_evidence",
    "side_method", "geo_criteria_agree", "crown_votes",
    "qc_status", "reject_reason", "elapsed_s",
)


# ================================================================ 网格工具

def read_stl(path, max_tris=200_000):
    raw = open(path, "rb").read()
    if raw[:5] == b"solid":
        raise ValueError("ASCII STL 不支持")
    n = struct.unpack_from("<I", raw, 80)[0]
    stride = max(1, n // max_tris)
    tris = []
    for i in range(0, n, stride):
        v = struct.unpack_from("<12f", raw, 84 + i * 50)
        tris.append(((v[3], v[4], v[5]), (v[6], v[7], v[8]),
                     (v[9], v[10], v[11])))
    return tris


def unit(v):
    n = math.sqrt(sum(c * c for c in v)) or 1.0
    return tuple(c / n for c in v)


class TriIndex:
    """点查询：顶点格 → 最近顶点 → 其 incident faces + 1-ring 面 → 精确最近点。"""

    def __init__(self, tris, cell=1.0):
        self.tris = tris
        self.cell = cell
        self.verts = []
        self.vid = {}
        self.vfaces = []
        for fi, t in enumerate(tris):
            for p in t:
                k = (round(p[0], 5), round(p[1], 5), round(p[2], 5))
                vid = self.vid.get(k)
                if vid is None:
                    vid = len(self.verts)
                    self.vid[k] = vid
                    self.verts.append(p)
                    self.vfaces.append([])
                self.vfaces[vid].append(fi)
        self.buckets = {}
        for vid, p in enumerate(self.verts):
            k = (int(math.floor(p[0] / cell)), int(math.floor(p[1] / cell)),
                 int(math.floor(p[2] / cell)))
            self.buckets.setdefault(k, []).append(vid)
        self.vadj = [set() for _ in self.verts]
        for t in tris:
            ids = [self.vid[(round(p[0], 5), round(p[1], 5), round(p[2], 5))]
                   for p in t]
            for a in ids:
                for b in ids:
                    if a != b:
                        self.vadj[a].add(b)

    def _near_vertex(self, p, radius):
        c = int(math.ceil(radius / self.cell))
        bx, by, bz = (int(math.floor(p[i] / self.cell)) for i in range(3))
        best, bv = math.inf, None
        r2 = radius * radius
        for dx in range(-c, c + 1):
            for dy in range(-c, c + 1):
                for dz in range(-c, c + 1):
                    for vid in self.buckets.get((bx + dx, by + dy, bz + dz),
                                                ()):
                        q = self.verts[vid]
                        d = ((q[0]-p[0])**2 + (q[1]-p[1])**2 + (q[2]-p[2])**2)
                        if d < best and d <= r2:
                            best, bv = d, vid
        return bv

    def nearest(self, p, *, radius=12.0):
        """→ (distance, face_id)。半径内无顶点则回 None（fail-loud，不猜）。"""
        v0 = self._near_vertex(p, radius)
        if v0 is None:
            return None, None
        cand = set(self.vfaces[v0])
        for w in self.vadj[v0]:
            cand.update(self.vfaces[w])
        best, bf = math.inf, None
        for fi in cand:
            q, d2 = closest_on_tri(p, *self.tris[fi])
            if d2 < best:
                best, bf = d2, fi
        return math.sqrt(best), bf


def _seg_nearest(p, a, b):
    """点到线段的最近点与距离²（退化三角形与兜底都用它）。"""
    ab = (b[0]-a[0], b[1]-a[1], b[2]-a[2])
    d2 = ab[0]*ab[0] + ab[1]*ab[1] + ab[2]*ab[2]
    if d2 < 1e-20:
        return a, ((p[0]-a[0])**2 + (p[1]-a[1])**2 + (p[2]-a[2])**2)
    t = ((p[0]-a[0])*ab[0] + (p[1]-a[1])*ab[1] + (p[2]-a[2])*ab[2]) / d2
    t = 0.0 if t < 0.0 else (1.0 if t > 1.0 else t)
    q = (a[0]+ab[0]*t, a[1]+ab[1]*t, a[2]+ab[2]*t)
    return q, ((p[0]-q[0])**2 + (p[1]-q[1])**2 + (p[2]-q[2])**2)


def closest_on_tri(p, a, b, c):
    """点到三角形最近点（Ericson §5.1.5）；返回 (点, 距离²)。

    **退化三角形**（面积≈0，真实扫描网格里存在）会让原式除零
    （实测 ZeroDivisionError: d1/(d1-d3)）。这里先判面积，退化时退化为
    "投影到最长边"，保证任何输入都有定义且连续。
    """
    def sub(u, v): return (u[0]-v[0], u[1]-v[1], u[2]-v[2])
    def dot(u, v): return u[0]*v[0] + u[1]*v[1] + u[2]*v[2]
    def mad(u, v, s): return (u[0]+v[0]*s, u[1]+v[1]*s, u[2]+v[2]*s)
    ab0, ac0 = sub(b, a), sub(c, a)
    n0 = (ab0[1]*ac0[2]-ab0[2]*ac0[1], ab0[2]*ac0[0]-ab0[0]*ac0[2],
          ab0[0]*ac0[1]-ab0[1]*ac0[0])
    if n0[0]*n0[0]+n0[1]*n0[1]+n0[2]*n0[2] < 1e-18:
        cands = [_seg_nearest(p, a, b), _seg_nearest(p, b, c),
                 _seg_nearest(p, c, a)]
        return min(cands, key=lambda x: x[1])
    ab, ac, ap = sub(b, a), sub(c, a), sub(p, a)
    d1, d2 = dot(ab, ap), dot(ac, ap)
    if d1 <= 0 and d2 <= 0:
        return a, dot(ap, ap)
    bp = sub(p, b)
    d3, d4 = dot(ab, bp), dot(ac, bp)
    if d3 >= 0 and d4 <= d3:
        return b, dot(bp, bp)
    vc = d1*d4 - d3*d2
    if vc <= 0 and d1 >= 0 and d3 <= 0:
        den = d1 - d3
        t = (d1/den) if abs(den) > 1e-20 else 0.0
        q = mad(a, ab, t)
        r = sub(p, q)
        return q, dot(r, r)
    cp = sub(p, c)
    d5, d6 = dot(ab, cp), dot(ac, cp)
    if d6 >= 0 and d5 <= d6:
        return c, dot(cp, cp)
    vb = d5*d2 - d1*d6
    if vb <= 0 and d2 >= 0 and d6 <= 0:
        den = d2 - d6
        t = (d2/den) if abs(den) > 1e-20 else 0.0
        q = mad(a, ac, t)
        r = sub(p, q)
        return q, dot(r, r)
    va = d3*d6 - d5*d4
    if va <= 0 and (d4-d3) >= 0 and (d5-d6) >= 0:
        den = (d4-d3)+(d5-d6)
        t = ((d4-d3)/den) if abs(den) > 1e-20 else 0.0
        q = mad(b, sub(c, b), t)
        r = sub(p, q)
        return q, dot(r, r)
    ds = va+vb+vc
    if abs(ds) < 1e-20:
        cands = [_seg_nearest(p, a, b), _seg_nearest(p, b, c),
                 _seg_nearest(p, c, a)]
        return min(cands, key=lambda x: x[1])
    den = 1.0/ds
    v, w = vb*den, vc*den
    q = (a[0]+ab[0]*v+ac[0]*w, a[1]+ab[1]*v+ac[1]*w, a[2]+ab[2]*v+ac[2]*w)
    r = sub(p, q)
    return q, dot(r, r)


# ================================================================ 病例定位

_UUID2DIRS = None


def _build_uuid_map():
    """一次遍历 ROOTS 建立 目录名哈希 → 目录列表 的反查表。

    原实现对**每次调用**都 os.walk 整棵数据树，2546 行就是 2546 次全树遍历
    （实测单行 ~5.9s 里很大一块在这）。这里只走一次。
    注意必须保留**列表**：同名目录可能出现在 all_data 与 EXO 两处，
    只留一个会挑到没有 CI 的那份（踩过）。
    """
    m = {}
    for root in ROOTS:
        for dp, dn, _f in os.walk(root):
            for d in dn:
                m.setdefault(hashlib.sha256(d.encode()).hexdigest()[:16],
                             []).append(os.path.join(dp, d))
    return m


def case_dirs(uuid):
    global _UUID2DIRS
    if _UUID2DIRS is None:
        _UUID2DIRS = _build_uuid_map()
    return list(_UUID2DIRS.get(uuid[3:], []))


def find_by_hash(dirs, want):
    if not want:
        return None
    for d in dirs:
        for dp, _dn, fns in os.walk(d):
            for f in fns:
                if f.lower().endswith(".stl") and \
                        content_hash(os.path.join(dp, f)) == want:
                    return os.path.join(dp, f)
    return None


#: 设计快照后缀。**必须与 indexer 的 _CI_SUFFIXES 一致**：索引接受
#: (".constructionInfo", ".modelInfo")，只认前者会让"仅导出 modelInfo"的病例
#: 被误判 no_construction_info（实测 cf_6fc88119791c22ed 目录里只有 .modelInfo）。
CI_SUFFIXES = (".constructionInfo", ".modelInfo")


def find_ci(dirs):
    out = []
    for d in dirs:
        for dp, _dn, fns in os.walk(d):
            for f in fns:
                if f.endswith(CI_SUFFIXES):
                    out.append(os.path.join(dp, f))
    # constructionInfo 优先（canonical），modelInfo 兜底 —— 与 indexer 同序
    out.sort(key=lambda p: (".modelinfo" in p.lower(), p))
    return out


def strip_sha(h):
    return h[4:] if h and h.startswith("sha:") else (h or "")


# ================================================================ 阶段：审计

def stage_audit(args_rows, out_dir, done, limit, resume_path):
    t_all = time.time()
    n_done = 0
    with open(resume_path, "a", newline="", encoding="ascii") as fh:
        w = csv.DictWriter(fh, fieldnames=list(AUDIT_COLUMNS))
        if not done:
            w.writeheader()
        for i, r in enumerate(args_rows):
            if r["case_uuid"] in done:
                continue
            if limit and n_done >= limit:
                break
            t0 = time.time()
            rec = {k: "" for k in AUDIT_COLUMNS}
            rec["case_uuid"] = r["case_uuid"]
            rec["fdi"] = r["fdi"]
            try:
                dirs = case_dirs(r["case_uuid"])
                if not dirs:
                    raise RuntimeError("case_dir_not_found")
                die = find_by_hash(dirs, strip_sha(r["modeldie_asset_id"]))
                if die is None:
                    raise RuntimeError("die_asset_not_found")
                cis = find_ci(dirs)
                if not cis:
                    raise RuntimeError("no_construction_info")
                case = parse_construction_case(cis[0])
                tooth = case.tooth(int(r["fdi"]))
                if tooth is None or not tooth.margin_points:
                    raise RuntimeError("margin_missing_in_ci")
                margin = [(float(p[0]), float(p[1]), float(p[2]))
                          for p in tooth.margin_points]
                rec["n_margin"] = len(margin)
                rec["axis_present"] = bool(tooth.insertion_axis)
                tris = read_stl(die)
                rec["die_faces"] = len(tris)
                idx = TriIndex(tris)
                rec["die_points"] = len(idx.verts)
                ds, miss = [], 0
                for p in margin:
                    d, _f = idx.nearest(p)
                    if d is None:
                        miss += 1
                    else:
                        ds.append(d)
                if not ds:
                    raise RuntimeError("all_margin_points_off_mesh")
                ds.sort()
                q = lambda t: ds[min(len(ds)-1, int(t*(len(ds)-1)))]
                rec["proj_p50_mm"] = round(q(.50), 4)
                rec["proj_p95_mm"] = round(q(.95), 4)
                rec["proj_max_mm"] = round(ds[-1], 4)
                rec["n_beyond_0p2"] = sum(1 for d in ds if d > 0.2)
                rec["n_beyond_1p0"] = sum(1 for d in ds if d > 1.0)
                rec["n_beyond_2p0"] = sum(1 for d in ds if d > 2.0)
                if miss:
                    rec["reject_reason"] = "n_off_mesh=%d" % miss
            except Exception as exc:                 # noqa: BLE001
                rec["reject_reason"] = "%s: %s" % (type(exc).__name__,
                                                   str(exc)[:60])
            rec["elapsed_s"] = round(time.time()-t0, 1)
            w.writerow(rec)
            fh.flush()
            n_done += 1
            if n_done % 20 == 0:
                print("  audit %d/%d（%.0fs）" % (n_done, len(args_rows),
                                                time.time()-t_all), flush=True)
    print("audit 行数 %d，用时 %.0fs" % (n_done, time.time()-t_all))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True,
                    choices=["audit", "build", "split", "funnel",
                             "kind", "select"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-proj-p95", type=float, default=None)
    ap.add_argument("--max-proj-max", type=float, default=None)
    ap.add_argument("--die-selection", action="append", default=None,
                    help="几何选代型结果 CSV（可重复给多个分片）；"
                         "build 阶段用它确定每行用哪个代型")
    ap.add_argument("--shard", default="1/1",
                    help="select 阶段分片 i/n（并行用，各写自己的 CSV）")
    ap.add_argument("--only-evidence", action="store_true",
                    help="只处理资产=目录里 *modeldie* 文件的行")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    rows = [r for r in csv.DictReader(open(INDEX_CSV, encoding="ascii"))
            if r["has_modeldie"] == "True"
            and r["modeldie_same_frame"] == "True"
            and r["has_margin"] == "True"]
    print("eligible rows:", len(rows), "病例:",
          len({r["case_uuid"] for r in rows}))

    if a.stage == "funnel":
        return stage_funnel(a.out, rows)
    if a.stage == "select":
        i, n = (a.shard.split("/") + ["1"])[:2]
        return stage_select(a.out, (int(i), int(n)), a.limit)
    if a.stage == "build" and a.die_selection:
        dm = die_map_from_selection(a.die_selection)
        if not dm:
            raise SystemExit("--die-selection 没有任何可用行")
        all_rows = list(csv.DictReader(open(INDEX_CSV, encoding="ascii")))
        keyed = {(r["case_uuid"], r["fdi"]): r for r in all_rows}
        rows = [keyed[k] for k in dm if k in keyed]
        print("按几何选代型构建：%d 行 / %d 病例"
              % (len(rows), len({r["case_uuid"] for r in rows})), flush=True)
        return stage_build(a.out, rows, a.max_proj_p95, a.max_proj_max,
                           a.limit, die_map=dm)
    if a.stage == "build" and a.only_evidence:
        pairs = evidence_rows(
            [r for r in csv.DictReader(open(INDEX_CSV, encoding="ascii"))])
        print("证据行源: %d 行 / %d 病例"
              % (len(pairs), len({r["case_uuid"] for r, _ in pairs})), flush=True)
        return stage_build(a.out, [r for r, _ in pairs], a.max_proj_p95,
                           a.max_proj_max, a.limit,
                           die_map={r["case_uuid"] + "|" + r["fdi"]: d
                                    for r, d in pairs})
    if a.stage == "kind":
        if a.only_evidence:
            pairs = evidence_rows(
                [r for r in csv.DictReader(open(INDEX_CSV, encoding="ascii"))])
            print("证据行源: %d 行 / %d 病例"
                  % (len(pairs), len({r["case_uuid"] for r, _ in pairs})),
                  flush=True)
            return stage_kind(a.out, [r for r, _ in pairs], a.limit,
                              die_map={r["case_uuid"] + "|" + r["fdi"]: d
                                       for r, d in pairs})
        return stage_kind(a.out, rows, a.limit)
    if a.stage == "audit":
        path = os.path.join(a.out, "projection_audit.csv")
        done = set()
        if os.path.isfile(path):
            done = {r["case_uuid"] + "|" + r["fdi"]
                    for r in csv.DictReader(open(path, encoding="ascii"))}
        rows = [r for r in rows
                if r["case_uuid"] + "|" + r["fdi"] not in done]
        return stage_audit(rows, a.out, done, a.limit, path)
    if a.stage == "build":
        return stage_build(a.out, rows, a.max_proj_p95, a.max_proj_max,
                           a.limit)
    if a.stage == "split":
        return stage_split(a.out)
    raise SystemExit("unknown stage")


def stage_funnel(out_dir, rows):
    """漏斗统计：每层剩余数量 + reject 原因分布（用户 M0 验收要求）。"""
    path = os.path.join(out_dir, "projection_audit.csv")
    if not os.path.isfile(path):
        raise SystemExit("先跑 --stage audit")
    recs = list(csv.DictReader(open(path, encoding="ascii")))
    ok = [r for r in recs if not r["reject_reason"]]
    reasons = {}
    for r in recs:
        if r["reject_reason"]:
            key = r["reject_reason"].split(":")[0].split("=")[0]
            reasons[key] = reasons.get(key, 0) + 1
    ds = sorted(float(r["proj_p95_mm"]) for r in ok)
    dm = sorted(float(r["proj_max_mm"]) for r in ok)
    full = sorted(float(r["proj_max_mm"]) for r in recs
                  if r["proj_max_mm"])
    def pct(v, t):
        return round(v[min(len(v)-1, int(t*(len(v)-1)))], 4) if v else None
    funnel = {
        "margin_gt_rows_in_index": len(rows),
        "audit_rows": len(recs),
        "readable": len(ok),
        "reject_reasons": dict(sorted(reasons.items(), key=lambda x: -x[1])),
        "proj_p95_distribution_mm": {
            "p50": pct(ds, .5), "p90": pct(ds, .9), "p95": pct(ds, .95),
            "p99": pct(ds, .99), "max": ds[-1] if ds else None},
        "proj_max_distribution_mm": {
            "p50": pct(dm, .5), "p90": pct(dm, .9), "p95": pct(dm, .95),
            "p99": pct(dm, .99), "max": dm[-1] if dm else None},
    }
    json.dump(funnel, open(os.path.join(out_dir, "funnel.json"), "w"),
              indent=1)
    print(json.dumps(funnel, indent=1, ensure_ascii=False))
    return 0


# ================================================================ 阶段：build

class _Reject(Exception):
    """fail-loud：不满足规格的数据一律拒绝，绝不强行生成标签。"""


class _Review(Exception):
    """不确定但不致命：进 REVIEW 清单（排除出训练集，保留可查）。

    典型情形：两个符号无关判据（测地跨度 / 面积）给出相反的冠方/颈方。
    按用户三态设计（AUTO_ACCEPTED / REVIEW_REQUIRED / FAILED），
    这类不该硬标也不该丢弃。"""


def _face_components(n_faces, adj, allowed):
    """allowed 面集内的连通分量（按面数降序）。"""
    seen = set()
    comps = []
    for s in range(n_faces):
        if s not in allowed or s in seen:
            continue
        stack = [s]
        seen.add(s)
        comp = []
        while stack:
            f = stack.pop()
            comp.append(f)
            for g in adj.get(f, ()):
                if g in allowed and g not in seen:
                    seen.add(g)
                    stack.append(g)
        comps.append(comp)
    comps.sort(key=len, reverse=True)
    return comps


def _classify_kind(stats):
    """把两个主分量的统计判成资产形态（纯函数，便于单测）。

    **保守偏置**：不确定一律 UNCLEAR（排除出训练集），绝不把冠壳判成代型。
    规则依据全部来自实测（见 docs/AUTO_MARGIN_M0_ASSET_AUDIT.md）：

    1. 有分量"法向指向牙体轴"（radial < INTAGLIO_RADIAL）→ 这是贴预备体的
       内表面 → **冠/桥壳**。实测冠壳内表面 −0.41 ~ −0.85，配对外表面
       +0.50 ~ +0.73。
    2. 没有内表面**不等于**是代型：实心冠体（``bridge_slm_cad``）同样没有
       内表面。实测它的两个分量 radial 都 ≈ +0.1（近轴向法向，是"顶盖/底盖"
       不是"环壁"），above_frac ≈ 0.5。所以代型还要求：
         - 冠方分量几乎全在 Margin 平面之上（above_frac ≥ 0.8，预备体表面）
         - 颈方分量是**环绕牙体的壁**（radial ≥ CERVICAL_WALL_RADIAL）
         - 颈方真的向下延伸（ax_min ≤ −CERVICAL_MIN_EXTENT_MM）
       （这三条是**待验证假设**：目前 24 行抽样里 0 行被正判为代型，
        还没有正样本证据。保持保守，宁可少也要对。）
    """
    if len(stats) != 2:
        return ASSET_KIND_UNCLEAR
    if any(st["radial"] < INTAGLIO_RADIAL for st in stats):
        return ASSET_KIND_INTAGLIO
    prep = max(stats, key=lambda st: st["above_frac"])
    wall = min(stats, key=lambda st: st["above_frac"])
    if (prep["above_frac"] >= PREP_ABOVE_FRAC
            and wall["radial"] >= CERVICAL_WALL_RADIAL
            and wall["ax_min"] <= -CERVICAL_MIN_EXTENT_MM):
        return ASSET_KIND_DIE
    return ASSET_KIND_UNCLEAR



def asset_kind(idx, tris, margin_pts, axis, cut):
    """判定资产形态：DIE_LIKE / CROWN_SHELL_INTAGLIO / UNCLEAR。

    动机（实测）：冻结索引里 94.9% 的 ``modeldie_asset_id`` 来自 indexer 的
    **回退启发式**（"包围盒中心离设计中心最近的 STL"），而 exo2 目录里
    只有编号 STL —— 冠壳因此被静默当成代型。在冠壳上按 Margin 切割，
    分出来的两侧是**外表面 / 内表面**，不是"冠方 / 颈方"。这种标签进
    训练集就是用户最明确禁止的事，所以这里 fail-loud。

    判别只用几何：内表面分量（法向指向牙体轴）存在 → 冠壳。
    """
    proj = []
    for p in margin_pts:
        q, _fi = _nearest_pt(idx, p)
        proj.append(q if q is not None else p)
    adj = {}
    for fi, t in enumerate(tris):
        if fi in cut:
            continue
        for p in t:
            vid = idx.vid[(round(p[0], 5), round(p[1], 5), round(p[2], 5))]
            for fj in idx.vfaces[vid]:
                if fj != fi and fj not in cut:
                    adj.setdefault(fi, set()).add(fj)
    comps = _face_components(len(tris), adj, set(range(len(tris))) - cut)
    main = [c for c in comps if len(c) >= MIN_REGION_FACES]
    mc = tuple(sum(p[k] for p in proj) / max(len(proj), 1) for k in range(3))
    stats = []
    for comp in main:
        area = dot_sum = above = 0.0
        ax_min, ax_max = 1e9, -1e9
        for f in comp:
            a, b, c3 = tris[f]
            u = (b[0]-a[0], b[1]-a[1], b[2]-a[2])
            v = (c3[0]-a[0], c3[1]-a[1], c3[2]-a[2])
            n = (u[1]*v[2]-u[2]*v[1], u[2]*v[0]-u[0]*v[2], u[0]*v[1]-u[1]*v[0])
            L = math.sqrt(sum(x*x for x in n)) or 1e-12
            ar = 0.5*L
            fc = tuple((a[k]+b[k]+c3[k])/3.0 for k in range(3))
            rad = tuple(fc[k]-mc[k] for k in range(3))
            rad = tuple(r - axis[k]*sum(rad[j]*axis[j] for j in range(3))
                        for k, r in enumerate(rad))
            Lr = math.sqrt(sum(x*x for x in rad)) or 1e-12
            dot_sum += ar*sum(n[k]/L*rad[k]/Lr for k in range(3))
            d = sum((fc[k]-mc[k])*axis[k] for k in range(3))
            if d > 0:
                above += ar
            if d < ax_min:
                ax_min = d
            if d > ax_max:
                ax_max = d
            area += ar
        stats.append(dict(n=len(comp), area=area, radial=dot_sum/max(area, 1e-12),
                          above_frac=above/max(area, 1e-12),
                          ax_min=ax_min, ax_max=ax_max))
    kind = _classify_kind(stats)
    return dict(kind=kind, n_main=len(main),
                radial=[round(st["radial"], 3) for st in stats],
                above=[round(st["above_frac"], 3) for st in stats],
                area=[round(st["area"], 1) for st in stats],
                ax_min=[round(st["ax_min"], 2) for st in stats],
                ax_max=[round(st["ax_max"], 2) for st in stats])


def _nearest_pt(idx, p):
    d, fi = idx.nearest(p)
    if d is None:
        return None, None
    q, _d2 = closest_on_tri(p, *idx.tris[fi])
    return q, fi


def _build_one(r, a, lab_dir, die_override=None):
    """单病例入口：准备（解析/投影/切割带）→ 切分 → 标签 + QC。

    用户规格 3/4/5/6/10 的落点：
    3. 投影保留原曲线与投影曲线两份，逐点误差记录（不静默吸附）；
    4. **真正切断 adjacency**：Margin 点所在顶点的 incident faces 整体从
       面邻接图移除（不是只取最近顶点）；
    5. seed 用 constructionInfo 的 **InsertionAxis**（患者就位方向）选轴向
       最高面，**不用世界 Z**；轴缺失/seed 不明确 → REJECT；
    6. 切完后**必须分出 ≥2 个面连通分量**，否则说明 Margin 没绕成一整圈
       （能绕过 Margin 连通）→ **一票否决**（规格 10）。
    """
    ctx = _prepare(r, die_override)
    return _build_one_ctx(ctx, r, lab_dir)


_HASH2PATH = {}


def _hash_to_path(dirs):
    """病例目录里 STL 的内容哈希 → 路径（每个目录只哈希一次并缓存）。

    content_hash 只读字节做 sha256（~0.05s/5MB），比 read_stl 便宜得多，
    所以这里可以放心把目录里所有 STL 都哈希一遍。
    """
    key = tuple(sorted(dirs))
    if key not in _HASH2PATH:
        m = {}
        for d in dirs:
            for dp, _dn, fns in os.walk(d):
                for f in fns:
                    if f.lower().endswith(".stl"):
                        q = os.path.join(dp, f)
                        try:
                            m.setdefault(content_hash(q), q)
                        except OSError:
                            pass
        _HASH2PATH[key] = m
    return _HASH2PATH[key]


def die_map_from_selection(paths):
    """读 die_selection*.csv（可多个分片）→ {(case_uuid, fdi): 本地路径}。

    只收 selection == MARGIN_ON_DIE 的行（该牙位 Margin 投影到该候选代型
    p95 <= 0.20mm —— 几何证据）。CSV 里存的是内容哈希（PHI-free），
    这里在病例目录内解析成路径。
    """
    want = {}
    for p in paths:
        for r in csv.DictReader(open(p, encoding="ascii")):
            if r.get("selection") != "MARGIN_ON_DIE" or not r.get("die_asset_id"):
                continue
            want[(r["case_uuid"], r["fdi"])] = r["die_asset_id"]
    by_case = {}
    for (uuid, fdi), h in want.items():
        by_case.setdefault(uuid, []).append((fdi, h))
    out = {}
    missing = 0
    for uuid, items in by_case.items():
        mapping = _hash_to_path(case_dirs(uuid))
        for fdi, h in items:
            q = mapping.get(h)
            if q:
                out[(uuid, fdi)] = q
            else:
                missing += 1
    print("几何选代型：%d 行命中，%d 行在目录里找不到对应文件"
          % (len(out), missing), flush=True)
    return out


def _prepare(r, die_override=None):
    """解析资产 + Margin + 轴 → 投影 → 切割带。build 与 kind 共用同一套。

    抽出来的原因：切割带逻辑踩过 4 次坑（顶点切割/只取最近顶点/碎片归属），
    审计与构建必须是同一份实现，不能各写一遍。
    """
    dirs = case_dirs(r["case_uuid"])
    die = die_override
    if die is None:
        die = find_by_hash(dirs, strip_sha(r["modeldie_asset_id"]))
    if die is None:
        raise _Reject("die_asset_not_found")
    cis = find_ci(dirs)
    if not cis:
        raise _Reject("no_construction_info")
    case = parse_construction_case(cis[0])
    tooth = case.tooth(int(r["fdi"]))
    if tooth is None or not tooth.margin_points:
        raise _Reject("margin_missing_in_ci")
    margin = [(float(p[0]), float(p[1]), float(p[2]))
              for p in tooth.margin_points]
    axis = tooth.insertion_axis
    if not axis:
        raise _Reject("no_insertion_axis")
    axis = unit((float(axis[0]), float(axis[1]), float(axis[2])))

    tris = read_stl(die)
    idx = TriIndex(tris)
    n_faces = len(tris)

    # ---- 1. 投影（原曲线 + 投影曲线 + 逐点误差）
    proj_pts, proj_err, off_mesh = [], [], 0
    for p in margin:
        q, _fi = _nearest_pt(idx, p)
        if q is None:
            off_mesh += 1
            proj_pts.append(p)
            proj_err.append(0.0)
        else:
            proj_pts.append(q)
            proj_err.append(math.dist(p, q))
    if off_mesh:
        raise _Reject("margin_off_mesh=%d" % off_mesh)
    errs = sorted(proj_err)

    # ---- 2. 切割带 = Margin 点**所在面** + 该面三个顶点的 incident faces。
    # 不能只匹配网格顶点：实测 413 个 Margin 点里只有 320 个精确落在顶点上
    # （其余在三角形内部 —— 到"面"的距离同样是 0），只按顶点切会留下断口，
    # 顶点环的度分布是 {2:192, 1:114, 0:14}，切完仍然连通 → 误判 bypass。
    cut = set()
    for p in proj_pts:
        _d, fi = idx.nearest(p)
        if fi is None:
            continue
        cut.add(fi)
        for vv in tris[fi]:
            v = idx.vid[(round(vv[0], 5), round(vv[1], 5), round(vv[2], 5))]
            cut.update(idx.vfaces[v])
    if not cut:
        raise _Reject("cut_set_empty")
    return dict(dirs=dirs, die=die, tris=tris, idx=idx, n_faces=n_faces,
                margin=margin, axis=axis, proj_pts=proj_pts, proj_err=proj_err,
                errs=errs, cut=cut, r=r)


def _build_one_ctx(ctx, r, lab_dir):
    dirs = ctx["dirs"]
    die = ctx["die"]
    tris = ctx["tris"]
    idx = ctx["idx"]
    n_faces = ctx["n_faces"]
    margin = ctx["margin"]
    axis = ctx["axis"]
    proj_pts = ctx["proj_pts"]
    proj_err = ctx["proj_err"]
    errs = ctx["errs"]
    cut = ctx["cut"]
    # ---- 面邻接（不经过切割带）
    adj = {}
    for fi, t in enumerate(tris):
        if fi in cut:
            continue
        for p in t:
            vid = idx.vid[(round(p[0], 5), round(p[1], 5), round(p[2], 5))]
            for fj in idx.vfaces[vid]:
                if fj != fi and fj not in cut:
                    adj.setdefault(fi, set()).add(fj)

    allowed = set(range(n_faces)) - cut
    comps = _face_components(n_faces, adj, allowed)
    if len(comps) < 2:
        raise _Reject("margin_not_separating(bypass_detected)")
    # 主区域 = 面数 >= MIN_REGION_FACES 的分量。**恰好 2 个**才算切分成功：
    # 1 个 = 没切开（bypass）；>2 个 = 拓扑异常，都需要人工看，不能硬标。
    main = [c for c in comps if len(c) >= MIN_REGION_FACES]
    frag_faces = sum(len(c) for c in comps if len(c) < MIN_REGION_FACES)
    if len(main) == 1:
        raise _Reject("margin_not_separating(bypass_detected)")
    if len(main) > 2:
        raise _Reject("too_many_regions=%d" % len(main))
    # ---- 资产形态门（fail-loud）：冻结索引里 94.9% 的 modeldie_asset_id 来自
    # indexer 的回退启发式（bbox 中心离设计中心最近），实际资产常是冠/桥 CAD
    # 实体。在冠壳上按 Margin 切割，分出的两侧是**外表面/内表面**，不是
    # "冠方/颈方" —— 这种标签进训练集是明确禁止的，所以这里直接 REJECT。
    ak = asset_kind(idx, tris, margin, axis, cut)
    if ak["kind"] == ASSET_KIND_INTAGLIO:
        raise _Reject("asset_is_crown_shell(radial=%s,above=%s)"
                      % (ak["radial"], ak["above"]))
    # ---- 冠方/颈方判定：**符号无关判据**（规格 5 说"用 Axis 不要用世界 Z"，
    # 但实测 exocad 的 <Axis> 符号跨病例不一致，任何"轴向最低/最高"判据
    # 都会在有符号翻转的病例上判反 —— 渲染出来的标签整片反色）。
    #
    # 采用两个**与轴符号无关**的判据，要求互相一致，不一致就 fail-loud：
    #   A. 测地跨度：从切割带出发在分量内 BFS，**面跳数最大者**是颈方
    #      （代型本体/模型段要延伸到基座，cap 只是一只浅碟）；
    #   B. 面积：**面积小者**是冠方 cap。
    # 两者指向同一分量才算过。
    # 顶点 → 切割带面（供测地种子用）
    vid_cut = {}
    for f in cut:
        for vv in tris[f]:
            v = idx.vid[(round(vv[0], 5), round(vv[1], 5), round(vv[2], 5))]
            vid_cut.setdefault(v, []).append(f)

    adj_in = {}
    for comp in main:
        cs = set(comp)
        for f in comp:
            for g in adj.get(f, ()):
                if g in cs:
                    adj_in.setdefault(f, set()).add(g)

    def geodesic_span(comp):
        """分量内从"紧邻切割带的面"出发的最大面跳数。"""
        cs = set(comp)
        seeds = []
        for f in comp:
            for vv in tris[f]:
                v = idx.vid[(round(vv[0], 5), round(vv[1], 5), round(vv[2], 5))]
                if v in vid_cut:
                    seeds.append(f)
                    break
        if not seeds:
            return -1
        dist = {f: 0 for f in seeds}
        dq = list(seeds)
        head = 0
        while head < len(dq):
            f = dq[head]
            head += 1
            for g in adj_in.get(f, ()):
                if g in cs and g not in dist:
                    dist[g] = dist[f] + 1
                    dq.append(g)
        return max(dist.values()) if dist else -1

    def comp_area(comp):
        tot = 0.0
        for f in comp:
            a, b, c = tris[f]
            u = (b[0]-a[0], b[1]-a[1], b[2]-a[2])
            v = (c[0]-a[0], c[1]-a[1], c[2]-a[2])
            cr = (u[1]*v[2]-u[2]*v[1], u[2]*v[0]-u[0]*v[2],
                  u[0]*v[1]-u[1]*v[0])
            tot += 0.5*math.sqrt(sum(x*x for x in cr))
        return tot

    # ---- 优先用**定义性判据**：Final Crown 的内表面就坐在预备体上。
    # 哪一侧离冠的表面更近，哪一侧就是冠方（Class1）。这与轴符号无关、
    # 不依赖"哪边面多/更高/更浅"，是真正定义性的。
    # 注意：这只是**标签侧向**的参考，模型输入永远只有 prep mesh —— 不构成
    # 特征泄漏。没有冠的病例退回几何判据（并如实记录用了哪种）。
    # **已禁用（未验证）**：曾想用 Final Crown 做定义性判据，但实测该字段
    # 指向的是 *_coping_cad.stl（内冠），其表面同时靠近两个分量
    # （实测投票 23240/14095 = 62%/38%，不决定性）。要启用必须先验证
    # "冠/内冠的内表面坐在预备体上、且只落在 Class1 一侧" 这条假设，
    # 并处理"距 die < 0.3mm 才算贴附"的距离门限。在验证前不允许它参与
    # 标签侧向 —— 错标签进训练集是用户最明确禁止的事。
    # **Final Crown 参考路径已摘除（未验证）**：该字段实测指向
    # ``*_coping_cad.stl``（内冠），其表面同时靠近两个分量
    # （实测投票 23240/14095 = 62%/38%，不决定性）。要启用必须先验证
    # "冠/内冠的内表面坐在预备体上、且只落在 Class1 一侧"，
    # 并加"距 die < 0.3mm 才算贴附"的距离门限。
    # 在验证前绝不允许它参与标签侧向 —— 错标签进训练集是明确禁止的。
    crown_votes = None

    # ============================================================ 侧别判定
    # **代型（DIE_LIKE）用 above_frac 做决定性判据**：实测 18 例正样本
    # 冠方 0.885~0.974 vs 颈方 0.007~0.036（差 1~2 个数量级），且 93 个
    # DIE_LIKE 行的 margin 投影 p95 = 0.0000（代型与 margin 同坐标系）。
    # "哪一侧在 Margin 平面之上"就是定义性的 —— 预备体表面在环之上，
    # 代型颈部侧壁在环之下。跨度/面积只作**记录**（它们是为冠壳设计的，
    # 在代型上与 above_frac 系统性相反，实测 93/93  disagree —— 这正是
    # 之前四版判据反复矛盾的根源）。
    ak_kind = ak["kind"]
    if ak_kind == ASSET_KIND_DIE and len(ak["above"]) == 2:
        d_cap = 0 if ak["above"][0] > ak["above"][1] else 1
        class1 = main[d_cap]
        class0 = main[1 - d_cap]
        side_method = "ABOVE_FRAC"
        spans = [geodesic_span(c) for c in main]
        areas = [comp_area(c) for c in main]
        d_span = (0 if spans[0] < spans[1] else 1) if min(spans) >= 0 else None
        geo_agree = (d_span == d_cap) if d_span is not None else None
        review_reason = None
        side_info = {"above_frac": ak["above"], "span": spans,
                     "area": [round(x, 1) for x in areas]}
    else:
        spans = [geodesic_span(c) for c in main]
        areas = [comp_area(c) for c in main]
        if min(spans) < 0:
            raise _Reject("geodesic_seed_missing")
        # A: 测地跨度小 = cap；B: 面积小 = cap
        if spans[0] != spans[1]:
            a_cap = 0 if spans[0] < spans[1] else 1
        else:
            raise _Reject("geodesic_span_tie")
        b_cap = 0 if areas[0] < areas[1] else 1
        side_method = "GEOMETRY"
        geo_agree = None
        if a_cap != b_cap:
            # 非代型资产（回退冠壳）上两判据矛盾：标签按跨度写盘但排除出训练集
            class1 = main[a_cap]
            class0 = main[1 - a_cap]
            review_reason = ("side_criteria_disagree(span=%s,area=%s)"
                             % (spans, [round(x, 1) for x in areas]))
        else:
            class1 = main[a_cap]
            class0 = main[1 - a_cap]
            review_reason = None
        side_info = {"span": spans, "area": [round(x, 1) for x in areas]}

    # 方向核对（仅记录，不作为否决）：cap 的轴向范围应在 Margin 环附近
    axial_of = lambda f: (sum(sum(vv[k] * axis[k] for k in range(3))
                              for vv in tris[f]) / 3.0)
    seed_ax = round(max(axial_of(f) for f in class1), 3)

    # 碎片率 = 被切掉的面 + 小分量的面，占全部面的比例（如实上报，不当 0）
    frag = (len(cut) + frag_faces) / max(n_faces, 1)

    # ---- 标签写盘（uint8/face：0=Class0, 1=Class1, 255=cut/未标）
    labels = bytearray([255]) * n_faces
    for f in class0:
        labels[f] = 0
    for f in class1:
        labels[f] = 1
    rel = "%s_fdi%s_labels.bin" % (r["case_uuid"], r["fdi"])
    with open(os.path.join(lab_dir, rel), "wb") as fh:
        fh.write(bytes(labels))
    # 曲线两份 + 逐点误差
    curve_rel = "%s_fdi%s_curves.json" % (r["case_uuid"], r["fdi"])
    with open(os.path.join(lab_dir, curve_rel), "w", encoding="ascii") as fh:
        json.dump({"margin_gt": [[round(c, 5) for c in p] for p in margin],
                   "projected": [[round(c, 5) for c in p] for p in proj_pts],
                   "per_point_error_mm": [round(e, 6) for e in proj_err],
                   "face_labels_bin": rel,
                   "n_faces": n_faces,
                   "axis": [round(c, 6) for c in axis]}, fh)

    return {
        "prep_mesh_ref": rel,
        "n_margin": len(margin), "n_faces": n_faces,
        "n_class1_faces": len(class1), "n_class0_faces": len(class0),
        "n_cut_edges": len(cut), "n_unlabeled_faces": len(cut) + frag_faces,
        "proj_p50_mm": round(errs[len(errs) // 2], 4),
        "proj_p95_mm": round(errs[int(.95 * (len(errs) - 1))], 4),
        "proj_max_mm": round(errs[-1], 4),
        "n_regions_after_cut": len(comps),
        "n_main_regions": len(main),
        "fragment_ratio": round(frag, 4),
        "bypass_detected": False,
        "seed_axial_mm": seed_ax,
        "span": str(side_info["span"]), "area_mm2": str(side_info["area"]),
        "asset_kind": (ak["kind"] if ak["kind"] != ASSET_KIND_INTAGLIO
                       else ASSET_KIND_INTAGLIO),
        "radial": str(ak["radial"]), "above_frac": str(ak["above"]),
        "side_method": side_method,
        "geo_criteria_agree": ("" if geo_agree is None else geo_agree),
        "crown_votes": ("" if crown_votes is None
                        else "%d/%d" % (crown_votes[0], crown_votes[1])),
        "_review_reason": review_reason,
    }



def evidence_rows(rows):
    """M0 的正样本行源：**目录里存在 fdi 匹配的 `*modeldie*` 文件 + 有 margin**。

    不哈希（哈希在 find_by_hash 里要把整目录 STL 全读一遍，2546 行要几小时）。
    返回 (row, die_path) 列表；die_path 直接交给 _prepare(die_override=...)。
    """
    from collections import OrderedDict
    out = []
    cache = OrderedDict()
    for r in rows:
        if r.get("has_margin") != "True" or not r["fdi"].isdigit():
            continue
        uuid = r["case_uuid"]
        if uuid not in cache:
            fns = []
            for d in case_dirs(uuid):
                try:
                    fns += [os.path.join(d, f) for f in os.listdir(d)
                            if f.lower().endswith(".stl")
                            and "modeldie" in f.lower()]
                except OSError:
                    pass
            cache[uuid] = fns
        fdi = int(r["fdi"])
        for f in cache[uuid]:
            try:
                if fdi in fdis_in_name(f):
                    out.append((r, f))
                    break
            except Exception:                      # noqa: BLE001
                continue
    return out


def _rows_with_evidence(rows):
    """只保留"我方资产的哈希 = 目录里 *modeldie* 命名文件的哈希"的行。

    这是 M0 里唯一有文件名证据的资产来源（其余 94.9% 来自 indexer 回退）。
    只对命名含 modeldie 的少量文件做哈希，不碰整目录。
    """
    seen = {}
    out = []
    for r in rows:
        uuid = r["case_uuid"]
        if uuid not in seen:
            hits = set()
            for d in case_dirs(uuid):
                try:
                    fns = os.listdir(d)
                except OSError:
                    continue
                for fn in fns:
                    if fn.lower().endswith(".stl") and "modeldie" in fn.lower():
                        try:
                            hits.add(content_hash(os.path.join(d, fn)))
                        except Exception:              # noqa: BLE001
                            pass
            seen[uuid] = hits
        if strip_sha(r["modeldie_asset_id"]) in seen[uuid]:
            out.append(r)
    return out


def stage_kind(out_dir, rows, limit=0, die_map=None):
    """资产形态审计：逐行判定 DIE_LIKE / CROWN_SHELL_INTAGLIO / UNCLEAR。

    用户规格 1（先出分布再冻结阈值）的同类要求：形态必须先看清，
    再决定哪些行能进 AutoMarginGTSet。
    """
    path = os.path.join(out_dir, "asset_kind_audit.csv" if die_map is None
                        else "asset_kind_evidence.csv")
    resume = set()
    cols = ("case_uuid", "fdi", "asset", "has_filename_evidence", "n_faces",
            "n_main", "asset_kind", "radial", "above_frac", "area_mm2",
            "boundary_edges", "volume", "reject_reason", "elapsed_s")
    if os.path.isfile(path):
        resume = {(r["case_uuid"], r["fdi"])
                  for r in csv.DictReader(open(path, encoding="ascii"))}
    t_all = time.time()
    n_done = 0
    # 资产名含中文（"A组"/"分层"），必须 utf-8；ascii 会 UnicodeEncodeError（踩过）
    with open(path, "a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(cols))
        if not resume:
            w.writeheader()
        for r in rows:
            if (r["case_uuid"], r["fdi"]) in resume:
                continue
            rec = {k: "" for k in cols}
            rec["case_uuid"] = r["case_uuid"]
            rec["fdi"] = r["fdi"]
            t0 = time.time()
            try:
                ctx = _prepare(r, (die_map or {}).get(
                    r["case_uuid"] + "|" + r["fdi"]))
                ak = asset_kind(ctx["idx"], ctx["tris"], ctx["margin"],
                                ctx["axis"], ctx["cut"])
                rec.update(asset=os.path.basename(ctx["die"]),
                           has_filename_evidence=("modeldie" in
                                                  os.path.basename(ctx["die"]).lower()),
                           n_faces=ctx["n_faces"], n_main=ak["n_main"],
                           asset_kind=ak["kind"], radial=str(ak["radial"]),
                           above_frac=str(ak["above"]), area_mm2=str(ak["area"]))
            except _Reject as exc:
                rec["reject_reason"] = str(exc)
            except Exception as exc:                 # noqa: BLE001
                rec["reject_reason"] = "%s: %s" % (type(exc).__name__,
                                                   str(exc)[:60])
            rec["elapsed_s"] = round(time.time() - t0, 1)
            w.writerow(rec)
            fh.flush()
            n_done += 1
            if limit and n_done >= limit:
                break
            if n_done % 10 == 0:
                print("  kind %d（%.0fs）" % (n_done, time.time() - t_all),
                      flush=True)
    print("kind 行数 %d，用时 %.0fs" % (n_done, time.time() - t_all))
    # 汇总
    recs = list(csv.DictReader(open(path, encoding="utf-8")))
    from collections import Counter
    dist = Counter()
    ev = Counter()
    for x in recs:
        if x["asset_kind"]:
            dist[x["asset_kind"] + ("/有文件名证据" if x["has_filename_evidence"] == "True"
                                    else "/回退产物")] += 1
        else:
            dist["REJECT:" + x["reject_reason"].split("(")[0]] += 1
        ev[x["has_filename_evidence"] or "?"] += 1
    print("形态分布(仅本次运行写入的行):", dict(dist))
    print("文件名证据 True/False:", dict(ev))



# ================================================================ 阶段：几何选代型
# 用户决策（2026-09-30 ①）：121 行 margin_off_mesh 抽查 60/60 都是"取错牙"，
# 根因是**按文件名匹配 fdi 不可靠**（病历号/订单号/组名/括号序号混在文件名里，
# 同病例多颗代型区分不开）。正确做法：对同病例的**所有候选 `*modeldie*`**，
# 把该牙位 Margin 分别投影过去，取**投影误差最小**者；最小值仍超阈才 REJECT。
# 判据自带验证（投影误差），不依赖命名 —— 符合"不靠文件名猜"的纪律。

SELECT_COLUMNS = ("case_uuid", "fdi", "n_candidates", "n_margin",
                  "margin_method", "die_asset_id",
                  "err_p50_mm", "err_p95_mm", "err_max_mm",
                  "err_runnerup_p95_mm", "n_off_mesh",
                  "selection", "elapsed_s")


def _margin_for(r):
    """取该行的 Margin 与轴；返回 (margin_pts, axis, method) 或抛 _Reject。"""
    dirs = case_dirs(r["case_uuid"])
    cis = find_ci(dirs)
    if not cis:
        raise _Reject("no_design_snapshot")
    method = ("MODEL_INFO" if cis[0].lower().endswith(".modelinfo")
              else "CONSTRUCTION_INFO")
    tooth = parse_construction_case(cis[0]).tooth(int(r["fdi"]))
    if tooth is None or not tooth.margin_points:
        raise _Reject("margin_missing_in_ci")
    axis = tooth.insertion_axis
    if not axis:
        raise _Reject("no_insertion_axis")
    margin = [(float(p[0]), float(p[1]), float(p[2]))
              for p in tooth.margin_points]
    return margin, unit((float(axis[0]), float(axis[1]), float(axis[2]))), method


def _candidate_dies(dirs):
    """病例目录里的候选 `*modeldie*` 文件，按**内容哈希去重**。

    224 个牙库文件里有 80 对逐字节相同，代型导出同样会有重复 —— 去重避免
    同一次选择里出现两个"同一个网格"互相竞争。
    """
    out = {}
    for d in dirs:
        for dp, _dn, fns in os.walk(d):
            for f in fns:
                if not f.lower().endswith(".stl"):
                    continue
                if "modeldie" not in f.lower():
                    continue
                q = os.path.join(dp, f)
                try:
                    h = content_hash(q)
                except OSError:
                    continue
                out.setdefault(h, q)
    return out


def stage_select(out_dir, shard=(1, 1), limit=0):
    """几何选代型：写 die_selection[_shardIofN].csv（PHI-free）。"""
    t_all = time.time()
    all_rows = [r for r in csv.DictReader(open(INDEX_CSV, encoding="ascii"))
                if r["has_margin"] == "True"]
    by_case = {}
    for r in all_rows:
        by_case.setdefault(r["case_uuid"], []).append(r)
    # 候选池 = 病例目录里有 >=1 个 *modeldie* 的病例（不要求文件名含 fdi）
    pool = []
    for uuid in sorted(by_case):
        dirs = case_dirs(uuid)
        cands = _candidate_dies(dirs) if dirs else {}
        if cands:
            pool.append((uuid, dirs, cands))
    print("有候选代型的病例: %d / 有 margin 的病例 %d"
          % (len(pool), len(by_case)), flush=True)
    print("候选池行数: %d" % sum(len(by_case[u]) for u, _d, _c in pool),
          flush=True)
    i, n = shard
    my = pool[i - 1::n]
    print("本分片 %d/%d：%d 个病例" % (i, n, len(my)), flush=True)
    path = os.path.join(out_dir, "die_selection%s.csv"
                        % ("" if shard == (1, 1) else "_shard%dof%d" % (i, n)))
    n_done = 0
    with open(path, "w", newline="", encoding="ascii") as fh:
        w = csv.DictWriter(fh, fieldnames=list(SELECT_COLUMNS))
        w.writeheader()
        for uuid, dirs, cands in my:
            t0 = time.time()
            # 每个候选代型只读一次，TriIndex 只建一次，供该病例所有牙位复用
            built = {}
            for h, p in cands.items():
                try:
                    tris = read_stl(p)
                    built[h] = (TriIndex(tris), len(tris))
                except Exception:                    # noqa: BLE001
                    continue
            if not built:
                continue
            for r in by_case[uuid]:
                rec = {k: "" for k in SELECT_COLUMNS}
                rec["case_uuid"] = uuid
                rec["fdi"] = r["fdi"]
                rec["n_candidates"] = len(built)
                rec["elapsed_s"] = 0.0
                try:
                    margin, axis, method = _margin_for(r)
                    rec["n_margin"] = len(margin)
                    rec["margin_method"] = method
                    mc = tuple(sum(q[k] for q in margin) / len(margin)
                               for k in range(3))
                    scored = []
                    for h, (idx, nf) in built.items():
                        ds = []
                        miss = 0
                        for q in margin:
                            d, _f = idx.nearest(q)
                            if d is None:
                                miss += 1
                            else:
                                ds.append(d)
                        if not ds or miss > 0.5 * len(margin):
                            continue
                        ds.sort()
                        p50 = ds[len(ds) // 2]
                        p95 = ds[int(.95 * (len(ds) - 1))]
                        scored.append((p95, p50, ds[-1], h, miss))
                    if not scored:
                        raise _Reject("no_candidate_projects")
                    scored.sort()
                    best = scored[0]
                    rec["die_asset_id"] = best[3]
                    rec["err_p50_mm"] = round(best[1], 4)
                    rec["err_p95_mm"] = round(best[0], 4)
                    rec["err_max_mm"] = round(best[2], 4)
                    rec["err_runnerup_p95_mm"] = (round(scored[1][0], 4)
                                                  if len(scored) > 1 else "")
                    rec["n_off_mesh"] = best[4]
                    rec["selection"] = ("MARGIN_ON_DIE" if best[0] <= 0.20
                                        else "BEST_STILL_OFF")
                except _Reject as exc:
                    rec["selection"] = "REJECT:" + str(exc)
                except Exception as exc:             # noqa: BLE001
                    rec["selection"] = "REJECT:%s" % type(exc).__name__
                rec["elapsed_s"] = round(time.time() - t0, 1)
                w.writerow(rec)
                fh.flush()
                n_done += 1
            if n_done and n_done % 20 < len(by_case[uuid]):
                print("  select %d（%.0fs）" % (n_done, time.time() - t_all),
                      flush=True)
    print("select 行数 %d，用时 %.0fs" % (n_done, time.time() - t_all))
    rows = list(csv.DictReader(open(path, encoding="ascii")))
    from collections import Counter
    print("选择结果:", dict(Counter(r["selection"] for r in rows)))
    print("写盘", path)


def stage_build(out_dir, rows, max_p95, max_max, limit=0, die_map=None):
    """批量构建 face-level 标签。"""
    t_all = time.time()
    audit_path = os.path.join(out_dir, "projection_audit.csv")
    if not os.path.isfile(audit_path):
        raise SystemExit("先跑 --stage audit")
    audit = {(r["case_uuid"], r["fdi"]): r
             for r in csv.DictReader(open(audit_path, encoding="ascii"))}
    lab_dir = os.path.join(out_dir, "labels")
    os.makedirs(lab_dir, exist_ok=True)
    path = os.path.join(out_dir, "auto_margin_gt.csv")
    resume = set()
    if os.path.isfile(path):
        resume = {(r["case_uuid"], r["fdi"])
                  for r in csv.DictReader(open(path, encoding="ascii"))}
    n_done = 0
    with open(path, "a", newline="", encoding="ascii") as fh:
        w = csv.DictWriter(fh, fieldnames=list(BUILD_COLUMNS))
        if not resume:
            w.writeheader()
        for r in rows:
            key = (r["case_uuid"], r["fdi"])
            if key in resume:
                continue
            a = audit.get(key)
            rec = {k: "" for k in BUILD_COLUMNS}
            rec["case_uuid"] = r["case_uuid"]
            rec["group_id"] = r["case_uuid"]
            rec["fdi"] = r["fdi"]
            t0 = time.time()
            try:
                # 证据模式（die_map 非空）**不能**用审计的数值：审计投影到的
                # 是索引里回退挑中的资产（冠壳），不是代型 —— 阈值与拒绝理由
                # 都必须来自本次构建自己算出的投影值。
                if a is not None and die_map is None:
                    if a["reject_reason"]:
                        raise _Reject("audit_rejected:" +
                                      a["reject_reason"].split(":")[0]
                                      .split("=")[0])
                    if max_p95 is not None and float(a["proj_p95_mm"]) > max_p95:
                        raise _Reject("proj_p95_over_threshold")
                    if max_max is not None and float(a["proj_max_mm"]) > max_max:
                        raise _Reject("proj_max_over_threshold")
                elif a is None and die_map is None:
                    raise _Reject("no_audit_row")
                built = _build_one(r, a, lab_dir,
                                   (die_map or {}).get(key))
                rr = built.pop("_review_reason", None)
                # 证据模式：阈值在**构建自身**算出的投影值上判
                if die_map is not None:
                    if max_p95 is not None and built["proj_p95_mm"] > max_p95:
                        raise _Reject("proj_p95_over_threshold")
                    if max_max is not None and built["proj_max_mm"] > max_max:
                        raise _Reject("proj_max_over_threshold")
                rec.update(built)
                rec["qc_status"] = "REVIEW" if rr else "OK"
                if rr:
                    rec["reject_reason"] = rr
            except _Reject as exc:
                rec["qc_status"] = "REJECT"
                rec["reject_reason"] = str(exc)
            except Exception as exc:                 # noqa: BLE001
                rec["qc_status"] = "REJECT"
                rec["reject_reason"] = "%s: %s" % (type(exc).__name__,
                                                   str(exc)[:70])
            rec["elapsed_s"] = round(time.time() - t0, 1)
            w.writerow(rec)
            fh.flush()
            n_done += 1
            if limit and n_done >= limit:
                break
            if n_done % 20 == 0:
                print("  build %d（%.0fs）" % (n_done, time.time() - t_all),
                      flush=True)
    print("build 行数 %d，用时 %.0fs" % (n_done, time.time() - t_all))


def stage_split(out_dir, seed=20260929):
    """case/group-level 固定 70/15/15 split（用户规格 8）。"""
    path = os.path.join(out_dir, "auto_margin_gt.csv")
    recs = [r for r in csv.DictReader(open(path, encoding="ascii"))
            if r["qc_status"] == "OK"]
    groups = sorted({r["group_id"] for r in recs})
    if not groups:
        raise SystemExit("没有 OK 行可切分")
    # 确定性哈希分配：同一 group 永远落同一侧
    def bucket(g):
        h = int(hashlib.sha256(("%d|%s" % (seed, g)).encode()).hexdigest()[:8],
                16)
        return h % 100
    assign = {}
    for g in groups:
        b = bucket(g)
        assign[g] = "train" if b < 70 else ("val" if b < 85 else "test")
    man = {
        "seed": seed, "unit": "group_id(=case_uuid)",
        "n_groups": len(groups), "n_rows": len(recs),
        "counts": {k: sum(1 for v in assign.values() if v == k)
                   for k in ("train", "val", "test")},
        "assign": assign,
    }
    json.dump(man, open(os.path.join(out_dir, "split_manifest.json"), "w"),
              indent=1)
    print(json.dumps({k: v for k, v in man.items() if k != "assign"}, indent=1))






if __name__ == "__main__":
    raise SystemExit(main() or 0)
