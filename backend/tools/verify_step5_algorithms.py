"""验证 Step5 牙冠设计是否**真的执行了设计算法**（不是只摆模板）。

四项：
  1) outer crown generation：margin curve 是否参与（margin_points_used / 边缘线到冠的距离 /
     颈缘贴合结果）
  2) 适配邻牙（近中/远中）：前后 mesh hash、顶点位移、双向 Hausdorff、接触指标
  3) 适配对颌：同上
  4) 任何一步没改几何 → 就是 workflow 接线问题

用法：python -m tools.verify_step5_algorithms [design_id]
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

B = "http://192.168.110.17:3277"
DEFAULT_DESIGN = "c973c6de213249bca4ff43ea7d0c197f"   # 【演示】FDI46 已生成牙冠


def sha_of_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:16]


def fetch_crown(client, did):
    d = client.get(f"/api/designs/{did}").json()
    raw = client.get(f"/api/designs/{did}/outer.stl").content
    mesh = trimesh.load(trimesh.util.wrap_as_stream(raw), file_type="stl", force="mesh")
    return d, raw, mesh


def symmetric_hausdorff(a, b):
    tree_a = cKDTree(np.asarray(a.vertices, float))
    tree_b = cKDTree(np.asarray(b.vertices, float))
    ab = tree_a.query(np.asarray(b.vertices, float))[0]
    ba = tree_b.query(np.asarray(a.vertices, float))[0]
    return float(max(ab.max(), ba.max()))


def vertex_delta(a, b):
    va, vb = np.asarray(a.vertices, float), np.asarray(b.vertices, float)
    if va.shape != vb.shape:
        return None
    return float(np.linalg.norm(va - vb, axis=1).max())


def margin_report(client, did, mesh):
    d = client.get(f"/api/designs/{did}").json()
    margin = np.asarray(json.loads(d["margin_json"]), float)
    from trimesh.proximity import closest_point
    _, dist, _ = closest_point(mesh, margin)
    return {"margin_points_used": int(len(margin)),
            "projected_margin_distance_mm": {"mean": round(float(dist.mean()), 4),
                                             "p95": round(float(np.quantile(dist, .95)), 4),
                                             "max": round(float(dist.max()), 4)}}


def analysis(client, did, kind):
    b = client.get(f"/api/designs/{did}/outer-analysis", params={"kind": kind}).json()
    return {"status": b.get("status"), "min_mm": b.get("min_mm"), "p05_mm": b.get("p05_mm"),
            "skipped": sorted((b.get("skipped") or {}).keys())}


def main():
    import httpx
    did = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_DESIGN
    client = httpx.Client(base_url=B, timeout=900)

    print("=" * 78)
    print("① outer crown generation：margin curve 是否参与")
    design, raw0, mesh0 = fetch_crown(client, did)
    print("  初始冠: faces=%d verts=%d hash=%s" % (len(mesh0.faces), len(mesh0.vertices), sha_of_bytes(raw0)))
    print("  边缘线参与情况:", json.dumps(margin_report(client, did, mesh0), ensure_ascii=False))
    margin = json.loads(design["margin_json"])
    axis = json.loads(design["axis_json"])
    placed = client.post(f"/api/designs/{did}/outer/auto",
                         json={"margin_points": margin, "insertion_axis": axis})
    print("  POST /outer/auto ->", placed.status_code)
    if placed.status_code == 200:
        _, raw1, mesh1 = fetch_crown(client, did)
        print("  重放后冠: faces=%d verts=%d hash=%s | 顶点最大位移=%s | Hausdorff=%s mm" % (
            len(mesh1.faces), len(mesh1.vertices), sha_of_bytes(raw1),
            vertex_delta(mesh0, mesh1), symmetrical(mesh0, mesh1) if False else round(symmetric_hausdorff(mesh0, mesh1), 4)))
        print("  边缘线参与情况(重放后):", json.dumps(margin_report(client, did, mesh1), ensure_ascii=False))
        print("  body:", json.dumps(placed.json(), ensure_ascii=False)[:260] if isinstance(placed.json(), dict) else "")

    print("=" * 78)
    print("②/③ 适配邻牙 / 适配对颌：几何是否真的被改")
    rows = []
    for op, label in (("adapt_mesial", "适配近中"), ("adapt_distal", "适配远中"), ("adapt_antagonist", "适配对颌")):
        before_design, before_raw, before_mesh = fetch_crown(client, did)
        before_prox = analysis(client, did, "neighbors")
        before_occ = analysis(client, did, "antagonist")
        r = client.post(f"/api/designs/{did}/edit-outer",
                        json={"operation": op, "base_outer_path": before_design["outer_path"],
                              "target_distance_mm": 0.05})
        ok = r.status_code == 200
        body = r.json() if ok else {}
        after_design, after_raw, after_mesh = fetch_crown(client, did)
        after_prox = analysis(client, did, "neighbors")
        after_occ = analysis(client, did, "antagonist")
        row = {"op": label, "http": r.status_code,
               "status": body.get("status"),
               "hash_before": sha_of_bytes(before_raw), "hash_after": sha_of_bytes(after_raw),
               "hash_changed": sha_of_bytes(before_raw) != sha_of_bytes(after_raw),
               "verts": "%d→%d" % (len(before_mesh.vertices), len(after_mesh.vertices)),
               "max_vertex_move_mm": vertex_delta(before_mesh, after_mesh),
               "hausdorff_mm": round(symmetric_hausdorff(before_mesh, after_mesh), 4),
               "prox_min": "%s→%s" % (before_prox.get("min_mm"), after_prox.get("min_mm")),
               "occ_min": "%s→%s" % (before_occ.get("min_mm"), after_occ.get("min_mm")),
               "skipped_sides": body.get("edit", {}).get("skipped_sides") if ok else None,
               "margin_disp": (body.get("edit") or {}).get("margin_displacement_mm") if ok else None,
               "detail": None if ok else r.text[:140]}
        rows.append(row)
        print("\n  [%s] http=%s status=%s hash %s→%s changed=%s" % (
            label, row["http"], row["status"], row["hash_before"], row["hash_after"], row["hash_changed"]))
        print("      verts %s | 最大顶点位移=%s mm | Hausdorff=%s mm" % (
            row["verts"], row["max_vertex_move_mm"], row["hausdorff_mm"]))
        print("      接触: 邻牙 min %s | 咬合 min %s | margin位移=%s | 跳过侧=%s" % (
            row["prox_min"], row["occ_min"], row["margin_disp"], row["skipped_sides"]))
        if row["detail"]:
            print("      detail:", row["detail"])

    print("\n" + "=" * 78)
    print("结论（几何是否被修改）")
    for row in rows:
        verdict = "✓ 修改了几何" if row["hash_changed"] else "✗ 几何未变（接线/可动区域问题）"
        print("  %-8s %s  Hausdorff=%s mm" % (row["op"], verdict, row["hausdorff_mm"]))
    Path("step5_verification.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print("\n明细: backend/step5_verification.json")


def symmetrical(a, b):  # 兼容占位（未使用）
    return symmetric_hausdorff(a, b)


if __name__ == "__main__":
    main()
