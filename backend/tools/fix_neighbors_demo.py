"""补齐演示病例的近中/远中邻牙与牙冠的"连接"（贴牙冠近接区取片）。

原来的取片是"以基台中心为圆心 + 半径"，选出来的面离牙冠 1.8mm 且近中那次被门拒绝
（选区含基牙重合面/非单连通）。这里改成：**以牙冠表面为参考**，取"离牙冠 < R、朝向
±frame.x、且离基台≥clearance"的单连通块，这才是真正的邻接接触区。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

B = "http://192.168.110.17:3277"
DID = "c973c6de213249bca4ff43ea7d0c197f"
CID = "0604e89f67f04dcc8170765e2f4c85fc"


def patch_near_crown(jaw, crown, abutment, origin, axis, *, sign, radius=5.0, clearance_mm=1.2,
                     facing_min=0.2, min_offset_mm=0.8):
    """近接裁片：**位置在确认方向的正确一侧**（(c-origin)·axis>0，后端硬判据）+
    贴着牙冠 + 面朝牙冠（法向指向内侧 = 邻牙的邻接面）+ 离基牙有间隙。"""
    from scipy.spatial import cKDTree
    from trimesh.graph import connected_components
    axis = np.asarray(axis, float) * sign
    centres = jaw.triangles_center
    distance_to_crown = cKDTree(np.asarray(crown.vertices, float)).query(centres)[0]
    normals = np.asarray(jaw.face_normals, float)
    offset = (centres - np.asarray(origin, float)) @ axis
    keep = ((offset >= min_offset_mm) & (distance_to_crown <= radius)
            & ((normals @ (-axis)) >= facing_min))
    if abutment is not None:
        keep &= cKDTree(np.asarray(abutment.vertices, float)).query(centres)[0] >= clearance_mm
    if keep.sum() < 30:
        return []
    edges = np.asarray(jaw.face_adjacency)
    inside = np.isin(edges, np.flatnonzero(keep)).all(axis=1)
    if not inside.any():
        return []
    best = max(connected_components(edges[inside]), key=len)
    return sorted(int(f) for f in best)


def main():
    import httpx
    client = httpx.Client(base_url=B, timeout=600)
    design = client.get(f"/api/designs/{DID}").json()
    status = client.get(f"/api/designs/{DID}/status").json()
    frame_doc = json.loads(design["autofit_frame_json"])
    axis = np.asarray(frame_doc["x"], float)          # 近中方向（已确认）
    case = client.get(f"/api/cases/{CID}").json()
    assets = {a["kind"]: a for a in case["assets"]}
    jaw = trimesh.load(assets["lower"]["path"], force="mesh")
    crown = trimesh.load(design["outer_path"], force="mesh")
    abutment = trimesh.load(assets["support"]["path"], force="mesh") if "support" in assets else None

    design_origin = np.asarray(json.loads(design["margin_json"]), float).mean(axis=0)

    def digest():
        return client.get(f"/api/designs/{DID}/status").json()["input_digest"]

    for side, sign in (("mesial", 1.0), ("distal", -1.0)):
        best = []
        for radius, clearance, facing in ((5.0, 1.2, 0.20), (7.0, 1.0, 0.10), (4.0, 1.5, 0.30)):
            best = patch_near_crown(jaw, crown, abutment, frame_doc['origin'] if 'origin' in frame_doc else design_origin, axis,
                                    sign=sign, radius=radius, clearance_mm=clearance, facing_min=facing)
            if len(best) >= 100:
                print(f"  {side}: radius={radius} clearance={clearance} facing={facing} -> {len(best)} faces")
                break
        if not best:
            print(f"  {side}: 找不到可用裁片，标 ABSENT")
        payload = {"status": "CONFIRMED" if best else "ABSENT", "source": "lower",
                   "instance_name": f"{side} neighbor (contact band)",
                   "pollution_reviewed": True, "expected_input_digest": digest(), "confirmed": True}
        if best:
            payload["face_ids"] = best[:20000]
            payload["outward_normal"] = (-sign * np.asarray(frame_doc["x"], float)).tolist()
        r = client.put(f"/api/designs/{DID}/auto-fit/neighbors/{side}", json=payload)
        print(f"  {side} PUT -> {r.status_code} {'' if r.status_code == 200 else r.text[:160]}")

    for kind in ("neighbors", "antagonist"):
        body = client.get(f"/api/designs/{DID}/outer-analysis", params={"kind": kind}).json()
        print("\n%s: status=%s skipped=%s" % (kind, body.get("status"), json.dumps(body.get("skipped"), ensure_ascii=False)))
        print("   min_mm=%.4f p05_mm=%.4f" % (body.get("min_mm", float("nan")), body.get("p05_mm", float("nan"))))


if __name__ == "__main__":
    main()
