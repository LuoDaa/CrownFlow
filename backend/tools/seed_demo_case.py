"""在**线上服务**里造一个"已生成牙冠"的演示病例（真实库，不是隔离库）。

用法（backend 目录）：
    python -m tools.seed_demo_case            # 默认走 http://127.0.0.1:3277
    python -m tools.seed_demo_case --base http://192.168.110.17:3277

流程与前端点击完全一致：
  建病例 → 上传下颌(工作颌)/上颌(对颌) → draft-preparation（自动提取基台支撑 + 肩线 + 轴）
  → 保存边缘线/轴 → 参数 → 确认方向帧 → 确认咬合 → 邻牙(可失败则 ABSENT)
  → outer/auto（生成牙冠）→ finalize（DESIGN_MODE）→ 导出 final.stl
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

CASE_DIR = Path(r"D:\code\牙冠自动生成\1.0测试数据\基台测试1-牙位46")
FDI = 46
CASE_NAME = "【演示】FDI46 已生成牙冠"


def faces_near(jaw, abutment, center, axis, *, sign, radius=6.0, clearance_mm=1.5, facing_min=0.5):
    from scipy.spatial import cKDTree
    from trimesh.graph import connected_components
    axis = np.asarray(axis, float) * sign
    normals = np.asarray(jaw.face_normals, float)
    centres = jaw.triangles_center
    offset = centres - np.asarray(center, float)
    keep = (np.linalg.norm(offset, axis=1) <= radius) & ((normals @ axis) >= facing_min)
    if abutment is not None:
        keep &= cKDTree(np.asarray(abutment.vertices, float)).query(centres)[0] >= clearance_mm
    if not keep.any():
        return []
    edges = np.asarray(jaw.face_adjacency)
    inside = np.isin(edges, np.flatnonzero(keep)).all(axis=1)
    if not inside.any():
        return []
    best = max(connected_components(edges[inside]), key=len)
    return sorted(int(f) for f in best)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="http://127.0.0.1:3277")
    args = parser.parse_args()
    import httpx

    client = httpx.Client(base_url=args.base, timeout=600.0)
    print("server:", args.base, "| health:", client.get("/api/health").json())

    lower = sorted(CASE_DIR.glob("下颌-基台牙位46*.stl"))[0]
    upper = sorted(CASE_DIR.glob("上颌*.stl"))[0]
    print("working jaw:", lower.name, "| antagonist:", upper.name)

    case_id = client.post("/api/cases", json={"name": CASE_NAME}).json()["id"]
    for kind, path in (("lower", lower), ("upper", upper)):
        r = client.post(f"/api/cases/{case_id}/assets/{kind}",
                        files={"file": (path.name, path.read_bytes(), "model/stl")})
        assert r.status_code == 200, (kind, r.text)
    design_id = client.post("/api/designs", json={"draft": True, "case_id": case_id,
                                                  "target_fdi": FDI, "support_type": "abutment"}).json()["id"]
    path = f"/api/designs/{design_id}"
    print("case:", case_id, "| design:", design_id)

    def digest():
        return client.get(path + "/status").json()["input_digest"]

    prepared = client.post(path + "/draft-preparation",
                           json={"radius_mm": 7, "restorative_abutment_confirmed": True})
    assert prepared.status_code == 200, prepared.text
    body = prepared.json()
    margin = [list(map(float, p)) for p in body["points"]]
    axis = list(map(float, body["axis"]))
    print("margin points:", len(margin), "| support:", (body.get("support_mesh") or {}))

    assert client.post(path + "/margin", json={"points": margin}).status_code == 200
    assert client.post(path + "/axis", json={"axis": axis}).status_code == 200
    profile = client.put(path + "/profile", json={"material": "演示配置", "coverage": "ENGINEERING_ONLY",
                                                  "bite_confirmed": True, "confirmed": True,
                                                  "parameters": {"machine_axes": 5, "minimum_thickness_mm": 0.5,
                                                                 "proximal_target_mm": 0.05,
                                                                 "occlusal_target_mm": 0.05}})
    assert profile.status_code == 200, profile.text

    from app.geometry.crown_outer.patient_frame import build_patient_frame
    frame = build_patient_frame(np.asarray(margin), np.asarray(axis))
    r = client.put(path + "/auto-fit/frame", json={"mesial": frame.x.tolist(), "buccal": frame.y.tolist(),
                                                   "units": "mm", "expected_input_digest": digest(),
                                                   "confirmed": True})
    assert r.status_code == 200, r.text
    r = client.put(path + "/auto-fit/bite", json={"transform": np.eye(4).tolist(),
                                                  "source_method": "REGISTERED_EXPORT", "units": "mm",
                                                  "expected_input_digest": digest(), "confirmed": True})
    assert r.status_code == 200, r.text
    print("frame + bite confirmed")

    # 邻牙（可失败 ⇒ 该侧标 ABSENT，不阻塞）
    case_detail = client.get(f"/api/cases/{case_id}").json()
    support_entry = next((a for a in case_detail.get("assets", []) if a["kind"] == "support"), None)
    jaw = trimesh.load(lower, force="mesh")
    abutment = trimesh.load(support_entry["path"], force="mesh") if support_entry else None
    center = np.asarray(body.get("implant_candidate", {}).get("center") or abutment.bounds.mean(axis=0))
    for side, sign in (("mesial", 1.0), ("distal", -1.0)):
        face_ids = faces_near(jaw, abutment, center, frame.x, sign=sign)
        payload = {"status": "CONFIRMED" if face_ids else "ABSENT", "source": "lower",
                   "instance_name": f"{side} neighbor", "pollution_reviewed": True,
                   "expected_input_digest": digest(), "confirmed": True}
        if face_ids:
            payload["face_ids"] = face_ids[:20000]
            payload["outward_normal"] = (-sign * np.asarray(frame.x, float)).tolist()
        r = client.put(f"{path}/auto-fit/neighbors/{side}", json=payload)
        if r.status_code != 200 and face_ids:
            r = client.put(f"{path}/auto-fit/neighbors/{side}",
                           json={**payload, "status": "ABSENT", "face_ids": None,
                                 "outward_normal": None})
        print(f"  neighbor {side}: {r.status_code} ({len(face_ids)} faces)")

    placed = client.post(path + "/outer/auto", json={"margin_points": margin, "insertion_axis": axis})
    assert placed.status_code == 200, placed.text
    design = client.get(path).json()
    crown = trimesh.load(design["outer_path"], force="mesh")
    print("CROWN:", len(crown.faces), "faces |", len(crown.vertices), "verts | watertight:", crown.is_watertight)

    status = client.get(path + "/status").json()
    final = client.post(path + "/finalize", json={"expected_input_digest": status["input_digest"],
                                                  **status["confirmed_revisions"]})
    print("finalize:", final.status_code)
    if final.status_code == 200:
        acceptance = (final.json().get("qc") or {}).get("final_acceptance")
        stl = client.get(path + "/final.stl")
        print("  QC acceptance:", json.dumps(acceptance, ensure_ascii=False)[:220])
        print("  final.stl:", stl.status_code, len(stl.content), "bytes")
        print("  final.stl path:", design.get("final_path"))
    else:
        print("  detail:", final.text[:300])
    print("\n翻开：", f"{args.base}/  → 最近病例里点《{CASE_NAME}》（FDI {FDI}）")


if __name__ == "__main__":
    main()
