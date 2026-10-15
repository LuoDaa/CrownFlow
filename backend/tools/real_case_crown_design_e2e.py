"""真实病例端到端：Crown Design 1.0 全链路（含自动语义区域与邻牙/对颌适配）。

用 `测试订单/人工设计基台` 里的**真实病例**跑一遍：
    载入病例 → 牙位 → Margin/Axis → 模板/参数 → 生成牙冠 → 自动语义区域
    → 适配邻牙 → 适配对颌 → 接触图 → 自由塑形 → 最终构建 → 导出 STL

用法（在 backend 目录）：
    python -m tools.real_case_crown_design_e2e [病例目录名]

数据与 DB 全部落在临时目录，**不碰工作区里的真实病例库**。
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

# 直接 `python tools/xxx.py` 时 sys.path[0] 是 tools/（会遮蔽 backend 里的包），
# 这里显式把 backend 目录放回最前，保证 `import app` 指向项目自身。
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import trimesh

CORPUS = Path(r"D:\code\牙冠自动生成\测试订单\人工设计基台\人工设计基台")
DEFAULT_CASE = "1026h徐志平"
CROP_HALF_MM = 18.0       # 工作区裁切半宽（mm）：真实几何、只删远处面片
FDI = 46
WORKING_JAW = "lower" if FDI >= 31 else "upper"      # FDI 46 ⇒ 工作颌 = 下颌
ANTAGONIST = "upper" if WORKING_JAW == "lower" else "lower"


def isolate(data_dir: Path):
    """把 DB 与数据目录指到临时路径（与 tests/integration/conftest.py 同一手法）。"""
    from app import config, db
    (data_dir / "cases").mkdir(parents=True, exist_ok=True)
    (data_dir / "templates").mkdir(parents=True, exist_ok=True)
    config.DATA_DIR = data_dir
    config.CASE_DIR = data_dir / "cases"
    config.TEMPLATE_DIR = data_dir / "templates"
    config.DB_PATH = data_dir / "e2e.sqlite3"
    for name, module in list(sys.modules.items()):
        if name.startswith("app."):
            for key, relative in (("CASE_DIR", "cases"), ("TEMPLATE_DIR", "templates"), ("DATA_DIR", "")):
                if hasattr(module, key):
                    setattr(module, key, data_dir / relative)
    db.init_db()


def pick_files(case_dir: Path):
    """按文件名角色挑出工作颌 / 对颌 / 基台（角色不绑定文件名，这里只用真实存在的三件）。"""
    stl = sorted(case_dir.glob("*.stl"))
    working, antagonist, support = None, None, None
    for path in stl:
        name = path.name.lower()
        if "lowerjaw" in name and WORKING_JAW == "lower":
            working = path
        elif "upperjaw" in name and WORKING_JAW == "upper":
            working = path
        elif "unsectionedmodel_lowerjaw" in name:
            working = path
        if "antag" in name:
            antagonist = path
        if "基台" in path.name and "定位器" not in path.name:
            support = path
    if working is None:
        for path in stl:
            if path.name.lower().startswith(case_dir.name.replace("_", "").lower()[:6]):
                working = path
                break
    return working, antagonist, support


def crop_near(mesh: trimesh.Trimesh, center, *, half_mm=18.0):
    """把扫描裁到目标牙位附近的工作区（**几何保真**：只删面，不改顶点）。

    真实口扫是 20 万面量级；本机进程内存预算撑不住"整颌 + 整对颌 + 冠"同时做点面查询。
    裁切是我们为这台机器做的**输入准备**，不是算法改动；口径记录在报告的 workspace 字段里。
    """
    centres = mesh.triangles_center
    keep = np.all(np.abs(centres - np.asarray(center, float)) <= half_mm, axis=1)
    if not keep.any():
        return mesh
    submesh = mesh.submesh([np.flatnonzero(keep)], append=True)
    return submesh if isinstance(submesh, trimesh.Trimesh) else mesh


def face_ids_near(jaw: trimesh.Trimesh, center, side_axis, *, abutment=None, radius=6.0, sign=1.0,
                  facing_min=0.5, clearance_mm=3.5):
    """工作颌上「目标侧、朝向一致、单连通、且**不与基牙重合**」的真实邻牙裁片。

    后端会拒绝跨牙 / 断裂 / 朝向不一致 / 与基牙重合的选片，所以这里要求：
    朝目标牙的那一面、离基牙表面至少 ``clearance_mm``、并取最大连通块。
    """
    from scipy.spatial import cKDTree
    from trimesh.graph import connected_components

    axis = np.asarray(side_axis, float) * sign
    normals = np.asarray(jaw.face_normals, float)
    centres = jaw.triangles_center
    offset = centres - np.asarray(center, float)
    keep = (np.linalg.norm(offset, axis=1) <= radius) & ((normals @ axis) >= facing_min)
    if abutment is not None:
        clearance = cKDTree(np.asarray(abutment.vertices, float)).query(centres)[0]
        keep &= clearance >= clearance_mm
    if not keep.any():
        return []
    edges = np.asarray(jaw.face_adjacency)
    inside = np.isin(edges, np.flatnonzero(keep)).all(axis=1)
    if not inside.any():
        return []
    best = max(connected_components(edges[inside]), key=len)
    return sorted(int(face) for face in best)


def contact_area_estimate(payload):
    """逐顶点有符号距离 → 接触区面积（顶点面积估计，工程口径）。"""
    signed = payload.get("signed_mm")
    vertices, faces = payload.get("vertices"), payload.get("faces")
    if not signed or not vertices or not faces:
        return None
    area = 0.0
    for a, b, c in faces:
        if signed[a] > 0 and signed[b] > 0 and signed[c] > 0:
            continue
        pa, pb, pc = (np.asarray(vertices[i], float) for i in (a, b, c))
        area += float(np.linalg.norm(np.cross(pb - pa, pc - pa)) / 2)
    return area


def main():
    case_name = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CASE
    case_dir = CORPUS / case_name
    if not case_dir.is_dir():
        raise SystemExit("找不到真实病例目录：%s" % case_dir)
    working_file, antagonist_file, support_file = pick_files(case_dir)
    print("病例        :", case_name)
    print("工作颌      :", working_file and working_file.name)
    print("对颌        :", antagonist_file and antagonist_file.name)
    print("基台        :", support_file and support_file.name)
    if not (working_file and antagonist_file and support_file):
        raise SystemExit("这件病例缺少工作颌 / 对颌 / 基台 STL")

    with tempfile.TemporaryDirectory() as tmp:
        isolate(Path(tmp))
        from fastapi.testclient import TestClient
        from app.main import app

        report = {}
        # 工作区裁切（几何保真、只删远处面片）：见 crop_near 说明
        jaw_full = trimesh.load(working_file, force="mesh")
        antagonist_full = trimesh.load(antagonist_file, force="mesh")
        abutment = trimesh.load(support_file, force="mesh")
        centre = abutment.bounds.mean(axis=0)
        jaw = crop_near(jaw_full, centre, half_mm=CROP_HALF_MM)
        antagonist = crop_near(antagonist_full, centre, half_mm=CROP_HALF_MM)
        report["workspace"] = {"half_mm": CROP_HALF_MM,
                               "working_jaw_faces": [len(jaw_full.faces), len(jaw.faces)],
                               "antagonist_faces": [len(antagonist_full.faces), len(antagonist.faces)]}
        print("工作区裁切  :", json.dumps(report["workspace"]))
        uploads = {WORKING_JAW: jaw, ANTAGONIST: antagonist, "support": abutment}

        with TestClient(app) as client:
            # PHI 纪律：病例名里**不写**任何患者/订单标识，只用中性编号（目录名只用于定位文件）。
            created = client.post("/api/cases", json={"name": "real-case-e2e-FDI%d" % FDI})
            assert created.status_code == 200, created.text
            case_id = created.json()["id"]
            for kind, mesh in uploads.items():
                response = client.post(f"/api/cases/{case_id}/assets/{kind}",
                                       files={"file": (f"{kind}.stl", mesh.export(file_type="stl"), "model/stl")})
                assert response.status_code == 200, (kind, response.text)
            design_id = client.post("/api/designs", json={"draft": True, "case_id": case_id,
                                                          "target_fdi": FDI, "support_type": "abutment"}).json()["id"]
            path = f"/api/designs/{design_id}"

            def digest():
                return client.get(path + "/status").json()["input_digest"]

            # ── Margin / Axis：真实基台肩线提取 ─────────────────────────────
            prepared = client.post(path + "/draft-preparation",
                                   json={"radius_mm": 7, "restorative_abutment_confirmed": True})
            print("draft      :", prepared.status_code, (prepared.text or "")[:160])
            assert prepared.status_code == 200, prepared.text
            body = prepared.json()
            margin = body.get("points")
            axis = body.get("axis")
            assert margin and axis, body
            margin = [list(map(float, p)) for p in margin]
            report["margin_points"] = len(margin)
            assert client.post(path + "/margin", json={"points": margin}).status_code == 200
            assert client.post(path + "/axis", json={"axis": list(map(float, axis))}).status_code == 200

            # ── 参数（必须在确认坐标帧之前：confirmed_inputs 要求三件都已确认）──
            profile = client.put(path + "/profile", json={"material": "real case e2e", "coverage": "ENGINEERING_ONLY",
                                                          "bite_confirmed": True, "confirmed": True,
                                                          "parameters": {"machine_axes": 5, "minimum_thickness_mm": 0.5,
                                                                         "proximal_target_mm": 0.05,
                                                                         "occlusal_target_mm": 0.05}})
            assert profile.status_code == 200, profile.text

            # ── 坐标帧：横向轴用边缘线投影 PCA（几何），z = 真实就位轴 ──────
            from app.geometry.crown_outer.patient_frame import build_patient_frame
            frame = build_patient_frame(np.asarray(margin), np.asarray(axis, float))
            frame_response = client.put(path + "/auto-fit/frame",
                                        json={"mesial": frame.x.tolist(), "buccal": frame.y.tolist(), "units": "mm",
                                              "expected_input_digest": digest(), "confirmed": True})
            assert frame_response.status_code == 200, frame_response.text

            # ── 对颌：病例自带咬合位姿（上下颌同帧 ⇒ 单位阵）────────────────
            bite = client.put(path + "/auto-fit/bite",
                              json={"transform": np.eye(4).tolist(), "source_method": "REGISTERED_EXPORT",
                                    "units": "mm", "expected_input_digest": digest(), "confirmed": True})
            assert bite.status_code == 200, bite.text

            # ── 邻牙：从真实工作颌上取目标侧附近的面片 ──────────────────────
            mesial_ids = face_ids_near(jaw, centre, frame.x, abutment=abutment, sign=1.0)
            distal_ids = face_ids_near(jaw, centre, frame.x, abutment=abutment, sign=-1.0)
            report["neighbor_face_ids"] = {"mesial": len(mesial_ids), "distal": len(distal_ids)}
            for side, ids in (("mesial", mesial_ids), ("distal", distal_ids)):
                sign = 1.0 if side == "mesial" else -1.0
                payload = {"status": "CONFIRMED" if ids else "ABSENT", "source": WORKING_JAW,
                           "instance_name": "%s neighbor" % side, "face_ids": ids[:20000],
                           "pollution_reviewed": True, "expected_input_digest": digest(), "confirmed": True}
                if ids:
                    # 开放裁片必须声明外向法向：邻牙裁片朝向目标牙（-sign·x）
                    payload["outward_normal"] = (-sign * np.asarray(frame.x, float)).tolist()
                else:
                    payload.pop("face_ids")
                response = client.put(f"{path}/auto-fit/neighbors/{side}", json=payload)
                assert response.status_code == 200, (side, response.text)

            # ── 生成牙冠（模板自动放置）───────────────────────────────────
            placed = client.post(path + "/outer/auto",
                                 json={"margin_points": margin, "insertion_axis": list(map(float, axis))})
            assert placed.status_code == 200, placed.text
            mesh_path = client.get(path).json()["outer_path"]
            crown = trimesh.load(mesh_path, force="mesh")
            report["crown"] = {"vertices": len(crown.vertices), "faces": len(crown.faces)}

            # ── 自动语义区域 ──────────────────────────────────────────────
            session = client.get(path + "/sculpt-session?regions=1").json()
            region_map = session["region_map"]
            report["region_map"] = {"counts": region_map["counts"], "domains": region_map["domain_counts"],
                                    "flags": region_map["flags"], "provenance": region_map["provenance"],
                                    "manual_annotations": session["manual_region_names"]}
            print("语义区域    :", json.dumps(region_map["counts"], ensure_ascii=False))
            print("区域 flags  :", region_map["flags"])

            def measure(kind):
                payload = client.get(f"{path}/outer-analysis?kind={kind}")
                return payload.json() if payload.status_code == 200 else None

            def edit(operation, **extra):
                return client.post(path + "/edit-outer",
                                   json={"operation": operation, "base_outer_path": client.get(path).json()["outer_path"],
                                         **extra})

            # ── 适配邻牙（近中）──────────────────────────────────────────
            before_neighbor = measure("neighbors")
            neighbor_before = min(before_neighbor["signed_mm"]) if before_neighbor else None
            neighbor_vertices_before = np.asarray(session["vertices"])
            adjusted = edit("adapt_mesial", target_distance_mm=0.05)
            report["adapt_mesial"] = {"status": adjusted.status_code, "body_status": adjusted.json().get("status")}
            assert adjusted.status_code == 200, adjusted.text
            if adjusted.json().get("status") == "APPLIED":
                after_neighbor = measure("neighbors")
                payload = adjusted.json()["edit"]
                report["neighbor_distance"] = {
                    "before_min_signed_mm": neighbor_before,
                    "after_min_signed_mm": min(after_neighbor["signed_mm"]) if after_neighbor else None,
                    "margin_displacement_mm": payload.get("margin_displacement_mm"),
                    "skipped": payload.get("skipped_sides"),
                }
                moved = trimesh.load(client.get(path).json()["outer_path"], force="mesh")
                report["neighbor_distance"]["max_vertex_move_mm"] = float(
                    np.linalg.norm(moved.vertices - neighbor_vertices_before, axis=1).max())
                assert payload.get("margin_displacement_mm") == 0.0

            # ── 适配对颌 ─────────────────────────────────────────────────
            before_antagonist = measure("antagonist")
            occlusion_before = min(before_antagonist["signed_mm"]) if before_antagonist else None
            area_before = contact_area_estimate(before_antagonist)
            vertices_before = np.asarray(trimesh.load(client.get(path).json()["outer_path"], force="mesh").vertices)
            protection = np.asarray(session["region_protection"])
            antagonist = edit("adapt_antagonist", target_distance_mm=0.05)
            report["adapt_antagonist"] = {"status": antagonist.status_code, "body_status": antagonist.json().get("status")}
            assert antagonist.status_code == 200, antagonist.text
            if antagonist.json().get("status") == "APPLIED":
                after_antagonist = measure("antagonist")
                after_mesh = trimesh.load(client.get(path).json()["outer_path"], force="mesh")
                displacement = np.linalg.norm(after_mesh.vertices - vertices_before, axis=1)
                report["occlusion"] = {
                    "before_min_signed_mm": occlusion_before,
                    "after_min_signed_mm": min(after_antagonist["signed_mm"]) if after_antagonist else None,
                    "penetration_before_mm": max(0.0, -(occlusion_before or 0.0)),
                    "penetration_after_mm": max(0.0, -min(after_antagonist["signed_mm"])) if after_antagonist else None,
                    "contact_area_before_mm2": area_before,
                    "contact_area_after_mm2": contact_area_estimate(after_antagonist),
                    "margin_displacement_mm": antagonist.json()["edit"].get("margin_displacement_mm"),
                    "protected_max_move_mm": float(displacement[protection <= 0].max()) if (protection <= 0).any() else 0.0,
                    "max_vertex_move_mm": float(displacement.max()),
                }

            # ── 自由塑形（保护顶点必须不动）───────────────────────────────
            session = client.get(path + "/sculpt-session?regions=1").json()
            current = np.asarray(trimesh.load(client.get(path).json()["outer_path"], force="mesh").vertices)
            occlusal = np.asarray(session["region_weights"]["occlusal"]) >= 0.5
            lock = np.asarray(session["region_protection"]) <= 0
            brush = edit("add", centers=[current[occlusal].mean(axis=0).tolist()], radius_mm=1.5, strength_mm=0.03)
            assert brush.status_code == 200, brush.text
            sculpted = trimesh.load(client.get(path).json()["outer_path"], force="mesh")
            moved = np.linalg.norm(sculpted.vertices - current, axis=1)
            report["freeform"] = {"max_move_mm": float(moved.max()),
                                  "protected_max_move_mm": float(moved[lock].max()) if lock.any() else 0.0}

            # ── 最终构建 / 导出 ───────────────────────────────────────────
            status = client.get(path + "/status").json()
            final = client.post(path + "/finalize", json={"expected_input_digest": status["input_digest"],
                                                          **status["confirmed_revisions"]})
            report["finalize"] = {"status": final.status_code, "qc": bool(final.status_code == 200 and final.json().get("qc"))}
            stl = client.get(path + "/final.stl")
            report["export"] = {"status": stl.status_code, "bytes": len(stl.content) if stl.status_code == 200 else 0}

        print("\n===== 真实病例端到端结果 =====")
        print(json.dumps(report, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
