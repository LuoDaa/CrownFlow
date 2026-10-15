"""三例真实基台病例：Margin → Support Extraction 逐例诊断 + 完整跑到 Final Crown STL。

病例 manifest（P0-15）：不使用 `基台测试1-46/world_result.stl` 这类旧路径，
直接读当前实际测试数据目录；缺文件就**报错**，不用 skip 掩盖数据改名。

用法（在 backend 目录）：
    python -m tools.diagnose_support_extraction            # 只做诊断
    python -m tools.diagnose_support_extraction --full      # 诊断 + 跑到导出
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

CORPUS = Path(r"D:\code\牙冠自动生成\1.0测试数据")

SUPPORT_ROI_MM = 8.0   # 支撑 ROI 半径：必须装得下整条肩线，否则切完的边界不可能是确认终止线

#: 固定 1.0 回归病例（目录名 → 牙位；文件角色在目录里用真实名字标识）
REAL_ABUTMENT_CASES = [
    {"case": "基台测试1-牙位46", "tooth": 46, "jaw_pattern": "下颌-基台牙位46*.stl", "antagonist_pattern": "上颌*.stl"},
    {"case": "基台测试2-牙位36", "tooth": 36, "jaw_pattern": "下颌-基台牙位36*.stl", "antagonist_pattern": "上颌*.stl"},
    {"case": "基台测试3-牙位46", "tooth": 46, "jaw_pattern": "下颌-基台牙位46*.stl", "antagonist_pattern": "上颌*.stl"},
]


def isolate(data_dir: Path):
    from app import config, db
    (data_dir / "cases").mkdir(parents=True, exist_ok=True)
    (data_dir / "templates").mkdir(parents=True, exist_ok=True)
    config.DATA_DIR = data_dir
    config.CASE_DIR = data_dir / "cases"
    config.TEMPLATE_DIR = data_dir / "templates"
    config.DB_PATH = data_dir / "diag.sqlite3"
    for name, module in list(sys.modules.items()):
        if name.startswith("app."):
            for key, relative in (("CASE_DIR", "cases"), ("TEMPLATE_DIR", "templates"), ("DATA_DIR", "")):
                if hasattr(module, key):
                    setattr(module, key, data_dir / relative)
    db.init_db()


def resolve(entry):
    folder = CORPUS / entry["case"]
    if not folder.is_dir():
        raise SystemExit("真实病例目录不存在（数据已改名？）：%s" % folder)
    jaw = sorted(folder.glob(entry["jaw_pattern"]))
    antagonist = sorted(folder.glob(entry["antagonist_pattern"]))
    if not jaw or not antagonist:
        raise SystemExit("%s 缺少工作颌或对颌 STL：%s" % (entry["case"], [p.name for p in folder.iterdir()]))
    return jaw[0], antagonist[0]


def boundary_loops(mesh):
    """边界环：只被一个面使用的边 → 连成环（孔洞/开放边界的计算口径）。"""
    edges = np.sort(np.asarray(mesh.edges_unique), axis=1)
    counts = np.asarray(mesh.edges_unique_length)  # 与 edges_unique 同序
    used = {}
    for tri in np.asarray(mesh.faces):
        for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
            key = (min(a, b), max(a, b))
            used[key] = used.get(key, 0) + 1
    open_edges = [key for key, count in used.items() if count == 1]
    if not open_edges:
        return 0, 0.0, []
    from trimesh.graph import connected_components
    loops = connected_components(np.asarray(open_edges))
    lengths = []
    for loop in loops:
        nodes = np.asarray(loop)
        lengths.append(float(np.ptp(mesh.vertices[nodes], axis=0).max()) if len(nodes) else 0.0)
    del counts
    return len(loops), float(max(lengths) if lengths else 0.0), [len(loop) for loop in loops]


def mesh_facts(mesh, label):
    components = sorted(mesh.split(only_watertight=False), key=lambda m: len(m.faces), reverse=True)
    holes, span, sizes = boundary_loops(mesh)
    return {
        "role": label, "faces": int(len(mesh.faces)), "vertices": int(len(mesh.vertices)),
        "watertight": bool(mesh.is_watertight), "component_count": len(components),
        "component_faces": [int(len(c.faces)) for c in components[:6]],
        "boundary_loop_count": holes, "boundary_span_mm": round(span, 2),
        "boundary_loop_sizes": sizes[:6],
    }


def margin_facts(margin, jaw):
    points = np.asarray(margin, float)
    from scipy.spatial import cKDTree
    distance = cKDTree(np.asarray(jaw.vertices, float)).query(points)[0]
    closed = float(np.linalg.norm(points[0] - points[-1])) if len(points) > 1 else None
    return {
        "points": int(len(points)),
        "closed_max_gap_mm": None if closed is None else round(closed, 4),
        "on_surface_max_mm": round(float(distance.max()), 4),
        "on_surface_median_mm": round(float(np.median(distance)), 4),
        "scale_mm": [round(float(v), 2) for v in np.ptp(points, axis=0)],
        "finite": bool(np.isfinite(points).all()),
    }


def run_case(client, entry, *, full: bool):
    from app.config import CASE_DIR
    jaw_file, antagonist_file = resolve(entry)
    report = {"case": entry["case"], "tooth": entry["tooth"],
              "jaw_file": jaw_file.name, "antagonist_file": antagonist_file.name}
    jaw = trimesh.load(jaw_file, force="mesh")
    antagonist = trimesh.load(antagonist_file, force="mesh")
    report["jaw"] = mesh_facts(jaw, "working_jaw")
    report["antagonist"] = mesh_facts(antagonist, "antagonist")

    case_id = client.post("/api/cases", json={"name": "real-abutment-%d" % entry["tooth"]}).json()["id"]
    for kind, path in (("upper", antagonist_file), ("lower", jaw_file)):
        response = client.post(f"/api/cases/{case_id}/assets/{kind}",
                               files={"file": (path.name, path.read_bytes(), "model/stl")})
        assert response.status_code == 200, (kind, response.text)

    candidates = client.get(f"/api/cases/{case_id}/implant-candidates?source=lower")
    report["implant_candidates"] = {"status": candidates.status_code,
                                    "body": candidates.json() if candidates.status_code == 200 else candidates.text}
    candidate_list = (candidates.json() or {}).get("candidates") or [] if candidates.status_code == 200 else []

    # **Abutment Support First**：不预先建 ROI 支撑（那会造出第二个 support 资产，
    # 与 draft-preparation 自己算终止线所用的网格不是同一张）。支撑由 draft-preparation 提取。
    report["extract_support"] = {"status": "DELEGATED_TO_DRAFT_PREPARATION",
                                 "body": "支撑资产与终止线在同一步、同一张网格上产生"}

    design = client.post("/api/designs", json={"draft": True, "case_id": case_id,
                                               "target_fdi": entry["tooth"], "support_type": "abutment"})
    assert design.status_code == 200, design.text
    path = "/api/designs/%s" % design.json()["id"]
    prepared = client.post(path + "/draft-preparation",
                           json={"radius_mm": 7, "restorative_abutment_confirmed": True})
    report["draft_preparation"] = {"status": prepared.status_code,
                                   "body": prepared.json() if prepared.status_code == 200 else prepared.text}
    if prepared.status_code == 200:
        body = prepared.json()
        margin = body.get("points")
        if margin:
            report["margin"] = margin_facts(margin, jaw)
        report["axis"] = body.get("axis")
        report["implant_candidate"] = body.get("implant_candidate")

    contract = client.get(path + "/prep-surface-contract")
    report["support_contract"] = contract.json() if contract.status_code == 200 else {"status": contract.status_code,
                                                                                     "body": contract.text}
    report["status_path"] = path
    report["case_id"] = case_id
    detail = client.get(f"/api/cases/{case_id}").json()
    supports = [a for a in detail.get("assets", []) if a["kind"] == "support"]
    report["support_assets"] = [{"id": a["id"], "original_name": a.get("original_name"),
                                 "faces": (json.loads(a.get("mesh_info_json") or "{}").get("faces")),
                                 "created_at": a.get("created_at")} for a in supports]
    if prepared.status_code == 200:
        support_entry = sorted(supports, key=lambda a: a.get("created_at") or "", reverse=True)[0] if supports else None
        if support_entry:
            info = json.loads(support_entry.get("mesh_info_json") or "{}")
            report["support_used"] = {"original_name": support_entry.get("original_name"),
                                      "vertices": info.get("vertices"), "faces": info.get("faces")}
            report["extraction_diagnostics"] = diagnose_extraction(
                support_entry["path"], prepared.json()["points"], prepared.json()["axis"])
    if full and prepared.status_code == 200:
        report["full_flow"] = run_full_flow(client, path, prepared.json(), entry)
    return report, client, path, jaw_file, antagonist_file, full


def run_full_flow(client, path, prepared_body, entry):
    """确认边缘线 / 插入轴 / 参数 / 方向帧 → 生成牙冠 → 语义区域 → Final Crown STL。"""
    import numpy as np

    def digest():
        return client.get(path + "/status").json()["input_digest"]

    steps = {}
    margin = prepared_body["points"]
    axis = prepared_body["axis"]

    def confirm(subpath, payload):
        response = client.put(f"{path}/auto-fit/{subpath}",
                              json={**payload, "expected_input_digest": digest(), "confirmed": True})
        steps[subpath] = {"status": response.status_code,
                          "body": response.json() if response.status_code == 200 else response.text[:200]}
        return response

    steps["margin_save"] = {"status": client.post(path + "/margin", json={"points": margin}).status_code}
    steps["axis_save"] = {"status": client.post(path + "/axis", json={"axis": axis}).status_code}
    profile = client.put(path + "/profile", json={"material": "real abutment e2e", "coverage": "ENGINEERING_ONLY",
                                                  "bite_confirmed": True, "confirmed": True,
                                                  "parameters": {"machine_axes": 5, "minimum_thickness_mm": 0.5,
                                                                 "proximal_target_mm": 0.05, "occlusal_target_mm": 0.05}})
    steps["profile"] = {"status": profile.status_code, "body": profile.text[:200] if profile.status_code != 200 else "OK"}

    from app.geometry.crown_outer.patient_frame import build_patient_frame
    frame = build_patient_frame(np.asarray(margin, float), np.asarray(axis, float))
    confirm("frame", {"mesial": frame.x.tolist(), "buccal": frame.y.tolist(), "units": "mm"})
    confirm("bite", {"transform": np.eye(4).tolist(), "source_method": "REGISTERED_EXPORT", "units": "mm"})

    contract = client.get(path + "/prep-surface-contract").json()
    steps["contract_after_confirm"] = {"readiness": contract["classification"]["readiness"],
                                       "rejection_code": contract["classification"]["rejection_code"],
                                       "component_count": contract["geometry"].get("component_count"),
                                       "boundary_loop_count": contract["geometry"].get("boundary_loop_count")}
    steps["contract_diagnostics"] = (contract.get("diagnostics") or {})

    placed = client.post(path + "/outer/auto", json={"margin_points": margin, "insertion_axis": axis})
    steps["outer_auto"] = {"status": placed.status_code, "body": placed.text[:200] if placed.status_code != 200 else "OK"}
    if placed.status_code != 200:
        return steps
    crown = trimesh.load(client.get(path).json()["outer_path"], force="mesh")
    steps["crown"] = mesh_facts(crown, "crown")

    session = client.get(path + "/sculpt-session").json()
    steps["regions"] = session["region_map"].get("counts") if isinstance(session["region_map"], dict) else None
    steps["region_flags"] = session["region_map"].get("flags")

    status = client.get(path + "/status").json()
    final = client.post(path + "/finalize", json={"expected_input_digest": status["input_digest"],
                                                  **status["confirmed_revisions"]})
    steps["finalize"] = {"status": final.status_code, "has_qc": bool(final.status_code == 200 and final.json().get("qc")),
                         "body": None if final.status_code == 200 else final.text}
    stl = client.get(path + "/final.stl")
    steps["export"] = {"status": stl.status_code, "bytes": len(stl.content) if stl.status_code == 200 else 0}
    if stl.status_code == 200:
        from pathlib import Path as _Path
        import tempfile as _tempfile
        target = _Path(tempfile.gettempdir()) / ("final_crown_%d_%s.stl" % (entry["tooth"], entry["case"]))
        target.write_bytes(stl.content)
        mesh = trimesh.load(target, force="mesh")
        steps["export"]["valid_mesh"] = bool(mesh.is_watertight or len(mesh.faces) > 100)
        steps["export"]["faces"] = int(len(mesh.faces))
        steps["export"]["path"] = str(target)
    return steps


def diagnose_extraction(support_path, margin, axis):
    """复刻 extract_abutment_support 的判据并**只读**报告候选支撑区域。

    对每个"切完的连通块 + 其边界环"给出：面积、质心、到就位轴距离、边界到确认终止线的
    p95/最大偏差 —— 用来分辨 0 匹配 / 多匹配（SUPPORT_EXTRACTION_AMBIGUOUS）到底是怎么来的。
    """
    from app.geometry.io import load_mesh
    from app.geometry.nonplanar_crown import MarginCurtain
    from app.geometry.crown_outer.collar_body_outer import _clip_by_scalar, _boundary_loops
    from app.geometry.abutment_support import margin_ribbon_field
    from app.geometry.margin_seam import _polyline_distances
    from trimesh.proximity import closest_point
    import trimesh as _tm

    support = load_mesh(Path(support_path))
    curtain = MarginCurtain(np.asarray(margin, float), np.asarray(axis, float))
    curve = curtain.points
    _, dist, margin_faces = closest_point(support, curve)
    facts = {"support_faces": int(len(support.faces)), "margin_max_dev_mm": round(float(dist.max()), 4),
             "margin_faces": int(len(np.unique(margin_faces)))}
    labels_src = _tm.graph.connected_component_labels(support.face_adjacency, node_count=len(support.faces))
    identities = np.unique(labels_src[margin_faces])
    facts["support_components_touching_margin"] = [int(i) for i in identities]
    if len(identities) != 1:
        facts["candidates"] = []
        facts["verdict"] = "MULTIPLE_SUPPORT_COMPONENTS"
        return facts
    chosen = int(identities[0])
    source_faces = np.flatnonzero(labels_src == chosen)
    phi, _ = margin_ribbon_field(support, curve)
    vertices, faces, _, mapping = _clip_by_scalar(support.vertices, support.faces[source_faces],
                                                  -phi, return_face_sources=True)
    cut = _tm.Trimesh(vertices, faces, process=False)
    labels = _tm.graph.connected_component_labels(cut.face_adjacency, node_count=len(faces))
    facts["cut_component_count"] = int(len(np.unique(labels)))
    candidates = []
    for label in np.unique(labels):
        ids = np.flatnonzero(labels == label)
        centred = cut.triangles_center[ids] - curtain.frame.origin
        local = centred @ curtain.frame.matrix()
        entry = {"label": int(label), "faces": int(len(ids)), "area_mm2": round(float(cut.area_faces[ids].sum()), 3),
                 "centroid_local": [round(float(v), 2) for v in local.mean(axis=0)],
                 "distance_to_axis_mm": round(float(np.linalg.norm(local.mean(axis=0)[:2])), 3),
                 "loops": []}
        for loop in _boundary_loops(faces[ids]):
            error = _polyline_distances(vertices[loop], curve)
            entry["loops"].append({"points": int(len(loop)),
                                   "to_margin_p95_mm": round(float(np.quantile(error, .95)), 4),
                                   "to_margin_max_mm": round(float(error.max()), 4),
                                   "matches_margin": bool(np.quantile(error, .95) < .05 and error.max() < .1)})
        candidates.append(entry)
    facts["candidates"] = candidates
    matches = [c for c in candidates if any(l["matches_margin"] for l in c["loops"])]
    facts["match_count"] = len(matches)
    facts["verdict"] = ("OK" if len(matches) == 1 else
                        "NO_MATCH（切完没有任何区域的边界与确认终止线重合）" if not matches else
                        "MULTIPLE_MATCH（多个区域都声称对应终止线）")
    return facts


def main():
    full = "--full" in sys.argv
    with tempfile.TemporaryDirectory() as tmp:
        isolate(Path(tmp))
        from fastapi.testclient import TestClient
        from app.main import app
        results = []
        with TestClient(app) as client:
            for entry in REAL_ABUTMENT_CASES:
                print("\n===== %s（FDI %d）=====" % (entry["case"], entry["tooth"]))
                report, client, path, jaw_file, antagonist_file, full = run_case(client, entry, full=full)
                results.append(report)
                print("工作颌     :", json.dumps(report["jaw"], ensure_ascii=False))
                print("对颌       :", json.dumps(report["antagonist"], ensure_ascii=False))
                print("种植候选   :", json.dumps(report["implant_candidates"], ensure_ascii=False)[:400])
                print("提取支撑   :", json.dumps(report["extract_support"], ensure_ascii=False)[:300])
                print("draft      :", json.dumps(report["draft_preparation"], ensure_ascii=False)[:300])
                print("Margin     :", json.dumps(report.get("margin"), ensure_ascii=False))
                print("支撑合同   :", json.dumps(report["support_contract"], ensure_ascii=False)[:400])
        # 把 debug 网格（红=自交面 / 黄=邻近 / 绿=正常）从临时目录拷到稳定目录，便于直接打开
        import shutil
        source = Path(tmp) / "debug"
        if source.is_dir():
            target = Path(__file__).resolve().parents[1] / "debug_out"
            target.mkdir(parents=True, exist_ok=True)
            shutil.copytree(source, target, dirs_exist_ok=True)
            print("\n调试网格已复制到:", target)
        # 整份运行产物（support / 牙冠 STL / QC / 状态）也拷出来，便于直接查看当前生成情况
        run_out = Path(__file__).resolve().parents[1] / "run_out"
        if run_out.exists():
            shutil.rmtree(run_out)
        shutil.copytree(Path(tmp), run_out, dirs_exist_ok=True)
        print("运行产物已复制到:", run_out)
        print("\n===== 汇总 =====")
        for report in results:
            contract = report["support_contract"] or {}
            print("%-16s margin=%-5s readiness=%-24s rejection=%-28s components=%s loops=%s" % (
                report["case"], (report.get("margin") or {}).get("points"),
                contract.get("readiness"), contract.get("rejection_code"),
                contract.get("component_count"), contract.get("boundary_loop_count")))
        Path("diag_support_extraction.json").write_text(json.dumps(results, ensure_ascii=False, indent=2, default=str),
                                                        encoding="utf-8")
        print("\n明细已写入 backend/diag_support_extraction.json")


if __name__ == "__main__":
    main()
