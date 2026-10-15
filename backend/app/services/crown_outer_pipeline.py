"""CrownOuter 全链路封装：把已冻结的几何引擎接成**可调用的阶段流水线**。

链路（每一段都是已验收的引擎，此处只做编排与产物落盘）：

    outer_h1        CrownOuter v0 + H.1 缩放解耦            → crown_h1.stl
    intaglio        预备体（代型）表面 + cement gap          → intaglio.stl
    contact         Contact v2 → I.1 → I.2（SDF 投影）        → crown_contact.stl
    occlusion       Occlusion v1（对颌碰撞消除）              → crown_occluded.stl
    thickness_audit Thickness v1（分区账 + 冲突归因，只测不修）→ audit.json + 热力图
    envelope_j1     J.1 Intaglio-Constrained（**实验**，负结果）  → crown_j1.stl
    collar_body     J.2.1 Fixed Collar + Implicit Body（**实验**）→ crown_j21.stl

**分层显示**（参考 3Shape 的"显示器"面板）由审计/接触/咬合的量直接生成顶点色 GLB：
平面 / 薄区域（内表面）/ 邻牙接触（外表面）/ 咬合间隙（外表面）。

输入 = 一个"病例包"目录（PHI-free：case_uuid + case.json + 网格）。
本模块只依赖 numpy/scipy/trimesh + app 内的几何引擎；**不写死任何临床默认值**
（阈值全部来自传入参数，缺省值只作为 fallback 并在 manifest 里标 provenance）。
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..geometry.crown_outer import (
    CrownOuterConfig,
    generate_crown_outer_v0,
    load_crown_template,
)
from ..geometry.crown_outer.contact import (
    harden_contacts,
    ContactParameters,
    NeighborPatch,
    optimize_contacts,
)
from ..geometry.crown_outer.contact_i2 import (
    I2Params,
    solve_proximal_projection,
)
from ..geometry.crown_outer.contact_refine import RefineParams, refine_contacts
from ..geometry.crown_outer.collar_body_outer import (
    CollarBodyParams,
    build_collar_body_outer,
)
from ..geometry.crown_outer.gt_compare import extract_outer_surface
from ..geometry.crown_outer.intaglio_constrained import (
    OuterV2Params,
    build_intaglio_constrained_outer,
)
from ..geometry.crown_outer.occlusion import (
    OcclusionParams,
    solve_occlusion_collision,
)
from ..geometry.crown_outer.patient_frame import PatientFrame
from ..geometry.crown_outer.thickness import (
    ThicknessParameters,
    audit_thickness,
    intaglio_from_prep,
)
from ..importers.tooth_library import discover_libraries

STAGES = ("outer_h1", "intaglio", "contact", "occlusion", "thickness_audit",
          "envelope_j1", "collar_body")
#: 已冻结/已验证的主链；后两个是研究阶段（负结果/未完成），前端标注"实验"
MAIN_STAGES = ("outer_h1", "intaglio", "contact", "occlusion", "thickness_audit")
EXPERIMENTAL_STAGES = ("envelope_j1", "collar_body")

#: 设计参数的缺省值（**不是临床建议**：仅当调用方没给时才用，manifest 会标 provenance）
DEFAULT_PARAMS = {
    "cement_gap_mm": 0.08,
    "min_thickness_mm": 0.60,
    "near_margin_thickness_mm": 0.20,
    "near_margin_distance_mm": 0.80,
    "near_margin_mode": "linear",
    "contact_gap_mm": 0.05,
    "occlusal_gap_mm": 0.05,
    "tooth_library": "炼金王",
}


class PipelineError(RuntimeError):
    pass


@dataclass
class CrownBundle:
    """一个病例包（路径级输入）。"""

    case_uuid: str
    case_dir: Path
    margin_points: np.ndarray
    insertion_axis: np.ndarray
    fdi: int
    prep_path: Path | None = None
    die_path: Path | None = None
    mesial_path: Path | None = None
    distal_path: Path | None = None
    antagonist_path: Path | None = None
    gt_crown_path: Path | None = None
    axis_mesial: np.ndarray | None = None
    axis_buccal: np.ndarray | None = None
    center: np.ndarray | None = None
    meta: dict = field(default_factory=dict)

    def frame(self) -> PatientFrame:
        return patient_frame_from_meta(self.meta)


def patient_frame_from_meta(meta: dict) -> PatientFrame:
    """与 tools/run_contact_v1_case.py 同一约定：X = AxisMesial ⊥ Z（不用 PCA）。"""
    margin = np.asarray(meta["margin_points"], dtype=float)
    z = np.asarray(meta["insertion_axis"], dtype=float)
    z = z / np.linalg.norm(z)
    m = np.asarray(meta["axis_mesial"], dtype=float)
    m = m - (m @ z) * z
    x = m / np.linalg.norm(m)
    y = np.cross(z, x)
    y = y / np.linalg.norm(y)
    return PatientFrame(origin=margin.mean(axis=0), z=z, x=x, y=y)


def load_bundle(case_dir: Path | str) -> CrownBundle:
    """读病例包目录（case.json + 各 STL；缺哪个就记 None，阶段里再判）。"""
    case_dir = Path(case_dir)
    meta = json.load(open(case_dir / "case.json", encoding="utf-8"))
    meta.setdefault("case_uuid", case_dir.name)
    meta.setdefault("margin_points", meta.get("margin_points"))
    if not meta.get("margin_points"):
        raise PipelineError("case.json 缺 margin_points")
    if not meta.get("insertion_axis"):
        raise PipelineError("case.json 缺 insertion_axis")

    def pick(*names):
        for n in names:
            p = case_dir / n
            if p.exists():
                return p
        return None

    return CrownBundle(
        case_uuid=case_dir.name,
        case_dir=case_dir,
        margin_points=np.asarray(meta["margin_points"], dtype=float),
        insertion_axis=np.asarray(meta["insertion_axis"], dtype=float),
        fdi=int(meta.get("fdi") or 46),
        prep_path=pick("prep.stl"),
        die_path=pick("die.stl"),
        mesial_path=pick("mesial_neighbor_patch.stl"),
        distal_path=pick("distal_neighbor_patch.stl"),
        antagonist_path=pick("antagonist_patch.stl"),
        gt_crown_path=pick("final_crown.stl"),
        axis_mesial=np.asarray(meta["axis_mesial"], float)
        if meta.get("axis_mesial") is not None else None,
        axis_buccal=np.asarray(meta["axis_buccal"], float)
        if meta.get("axis_buccal") is not None else None,
        center=np.asarray(meta["center"], float)
        if meta.get("center") is not None else None,
        meta=meta,
    )


def merged_params(overrides: dict | None) -> tuple[dict, dict]:
    """合并参数并给出 provenance（哪些是调用方给的、哪些走了缺省）。"""
    p = dict(DEFAULT_PARAMS)
    given = {}
    for k, v in (overrides or {}).items():
        if k in p and v is not None:
            p[k] = v
            given[k] = v
    provenance = {"given": sorted(given), "defaulted": sorted(
        k for k in DEFAULT_PARAMS if k not in given)}
    return p, provenance


# ---------------------------------------------------------------- 分层热力图

_CMAP = {
    "thin": ((0.86, 0.20, 0.16), (0.97, 0.80, 0.20), (0.24, 0.72, 0.40)),
    "contact": ((0.88, 0.22, 0.18), (0.95, 0.75, 0.25), (0.25, 0.70, 0.45)),
    "occlusal": ((0.90, 0.25, 0.15), (0.95, 0.80, 0.25), (0.30, 0.68, 0.45)),
}


def _lerp3(c0, c1, t):
    t = np.clip(t, 0.0, 1.0)[:, None]
    return np.asarray(c0)[None, :] * (1 - t) + np.asarray(c1)[None, :] * t


def _ramp(values, vmin, vmax, key):
    stops = _CMAP[key]
    t = (np.asarray(values, float) - vmin) / max(vmax - vmin, 1e-9)
    t = np.clip(t, 0.0, 1.0)
    lo = _lerp3(stops[0], stops[1], t * 2.0)
    hi = _lerp3(stops[1], stops[2], (t - 0.5) * 2.0)
    return np.where((t < 0.5)[:, None], lo, hi)


def write_layer_glb(mesh_v, mesh_f, colors, path: Path):
    """顶点色 GLB（前端用 GLTFLoader 直接叠在 STL 上做热力图）。"""
    import trimesh
    m = trimesh.Trimesh(np.asarray(mesh_v, float), np.asarray(mesh_f, np.int64),
                        process=False)
    rgba = np.concatenate([np.clip(colors, 0, 1),
                           np.ones((len(colors), 1))], axis=1)
    m.visual = trimesh.visual.ColorVisuals(m, vertex_colors=(rgba * 255)
                                           .astype(np.uint8))
    m.export(path)


# ---------------------------------------------------------------- 各阶段

def _mesh(path):
    import trimesh
    return trimesh.load(str(path), force="mesh")


def _h1_template(params: dict, fdi: int):
    libs = {l.name: l for l in discover_libraries()}
    name = params.get("tooth_library") or DEFAULT_PARAMS["tooth_library"]
    if name not in libs:
        name = sorted(libs)[0]
    return load_crown_template(libs[name], fdi), name


def stage_outer_h1(bundle: CrownBundle, out_dir: Path, params: dict,
                   prev: dict) -> dict:
    """CrownOuter v0 + H.1（模板 → 患者 Margin 的拉普拉斯形变）。"""
    template, lib_name = _h1_template(params, bundle.fdi)
    res = generate_crown_outer_v0(template, bundle.margin_points,
                                  bundle.insertion_axis,
                                  config=CrownOuterConfig(
                                      body_scale_decoupling=True))
    if res.mesh is None:
        raise PipelineError("Crown Outer 生成失败：%s" % (res.notes or res.status))
    out = out_dir / "crown_h1.stl"
    res.mesh.export(str(out))
    qc = {k: v for k, v in (res.mesh_qc or {}).items()}
    return {
        "artifact": str(out),
        "mesh": {"n_vertices": int(len(res.mesh.vertices)),
                 "n_faces": int(len(res.mesh.faces))},
        "template": {"library": lib_name, "fdi": bundle.fdi,
                     "margin_source": template.margin_source},
        "uniform_scale": float(res.uniform_scale or 1.0),
        "boundary_shift": int(getattr(res, "boundary_shift", 0) or 0),
        "qc": qc,
        "status": res.status,
    }


def stage_intaglio(bundle: CrownBundle, out_dir: Path, params: dict,
                   prev: dict) -> dict:
    """内表面 = 预备体表面 + cement gap。

    优先用病例包里的 prep.stl；只有代型（die）时必须先切出 Margin 以上的预备体面。
    """
    from scipy.spatial import cKDTree
    gap = float(params["cement_gap_mm"])
    src = bundle.prep_path or bundle.die_path
    if src is None:
        raise PipelineError("病例包缺 prep.stl / die.stl —— 无法生成内表面")
    mesh = _mesh(src)
    v = np.asarray(mesh.vertices, float)
    f = np.asarray(mesh.faces, np.int64)
    meta = dict(bundle.meta)
    meta.setdefault("margin_points", bundle.margin_points.tolist())
    axis = bundle.insertion_axis / np.linalg.norm(bundle.insertion_axis)

    prep_note = "病例包自带 prep.stl（已是对齐后的预备体表面）"
    if bundle.prep_path is None:
        prep_note = "从代型切出 Margin 以上区域（轴向偏移 + 连通分量）"
        tree = cKDTree(bundle.margin_points)
        _d, idx = tree.query(v, k=1)
        near = bundle.margin_points[idx]
        s = np.einsum("ij,j->i", v - near, axis)
        above = s > -0.02
        adj: dict = {}
        for a, b, c in f:
            for u, w in ((int(a), int(b)), (int(b), int(c)), (int(c), int(a))):
                adj.setdefault(u, set()).add(w)
                adj.setdefault(w, set()).add(u)
        apex = int(np.argmax(v @ axis))
        if not above[apex]:
            raise PipelineError("代型最上方顶点不在 Margin 之上 —— 轴向或 margin 有误")
        seen, stack = {apex}, [apex]
        while stack:
            u = stack.pop()
            for w in adj.get(u, ()):
                if above[w] and w not in seen:
                    seen.add(w)
                    stack.append(w)
        vm = np.zeros(len(v), bool)
        vm[sorted(seen)] = True
        fm = vm[f].all(axis=1)
        v = v[vm]
        remap = np.full(len(mesh.vertices), -1, np.int64)
        remap[np.flatnonzero(vm)] = np.arange(int(vm.sum()))
        f = remap[f[fm]]

    iv, ifa, info = intaglio_from_prep(v, f, gap, axis=axis)
    out = out_dir / "intaglio.stl"
    import trimesh
    trimesh.Trimesh(iv, ifa, process=False).export(str(out))
    return {
        "artifact": str(out),
        "source": str(src),
        "cement_gap_mm": gap,
        "method": prep_note,
        "normal_flipped": bool(info.get("normal_flipped")),
        "mesh": {"n_vertices": int(len(iv)), "n_faces": int(len(ifa))},
    }


def _patches(bundle: CrownBundle):
    if bundle.mesial_path is None or bundle.distal_path is None:
        return None
    return {"mesial": NeighborPatch(*(lambda m: (m.vertices, m.faces))(
                _mesh(bundle.mesial_path))),
            "distal": NeighborPatch(*(lambda m: (m.vertices, m.faces))(
                _mesh(bundle.distal_path)))}


#: 研究管线里对颌/邻牙朝向的**声明标签**：这是算法假设，不是人工复核。
ALGORITHMIC_OUTWARD_DECLARATION = "ALGORITHMIC_CENTROID_ASSUMPTION_NOT_HUMAN_REVIEWED"


def _declare_outward(reference, crown_v) -> list[float]:
    """按质心方向**声明**一个外向法向并返回（O1 起必须显式声明，不再由求解器自己猜）。

    返回值喂给 ``solve_occlusion_collision(..., antagonist_outward=)``；
    报告里同时标注 :data:`ALGORITHMIC_OUTWARD_DECLARATION`，避免被误当成人工确认。
    """
    centroid = np.asarray(crown_v, float).mean(axis=0)
    fc = reference.vertices[reference.faces].mean(axis=1)
    w = np.einsum("ij,ij->i", centroid - fc, reference.normals)
    sign = 1.0 if w.sum() > 0 else -1.0
    normal = reference.normals.mean(axis=0)
    normal = normal / max(float(np.linalg.norm(normal)), 1e-12) * sign
    return [float(v) for v in normal]


def _orient_patches(patches, crown_v):
    centroid = np.asarray(crown_v, float).mean(axis=0)
    for patch in patches.values():
        fc = patch.vertices[patch.faces].mean(axis=1)
        w = np.einsum("ij,ij->i", centroid - fc, patch.normals)
        patch.normal_sign = 1.0 if w.sum() > 0 else -1.0


def stage_contact(bundle: CrownBundle, out_dir: Path, params: dict,
                  prev: dict) -> dict:
    """Contact v2（局部带硬锁）→ I.1 精修 → I.2（SDF 一步投影）。

    **硬化（Phase K，用户 2026-10-08 规格）**：每一步都过几何硬门
    （新增自交 = 0 / 非流形不新增 / 翻面 = 0 / margin 与咬合锁定区位移 = 0）；
    α line search 全失败 → CONTACT_CONSTRAINT_CONFLICT，绝不折叠换清零。
    产出固定三件：contact_hardened.stl / contact_report.json /
    geometry_qc_after_contact.json。
    """
    patches = _patches(bundle)
    if patches is None:
        raise PipelineError("病例包缺近中/远中邻牙 patch —— 接触阶段无法运行")
    start = Path(prev["outer_h1"]["artifact"])
    h1 = _mesh(start)
    _orient_patches(patches, h1.vertices)
    frame = bundle.frame()

    # 形变带宽：H.1 在邻接面需要 ~1mm 级收窄，3 环过渡太窄必然折叠
    # （实测 α=1 翻面 167 片）。加宽到 8 环让应变摊开 —— 这是**带宽**参数，
    # 不是把几何门的容差放水。
    # ⚠️ I.1（refine_contacts）会**原地修改传入网格**：若不再开一份快照，
    # 后面所有"相对 h1 的位移"都会量成 0（自己跟自己比）。
    import trimesh as _tm
    h1_before = _tm.Trimesh(vertices=np.array(h1.vertices, dtype=float),
                            faces=np.array(h1.faces, dtype=np.int64),
                            process=False)
    rings = int(params.get("contact_transition_rings", 8))
    hard = harden_contacts(h1, frame, patches["mesial"], patches["distal"],
                           ContactParameters(
                               target_gap_mm=float(params["contact_gap_mm"]),
                               design_target_mm=0.0,
                               transition_rings=rings))
    v2 = hard["result"]
    # v2 输出冻结一份：I.1 会**原地改输入网格**，不冻结的话下面所有
    # "相对 v2 的位移/几何门"都会变成自己跟自己比（实测放过 0.156mm 拖动）。
    v2_frozen = _tm.Trimesh(vertices=np.array(v2.mesh.vertices, dtype=float),
                            faces=np.array(v2.mesh.faces, dtype=np.int64),
                            process=False)
    def _fresh(mesh):
        """求解器隔离副本：本阶段已实测有子步会**原地写输入网格**，
        共享同一份会让后续锁定集判据悄悄漂移（放过 0.156mm 拖动）。"""
        return _tm.Trimesh(vertices=np.array(mesh.vertices, dtype=float),
                           faces=np.array(mesh.faces, dtype=np.int64),
                           process=False)

    i1 = refine_contacts(_fresh(v2.mesh), frame, patches["mesial"],
                         patches["distal"],
                         RefineParams(target_gap_mm=float(params["contact_gap_mm"])),
                         start_from=_fresh(v2.mesh))
    # **I.1 同样要过不变量**：实测它会把咬合锁定面拖 0.156mm（v1 教训的缩小版）。
    # 违反就丢弃 I.1 的结果、保留 v2 输出，并如实记账（不是静默忽略）。
    i1_gate = _invariants_only(v2_frozen, i1.mesh, frame, patches, params)
    if not i1_gate["ok"]:
        i1_used = False
        stage_notes_i1 = ("I.1 被丢弃：%s（保留 Contact v2 输出）"
                          % "；".join(i1_gate["violations"]))
    else:
        i1_used = True
        stage_notes_i1 = "I.1 通过不变量"
    occl = None
    if bundle.antagonist_path is not None:
        occl = NeighborPatch(*(lambda m: (m.vertices, m.faces))(
            _mesh(bundle.antagonist_path)))
        fc = occl.vertices[occl.faces].mean(axis=1)
        w = np.einsum("ij,ij->i", np.asarray(i1.mesh.vertices, float).mean(axis=0) - fc,
                      occl.normals)
        occl.normal_sign = 1.0 if w.sum() > 0 else -1.0
    i1_mesh = _fresh(i1.mesh if i1_used else v2.mesh)
    i2 = solve_proximal_projection(i1_mesh, frame, patches["mesial"],
                                   patches["distal"],
                                   I2Params(target_gap_mm=float(
                                       params["contact_gap_mm"])))
    out = out_dir / "crown_contact.stl"
    hardened_out = out_dir / "contact_hardened.stl"
    i2.mesh.export(str(out))
    i2.mesh.export(str(hardened_out))
    # ---- Phase K 汇报：逐侧 editable p50/p95/max、残留去向、最终几何 QC
    final_rep = _contact_report(h1_before, i2.mesh, frame, patches, params)
    geom_qc = _absolute_geometry_qc(i2.mesh, params)
    (out_dir / "contact_report.json").write_text(
        _json_dump(final_rep), encoding="utf-8")
    (out_dir / "geometry_qc_after_contact.json").write_text(
        _json_dump(geom_qc), encoding="utf-8")
    gate = {
        "geometry_clean": bool(geom_qc["ok"]),
        "self_intersections": geom_qc["metrics"].get("self_intersections"),
        "nonmanifold_edges": geom_qc["metrics"].get("nonmanifold_edges"),
        "flipped_faces": geom_qc["metrics"].get("flipped_faces"),
        "margin_displacement_mm": final_rep["margin_disp_mm"],
        "occlusal_locked_displacement_mm": final_rep["occlusal_locked_disp_mm"],
        "residual_editable_p95_mm": final_rep["editable"]["p95"],
        "residual_allow_mm": final_rep["residual_allow_mm"],
        "within_allowance": bool(final_rep["editable"]["p95"]
                                 <= final_rep["residual_allow_mm"]),
        # 用户 K Gate 的另外两条硬不变量（位移必须精确 0）
        "margin_preserved": bool(final_rep["margin_disp_mm"] == 0.0),
        "occlusal_locked_preserved": bool(
            final_rep["occlusal_locked_disp_mm"] == 0.0),
    }
    if not gate["margin_preserved"] or not gate["occlusal_locked_preserved"]:
        gate["status"] = "REVIEW_CONTACT"
    elif hard["conflict"]:
        gate["status"] = "CONTACT_CONSTRAINT_CONFLICT"
    elif not gate["geometry_clean"] or not gate["within_allowance"]:
        gate["status"] = "REVIEW_CONTACT"
    else:
        gate["status"] = "PASS"
    # ⚠️ I2Result 没有 summary()（只有 ContactResult/RefineResult 有）——
    # 直接读字段，别再签名猜 API。
    last = (i2.history or [{}])[-1]
    return {
        "artifact": str(out),
        "hardened_artifact": str(hardened_out),
        "hardening": {"alpha": hard["alpha"], "conflict": hard["conflict"],
                      "trials": hard["trials"],
                      "note": hard.get("note")},
        "gate": gate,
        "report": final_rep,
        "geometry_qc": geom_qc,
        "v2": v2.summary(),
        "i1": {"status": i1.status, "used": i1_used,
               "gate": i1_gate, "note": stage_notes_i1,
               "margin_max_displacement_mm": i1.margin_max_displacement_mm,
               "occlusal_p95_displacement_mm": i1.occlusal_p95_displacement_mm,
               "n_flipped_faces": i1.n_flipped_faces},
        "i2": {"status": i2.status, "iterations": i2.iterations,
               "alphas_accepted": i2.alphas_accepted,
               "margin_max_displacement_mm": i2.margin_max_displacement_mm,
               "occlusal_p95_displacement_mm": i2.occlusal_p95_displacement_mm,
               "n_flipped_faces": i2.n_flipped_faces,
               "proximal_shrink": i2.proximal_shrink,
               "last_round": {k: v for k, v in last.items()
                              if k != "iteration"} if last else {}},
        "handoff_to_occlusion": i2.handoff_to_occlusion,
        "notes": i2.notes,
        "geometry": {"n_vertices": int(len(i2.mesh.vertices)),
                     "n_faces": int(len(i2.mesh.faces))},
    }


def stage_occlusion(bundle: CrownBundle, out_dir: Path, params: dict,
                    prev: dict) -> dict:
    """Occlusion v1：对颌碰撞消除（只动咬合面，颈缘/邻接核心由构造锁死）。"""
    if bundle.antagonist_path is None:
        raise PipelineError("病例包缺 antagonist_patch.stl —— 咬合阶段无法运行")
    ant = NeighborPatch(*(lambda m: (m.vertices, m.faces))(
        _mesh(bundle.antagonist_path)))
    crown = _mesh(Path(prev["contact"]["artifact"]))
    # O1：符号来源必须显式声明（闭合实体由求解器判内外；开放裁片由这里声明法向）。
    antagonist_outward = _declare_outward(ant, crown.vertices)
    patches = _patches(bundle)
    if patches:
        _orient_patches(patches, crown.vertices)
    contact_outward = ({side: _declare_outward(patch, crown.vertices)
                        for side, patch in patches.items()} if patches else None)
    res = solve_occlusion_collision(
        crown, bundle.frame(), ant,
        contact_patches=patches,
        params=OcclusionParams(target_gap_mm=float(params["occlusal_gap_mm"])),
        antagonist_outward=antagonist_outward,
        contact_outward=contact_outward)
    out = out_dir / "crown_occluded.stl"
    res.mesh.export(str(out))
    return {
        "artifact": str(out),
        "status": res.status,
        "sign_method": res.sign_method,
        "signed_available": res.signed_available,
        "outward_declaration": ALGORITHMIC_OUTWARD_DECLARATION,
        "iterations": int(res.iterations),
        "history": res.history,
        "contact_preservation": res.contact_preservation,
        "margin_max_displacement_mm": float(res.margin_max_displacement_mm),
        "proximal_core_max_displacement_mm": float(
            res.proximal_core_max_displacement_mm),
        "n_flipped_faces": int(res.n_flipped_faces),
        "notes": res.notes,
    }


def stage_thickness_audit(bundle: CrownBundle, out_dir: Path, params: dict,
                          prev: dict) -> dict:
    """Thickness v1 审计 + 冲突归因（只测不修），并产出分层热力图。"""
    outer = _mesh(Path(prev["occlusion"]["artifact"]))
    inner = _mesh(Path(prev["intaglio"]["artifact"]))
    frame = bundle.frame()
    tp = ThicknessParameters(
        min_thickness_mm=float(params["min_thickness_mm"]),
        near_margin_thickness_mm=params["near_margin_thickness_mm"],
        near_margin_distance_mm=params["near_margin_distance_mm"],
        near_margin_mode=params["near_margin_mode"],
        cement_gap_mm=float(params["cement_gap_mm"]))

    patches = _patches(bundle)
    if patches:
        _orient_patches(patches, outer.vertices)
    ant = None
    if bundle.antagonist_path is not None:
        ant = NeighborPatch(*(lambda m: (m.vertices, m.faces))(
            _mesh(bundle.antagonist_path)))
        fc = ant.vertices[ant.faces].mean(axis=1)
        w = np.einsum("ij,ij->i", np.asarray(outer.vertices, float).mean(axis=0) - fc,
                      ant.normals)
        ant.normal_sign = 1.0 if w.sum() > 0 else -1.0

    ref = None
    if (prev.get("outer_h1") or {}).get("artifact"):
        ref = _mesh(Path(prev["outer_h1"]["artifact"]))
    res = audit_thickness(
        outer, inner, frame,
        axis_mesial=bundle.axis_mesial, axis_buccal=bundle.axis_buccal,
        contact_patches=patches, antagonist_patch=ant,
        contact_target_mm=0.0, antagonist_target_mm=0.0,
        reference_mesh=ref, params=tp)

    report = {"status": res.status, "summary": res.summary,
              "input_qc": res.input_qc, "regions": res.regions,
              "notes": res.notes}
    (out_dir / "thickness_audit.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")

    # ---- 热力图：薄区域（内表面域）/ 邻牙接触 + 咬合间隙（外表面域）
    tm = res.thickness_map
    iv = np.asarray(tm["vertices"], float)
    ifaces = np.asarray(inner.faces, np.int64)
    thr = np.asarray(tm["threshold_mm"], float)
    tprim = np.asarray(tm["t_primary_mm"], float)
    # 0 = 完全没材料（红），1 = 达标（绿），>1 截断
    ratio = np.clip(tprim / np.maximum(thr, 1e-9), 0.0, 1.0)
    write_layer_glb(iv, ifaces, _ramp(ratio, 0.0, 1.0, "thin"),
                    out_dir / "layer_thin.glb")

    layers = {"thin": str(out_dir / "layer_thin.glb")}
    ov = np.asarray(outer.vertices, float)
    of = np.asarray(outer.faces, np.int64)
    if patches:
        from ..geometry.crown_outer.contact import NeighborPatch as _NP
        dist = np.full(len(ov), np.inf)
        for patch in patches.values():
            # 统一口径（决策 ③）：shader 层的间隙也用同一把尺
            signed, valid, _cp, _f, _parts = patch.clearance_detail(ov)
            dist = np.minimum(dist, np.where(valid, signed, np.inf))
        finite = np.isfinite(dist)
        v_contact = np.where(finite, dist, 1.0)
        write_layer_glb(ov, of, _ramp(v_contact, -0.30, 0.30, "contact"),
                        out_dir / "layer_contact.glb")
        layers["contact"] = str(out_dir / "layer_contact.glb")
    if ant is not None:
        _cp, _f, signed, _d, valid = ant.closest_with_validity(
            ov, boundary_safety_mm=0.6)
        v_occ = np.where(valid, np.clip(signed, -1.0, 1.0), 1.0)
        write_layer_glb(ov, of, _ramp(v_occ, -0.50, 0.50, "occlusal"),
                        out_dir / "layer_occlusal.glb")
        layers["occlusal"] = str(out_dir / "layer_occlusal.glb")

    return {"artifact": report_path_str(out_dir / "thickness_audit.json"),
            "report": report, "layers": layers,
            "region_table": res.regions,
            "status": res.status}


def report_path_str(p: Path) -> str:
    return str(p)


def _json_dump(obj) -> str:
    import json

    def _default(o):                     # numpy / Path 兜底
        try:
            return o.tolist()
        except Exception:                # noqa: BLE001
            return str(o)

    return json.dumps(obj, indent=1, ensure_ascii=False, default=_default)


def _contact_report(before_mesh, after_mesh, frame, patches, params: dict) -> dict:
    """Phase K 汇报：逐侧 editable p50/p95/max + 残留去向 + 锁定位移。"""
    import numpy as np
    from ..geometry.crown_outer.contact_i2 import I2Params, measure, _zones

    p = I2Params(target_gap_mm=float(params["contact_gap_mm"]))
    v0 = np.asarray(before_mesh.vertices, float)
    v1 = np.asarray(after_mesh.vertices, float)
    faces = np.asarray(after_mesh.faces, np.int64)
    loop_idx, loop_set, occ_t, cervical, editable = _zones(v1, faces, frame, p)
    # ⚠️ 位移不变量必须按**求解时锁定的那套集合**量：occlusal table 的判据依赖
    # 形变后的 z 分位/法向，用"形变后重算的集合"去量会把新进集合的接触区顶点
    # 误报成"拖了咬合面"（实测 0.156mm 假报警）。所以这里用 v0 重算锁定集。
    _, _, occ_t0, _, _ = _zones(v0, faces, frame, p)
    domains = {s: (1.0 if s == "mesial" else -1.0)
               * ((v1 - frame.origin) @ frame.x) >= 0.0
               for s in ("mesial", "distal")}
    m = measure(v1, faces, patches, domains, editable, occ_t, cervical,
                frame, p, vertices_ref=v0)
    both = np.concatenate([np.asarray(m[s]["ids"], dtype=np.int64)
                           for s in ("mesial", "distal")]) \
        if (len(m["mesial"]["ids"]) or len(m["distal"]["ids"])) \
        else np.array([], dtype=np.int64)
    # 残留 = 仍为负（真穿透）且未修到的顶点；按"为什么没修"归类
    occ_set = set(int(i) for i in occ_t)
    cerv_set = set(int(i) for i in np.flatnonzero(cervical))
    loop_set_ids = set(int(i) for i in loop_idx)   # 颈环拓扑级不变（原顶点）
    routed = {"occlusal_handoff": [], "cervical_handoff": [],
              "editable_residual": []}
    for i in both:
        i = int(i)
        if i in occ_set:
            routed["occlusal_handoff"].append(i)
        elif i in cerv_set or i in loop_set_ids:
            routed["cervical_handoff"].append(i)
        else:
            routed["editable_residual"].append(i)
    d_margin = float(np.max(np.linalg.norm(v1[loop_idx] - v0[loop_idx],
                                           axis=1)))
    # ⚠️ 必须用**形变前**重算的锁定集（occ_t0）：occlusal table 的判据依赖
    # z 分位与法向，两者都随形变移动 → 用形变后的 occ_t 会把"本次接触修正
    # 波进来的顶点"算成"拖了咬合面"。实测两集 1297 vs 1305（差 18 个顶点），
    # 在 v0 集上位移恒为 0.000，在 v1 集上会假报 0.156mm。
    d_occ = float(np.max(np.linalg.norm(v1[occ_t0] - v0[occ_t0], axis=1))) \
        if len(occ_t0) else 0.0
    return {
        "target_gap_mm": p.target_gap_mm,
        "editable": {k: m["mesial"]["editable"][k] for k in ("p50", "p95",
                                                             "max")},
        "per_side": {s: {"editable": m[s]["editable"],
                         "occlusal": m[s]["occlusal"],
                         "cervical": m[s]["cervical"]}
                     for s in ("mesial", "distal")},
        "residual_vertices": {k: len(v) for k, v in routed.items()},
        "residual_detail": {k: v[:20] for k, v in routed.items()},
        "margin_disp_mm": d_margin,
        "occlusal_locked_disp_mm": d_occ,
        "residual_allow_mm": float(params.get("contact_residual_allow_mm",
                                              0.20)),
        "note": ("residual_allow_mm 是**构造门限**（本阶段允许多少残留穿透），"
                 "不是临床建议；改它要显式传参并进 manifest provenance"),
    }


def _invariants_only(before_mesh, after_mesh, frame, patches,
                     params: dict) -> dict:
    """I.1 这类**未硬化**子步的门：只查不变量（位移为 0 + 零新增自交/翻面）。"""
    import numpy as np
    from ..geometry.crown_outer.contact_refine import occ_table_mask
    from ..geometry.crown_outer.deform_gate import deformation_gate

    v0 = np.asarray(before_mesh.vertices, float)
    v1 = np.asarray(after_mesh.vertices, float)
    faces = np.asarray(after_mesh.faces, np.int64)
    if len(v1) != len(v0):
        # 顶点数变了 = 重网格了 —— 本项目纪律是"不许自由 remesh"，
        # 更不可能再谈"锁定面零位移"，直接判不合格。
        return {"ok": False, "violations": ["topology_changed(%d→%d)"
                                            % (len(v0), len(v1))],
                "metrics": {}}
    from ..geometry.crown_outer.contact_i2 import _loop_of
    loop_idx = _loop_of(v1, faces)
    # ⚠️ 锁定集合必须用**形变前**的网格算：分位/法向判据依赖当前位置，
    # 用形变后的网格重算会把"新进集合"的顶点当成没被锁过（实测放过了
    # I.1 拖咬合 0.156mm）。
    occ_t = occ_table_mask(v0, faces, frame,
                           I2Params(target_gap_mm=float(
                               params["contact_gap_mm"])))
    locked = set(int(i) for i in loop_idx) | set(int(i) for i in occ_t)
    moved = np.flatnonzero(np.abs(v1 - v0).max(axis=1) > 1e-12)
    focus = np.flatnonzero(np.isin(faces, moved).any(axis=1))
    g = deformation_gate(v0, v1, faces, scale_mm=float(
        params.get("gate_scale_mm", 0.10)), focus_faces=focus,
        locked_ids=locked)
    return g.as_dict()


def _absolute_geometry_qc(mesh, params: dict) -> dict:
    """阶段门用的**绝对**几何 QC（对最终产物，不是形变前后差分）。"""
    import numpy as np
    from ..geometry.crown_outer.deform_gate import (flipped_faces,
                                                    nonmanifold_edges)
    from ..geometry.crown_outer.thickness import (geometry_safety_gate,
                                                  self_intersections)

    v = np.asarray(mesh.vertices, float)
    f = np.asarray(mesh.faces, np.int64)
    scale = float(params.get("gate_scale_mm", 0.10))
    thr = max((scale / 2.0) ** 2, 1e-9)
    ev = self_intersections(v, f, severe_min_area_mm2=thr,
                            severe_max_normal_dot=0.999)
    gate = geometry_safety_gate(v, f, max_flip_ratio=None,
                                max_self_intersections=None,
                                expect_open_boundary=True)
    nm = nonmanifold_edges(f)
    fl = flipped_faces(v, v, f, scale_mm=scale)      # 参考网格 = 自身
    metrics = {
        "n_vertices": int(len(v)), "n_faces": int(len(f)),
        "self_intersections": int(ev.get("self_intersections") or 0),
        "self_intersections_micro_defect": int(
            ev.get("self_intersections_micro_defect") or 0),
        "nonmanifold_edges": nm,
        "winding_inconsistent_edges": int(
            gate["metrics"].get("winding_inconsistent_edges") or 0),
        "degenerate_faces": int(gate["metrics"].get("degenerate_faces") or 0),
        "boundary_edges": int(gate["metrics"].get("boundary_edges") or 0),
        "flipped_faces": fl["n_flipped_total"] if "n_flipped_total" in fl
        else 0,
    }
    violations = [k for k, val in (
        ("self_intersections", metrics["self_intersections"]),
        ("nonmanifold_edges", metrics["nonmanifold_edges"]),
        ("winding_inconsistent_edges", metrics["winding_inconsistent_edges"]),
        ("degenerate_faces", metrics["degenerate_faces"]),
    ) if val]
    return {"ok": not violations, "violations": violations,
            "metrics": metrics,
            "criteria": {"micro_area_threshold_mm2": thr,
                         "note": "绝对口径：新增/旧有一起算"}}


def stage_envelope_j1(bundle: CrownBundle, out_dir: Path, params: dict,
                      prev: dict) -> dict:
    """J.1 Intaglio-Constrained（**实验阶段**：已冻结为"负结果但机制验证成功"）。"""
    anatomy = _mesh(Path(prev["outer_h1"]["artifact"]))
    inner = _mesh(Path(prev["intaglio"]["artifact"]))
    tp = ThicknessParameters(
        min_thickness_mm=float(params["min_thickness_mm"]),
        near_margin_thickness_mm=params["near_margin_thickness_mm"],
        near_margin_distance_mm=params["near_margin_distance_mm"],
        near_margin_mode=params["near_margin_mode"],
        cement_gap_mm=float(params["cement_gap_mm"]))
    res = build_intaglio_constrained_outer(
        anatomy, inner, OuterV2Params(thickness=tp, safety_mm=0.02))
    out = out_dir / "crown_j1.stl"
    import trimesh
    trimesh.Trimesh(np.asarray(res.mesh.vertices, float),
                    np.asarray(res.mesh.faces, np.int64),
                    process=False).export(str(out))
    return {"artifact": str(out), "experimental": True,
            "summary": res.summary(), "notes": res.notes}


def stage_collar_body(bundle: CrownBundle, out_dir: Path, params: dict,
                      prev: dict) -> dict:
    """J.2.1 Fixed Collar + Implicit Body（**实验阶段**：collar ✓ / body ✗）。"""
    anatomy = _mesh(Path(prev["outer_h1"]["artifact"]))
    inner = _mesh(Path(prev["intaglio"]["artifact"]))
    tp = ThicknessParameters(
        min_thickness_mm=float(params["min_thickness_mm"]),
        near_margin_thickness_mm=params["near_margin_thickness_mm"],
        near_margin_distance_mm=params["near_margin_distance_mm"],
        near_margin_mode=params["near_margin_mode"],
        cement_gap_mm=float(params["cement_gap_mm"]))
    res = build_collar_body_outer(
        anatomy, inner, bundle.margin_points, bundle.insertion_axis,
        CollarBodyParams(thickness=tp))
    out = out_dir / "crown_j21.stl"
    v = np.asarray(res.mesh.vertices, float)
    f = np.asarray(res.mesh.faces, np.int64)
    import trimesh
    trimesh.Trimesh(v, f, process=False).export(str(out))
    return {"artifact": str(out), "experimental": True,
            "summary": res.summary(), "notes": res.notes}


_STAGE_FN = {
    "outer_h1": stage_outer_h1,
    "intaglio": stage_intaglio,
    "contact": stage_contact,
    "occlusion": stage_occlusion,
    "thickness_audit": stage_thickness_audit,
    "envelope_j1": stage_envelope_j1,
    "collar_body": stage_collar_body,
}
#: 阶段依赖（跑某阶段前必须先跑这些）
_DEPS = {
    "contact": ("outer_h1",),
    "occlusion": ("outer_h1", "contact"),
    "thickness_audit": ("outer_h1", "intaglio", "occlusion"),
    "envelope_j1": ("outer_h1", "intaglio"),
    "collar_body": ("outer_h1", "intaglio"),
}


def resolve_stages(requested) -> list[str]:
    """补齐依赖并保持 STAGES 顺序；未知阶段直接报错。"""
    want = list(requested or MAIN_STAGES)
    for s in want:
        if s not in STAGES:
            raise PipelineError("未知阶段 %r（可选：%s）" % (s, ", ".join(STAGES)))
    out = set(want)
    changed = True
    while changed:
        changed = False
        for s in list(out):
            for dep in _DEPS.get(s, ()):
                if dep not in out:
                    out.add(dep)
                    changed = True
    return [s for s in STAGES if s in out]
