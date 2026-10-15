from pathlib import Path
import json
import numpy as np

from ..geometry.io import load_mesh, save_mesh, repair_mesh, mesh_info
from ..geometry.registration import rigid_then_rbf, initialize_from_margin
from ..geometry.margin import suggest_margin_from_open_boundary
from ..geometry.axis import suggest_insertion_axis
from ..geometry.insertion import AxisStatus, InsertionAxisResult, solve_insertion_axis
from ..geometry.intaglio import BlockoutPolicy, GapFieldParams, apply_gap_field, build_blockout
from ..geometry.boolean import require_exact_boolean_backend
from ..geometry.qc import build_qc
from ..geometry.nonplanar_crown import build_nonplanar_crown, fitting_support_above_margin, MarginCurtain
from ..geometry.manufacturing_qc import audit_manufacturing


def generate_outer(template_path: Path, output_path: Path, template_points, patient_points,
                   morph_template=None, morph_patient=None, morph_strength=0.0):
    template = load_mesh(template_path)
    outer, matrix = rigid_then_rbf(
        template, template_points, patient_points,
        morph_template, morph_patient, morph_strength,
    )
    outer = repair_mesh(outer)
    save_mesh(outer, output_path)
    return {"matrix": matrix.tolist(), "mesh": mesh_info(outer)}


def generate_initialized_outer(template_path: Path, output_path: Path, margin_points, insertion_axis,
                               *, patient_frame=None):
    """模板初稿放置；live 路径应传入已确认 PatientFrame。"""
    outer = initialize_from_margin(
        load_mesh(template_path), margin_points, insertion_axis,
        patient_frame=patient_frame,
    )
    save_mesh(outer, output_path)
    result = {
        "mesh": mesh_info(outer),
        "generator": "margin_initialized_template",
        "review_required": True,
        "frame_source": "CONFIRMED_FRAME" if patient_frame is not None else "WORLD_X_FALLBACK",
    }
    return result


def _design_mode(params: dict) -> bool:
    return str(params.get("quality_mode", "DESIGN")).upper().startswith("DESIGN")


def _engineering_shell_requirement(outer, params):
    """Return outer-to-intaglio separation and its provenance.

    A confirmed minimum thickness is authoritative.  If it is absent we still
    need a small *topological* separation to prevent OUTER×INTAGLIO surfaces from
    occupying the same volume; that derived value is explicitly not a clinical
    thickness target.
    """
    value = params.get("minimum_thickness_mm")
    if value is not None:
        try:
            value = float(value)
        except (TypeError, ValueError):
            value = None
    if value is not None and np.isfinite(value) and value > 0:
        return value, "CONFIRMED_MINIMUM_THICKNESS"
    edges = np.asarray(getattr(outer, "edges_unique_length", []), float)
    edges = edges[np.isfinite(edges) & (edges > 1e-8)]
    median_edge = float(np.median(edges)) if len(edges) else 0.06
    derived = float(np.clip(2.0 * median_edge, 0.12, 0.30))
    return derived, "DERIVED_TOPOLOGY_SEPARATION_NOT_CLINICAL_THICKNESS"


def finalize_crown(outer_path: Path, support_path: Path, final_path: Path, cutter_path: Path,
                   margin_points, insertion_axis, params: dict, *, prepared_support=None,
                   support_contract=None, protected_surfaces=()):
    """Build intaglio + shared 3D Margin seam and return a final crown artifact.

    Before Boolean/stitch, the actual blockout+gap cutter is treated as an inner
    spatial envelope.  The external crown is expanded (never the intaglio) if it
    is too small to contain that envelope.  This prevents the former mid-body /
    occlusal OUTER×INTAGLIO intersection ring from reaching the stitch stage.
    """
    require_exact_boolean_backend()
    indication = params.get("indication")
    if indication in ("CUSTOM_IMPLANT_ABUTMENT", "SCREW_RETAINED_CROWN"):
        raise ValueError(
            "CUSTOM_ABUTMENT_NOT_SUPPORTED"
            if indication == "CUSTOM_IMPLANT_ABUTMENT" else "UNSUPPORTED_INDICATION"
        )
    if indication == "CROWN_ON_EXISTING_ABUTMENT" and prepared_support is None:
        raise ValueError("已有基台冠必须消费已准备的 PreparedAbutmentSupport，不能退回原始 Prep 路径")

    outer = load_mesh(outer_path)
    support = load_mesh(support_path)
    margin_points = np.asarray(margin_points, float)
    if np.array_equal(margin_points[0], margin_points[-1]):
        margin_points = margin_points[:-1]

    if params.get("margin_extension_mm", 0) != 0 or params.get("margin_gap_mm", 0) != 0:
        raise ValueError(
            "共享三维 Margin 当前仅支持 margin_extension_mm=0、margin_gap_mm=0；"
            "请明确重新确认配置，不能静默改变边缘"
        )

    policy = BlockoutPolicy(
        machine_axes=int(params.get("machine_axes") or 0),
        additional_blockout_after_gap=None,
        untouchable_zone_mm=float(params.get("untouchable_zone_mm") or 0.0),
        provenance={
            "source": "confirmed_design_profile",
            "machine_axes_explicit": params.get("machine_axes") is not None,
        },
    )

    if prepared_support is not None:
        from ..geometry.abutment_support import mesh_digest
        from .design_dependencies import file_digest
        if (
            not support_contract
            or not support_contract.ready
            or file_digest(support_path) != support_contract.source_asset_sha256
        ):
            raise ValueError("实体基台支撑契约缺失、未准备完成或源文件已变化")
        if mesh_digest(prepared_support.mesh) != support_contract.conditioned_mesh_sha256:
            raise ValueError("PreparedAbutmentSupport 与确认契约的哈希不一致")
        fitting_support = prepared_support.mesh
        fitting_domain = prepared_support.report
        actual_support = prepared_support.actual_surface
    else:
        fitting_support, fitting_domain = fitting_support_above_margin(
            support, margin_points, insertion_axis,
        )
        actual_support = fitting_support

    # A: insertion-direction blockout. B: variable cement-gap field.
    blockout = build_blockout(fitting_support, margin_points, insertion_axis, policy)
    gap_params = GapFieldParams(
        cement_gap_mm=params["cement_gap_mm"],
        cement_gap_start_mm=params.get("cement_gap_start_mm"),
        cement_gap_top_mm=params.get("cement_gap_top_mm"),
        additional_spacing_xyz_mm=None,
        additional_spacing_z_on_top_mm=None,
        provenance={"source": "confirmed_design_profile"},
    )
    cutter = apply_gap_field(
        blockout, fitting_support, margin_points, insertion_axis, gap_params,
    )

    # Artificial cavity closure must remain below the true Margin region.
    direction = np.asarray(insertion_axis, float)
    direction /= np.linalg.norm(direction)
    cavity_z = np.asarray(cutter.vertices) @ direction
    outer_bottom = float((np.asarray(outer.vertices) @ direction).min())
    closure_extension = max(0.0, float(cavity_z.min()) - outer_bottom + 0.1)
    if closure_extension:
        if cavity_z.min() >= float((margin_points @ direction).min()) - 0.1:
            raise ValueError("Cavity 下封口进入真实终止线区域，不能延伸")
        lower = np.isclose(cavity_z, cavity_z.min(), atol=1e-7, rtol=0)
        cutter.vertices[lower] -= closure_extension * direction

    # Full-body shell guard.  Previous code corrected only a ~2mm cervical band,
    # while real failures were concentrated 3-7mm above Margin around the abutment
    # top.  The actual cutter is the inner truth, so move only the outer surface.
    from ..geometry.crown_outer.space_morph import enforce_inner_envelope_clearance
    curtain = MarginCurtain(margin_points, insertion_axis)
    shell_required, shell_source = _engineering_shell_requirement(outer, params)
    outer, shell_guard = enforce_inner_envelope_clearance(
        outer, cutter, curtain.frame, margin_points, shell_required,
    )
    shell_guard["requirement_source"] = shell_source
    shell_guard["clinical_thickness_claim"] = shell_source == "CONFIRMED_MINIMUM_THICKNESS"

    blockout_diag = blockout.as_dict()
    blockout_diag["fitting_domain"] = fitting_domain
    blockout_diag["cavity_lower_closure_extension_mm"] = closure_extension
    blockout_diag["outer_shell_space_guard"] = shell_guard

    quality_mode = params.get("quality_mode", "DESIGN")
    design_mode = _design_mode(params)
    built = build_nonplanar_crown(
        outer,
        cutter,
        fitting_support if prepared_support else support,
        margin_points,
        insertion_axis,
        transition_mm=params["margin_band_mm"],
        prepared_support=prepared_support is not None,
        parameters=params,
        protected_surfaces=protected_surfaces,
        quality_mode=quality_mode,
    )
    final = built.final
    trimmed = built.outer_surface

    save_mesh(cutter, cutter_path)
    save_mesh(final, final_path)
    outer_surface_path = final_path.with_name(final_path.stem + "_outer_surface.stl")
    inner_surface_path = final_path.with_name(final_path.stem + "_intaglio_surface.stl")
    audit_path = final_path.with_name(final_path.stem + "_manufacturing_qc.json")
    save_mesh(built.outer_surface, outer_surface_path)
    save_mesh(built.intaglio_surface, inner_surface_path)

    # Re-read exported STL: quantization/export is part of the actual artifact.
    from scipy.spatial import cKDTree
    from ..geometry.margin_seam import audit_margin_seam
    from ..geometry.crown_outer.thickness import input_qc

    persisted = load_mesh(final_path)
    ids = cKDTree(persisted.vertices).query(np.asarray(margin_points), k=1)[1]
    readback = audit_margin_seam(
        persisted.vertices, persisted.faces, ids, margin_points,
    )
    if not persisted.is_watertight or not readback["shared_seam_edges_incidence_two"]:
        raise ValueError("最终 STL 重读后的共享 Margin 拓扑失效")

    exported_geometry = input_qc(
        np.asarray(persisted.vertices), np.asarray(persisted.faces),
    )
    topology_broken = bool(
        exported_geometry["nonmanifold_edges"]
        or exported_geometry["winding_inconsistent_edges"]
        or exported_geometry["degenerate_faces"]
    )
    if topology_broken:
        raise ValueError("最终 STL 重读后的拓扑检查失败")

    self_intersection_flag = bool(
        exported_geometry.get("self_intersections_truncated")
        or exported_geometry.get("self_intersections_severe", 0)
    )
    if self_intersection_flag and not design_mode:
        raise ValueError("STRICT_MODE：最终 STL 存在自交，拒绝通过")

    qc, heatmap = audit_manufacturing(
        persisted,
        load_mesh(outer_surface_path),
        load_mesh(inner_surface_path),
        actual_support,
        blockout,
        margin_points,
        insertion_axis,
        params,
        exported_geometry,
        virtual_support=fitting_support if prepared_support else None,
        virtual_cap_face_ids=(
            prepared_support.report["virtual_cap_face_ids"] if prepared_support else ()
        ),
    )
    qc["OuterShellSpaceGuard"] = shell_guard

    design_warnings = list(built.report.get("stitch_warnings") or [])
    if self_intersection_flag:
        design_warnings.append("EXPORTED_SELF_INTERSECTION_REVIEW_REQUIRED")
    if (shell_guard.get("audit_after") or {}).get("status") == "REVIEW_REQUIRED":
        design_warnings.append("INNER_SHELL_SPACE_CONSTRAINT_REVIEW_REQUIRED")
    design_warnings = list(dict.fromkeys(design_warnings))
    qc["final_acceptance"] = {
        "status": "DESIGN_READY" if design_mode else "STRICT_VERIFIED",
        "quality_mode": quality_mode,
        "warnings": design_warnings,
        "manufacturing_check": False,
        "cervical_stitch_exception": built.report.get("design_mode_exception"),
        "exported_self_intersection_review_required": bool(self_intersection_flag),
        "note": (
            "DESIGN_MODE：允许保留明确标注的工程自交警告供 3D 复核；"
            "共享 Margin、watertight、非流形、绕向、退化面仍为硬门。"
            if design_mode else
            "STRICT_MODE：自交与基础拓扑均为硬门。"
        ),
    }

    support_paths = {}
    if prepared_support:
        truth_path = final_path.with_name(final_path.stem + "_prepared_support.stl")
        actual_path = final_path.with_name(final_path.stem + "_actual_support.stl")
        save_mesh(fitting_support, truth_path)
        save_mesh(actual_support, actual_path)
        from .design_dependencies import file_digest
        if file_digest(truth_path) != support_contract.conditioned_mesh_sha256:
            raise ValueError("持久化支撑哈希与实际构造来源不一致")
        qc["SupportSurfaceAudit"] = support_contract.as_report()
        qc["VirtualCapDiagnostics"] = {
            "capped_region_area_mm2": prepared_support.report["capped_region_area_mm2"],
            "operations": prepared_support.report["virtual_cap_operations"],
            "clinical_gap_excluded": True,
            "unresolved_support_holes": prepared_support.report["unresolved_support_holes"],
        }
        support_paths = {
            "prepared_support": str(truth_path),
            "actual_support": str(actual_path),
        }

    qc["NonplanarMarginBuild"] = {**built.report, "exported_stl_seam": readback}
    qc["CrownBorderAudit"] = (
        built.report["crown_border"]
        or {"status": "NOT_ASSESSED", "reason": "LEGACY_KNIFE_EDGE"}
    )
    qc["MinimumThicknessEnforcement"] = (
        built.report["minimum_thickness_enforcement"]
        or {"status": "NOT_REQUESTED"}
    )
    if built.report["minimum_thickness_enforcement"]:
        enforcement = qc["MinimumThicknessEnforcement"]
        enforcement["postbuild_body_thin_area_mm2"] = qc[
            "FinalCrownThicknessAudit"
        ]["body_thin_face_sample_area_mm2"]
        enforcement["postbuild_body_conservative"] = qc[
            "FinalCrownThicknessAudit"
        ]["body_conservative"]
        if enforcement["postbuild_body_thin_area_mm2"] > 0:
            enforcement["status"] = "REVIEW_REQUIRED"
            enforcement["postbuild_reason"] = "FINAL_SHELL_STILL_HAS_THIN_BODY_SAMPLES"

    audit_path.write_text(
        json.dumps({"qc": qc, "thickness_heatmap": heatmap}, ensure_ascii=False),
        encoding="utf-8",
    )
    blockout_diag["w3"] = {
        "qc": qc,
        "artifact_paths": {
            "outer_surface": str(outer_surface_path),
            "intaglio_surface": str(inner_surface_path),
            "manufacturing_qc": str(audit_path),
            **support_paths,
        },
    }
    return trimmed, cutter, final, blockout_diag


def auto_margin(support_path: Path):
    mesh = load_mesh(support_path)
    return suggest_margin_from_open_boundary(mesh).tolist()


def auto_margin_m1(support_path: Path, axis=None, mode="SUPRA", radius_mm=1.0, apply_snap=True):
    """工作台用的 M1 自动 Margin（geometry/auto_margin 冻结配置 v2）。"""
    from ..geometry.auto_margin.engine import predict_margin
    from ..geometry.auto_margin.models import STATUS_AUTO_ACCEPTED
    from ..geometry.margin_feature_snap import snap_to_feature

    mesh = load_mesh(support_path)
    tris = np.asarray(mesh.triangles, float)
    axis_source = "confirmed"
    if axis is None:
        axis = list(map(float, suggest_insertion_axis(mesh, None)))
        axis_source = "provisional_sidewall_fit"
    axis = [float(v) for v in axis]
    norm = float(np.linalg.norm(axis))
    if norm < 1e-9:
        raise ValueError("就位轴无效")
    axis = [v / norm for v in axis]
    res = predict_margin(tris, axis)
    out = {
        "points": [],
        "status": res.status,
        "reason": res.reason,
        "config_id": res.config_id,
        "solver": res.solver,
        "quality": res.quality,
        "confidence": res.confidence,
        "candidate_rank_stats": res.candidate_rank_stats,
        "axis": axis,
        "axis_source": axis_source,
        "elapsed_s": res.elapsed_s,
        "mode": mode,
        "radius_mm": float(radius_mm),
        "snap": None,
    }
    if res.status == STATUS_AUTO_ACCEPTED and res.margin is not None and res.margin.points:
        pts = [[float(v) for v in p] for p in res.margin.points]
        out["points"] = pts
        if apply_snap and pts:
            out["snap"] = snap_to_feature(tris, pts, axis, mode, radius_mm)
            if out["snap"]["status"] == "APPLIED":
                out["points"] = out["snap"]["points"]
    return out


def auto_axis(support_path: Path, margin_points=None):
    """求解就位方向并返回状态、置信度、倒凹指标与诊断。"""
    mesh = load_mesh(support_path)
    if margin_points is None or len(margin_points) < 3:
        axis = suggest_insertion_axis(mesh, None)
        return InsertionAxisResult(
            axis=[float(v) for v in axis],
            initial_axis=[float(v) for v in axis],
            score=float("nan"),
            undercut_area_mm2=0.0,
            undercut_area_ratio=float("nan"),
            max_undercut_depth_mm=0.0,
            mean_undercut_depth_mm=0.0,
            margin_visibility_ratio=float("nan"),
            deviation_deg=0.0,
            confidence=0.0,
            status=AxisStatus.REVIEW_REQUIRED,
            diagnostics={
                "reason": (
                    "没有可用 Margin，退化为侧壁法向拟合轴；"
                    "请标注并确认 Margin 后重新求解以获得倒凹指标与置信度"
                ),
                "degraded_without_margin": True,
            },
        )
    return solve_insertion_axis(mesh, margin_points)


def qc_from_paths(final_path, outer_path, cutter_path, support_path,
                  same_jaw_path=None, antagonist_path=None, outer_mesh=None):
    final = load_mesh(final_path)
    outer = outer_mesh if outer_mesh is not None else load_mesh(outer_path)
    cutter = load_mesh(cutter_path)
    support = load_mesh(support_path)
    same = load_mesh(same_jaw_path) if same_jaw_path else None
    ant = load_mesh(antagonist_path) if antagonist_path else None
    return build_qc(final, outer, cutter, support, same, ant)
