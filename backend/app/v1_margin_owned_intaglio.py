"""CrownFlow 1.0 Margin-owned final intaglio assembly.

Outer design is already the accepted anatomy.  Finalization must therefore be
one-way: accepted outer -> patient intaglio -> shared confirmed Margin.  It must
not use a Boolean result to rediscover the two clinical source surfaces and it
must not deform the accepted outer while building the inner crown.

This policy deliberately reuses the established geometry kernels:
- Prepared support -> directional blockout -> cement-gap cutter remains the
  patient-owned cavity truth.
- MarginCurtain + triangle-level scalar clipping extracts supramarginal source
  skins directly from accepted outer and cutter.
- stitch_cervical_surfaces supplies the narrow generated collars and keeps every
  confirmed 3D Margin point verbatim in the final topology.
- Outer-minus-cutter Boolean is retained only as a diagnostic/audit artifact.

The module is imported before services.auto_fit captures pipeline symbols.
"""
from __future__ import annotations

from contextvars import ContextVar
import numpy as np
import trimesh
from trimesh.proximity import closest_point

from .geometry import nonplanar_crown as _npc
from .geometry.crown_outer import space_morph as _space


_FINALIZATION_OUTER_FROZEN: ContextVar[bool] = ContextVar(
    "crownflow_finalization_outer_frozen", default=False
)
_ORIGINAL_CLEARANCE_ENFORCER = _space.enforce_inner_envelope_clearance


def _audit_only_clearance(mesh, inner_envelope_mesh, frame, margin, required_mm, *args, **kwargs):
    """Keep accepted outer byte-for-byte fixed during final intaglio construction."""
    if not _FINALIZATION_OUTER_FROZEN.get():
        return _ORIGINAL_CLEARANCE_ENFORCER(
            mesh, inner_envelope_mesh, frame, margin, required_mm, *args, **kwargs
        )
    audit = _space.audit_inner_envelope_clearance(
        mesh, inner_envelope_mesh, frame, margin, required_mm, *args, **kwargs
    )
    return mesh.copy(), {
        "status": "AUDIT_ONLY_OUTER_FROZEN",
        "method": "ACCEPTED_OUTER_IMMUTABLE_DURING_FINAL_INTAGLIO",
        "required_outer_to_intaglio_mm": float(required_mm),
        "moved_vertices": 0,
        "max_push_per_pass_mm": 0.0,
        "audit_after": audit,
        "outer_modified": False,
        "intaglio_modified": False,
    }


# auto_fit may bind this function by value later.  Outside the finalization
# context the wrapper delegates exactly to the already-active design-stage owner.
_space.enforce_inner_envelope_clearance = _audit_only_clearance


def _boolean_audit(outer, cutter):
    """Run Boolean only as a diagnostic; never use it to define clinical skins."""
    report = {
        "role": "AUDIT_ONLY_NOT_SOURCE_SURFACE",
        "status": "NOT_RUN",
        "outer_unchanged": True,
        "cutter_unchanged": True,
    }
    try:
        boolean = _npc.crown_minus_cutter(outer, cutter)
        report.update({
            "status": "COMPUTED",
            "faces": int(len(boolean.faces)),
            "vertices": int(len(boolean.vertices)),
            "watertight": bool(boolean.is_watertight),
            "component_count": int(len(boolean.split(only_watertight=False))),
            "volume_mm3": float(abs(boolean.volume)) if boolean.is_volume else None,
        })
        try:
            from .geometry.boolean_healing import boolean_healing_stage
            healed, healing = boolean_healing_stage(boolean)
            report["healing"] = healing
            report["healed_faces"] = int(len(healed.faces))
        except Exception as exc:
            report["healing"] = {"status": "ERROR_IGNORED", "error": str(exc)[:200]}
    except Exception as exc:
        report.update({"status": "FAILED_NONBLOCKING", "error": str(exc)[:300]})
    return report


def _margin_owned_build(outer, cutter, support, margin, axis, *, transition_mm,
                        prepared_support=False, parameters=None, protected_surfaces=(),
                        quality_mode="DESIGN"):
    """Build final crown from direct source skins and one exact 3D Margin seam.

    No vertex of the accepted anatomical outer source is moved here.  The only
    newly generated external geometry is the narrow cervical collar between the
    supramarginal outer ring and the prescribed Margin vertices.
    """
    curtain = _npc.MarginCurtain(margin, axis)
    if not np.isfinite(transition_mm) or float(transition_mm) <= 0:
        raise _npc.MarginBuildError("三维颈部过渡高度 margin_band_mm 必须为正值")

    scan = _npc.audit_margin_on_scan(
        support, curtain.points, expected_margin_boundary=prepared_support
    )
    from .geometry.topology_validation import record_topology
    record_topology("01_input_outer", outer)
    record_topology("02_input_cutter", cutter)

    # Authoritative clinical surfaces: source outer and patient-derived cutter.
    # Both are triangle-clipped at a safe supramarginal transition; everything
    # below that transition is discarded from final clinical surface assembly.
    outside, outer_extract = _npc._single_open_surface_above_margin(
        outer, curtain, float(transition_mm), "OUTER", invert=False
    )
    inside, inner_extract = _npc._single_open_surface_above_margin(
        cutter, curtain, float(transition_mm), "INTAGLIO", invert=True
    )
    record_topology("05_direct_outer_supramarginal", outside)
    record_topology("06_direct_intaglio_supramarginal", inside)

    # Read-only proximity audit.  Do not revive the historical behavior that
    # pushed the accepted outer outwards during W3 assembly.
    outer_loop = _npc._boundary_loops(np.asarray(outside.faces, np.int64))[0]
    inner_loop = _npc._boundary_loops(np.asarray(inside.faces, np.int64))[0]
    outer_ring = np.asarray(outside.vertices, float)[np.asarray(outer_loop, np.int64)]
    inner_ring = np.asarray(inside.vertices, float)[np.asarray(inner_loop, np.int64)]
    if len(outer_ring) and len(inner_ring):
        _, distances, _ = closest_point(inside, outer_ring)
        distances = np.asarray(distances, float)
        source_separation = {
            "status": "MEASURED_NO_GEOMETRY_CHANGE",
            "outer_ring_to_intaglio_min_mm": float(distances.min()),
            "outer_ring_to_intaglio_p05_mm": float(np.quantile(distances, .05)),
            "outer_modified": False,
            "intaglio_modified": False,
        }
    else:
        source_separation = {
            "status": "NOT_ASSESSED",
            "reason": "EMPTY_CERVICAL_RING",
            "outer_modified": False,
            "intaglio_modified": False,
        }

    params = parameters or {}
    # Final-stage thickness is audit-only.  The design-stage shell solver had the
    # opportunity to grow the outer before acceptance; W3 may not reshape it.
    thickness_policy = None
    if params.get("enforce_minimum_thickness"):
        thickness_policy = {
            "status": "AUDIT_ONLY_OUTER_FROZEN",
            "requested_mm": float(params.get("minimum_thickness_mm") or 0.0),
            "geometry_changed": False,
        }

    from .geometry.cervical_cleanup import mesh_diagnostics
    cervical_report = {
        "status": "DIRECT_SOURCE_CLIP_TO_SHARED_MARGIN",
        "outer_before": mesh_diagnostics(outside, curtain.points),
        "inner_before": mesh_diagnostics(inside, curtain.points),
        "scope": "CROWN_CERVICAL_REGION_ONLY",
        "margin_curve_untouched": True,
        "accepted_outer_body_untouched": True,
    }

    final, seam = _npc.stitch_cervical_surfaces(
        outside, inside, curtain.points, quality_mode=quality_mode
    )

    # Re-extract the two open clinical surfaces from the assembled indexing while
    # keeping the shared Margin/collar topology in `final` only.
    n_outer = len(outside.vertices)
    n_inner = len(inside.vertices)
    seam_start = n_outer + n_inner
    faces = np.asarray(final.faces)
    outer_faces = faces[~np.any((faces >= n_outer) & (faces < seam_start), axis=1)]
    inner_faces = faces[~np.any(faces < n_outer, axis=1)]
    surfaces = []
    for selected in (outer_faces, inner_faces):
        mesh = trimesh.Trimesh(np.asarray(final.vertices).copy(), selected, process=False)
        mesh.remove_unreferenced_vertices()
        surfaces.append(mesh)

    preservation = np.concatenate([
        closest_point(outer, np.asarray(outside.vertices)[i:i + 512])[1]
        for i in range(0, len(outside.vertices), 512)
    ]) if len(outside.vertices) else np.zeros(0, float)

    report = {
        "status": "ENGINEERING_ONLY",
        "method": "MARGIN_OWNED_DIRECT_SOURCE_CLIP_AND_SHARED_3D_SEAM_V1",
        "assembly_order": [
            "ACCEPTED_OUTER_FROZEN",
            "PATIENT_BLOCKOUT_GAP_CUTTER",
            "DIRECT_OUTER_SUPRAMARGINAL_CLIP",
            "DIRECT_INTAGLIO_SUPRAMARGINAL_CLIP",
            "EXACT_SHARED_3D_MARGIN_COLLARS",
            "BOOLEAN_AUDIT_ONLY",
        ],
        "margin": _npc.margin_plane_diagnostics(curtain.points),
        "scan_adherence": scan,
        "seam": seam,
        "transition_mm": float(transition_mm),
        "outer_frozen": True,
        "body_vertices_moved": 0,
        "body_outer_vertex_count": int(len(outside.vertices)),
        "body_to_original_outer_p95_mm": float(np.quantile(preservation, .95)) if len(preservation) else 0.0,
        "body_to_original_outer_max_mm": float(preservation.max()) if len(preservation) else 0.0,
        "geometry_above_cervical_transition_preserved": True,
        "margin_profile": "SHARED_CONFIRMED_3D_MARGIN",
        "crown_border": None,
        "minimum_thickness_enforcement": None,
        "final_stage_thickness_policy": thickness_policy,
        "cervical_cleanup": cervical_report,
        "surface_extraction": {
            "method": "DIRECT_SOURCE_CLIP_AUTHORITATIVE",
            "outer": outer_extract,
            "intaglio": inner_extract,
            "cervical_separation": source_separation,
            "boolean_source_classification_used": False,
        },
        "boolean_audit": _boolean_audit(outer, cutter),
        "quality_mode": quality_mode,
        "design_mode_exception": seam.get("design_mode_exception"),
        "stitch_warnings": (
            [] if not seam.get("design_mode_exception")
            else ["Cervical stitch minor self intersection"]
        ),
    }
    return _npc.NonplanarCrownBuild(final, surfaces[0], surfaces[1], report)


# Install direct source assembly before pipeline imports the symbol by value.
_npc.build_nonplanar_crown = _margin_owned_build


# Freeze the accepted outer only while pipeline.finalize_crown is running.  The
# design-stage INNER_SHELL_RECHECK still delegates to its original solver.
from .services import pipeline as _pipeline
_ORIGINAL_FINALIZE = _pipeline.finalize_crown


def _finalize_outer_then_inner(*args, **kwargs):
    token = _FINALIZATION_OUTER_FROZEN.set(True)
    try:
        return _ORIGINAL_FINALIZE(*args, **kwargs)
    finally:
        _FINALIZATION_OUTER_FROZEN.reset(token)


_finalize_outer_then_inner._crownflow_outer_then_inner = True
_pipeline.finalize_crown = _finalize_outer_then_inner
