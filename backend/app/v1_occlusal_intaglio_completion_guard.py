"""CrownFlow 1.0 local-occlusion + complete-intaglio policy.

Two real-case regressions are addressed here without touching M/D, B/L, Margin or
proximal geometry:

1. Broad antagonist height prefit can lower a whole posterior crown even though
   only local cusp regions collide.  In 1.0 the local OCCLUSAL solver owns the
   antagonist ceiling; the broad height prefit becomes a no-op.
2. AUTO generation already builds a clinically meaningful intaglio preview, but
   the workflow explicitly deferred exact W3 crown construction until after
   candidate acceptance.  That made the sculpt canvas show an outer-only crown.
   Force W3 construction in the same generation job so outer + true intaglio are
   produced together.  Engineering QC may still be REVIEW_REQUIRED; geometry is
   preserved for review/export instead of being hidden.
"""
from __future__ import annotations

from .geometry.crown_outer import space_morph as _space
from .services import generation_workflow as _workflow


_ORIGINAL_HEIGHT_OWNER = _space._fit_antagonist_height
_ORIGINAL_GENERATE_CANDIDATE = _workflow.generate_candidate


def _local_solver_owns_antagonist_height(current, frame, margin, antagonist_surface, target, cfg):
    """Do not globally scale/drop the crown against the antagonist.

    The downstream OCCLUSAL stage has the confirmed antagonist surface, protected
    proximal contacts, the immutable cervical band, line-search deformation gates
    and the real local-normal target.  Returning the crown unchanged here avoids
    the previous whole-table drop while keeping a transparent audit record.
    """
    return current, {
        "applied": False,
        "mode": "LOCAL_OCCLUSAL_SOLVER_OWNS_ANTAGONIST_HEIGHT_V1",
        "reason": "BROAD_HEIGHT_PREFIT_DISABLED_TO_PRESERVE_POSTERIOR_ANATOMY",
        "target_clearance_mm": float(target),
        "target_reached": False,
        "global_penetration_guard_used": False,
        "passes": [],
        "downstream_owner": "OCCLUSAL_LOCAL_SURFACE_SOLVER",
    }


def _complete_auto_generation(d, kind):
    """Force exact W3/intaglio construction in the same AUTO generation run."""
    # generation_workflow imports propose locally inside generate_candidate, so
    # install the wrapper immediately before the original function resolves it.
    from .services import auto_fit as _auto_fit

    live = _auto_fit.propose
    if not getattr(live, "_crownflow_force_w3", False):
        original_propose = live

        def _propose_with_w3(*args, **kwargs):
            kwargs["run_w3_preflight"] = True
            return original_propose(*args, **kwargs)

        _propose_with_w3._crownflow_force_w3 = True
        _propose_with_w3._crownflow_original = original_propose
        _auto_fit.propose = _propose_with_w3

    report, artifacts = _ORIGINAL_GENERATE_CANDIDATE(d, kind)
    report.setdefault("generation_policy_overrides", {})[
        "occlusal_and_intaglio_completion"
    ] = {
        "antagonist_height_owner": "LOCAL_OCCLUSAL_SOLVER",
        "broad_height_prefit": "DISABLED",
        "w3_generation": "SAME_JOB_AS_OUTER",
        "inner_surface_required_for_complete_crown": True,
        "qc_policy": "RECORD_WARNINGS_DO_NOT_HIDE_GENERATED_GEOMETRY",
    }
    return report, artifacts


_space._fit_antagonist_height = _local_solver_owns_antagonist_height
_workflow.generate_candidate = _complete_auto_generation
