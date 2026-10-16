"""CrownFlow 1.0 last-mile geometry tuning.

This module only tightens three still-visible real-case defects without replacing
already-stable owners (Margin/cervical lock, established opposite proximal contact,
or template anatomy):

* fuller patient-derived buccolingual body width;
* stronger guarded closure of the single remaining open proximal side;
* antagonist-owned occlusal relief that follows the actual opposing surface normal
  enough to escape oblique collisions, while keeping cusp/fossa anatomy dominant.

It is imported after :mod:`v1_test_geometry_overrides` and before the API/service
graph, so the existing pipeline uses these values without a second geometry pass.
"""
from __future__ import annotations

import numpy as np

from . import v1_test_geometry_overrides as _width
from .geometry.crown_outer import auto_fit_adapter as _fit
from .geometry.crown_outer import proximal_v9 as _prox


# --- B/L fullness ---------------------------------------------------------
# The target itself remains the requested 50/50 gingiva+neighbor blend.  What was
# too conservative was the amount the crown was allowed to travel toward it.
_width.BL_MAX_EXPANSION_SCALE = 1.24
_width.BL_ENVELOPE_SAFETY_PAD_MM = 1.20
_width.BL_FULL_BODY_START_MM = 0.90
_width.BL_OCCLUSAL_CORE_WEIGHT = 0.48


# --- one-side proximal closure -------------------------------------------
# Keep V9's opposite-side protection and penetration guards, but allow a visibly
# open side to enter the final-kiss stage even when its remaining p95 is >0.7mm.
_prox.FINAL_KISS_TRIGGER_P95_MM = 1.50
_prox.FINAL_KISS_TARGET_BIAS_MM = 0.020
_prox.FINAL_KISS_MAX_MOVE_MM = 0.45
_prox.FINAL_KISS_TRANSITION_RINGS = 6
_prox.FINAL_KISS_MAX_PASSES = 3
_prox.FINAL_KISS_MAX_EXTRA_PENETRATION_MM = 0.10


# --- antagonist-owned occlusal relief -----------------------------------
# Existing code used only 8% of the non-axial antagonist target vector.  That is
# safe for height seeking, but on an oblique antagonist face it can leave a cusp
# visibly inside the opposing tooth even after several accepted axial moves.
_fit.OCCLUSAL_CONTACT_FRACTION = 0.32
_fit.OCCLUSAL_MAX_REACH_MM = 5.5
_ORIGINAL_DESIRED_TARGETS = _fit._desired_targets
_ORIGINAL_FIT_STAGE = _fit.fit_stage


def _antagonist_owned_targets(reference, ids, cp, fids, signed, distance, valid,
                              surface, target, stage, side, frame, cut_only):
    seed_ids, targets, blocked = _ORIGINAL_DESIRED_TARGETS(
        reference, ids, cp, fids, signed, distance, valid,
        surface, target, stage, side, frame, cut_only,
    )
    if stage != "OCCLUSAL" or not len(seed_ids):
        return seed_ids, targets, blocked

    # Map returned global seed ids back to this surface-query subset.  The
    # selection itself stays exactly the proven CAD-1 selection; only its target
    # direction is improved.
    positions = {int(vertex_id): pos for pos, vertex_id in enumerate(np.asarray(ids, int))}
    local = np.asarray([positions.get(int(vertex_id), -1) for vertex_id in seed_ids], int)
    usable = local >= 0
    if not np.all(usable):
        return seed_ids, targets, blocked

    normals = np.asarray(surface.patch.normals, float)[np.asarray(fids, int)[local]]
    normals *= float(surface.patch.normal_sign)
    full_target = np.asarray(cp, float)[local] + normals * float(target)
    src = np.asarray(reference, float)[np.asarray(seed_ids, int)]
    delta = full_target - src

    axis = np.asarray(frame.z, float)
    axis /= max(float(np.linalg.norm(axis)), 1e-12)
    axial = (delta @ axis)[:, None] * axis[None, :]

    # True collision/clearance deficit needs a stronger antagonist-normal
    # component to get out of angled opposing surfaces.  Height seeking remains
    # anatomy-first so the crown does not simply copy the antagonist.
    local_signed = np.asarray(signed, float)[local]
    collision = local_signed < float(target) - float(_fit.TARGET_TOL_MM)
    lateral_mix = np.where(collision, 0.65, 0.28)
    targets = src + axial + lateral_mix[:, None] * (delta - axial)
    return seed_ids, targets, blocked


def _patient_space_fit_stage(mesh, frame, margin, surfaces, targets, stage, *,
                             protected=None, domain_mesh=None, domain_masks=None,
                             cut_only=False, transition_rings=None, max_passes=None):
    """Use extra iterations only for the main occlusal solve.

    ``fit_stage``'s original default is bound at function-definition time, so
    changing the module constant alone would not affect live calls.  Explicit
    recheck budgets (for example max_passes=2 after shell repair) are preserved.
    """
    passes = max_passes
    if passes is None:
        passes = 7 if stage == "OCCLUSAL" else 5
    return _ORIGINAL_FIT_STAGE(
        mesh, frame, margin, surfaces, targets, stage,
        protected=protected,
        domain_mesh=domain_mesh,
        domain_masks=domain_masks,
        cut_only=cut_only,
        transition_rings=transition_rings,
        max_passes=passes,
    )


_fit._desired_targets = _antagonist_owned_targets
_fit.fit_stage = _patient_space_fit_stage
