"""Broad proximal shape fit for Step-5 crown design.

The previous implementation reused ``fit_stage`` repeatedly.  That solver is
excellent for the last few tenths of a millimetre, but when asked to bridge a
large proximal gap it can pull a small seed patch into a long spike.  Coarse
fit here deliberately solves a different problem: move a *broad proximal wall*
mainly along the confirmed mesial-distal axis, with smooth height falloff and
hard cervical protection.  Fine contact remains the responsibility of the
existing local-band solver.
"""
from __future__ import annotations

import numpy as np

from ..geometry.nonplanar_crown import MarginCurtain
from ..geometry.crown_shape_quality import deformation_shape_report, shape_gate
from ..geometry.crown_outer.deform_gate import deformation_gate


COARSE_TARGET_MM = 0.75
MAX_COARSE_SHIFT_MM = 1.50


def _smoothstep(x):
    x = np.clip(np.asarray(x, float), 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def _signed_min(surface, points):
    if surface is None:
        return None
    _, _, signed, _, valid = surface.query(points)
    return float(np.min(signed[valid])) if np.any(valid) else None


def _contact_face_metrics(mesh, surface, vertex_domain, *, low=-.05, high=.20):
    """Area/components of a contact *patch*, sampled on face centroids.

    Counting a face only when all three vertices happen to be in contact badly
    underestimates small contacts.  Centroid sampling is still an engineering
    approximation, but is stable and is suitable for Step-5 feedback.
    """
    domain = np.asarray(vertex_domain, bool)
    face_ids = np.flatnonzero(np.any(domain[mesh.faces], axis=1))
    if not len(face_ids):
        return 0.0, 0, 0
    centers = np.asarray(mesh.triangles_center)[face_ids]
    _, _, signed, _, valid = surface.query(centers)
    hit_local = np.asarray(valid, bool) & (np.asarray(signed) >= low) & (np.asarray(signed) <= high)
    hit = face_ids[hit_local]
    if not len(hit):
        return 0.0, 0, int(np.count_nonzero(valid))
    hit_set = set(map(int, hit))
    adj = {int(f): set() for f in hit}
    for a, b in np.asarray(mesh.face_adjacency, int):
        if int(a) in hit_set and int(b) in hit_set:
            adj[int(a)].add(int(b)); adj[int(b)].add(int(a))
    components = 0; seen = set()
    for f in hit:
        f = int(f)
        if f in seen: continue
        components += 1
        stack = [f]; seen.add(f)
        while stack:
            u = stack.pop()
            for v in adj.get(u, ()):
                if v not in seen:
                    seen.add(v); stack.append(v)
    return float(np.sum(mesh.area_faces[hit])), int(components), int(np.count_nonzero(valid))


def proximal_snapshot(mesh, margin, frame, surface, domain, *, reference=None, antagonist=None):
    vertices = np.asarray(mesh.vertices, float)
    _, _, signed, distance, valid = surface.query(vertices)
    in_domain = np.asarray(domain, bool) & valid
    area, components, face_valid = _contact_face_metrics(mesh, surface, domain)
    height = MarginCurtain(np.asarray(margin, float), frame.z).axial_above(vertices)
    return {
        'neighbor_distance_mm': float(np.min(distance)),
        'domain_signed_min_mm': float(np.min(signed[in_domain])) if np.any(in_domain) else None,
        'valid_count': int(np.count_nonzero(in_domain)),
        'contact_patch_area_mm2': area,
        'contact_patch_components': components,
        'contact_face_valid_count': face_valid,
        'crown_height_mm': float(np.max(height)),
        'margin_drift_mm': (float(np.max(np.linalg.norm(vertices[np.asarray(reference['locked'], bool)]
                            - reference['vertices'][np.asarray(reference['locked'], bool)], axis=1)))
                            if reference is not None and np.any(reference['locked']) else 0.0),
        'antagonist_min_mm': _signed_min(antagonist, vertices),
    }


def _broad_wall_weights(mesh, margin, frame, region_map, side, protected):
    """Continuous broad proximal wall weight, with cervical/top falloff."""
    v = np.asarray(mesh.vertices, float)
    base = np.asarray(region_map.deformation_weight(side), float)
    protection = np.asarray(region_map.protection, float)
    protected = np.asarray(protected, bool)
    height = MarginCurtain(np.asarray(margin, float), frame.z).axial_above(v)
    top = max(float(np.max(height)), 1e-6)

    # Broaden the continuous semantic field instead of thresholding it at .5.
    lateral = np.sqrt(np.clip(base, 0.0, 1.0))
    # Keep the confirmed cervical zone fixed, smoothly engage the body, then
    # reduce influence close to cusp tips so contact fitting does not tilt or
    # lengthen the whole crown.
    lower = _smoothstep((height - .75) / .90)
    upper_start = max(1.5, top - 1.35)
    upper = 1.0 - .80 * _smoothstep((height - upper_start) / max(top - upper_start, .35))
    weight = lateral * lower * upper * protection
    weight[protected] = 0.0

    # Diffuse the scalar field over the proximal wall to avoid a narrow bridge.
    # The hard protected area remains zero throughout.
    adj = [[] for _ in range(len(v))]
    for a, b in np.asarray(mesh.edges_unique, int):
        adj[int(a)].append(int(b)); adj[int(b)].append(int(a))
    for _ in range(4):
        nxt = weight.copy()
        active = np.flatnonzero((weight > 0) & ~protected)
        for i in active:
            nb = adj[int(i)]
            if nb:
                nxt[i] = max(weight[i], 0.70 * weight[i] + 0.30 * float(np.mean(weight[nb])))
        weight = np.clip(nxt, 0.0, 1.0)
        weight[protected] = 0.0
    return weight


def proximal_coarse_fit(mesh, margin, frame, region_map, side, surface, domain, *,
                        protected, antagonist=None, opposite_domain=None, opposite_surface=None,
                        goal_mm=COARSE_TARGET_MM):
    """Broad lateral morph to bring a distant proximal wall into fine-fit reach.

    It does *not* chase the final 0.05 mm contact.  The operation is rejected if
    it distorts local topology/shape or violates margin, height, antagonist or
    opposite-side invariants.
    """
    original = np.asarray(mesh.vertices, float).copy()
    faces = np.asarray(mesh.faces, int)
    locked = np.asarray(protected, bool)
    opposite = np.asarray(opposite_domain, bool) if opposite_domain is not None else np.zeros(len(original), bool)
    before = proximal_snapshot(mesh, margin, frame, surface, domain, antagonist=antagonist)
    opposite_before = (float(np.min(opposite_surface.query(original)[3]))
                       if opposite_surface is not None else None)
    if before['domain_signed_min_mm'] is None:
        return mesh.copy(), {'status': 'CONSTRAINT_CONFLICT', 'reason': 'NO_VALID_SIGNED_SAMPLES', 'before': before}
    if before['domain_signed_min_mm'] <= goal_mm:
        return mesh.copy(), {'status': 'NO_CORRECTION_NEEDED', 'before': before, 'after': before}

    weight = _broad_wall_weights(mesh, margin, frame, region_map, side, locked)
    effective = weight > .05
    if not np.any(effective):
        return mesh.copy(), {'status': 'CONSTRAINT_CONFLICT', 'reason': 'BROAD_PROXIMAL_DOMAIN_EMPTY', 'before': before}

    md = np.asarray(region_map.axes['mesial'], float) * (1.0 if side == 'mesial' else -1.0)
    md /= max(float(np.linalg.norm(md)), 1e-12)
    # The coarse pass intentionally stops short of contact.  A cap prevents a
    # bad/remote patch from causing a multi-millimetre pull in one operation.
    requested = max(0.0, float(before['domain_signed_min_mm']) - float(goal_mm))
    base_shift = min(requested, MAX_COARSE_SHIFT_MM)
    if base_shift <= 1e-6:
        return mesh.copy(), {'status': 'NO_CORRECTION_NEEDED', 'before': before, 'after': before}

    focus_faces = np.flatnonzero(np.any(effective[faces], axis=1))
    trials = []
    ant0 = before['antagonist_min_mm']
    for alpha in (1.0, .75, .5, .25, .125):
        displacement = (alpha * base_shift * weight)[:, None] * md[None, :]
        candidate = mesh.copy()
        candidate.vertices = original + displacement
        moved = np.linalg.norm(displacement, axis=1)
        gate = deformation_gate(original, candidate.vertices, faces, scale_mm=.05,
                                locked_ids=np.flatnonzero(locked), focus_faces=focus_faces)
        shape = deformation_shape_report(original, candidate.vertices, faces, focus_faces=focus_faces)
        shape_check = shape_gate(shape)
        after = proximal_snapshot(candidate, margin, frame, surface, domain, antagonist=antagonist)
        ant1 = after['antagonist_min_mm']
        ant_drift = abs(ant1 - ant0) if ant0 is not None and ant1 is not None else 0.0
        opposite_drift = float(np.max(moved[opposite])) if opposite.any() else 0.0
        opposite_distance_drift = (abs(float(np.min(opposite_surface.query(candidate.vertices)[3])) - opposite_before)
                                   if opposite_surface is not None else 0.0)
        height_drift = abs(after['crown_height_mm'] - before['crown_height_mm'])
        improved = (after['domain_signed_min_mm'] is not None
                    and after['domain_signed_min_mm'] < before['domain_signed_min_mm'] - .02)
        invariants_ok = (height_drift < .05 and ant_drift < .10 and opposite_drift < .05
                         and opposite_distance_drift < .05
                         and (not locked.any() or float(np.max(moved[locked])) < 1e-9))
        accepted = bool(gate.ok and shape_check['ok'] and improved and invariants_ok)
        trials.append({
            'alpha': alpha, 'requested_shift_mm': base_shift,
            'maximum_displacement_mm': float(np.max(moved)),
            'broad_domain_vertex_count': int(np.count_nonzero(effective)),
            'before_signed_min_mm': before['domain_signed_min_mm'],
            'after_signed_min_mm': after['domain_signed_min_mm'],
            'height_drift_mm': height_drift, 'antagonist_drift_mm': ant_drift,
            'opposite_side_drift_mm': opposite_drift,
            'opposite_distance_drift_mm': opposite_distance_drift,
            'geometry_gate': gate.as_dict(), 'shape_quality': shape,
            'shape_gate': shape_check, 'accepted': accepted,
        })
        if accepted:
            status = 'APPLIED' if after['domain_signed_min_mm'] <= goal_mm + .20 else 'APPLIED_PARTIAL'
            return candidate, {
                'status': status, 'method': 'broad_proximal_md_morph/1',
                'before': before, 'after': after, 'goal_mm': float(goal_mm),
                'requested_shift_mm': base_shift, 'trials': trials,
                'shape_quality': shape, 'broad_domain_vertex_count': int(np.count_nonzero(effective)),
            }

    return mesh.copy(), {
        'status': 'CONSTRAINT_CONFLICT', 'reason': 'COARSE_SHAPE_OR_INVARIANT_GATE',
        'before': before, 'goal_mm': float(goal_mm), 'requested_shift_mm': base_shift, 'trials': trials,
    }
