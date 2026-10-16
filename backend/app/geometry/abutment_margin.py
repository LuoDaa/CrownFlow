"""Nonplanar outer shoulder on an existing restorative abutment.

CAD-1 is an engineering iteration workflow: once a usable shoulder candidate can
be formed, quality diagnostics must not stop the whole crown pipeline.  The
extractor therefore repairs short weak arcs and reports warnings instead of
turning prominence/continuity heuristics into hard workflow gates.
"""
import numpy as np
import trimesh

ALGORITHM_VERSION = 'welooai/abutment-shoulder/1.1'


def _cyclic_path(candidates):
    best = None
    for start in range(len(candidates[0])):
        cost = np.full(len(candidates[0]), np.inf)
        cost[start] = -4 * np.linalg.norm(candidates[0][start, :2])
        parents = []
        for previous, current in zip(candidates[:-1], candidates[1:]):
            transition = 20 * np.sum((previous[:, None] - current[None]) ** 2, axis=2)
            table = cost[:, None] + transition
            parent = np.argmin(table, axis=0)
            parents.append(parent)
            cost = table[parent, np.arange(len(current))] - 4 * np.linalg.norm(current[:, :2], axis=1)
        closed_cost = cost + 20 * np.sum((candidates[-1] - candidates[0][start]) ** 2, axis=1)
        end = int(np.argmin(closed_cost))
        path = [end]
        for parent in parents[::-1]:
            path.append(int(parent[path[-1]]))
        if best is None or closed_cost[end] < best[0]:
            best = (float(closed_cost[end]), path[::-1])
    return np.asarray([c[p] for c, p in zip(candidates, best[1])]), best[0]


def _prominence(rim, sections):
    values = []
    for peak, (segments, radial) in zip(rim, sections):
        dz = segments[:, 1, 2] - segments[:, 0, 2]
        sides = []
        for height in (peak[2] - .75, peak[2] + .75):
            fraction = np.divide(
                height - segments[:, 0, 2], dz,
                out=np.full(len(dz), np.nan), where=np.abs(dz) > 1e-10,
            )
            ok = (fraction >= 0) & (fraction <= 1)
            crossing = segments[ok, 0] + fraction[ok, None] * (segments[ok, 1] - segments[ok, 0])
            sides.append(float(peak @ radial - np.max(crossing @ radial)) if len(crossing) else np.nan)
        values.append(sides)
    return np.asarray(values, float)


def _prominence_valid(prominence):
    return np.all(np.isfinite(prominence) & (prominence > .05), axis=1)


def _repair_weak_arc(rim, candidates, valid):
    """Repair weak DP excursions using the neighboring valid shoulder trajectory.

    A common IOS case has a short angular interval where the cyclic DP climbs to
    the abutment top.  The actual cervical shoulder still exists in the same ray's
    feature candidates.  For those rays we interpolate the expected radius/height
    from the nearest valid rays and choose the candidate closest to that expected
    cervical trajectory.  This preserves a real feature point instead of drawing
    a synthetic free-space bridge.
    """
    result = np.array(rim, copy=True)
    count = len(result)
    if count < 3 or int(np.count_nonzero(valid)) < 2:
        return result, 0
    repaired = 0
    for index in np.flatnonzero(~valid):
        left = right = None
        for step in range(1, count):
            probe = (index - step) % count
            if valid[probe]:
                left = probe
                break
        for step in range(1, count):
            probe = (index + step) % count
            if valid[probe]:
                right = probe
                break
        if left is None or right is None or left == right or not len(candidates[index]):
            continue
        span = (right - left) % count
        offset = (index - left) % count
        if span <= 0 or offset <= 0 or offset >= span:
            continue
        t = offset / span
        radius = (1 - t) * np.linalg.norm(rim[left, :2]) + t * np.linalg.norm(rim[right, :2])
        height = (1 - t) * rim[left, 2] + t * rim[right, 2]
        pool = np.asarray(candidates[index], float)
        pool_radius = np.linalg.norm(pool[:, :2], axis=1)
        # Height is weighted more strongly because the failure mode is a climb
        # from the cervical shoulder onto the occlusal/top interface.
        score = (pool_radius - radius) ** 2 + 4.0 * (pool[:, 2] - height) ** 2
        result[index] = pool[int(np.argmin(score))]
        repaired += 1
    return result, repaired


def extract_abutment_margin(mesh, axis, origin=None, *, strict_qc=False):
    """Return a surface-bound shoulder candidate plus measured diagnostics.

    ``strict_qc=False`` is the CAD-1 default: quality heuristics are warnings and
    never block a test-generation run once a complete finite candidate exists.
    Set ``strict_qc=True`` only for an explicit review workflow that wants the old
    fail-closed behavior.
    """
    axis = np.asarray(axis, float)
    origin = np.asarray(mesh.centroid if origin is None else origin, float)
    if (axis.shape != (3,) or origin.shape != (3,) or
            not np.isfinite([axis, origin]).all() or np.linalg.norm(axis) < 1e-8):
        raise ValueError('基台中心或方向无效')
    axis = axis / np.linalg.norm(axis)
    reference = np.eye(3)[np.argmin(np.abs(axis))]
    x = np.cross(axis, reference); x /= np.linalg.norm(x)
    basis = np.stack([x, np.cross(axis, x), axis], axis=1)
    local = trimesh.Trimesh((mesh.vertices - origin) @ basis, mesh.faces, process=False)
    if not len(local.face_adjacency):
        raise ValueError('基台缺少有效面邻接，请检查独立基台网格')
    selected = (np.degrees(local.face_adjacency_angles) >= 20) & local.face_adjacency_convex
    features = local.vertices[local.face_adjacency_edges[selected]]
    if not len(features):
        raise ValueError('未找到可用的基台肩部特征')

    candidates, sections = [], []
    angles = np.linspace(0, 2 * np.pi, 128, endpoint=False)
    for theta in angles:
        radial = np.array([np.cos(theta), np.sin(theta), 0.])
        normal = np.array([-np.sin(theta), np.cos(theta), 0.])
        delta = (features[:, 1] - features[:, 0]) @ normal
        fraction = np.divide(-features[:, 0] @ normal, delta,
                             out=np.full(len(delta), np.nan), where=np.abs(delta) > 1e-12)
        ok = (fraction >= -1e-8) & (fraction <= 1 + 1e-8)
        crossing = features[ok, 0] + np.clip(fraction[ok], 0, 1)[:, None] * (features[ok, 1] - features[ok, 0])
        positive = crossing[crossing @ radial > 0]
        if not len(positive):
            # This is a true geometry failure: there is no point from which a
            # closed 128-ray shoulder can be constructed.
            raise ValueError('基台肩部特征在部分方向完全缺失，无法形成闭合候选')
        candidates.append(positive)
        sections.append((trimesh.intersections.mesh_plane(local, normal, np.zeros(3)), radial))

    rim, energy = _cyclic_path(candidates)
    prominence_before = _prominence(rim, sections)
    valid_before = _prominence_valid(prominence_before)
    weak_before = np.flatnonzero(~valid_before)

    repaired_count = 0
    if len(weak_before):
        repaired_rim, repaired_count = _repair_weak_arc(rim, candidates, valid_before)
        if repaired_count:
            rim = repaired_rim

    prominence = _prominence(rim, sections)
    valid_after = _prominence_valid(prominence)
    weak_after = np.flatnonzero(~valid_after)
    warnings = []
    if len(weak_before):
        warnings.append({
            'code': 'SHOULDER_PROMINENCE_WEAK_ARC',
            'count_before': int(len(weak_before)),
            'count_after': int(len(weak_after)),
            'repaired_points': int(repaired_count),
            'indices_before': weak_before.tolist(),
            'indices_after': weak_after.tolist(),
            'note': '弱肩台弧已按相邻有效颈部轨迹回选真实特征点；CAD-1 不因该审核项中断生成',
        })
    if strict_qc and len(weak_after):
        raise ValueError('候选未形成完整肩部外缘，请人工核对描线')

    world = rim @ basis.T + origin
    following = np.roll(world, -1, axis=0)
    max_step = float(np.linalg.norm(following - world, axis=1).max())
    if max_step >= .5:
        warnings.append({
            'code': 'SHOULDER_LARGE_NEIGHBOR_STEP',
            'value_mm': max_step,
            'limit_mm': .5,
            'note': '仅作为工程提示，不阻断 CAD-1 生成',
        })
        if strict_qc:
            raise ValueError('肩线存在不连续跳跃，请手动纠正')

    _, distance, _ = trimesh.proximity.closest_point(mesh, world)
    samples = np.concatenate([world + t * (following - world) for t in (.25, .5, .75)])
    _, interior_distance, _ = trimesh.proximity.closest_point(mesh, samples)
    surface_warning = bool(distance.max() > 1e-6 or interior_distance.max() >= .03)
    if surface_warning:
        warnings.append({
            'code': 'SHOULDER_SURFACE_ADHERENCE_REVIEW',
            'point_max_mm': float(distance.max()),
            'segment_sample_max_mm': float(interior_distance.max()),
            'note': '候选仍可用于工程生成；最终结果在 3D 中复核',
        })
        if strict_qc:
            raise ValueError('肩线未可靠贴合基台表面，请手动纠正')

    finite_prominence = prominence[np.isfinite(prominence)]
    min_pair = []
    for column in range(2):
        values = prominence[:, column]
        values = values[np.isfinite(values)]
        min_pair.append(float(values.min()) if len(values) else None)

    return world.tolist(), {
        'algorithm_version': ALGORITHM_VERSION,
        'status': 'REVIEW_REQUIRED' if warnings else 'PASS_ENGINEERING',
        'workflow_blocked': False,
        'strict_qc': bool(strict_qc),
        'point_count': len(world),
        'height_span_mm': float(np.ptp(rim[:, 2])),
        'height_min_mm': float(rim[:, 2].min()),
        'height_max_mm': float(rim[:, 2].max()),
        'radius_min_mm': float(np.linalg.norm(rim[:, :2], axis=1).min()),
        'radius_max_mm': float(np.linalg.norm(rim[:, :2], axis=1).max()),
        'max_neighbor_step_mm': max_step,
        'point_surface_distance_max_mm': float(distance.max()),
        'segment_sample_surface_distance_max_mm': float(interior_distance.max()),
        'two_sided_prominence_min_mm': min_pair,
        'weak_prominence_count_before': int(len(weak_before)),
        'weak_prominence_count_after': int(len(weak_after)),
        'repaired_weak_arc_points': int(repaired_count),
        'warnings': warnings,
        'loop_energy': energy,
        'axis': axis.tolist(),
        'origin': origin.tolist(),
        'clinical_accuracy_assessed': False,
        'release_status': 'ENGINEERING_CANDIDATE',
    }
