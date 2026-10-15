"""Nonplanar outer shoulder on an existing restorative abutment, always reviewed."""
import numpy as np
import trimesh

ALGORITHM_VERSION = 'welooai/abutment-shoulder/1.0'


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


def extract_abutment_margin(mesh, axis, origin=None):
    """Return surface-bound candidate + measured QC; do not change input geometry.

    shortcut: validated on clear outward shoulders, rounded/ambiguous shoulders require manual editing.
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
        raise ValueError('未找到清晰的基台肩部外缘，请手动描线')
    candidates, sections = [], []
    for theta in np.linspace(0, 2 * np.pi, 128, endpoint=False):
        radial = np.array([np.cos(theta), np.sin(theta), 0.])
        normal = np.array([-np.sin(theta), np.cos(theta), 0.])
        delta = (features[:, 1] - features[:, 0]) @ normal
        fraction = np.divide(-features[:, 0] @ normal, delta,
                             out=np.full(len(delta), np.nan), where=np.abs(delta) > 1e-12)
        ok = (fraction >= -1e-8) & (fraction <= 1 + 1e-8)
        crossing = features[ok, 0] + np.clip(fraction[ok], 0, 1)[:, None] * (features[ok, 1] - features[ok, 0])
        positive = crossing[crossing @ radial > 0]
        if not len(positive):
            raise ValueError('基台肩部特征不完整，请补充描线；未生成替代截面')
        candidates.append(positive)
        sections.append((trimesh.intersections.mesh_plane(local, normal, np.zeros(3)), radial))
    rim, energy = _cyclic_path(candidates)
    prominence = []
    for peak, (segments, radial) in zip(rim, sections):
        dz = segments[:, 1, 2] - segments[:, 0, 2]
        sides = []
        for height in (peak[2] - .75, peak[2] + .75):
            fraction = np.divide(height - segments[:, 0, 2], dz,
                                 out=np.full(len(dz), np.nan), where=np.abs(dz) > 1e-10)
            ok = (fraction >= 0) & (fraction <= 1)
            crossing = segments[ok, 0] + fraction[ok, None] * (segments[ok, 1] - segments[ok, 0])
            sides.append(float(peak @ radial - np.max(crossing @ radial)) if len(crossing) else np.nan)
        prominence.append(sides)
    prominence = np.asarray(prominence)
    if not np.all(np.isfinite(prominence) & (prominence > .05)):
        raise ValueError('候选未形成完整肩部外缘，可能落在孔口或接口；请手动核对描线')
    world = rim @ basis.T + origin
    following = np.roll(world, -1, axis=0)
    max_step = float(np.linalg.norm(following - world, axis=1).max())
    if max_step >= .5:
        raise ValueError('肩线存在不连续跳跃，请手动纠正；未自动连接缺失段')
    _, distance, _ = trimesh.proximity.closest_point(mesh, world)
    samples = np.concatenate([world + t * (following - world) for t in (.25, .5, .75)])
    _, interior_distance, _ = trimesh.proximity.closest_point(mesh, samples)
    if distance.max() > 1e-6 or interior_distance.max() >= .03:
        raise ValueError('肩线未可靠贴合基台表面，请手动纠正')
    return world.tolist(), {
        'algorithm_version': ALGORITHM_VERSION, 'status': 'REVIEW_REQUIRED',
        'point_count': len(world), 'height_span_mm': float(np.ptp(rim[:, 2])),
        'max_neighbor_step_mm': max_step, 'point_surface_distance_max_mm': float(distance.max()),
        'segment_sample_surface_distance_max_mm': float(interior_distance.max()),
        'two_sided_prominence_min_mm': prominence.min(axis=0).tolist(),
        'loop_energy': energy, 'axis': axis.tolist(), 'origin': origin.tolist(),
        'clinical_accuracy_assessed': False,
    }
