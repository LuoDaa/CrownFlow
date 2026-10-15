"""Conservative geometric localization of disconnected implant components.

The scan cannot distinguish a scan body from a definitive restorative abutment.
This module only locates a component and its tentative axis; it never supplies a
clinical margin or an implant system/SKU.
"""

import numpy as np
import trimesh
from scipy.spatial import cKDTree
from trimesh.proximity import closest_point


def component_at_point(mesh, point):
    """Return the connected surface actually picked, or the original mesh."""
    components = mesh.split(only_watertight=False)
    if len(components) < 2:
        return mesh
    _, _, face = closest_point(mesh, np.asarray(point, dtype=float).reshape(1, 3))
    labels = trimesh.graph.connected_component_labels(mesh.face_adjacency, node_count=len(mesh.faces))
    picked = mesh.submesh([np.flatnonzero(labels == labels[int(face[0])])], append=True, repair=False)
    picked.remove_unreferenced_vertices()
    return picked


def _unit(value):
    length = np.linalg.norm(value)
    return value / length if length > 1e-9 else None


def _fit_axis(vertices, normals, initial):
    axis = _unit(initial)
    center = vertices.mean(axis=0)
    for _ in range(20):
        wall = np.abs(normals @ axis) < 0.35
        if wall.sum() < 50:
            break
        points = vertices[wall]
        projection = (points - points.mean(axis=0)) @ axis
        lo, hi = np.percentile(projection, [5, 95])
        points = points[(projection > lo) & (projection < hi)]
        projection = (points - points.mean(axis=0)) @ axis
        edges = np.linspace(projection.min(), projection.max(), 7)
        centers = [points[(projection >= edges[i]) & (projection < edges[i + 1])].mean(axis=0)
                   for i in range(6) if ((projection >= edges[i]) & (projection < edges[i + 1])).sum() > 20]
        if len(centers) < 3:
            break
        _, _, basis = np.linalg.svd(np.asarray(centers) - np.mean(centers, axis=0), full_matrices=False)
        revised = _unit(basis[0])
        if revised is None:
            break
        axis = revised
    projection = (vertices - center) @ axis
    radius = np.linalg.norm(vertices - center - np.outer(projection, axis), axis=1)
    wall = np.abs(normals @ axis) < 0.35
    if wall.sum() < 50:
        return None
    return axis, center, float(np.median(radius[wall])), float(np.std(radius[wall])), float(wall.mean())


def _symmetry(vertices, axis, center):
    points = vertices[::max(1, len(vertices) // 1400)]
    source = cKDTree(points)
    distance = []
    for degrees in (60, 120, 180):
        angle = np.radians(degrees)
        relative = points - center
        rotated = (center + relative * np.cos(angle)
                   + np.cross(axis, relative) * np.sin(angle)
                   + np.outer(relative @ axis, axis) * (1 - np.cos(angle)))
        rotated_tree = cKDTree(rotated)
        distance.append(float(np.median(np.r_[source.query(rotated)[0], rotated_tree.query(points)[0]])))
    return float(np.mean(distance))


def _solid_range(projection):
    """Remove sparse scan tails before reporting the component's physical ends."""
    counts, edges = np.histogram(projection, bins=24)
    occupied = np.flatnonzero(counts >= max(12, 0.18 * counts.max()))
    if len(occupied) < 4:
        return float(np.percentile(projection, 2)), float(np.percentile(projection, 98))
    return float(edges[occupied[0]]), float(edges[occupied[-1] + 1])


def implant_candidates(jaw, opposing=None):
    """Return [(isolated mesh, JSON-ready evidence)], strongest candidate first."""
    pieces = sorted(jaw.split(only_watertight=False), key=lambda mesh: len(mesh.faces), reverse=True)
    if len(pieces) < 2:
        return []
    main = pieces[0]
    occlusal = _unit(opposing.centroid - main.centroid) if opposing is not None else None
    found = []
    for component in pieces[1:]:
        if len(component.faces) < 800 or len(component.faces) >= 0.8 * len(main.faces):
            continue
        vertices = np.asarray(component.vertices, dtype=float)
        normals = np.asarray(component.vertex_normals, dtype=float)
        if len(vertices) < 100 or not np.isfinite(vertices).all():
            continue
        eigenvalues, eigenvectors = np.linalg.eigh(np.cov(vertices.T))
        fits = [_fit_axis(vertices, normals, eigenvectors[:, i]) for i in range(3)]
        fits = [fit for fit in fits if fit is not None and fit[2] > 0]
        if not fits:
            continue
        axis, center, radius, spread, wall_ratio = min(
            fits, key=lambda fit: _symmetry(vertices, fit[0], fit[1]))
        projection = (vertices - center) @ axis
        lo, hi = _solid_range(projection)
        length = hi - lo
        if not (1.3 <= radius <= 3.6 and 4.5 <= length <= 16 and spread / radius <= 0.32):
            continue
        wall_projection = projection[np.abs(normals @ axis) < 0.35]
        wall_span = float(np.diff(np.percentile(wall_projection, [5, 95]))[0] / length)
        symmetry = _symmetry(vertices, axis, center)
        if symmetry > 0.48 or wall_ratio < 0.28:
            continue
        samples = vertices[::max(1, len(vertices) // 800)]
        _, distance, _ = closest_point(main, samples)
        minimum, median = float(distance.min()), float(np.median(distance))
        if minimum > 0.25 or median > 2.0:
            continue
        bottom = vertices[(projection >= lo) & (projection < lo + 0.8)]
        top = vertices[(projection <= hi) & (projection > hi - 0.5)]
        if len(bottom) < 8 or len(top) < 8:
            continue
        base_center, top_center = bottom.mean(axis=0), top.mean(axis=0)
        _, base_distance, _ = closest_point(main, base_center.reshape(1, 3))
        _, top_distance, _ = closest_point(main, top_center.reshape(1, 3))
        if base_distance[0] > top_distance[0]:
            axis = -axis
            base_center, top_center = top_center, base_center
        angle = None
        if occlusal is not None:
            angle = float(np.degrees(np.arccos(np.clip(axis @ occlusal, -1, 1))))
        # shortcut: geometry locates a component but cannot prove its restorative role; upgrade with validated order/library identity.
        high = (symmetry <= 0.42 and wall_ratio >= 0.35 and wall_span >= 0.54 and spread / radius <= 0.22
                and minimum <= 0.08 and (angle is None or angle <= 65))
        summary = {
            'center': [float(v) for v in component.centroid],
            'base_center': [float(v) for v in base_center],
            'top_center': [float(v) for v in top_center],
            'axis': [float(v) for v in axis],
            'radius_mm': radius, 'length_mm': length,
            'classification': 'implant_component_candidate',
            'confidence': 'high' if high else 'review',
            'evidence': {'component_faces': int(len(component.faces)),
                         'jaw_contact_min_mm': minimum, 'jaw_contact_median_mm': median,
                         'cylinder_wall_ratio': wall_ratio,
                         'cylinder_wall_span_ratio': wall_span,
                         'radius_std_mm': spread,
                         'rotational_symmetry_chamfer_mm': symmetry,
                         'axis_angle_to_opposing_deg': angle,
                         'length_estimate_method': 'occupied_axial_bins',
                         'full_projected_length_mm': float(projection.max()-projection.min())},
        }
        found.append((component, summary))
    return sorted(found, key=lambda item: (item[1]['confidence'] != 'high',
                                           item[1]['evidence']['rotational_symmetry_chamfer_mm']))


def uploaded_support_candidate(support, jaw, opposing=None):
    """Locate a supplied support for review when the jaw has no separate piece."""
    vertices = np.asarray(support.vertices, dtype=float)
    if len(vertices) < 100 or not np.isfinite(vertices).all():
        raise ValueError('上传的独立基台网格无效')
    center = vertices.mean(axis=0)
    _, _, basis = np.linalg.svd(vertices - center, full_matrices=False)
    axis = basis[0]
    occlusal = _unit(opposing.centroid - jaw.centroid) if opposing is not None else None
    projection = (vertices - center) @ axis
    lo, hi = _solid_range(projection)
    base = vertices[(projection >= lo) & (projection < lo + 0.8)]
    top = vertices[(projection <= hi) & (projection > hi - 0.5)]
    if len(base) < 4 or len(top) < 4:
        raise ValueError('上传基台缺少足够的端部网格，无法确定位置')
    base_center, top_center = base.mean(axis=0), top.mean(axis=0)
    _, bd, _ = closest_point(jaw, base_center.reshape(1, 3))
    _, td, _ = closest_point(jaw, top_center.reshape(1, 3))
    if bd[0] > td[0] or (abs(bd[0]-td[0]) < 0.1 and occlusal is not None and axis @ occlusal < 0):
        axis = -axis
        base_center, top_center = top_center, base_center
    radius = np.linalg.norm(vertices - center - np.outer(projection, axis), axis=1)
    summary = {'center':[float(v) for v in support.centroid],
               'base_center':[float(v) for v in base_center],
               'top_center':[float(v) for v in top_center],
               'axis':[float(v) for v in axis],
               'radius_mm':float(np.median(radius)), 'length_mm':hi-lo,
               'classification':'uploaded_restorative_abutment_candidate',
               'confidence':'review',
               'evidence':{'component_faces':int(len(support.faces)),
                           'source':'user_uploaded_support',
                           'identity_and_margin_verified':False}}
    return support, summary
