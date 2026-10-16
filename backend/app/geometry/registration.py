import warnings

import numpy as np
import trimesh
from scipy.interpolate import RBFInterpolator


DEFAULT_CERVICAL_PADDING_MM = 0.20
DEFAULT_CERVICAL_TRANSITION_MM = 2.20


def kabsch(source_points, target_points):
    src = np.asarray(source_points, dtype=float)
    dst = np.asarray(target_points, dtype=float)
    if src.shape[0] < 3 or dst.shape != src.shape or src.shape[1] != 3:
        raise ValueError("模板点与患者点至少各需要 3 个一一对应的 XYZ 点")
    if np.linalg.matrix_rank(src - src.mean(0)) < 2:
        raise ValueError("模板配准点不能共线")
    if np.linalg.matrix_rank(dst - dst.mean(0)) < 2:
        raise ValueError("患者配准点不能共线")

    cs, ct = src.mean(0), dst.mean(0)
    a, b = src - cs, dst - ct
    u, _, vt = np.linalg.svd(a.T @ b)
    r = vt.T @ u.T
    if np.linalg.det(r) < 0:
        vt[-1] *= -1
        r = vt.T @ u.T
    t = ct - r @ cs

    m = np.eye(4)
    m[:3, :3] = r
    m[:3, 3] = t
    return m


def rigid_then_rbf(mesh, template_points, patient_points,
                   morph_template=None, morph_patient=None, morph_strength=0.0):
    """Optional coarse landmark morphing.

    CAD-1 does not use RBF for neighbor/occlusal contact solving.  Thin-plate RBF
    remains useful only as a coarse anatomy alignment tool when explicit landmark
    correspondences exist; local functional adaptation is handled downstream by
    the constrained Laplacian AutoFit path.
    """
    m = mesh.copy()
    matrix = kabsch(template_points, patient_points)
    m.apply_transform(matrix)

    if morph_strength <= 0 or not morph_template or not morph_patient:
        return m, matrix

    st = np.asarray(morph_template, float)
    tp = np.asarray(morph_patient, float)
    if len(st) < 4 or len(st) != len(tp):
        raise ValueError("RBF 形变至少需要 4 组对应点")

    st_h = np.c_[st, np.ones(len(st))]
    st_world = (matrix @ st_h.T).T[:, :3]
    delta = (tp - st_world) * float(morph_strength)

    rbf = RBFInterpolator(st_world, delta, kernel="thin_plate_spline", degree=1)
    verts = np.asarray(m.vertices)
    m.vertices = verts + rbf(verts)
    return m, matrix


def _smoothstep01(value):
    x = np.clip(np.asarray(value, dtype=float), 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def _tapered_cervical_scale(local, ring_xy, factor, cervical_z, transition_mm):
    """Fit the cervical footprint without globally scaling coronal anatomy.

    Margin geometry owns the emergence/root footprint only.  Scaling the full
    crown in X/Y by the cervical factor couples a narrow preparation to the whole
    bucco-lingual and mesio-distal anatomy and is a direct flattening mechanism.
    Apply the full factor at the cervical reference, then smoothly release it to
    identity over the emergence zone.  The coronal two thirds therefore remain a
    morphology prior unless patient-space constraints later require compression.
    """
    out = np.asarray(local, dtype=float).copy()
    center = np.asarray(ring_xy, dtype=float).mean(axis=0)
    height = np.maximum(out[:, 2] - float(cervical_z), 0.0)
    release = _smoothstep01(height / max(float(transition_mm), 1e-6))
    cervical_weight = 1.0 - release
    per_vertex_factor = 1.0 + cervical_weight[:, None] * (np.asarray(factor, float)[None, :] - 1.0)
    out[:, :2] = center + (out[:, :2] - center) * per_vertex_factor
    return out, cervical_weight


def initialize_from_margin(template, margin_points, insertion_axis, *, template_cervical_points=None,
                           patient_frame=None, cervical_padding_mm=DEFAULT_CERVICAL_PADDING_MM,
                           cervical_transition_mm=DEFAULT_CERVICAL_TRANSITION_MM):
    """Place a library tooth while keeping cervical and coronal ownership separate.

    Responsibilities stop deliberately at initial anatomy placement:

    * +Z template axis -> confirmed insertion axis;
    * template X/Y -> confirmed mesial/buccal patient frame;
    * cervical footprint -> patient Margin footprint, with a tapered emergence fit;
    * cervical reference -> Margin centroid/height.

    Crucially, the Margin footprint is **not** a requested whole-crown M/D or B/L
    size.  Its scale fades to identity through the emergence zone so a narrow
    preparation cannot flatten the buccal/lingual or occlusal anatomy.  Adjacent
    teeth and antagonist remain downstream feasibility/contact constraints.
    """
    margin = np.asarray(margin_points, dtype=float)
    axis = np.asarray(insertion_axis, dtype=float)
    if margin.ndim != 2 or margin.shape[0] < 3 or margin.shape[1] != 3 or not np.isfinite(margin).all():
        raise ValueError("自动提案需要至少 3 个有效 Margin 点")
    if axis.shape != (3,) or not np.isfinite(axis).all() or np.linalg.norm(axis) < 1e-8:
        raise ValueError("就位方向无效")
    if not np.isfinite(cervical_padding_mm) or cervical_padding_mm < 0 or cervical_padding_mm > 1.0:
        raise ValueError("cervical_padding_mm 必须位于 0–1 mm")
    if (not np.isfinite(cervical_transition_mm)
            or cervical_transition_mm < 0.5 or cervical_transition_mm > 5.0):
        raise ValueError("cervical_transition_mm 必须位于 0.5–5.0 mm")

    axis /= np.linalg.norm(axis)
    m = template.copy()
    transform = trimesh.geometry.align_vectors([0, 0, 1], axis)
    m.apply_transform(transform)
    vertices = np.asarray(m.vertices)
    center = margin.mean(axis=0)
    projection = vertices @ axis

    if patient_frame is not None:
        basis = np.asarray(patient_frame.matrix(), dtype=float)
        if basis.shape != (3, 3):
            raise ValueError("patient_frame.matrix() 必须是 3x3")
        frame_z = basis[:, 2] / max(float(np.linalg.norm(basis[:, 2])), 1e-12)
        if abs(float(frame_z @ axis)) < .999:
            raise ValueError("patient_frame 的 Z 与 insertion_axis 不一致：帧与轴必须来自同一次确认")
    else:
        warnings.warn(
            "initialize_from_margin 正在用 world-X 推断横向轴（legacy 路径）；live 生成必须传入 patient_frame",
            DeprecationWarning,
            stacklevel=2,
        )
        x = np.array([1., 0, 0])
        x -= axis * np.dot(x, axis)
        if np.linalg.norm(x) < .1:
            x = np.cross([0, 1, 0], axis)
        x /= np.linalg.norm(x)
        y = np.cross(axis, x)
        basis = np.column_stack((x, y, axis))

    local = vertices @ basis
    ring = local[local[:, 2] <= local[:, 2].min() + .1, :2]
    cuff = None
    cuff_local = None
    if template_cervical_points is not None:
        cuff = np.asarray(template_cervical_points, float)
        if cuff.ndim != 2 or cuff.shape[1] != 3 or len(cuff) < 3 or not np.isfinite(cuff).all():
            raise ValueError("模板颈缘必须为有效三维闭环点")
        cuff_world = trimesh.transform_points(cuff, transform)
        cuff_local = cuff_world @ basis
        ring = cuff_local[:, :2]

    target = margin @ basis
    if len(ring) >= 3:
        extent = np.ptp(ring, axis=0)
        target_extent = np.ptp(target[:, :2], axis=0)
        if np.any(extent < .1) or np.any(target_extent < .1):
            raise ValueError("模板或 Margin 颈缘尺寸退化")
        factor = (target_extent + 2.0 * float(cervical_padding_mm)) / extent
        if np.any(factor < .4) or np.any(factor > 2.5):
            raise ValueError("模板与基牙尺寸差异过大，请检查模板朝向和 Margin")
        cervical_z = float(np.mean(cuff_local[:, 2])) if cuff_local is not None else float(local[:, 2].min())
        local, _cervical_weight = _tapered_cervical_scale(
            local, ring, factor, cervical_z, float(cervical_transition_mm)
        )
        m.vertices = local @ basis.T
        vertices = np.asarray(m.vertices)
        projection = vertices @ axis

    horizontal = vertices.mean(axis=0) - axis * projection.mean()
    cervical_height = float(projection.min())
    if cuff is not None:
        horizontal = ring.mean(0) @ basis[:, :2].T
        cervical_height = float((cuff_world @ axis).mean())

    target_horizontal = center - axis * (center @ axis)
    translation = target_horizontal - horizontal + axis * (center @ axis - cervical_height)
    m.apply_translation(translation)
    return m