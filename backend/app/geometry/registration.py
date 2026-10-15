import warnings

import numpy as np
import trimesh
from scipy.interpolate import RBFInterpolator

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

def initialize_from_margin(template, margin_points, insertion_axis, *, template_cervical_points=None,
                           patient_frame=None):
    """Place a +Z-oriented anatomy template on the confirmed margin plane.

    **A0：横向轴只能来自确认帧。** 传 ``patient_frame``（confirmed mesial/buccal/axis，
    见 ``crown_outer.patient_frame``）时，本函数**不再**自己推断 X/Y —— 直接用该帧的
    x/y/z 作基（并要求帧的 Z 与 ``insertion_axis`` 一致，否则拒绝）。
    只有不传帧的 legacy 路径才回退到"world-X 投影"，且会发 ``DeprecationWarning``：
    那条路径在全局旋转下不等变，live 生成不得再走。
    """
    margin = np.asarray(margin_points, dtype=float)
    axis = np.asarray(insertion_axis, dtype=float)
    if margin.ndim != 2 or margin.shape[0] < 3 or margin.shape[1] != 3 or not np.isfinite(margin).all():
        raise ValueError("自动提案需要至少 3 个有效 Margin 点")
    if axis.shape != (3,) or not np.isfinite(axis).all() or np.linalg.norm(axis) < 1e-8:
        raise ValueError("就位方向无效")
    axis /= np.linalg.norm(axis)
    m = template.copy()
    transform = trimesh.geometry.align_vectors([0, 0, 1], axis)
    m.apply_transform(transform)
    vertices = np.asarray(m.vertices)
    center = margin.mean(axis=0)
    projection = vertices @ axis
    # Fit the cervical footprint first; a translated template can otherwise be narrower
    # than the preparation and produce an empty or disconnected boolean shell.
    if patient_frame is not None:
        basis=np.asarray(patient_frame.matrix(),dtype=float)
        if basis.shape!=(3,3):
            raise ValueError("patient_frame.matrix() 必须是 3x3")
        frame_z=basis[:,2]/max(float(np.linalg.norm(basis[:,2])),1e-12)
        if abs(float(frame_z@axis))<.999:
            raise ValueError("patient_frame 的 Z 与 insertion_axis 不一致：帧与轴必须来自同一次确认")
        # 帧的 X/Y 就是确认的 mesial/buccal，不再从 world 轴推断
    else:
        warnings.warn("initialize_from_margin 正在用 world-X 推断横向轴（legacy 路径）；"
                      "live 生成必须传入 patient_frame", DeprecationWarning, stacklevel=2)
        x=np.array([1.,0,0]);x-=axis*np.dot(x,axis)
        if np.linalg.norm(x)<.1:x=np.cross([0,1,0],axis)
        x/=np.linalg.norm(x);y=np.cross(axis,x)
        basis=np.column_stack((x,y,axis))
    local=vertices@basis
    ring=local[local[:,2]<=local[:,2].min()+.1,:2]
    if template_cervical_points is not None:
        cuff=np.asarray(template_cervical_points,float)
        if cuff.ndim!=2 or cuff.shape[1]!=3 or len(cuff)<3 or not np.isfinite(cuff).all():
            raise ValueError('模板颈缘必须为有效三维闭环点')
        ring=(trimesh.transform_points(cuff,transform)@basis)[:,:2]
    target=margin@basis
    if len(ring)>=3:
        extent=np.ptp(ring,axis=0)
        target_extent=np.ptp(target[:,:2],axis=0)
        if np.any(extent<.1) or np.any(target_extent<.1):raise ValueError('模板或 Margin 颈缘尺寸退化')
        factor=(target_extent+.4)/extent
        if np.any(factor<.4) or np.any(factor>2.5):raise ValueError('模板与基牙尺寸差异过大，请检查模板朝向和 Margin')
        local[:,:2]=(local[:,:2]-ring.mean(0))*factor+ring.mean(0)
        m.vertices=local@basis.T
        vertices=np.asarray(m.vertices);projection=vertices@axis
    horizontal = vertices.mean(axis=0) - axis * projection.mean()
    cervical_height=float(projection.min())
    if template_cervical_points is not None:
        # A known cervical loop is the placement reference, not the anatomy's biased mass center or lowest tip.
        cuff_world=trimesh.transform_points(cuff,transform)
        horizontal=ring.mean(0)@basis[:,:2].T
        cervical_height=float((cuff_world@axis).mean())
    target_horizontal = center - axis * (center @ axis)
    translation = target_horizontal - horizontal + axis * (center @ axis - cervical_height)
    m.apply_translation(translation)
    return m
