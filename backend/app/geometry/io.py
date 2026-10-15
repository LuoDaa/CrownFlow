from pathlib import Path
import numpy as np
import trimesh

def load_mesh(path: Path, process: bool = True, *, file_type: str | None = None) -> trimesh.Trimesh:
    """读取网格。

    ``file_type`` 显式指定格式（如 ``"stl"``），用于**文件名没有扩展名**的场景。
    这不是假想情况：3Shape 订单包里物理文件名就是 ``{GUID}`` 本身、不带后缀，
    靠扩展名分派会直接 ``NotImplementedError: file_type '...{A6}' not supported``。
    调用方有逻辑名（AliasFilenames 还原出来的）时，应当把后缀传进来。
    """
    suffix = (file_type or path.suffix).lower().lstrip(".")
    if suffix == "dcm":
        from hpsdecode import load_hps
        try:
            _, scan = load_hps(str(path))
        except Exception as exc:
            if "Encryption key not found" in str(exc):
                raise ValueError("此 DCM 使用 3Shape CE 加密；请配置合法的 HPS_ENCRYPTION_KEY，或从原软件导出 STL") from exc
            raise ValueError("无法读取此 DCM；仅支持 3Shape HPS 网格扫描，不支持 DICOM 影像") from exc
        obj = trimesh.Trimesh(vertices=scan.vertices, faces=scan.faces, process=process)
    elif file_type:
        obj = trimesh.load(path, file_type=file_type, force="mesh", process=process)
    else:
        obj = trimesh.load(path, force="mesh", process=process)
    if not isinstance(obj, trimesh.Trimesh) or len(obj.faces) == 0:
        raise ValueError("不是有效的三角网格")
    return obj

def save_mesh(mesh: trimesh.Trimesh, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(path)

def mesh_info(mesh_or_path) -> dict:
    mesh = load_mesh(mesh_or_path) if isinstance(mesh_or_path, Path) else mesh_or_path
    ext = mesh.bounds[1] - mesh.bounds[0]
    try:
        comps = len(mesh.split(only_watertight=False))
    except Exception:
        comps = None
    return {
        "vertices": int(len(mesh.vertices)),
        "faces": int(len(mesh.faces)),
        "extents_mm": [float(x) for x in ext],
        "bounds_min": [float(x) for x in mesh.bounds[0]],
        "bounds_max": [float(x) for x in mesh.bounds[1]],
        "components": comps,
        "watertight": bool(mesh.is_watertight),
        "volume_mm3": float(abs(mesh.volume)) if mesh.is_watertight else None,
    }

def repair_mesh(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    m = mesh.copy()
    m.remove_unreferenced_vertices()
    m.merge_vertices()
    try:
        trimesh.repair.fix_normals(m, multibody=True)
        trimesh.repair.fix_winding(m)
        trimesh.repair.fill_holes(m)
    except Exception:
        pass
    return m

def transform_points(points, matrix):
    pts = np.asarray(points, dtype=float)
    hom = np.c_[pts, np.ones(len(pts))]
    return (np.asarray(matrix, dtype=float) @ hom.T).T[:, :3]
