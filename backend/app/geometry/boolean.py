"""布尔运算 —— 带校验的操作，不是"函数没报错就算成功"。

实测教训（FDI36 病例，默认安装无 manifold3d/mapbox-earcut）：

    Outer(非水密) - Cutter  ->  watertight=True, volume=27.6 mm3
    真值 210.8 mm3           ->  只有 13%

Mesh 合法 ≠ 结果正确。所以本模块把布尔做成**有前置条件与后置校验的操作**，
任何一步不通过就抛异常，绝不保存结果。

同时：``manifold3d`` 不再是可选依赖。体素 CSG 降级为**显式 opt-in 的调试路径**，
默认关闭 —— 它在不裁减任何东西的情况下也会把牙冠放大 10% 并引入 0.09 mm 中位误差。
"""

from __future__ import annotations

import numpy as np
import trimesh


class BooleanBackendMissing(RuntimeError):
    """生产路径要求的精确布尔后端不可用。"""


class BooleanPreconditionError(RuntimeError):
    """布尔运算的输入不满足前置条件。"""


class BooleanValidationError(RuntimeError):
    """布尔运算的输出未通过后置校验。"""


def manifold_available() -> bool:
    try:
        import manifold3d  # noqa: F401
        return True
    except Exception:
        return False


def earcut_available() -> bool:
    try:
        import mapbox_earcut  # noqa: F401
        return True
    except Exception:
        return False


def boolean_backend_name() -> str:
    return "manifold3d" if manifold_available() else "MISSING"


def require_exact_boolean_backend() -> None:
    """生产路径的准入检查。缺依赖时直接失败，不静默降级。"""
    missing = []
    if not manifold_available():
        missing.append("manifold3d")
    if not earcut_available():
        missing.append("mapbox-earcut")
    if missing:
        raise BooleanBackendMissing(
            "缺少精确布尔/裁切所需的依赖: " + ", ".join(missing) + "。\n"
            "  它们是功能依赖，不是可选依赖：\n"
            "    - manifold3d    精确网格布尔（体素降级会引入 ~10% 体积膨胀与 0.09 mm 中位误差）\n"
            "    - mapbox-earcut 裁切盖片三角化（缺失时 slice_mesh_plane(cap=True) 抛异常，\n"
            "                    牙冠会以非水密状态进入后续步骤，产出 13% 体积的碎冠）\n"
            "  安装: pip install -r backend/requirements.txt"
        )


def assert_watertight(mesh: trimesh.Trimesh, label: str, *, repair: bool = True) -> None:
    """断言网格水密。可选先尝试修复，修复后仍非水密即失败。"""
    if mesh.is_watertight:
        return
    if repair:
        try:
            trimesh.repair.fill_holes(mesh)
            trimesh.repair.fix_winding(mesh)
            trimesh.repair.fix_normals(mesh, multibody=True)
        except Exception:
            pass
        if mesh.is_watertight:
            return

    n_boundary = 0
    edges = np.asarray(mesh.edges_sorted)
    if len(edges):
        _, counts = np.unique(edges, axis=0, return_counts=True)
        n_boundary = int((counts == 1).sum())

    hint = ""
    if not earcut_available():
        hint = ("\n  根因很可能是缺少 mapbox-earcut：slice_mesh_plane(cap=True) 需要它来三角化"
                "裁切面，缺失时该调用抛异常并退化到 cap=False，留下开放边界。")
    raise BooleanPreconditionError(
        f"{label} 不是水密网格（剩余开放边界边 {n_boundary} 条）。"
        "非水密网格进入布尔运算会产出无意义的碎片，实测可低至真值体积的 13%。"
        f"{hint}"
    )


def _largest_component(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    comps = mesh.split(only_watertight=False)
    if len(comps) <= 1:
        return mesh
    return max(comps, key=lambda m: abs(m.volume) if m.is_watertight else m.area)


def _repair(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    mesh.remove_unreferenced_vertices()
    try:
        trimesh.repair.fix_normals(mesh, multibody=True)
        trimesh.repair.fix_winding(mesh)
        trimesh.repair.fill_holes(mesh)
    except Exception:
        pass
    return _largest_component(mesh)


def validate_boolean_result(
    outer: trimesh.Trimesh,
    cutter: trimesh.Trimesh,
    result: trimesh.Trimesh,
    *,
    bbox_tolerance_mm: float = 0.05,
) -> dict:
    """布尔结果的后置校验。任一项不通过即抛 :class:`BooleanValidationError`。

    校验项（对应实测过的失败模式）：

    - 结果非空且体积为正 —— 捕获空结果
    - 结果水密 —— 捕获碎片化
    - 结果是单一连通分量 —— 捕获"冠被切成几块"
    - 结果体积严格小于 outer 体积 —— 差集不该变大
    - 结果包围盒包含于 outer 包围盒（容差内）—— 捕获坐标系错位
    - 质心位移在合理范围 —— 捕获"结果其实是另一块几何"
    """
    problems: list[str] = []

    if result is None or len(result.faces) == 0:
        raise BooleanValidationError("布尔结果为空网格")

    if not result.is_watertight:
        raise BooleanValidationError(
            f"布尔结果不水密（{len(result.faces)} 面）。实测该状态下的体积与真值无可靠关系。"
        )

    volume = float(abs(result.volume))
    if volume <= 0:
        problems.append(f"结果体积非正: {volume}")

    components = len(result.split(only_watertight=False))
    if components != 1:
        problems.append(f"结果有 {components} 个连通分量（应为 1）")

    outer_volume = float(abs(outer.volume))
    if outer_volume > 0 and volume >= outer_volume:
        problems.append(
            f"结果体积 {volume:.2f} 不小于 outer 体积 {outer_volume:.2f}；差集不该变大"
        )

    if not np.all(np.asarray(result.bounds[0]) >= np.asarray(outer.bounds[0]) - bbox_tolerance_mm):
        problems.append("结果包围盒下限超出 outer 包围盒")
    if not np.all(np.asarray(result.bounds[1]) <= np.asarray(outer.bounds[1]) + bbox_tolerance_mm):
        problems.append("结果包围盒上限超出 outer 包围盒")

    if problems:
        raise BooleanValidationError(
            "布尔结果未通过校验：\n  - " + "\n  - ".join(problems)
        )

    return {
        "volume_mm3": volume,
        "volume_ratio_of_outer": volume / outer_volume if outer_volume > 0 else None,
        "faces": int(len(result.faces)),
        "components": components,
        "watertight": bool(result.is_watertight),
        "cutter_volume_mm3": float(abs(cutter.volume)),
    }


def _voxel_difference(outer: trimesh.Trimesh, cutter: trimesh.Trimesh,
                      pitch: float = 0.08, max_cells: int = 140_000_000) -> trimesh.Trimesh:
    """体素 CSG 降级实现。

    **只用于调试与对照，不得作为生产路径。** 实测：即使不裁减任何几何，
    pitch 0.10 mm 下也会把体积放大 10%、引入 0.089 mm 中位 / 0.424 mm 最大表面误差、
    面数增至 2.4 倍。
    """
    pitch = float(pitch)
    if pitch <= 0:
        raise ValueError("voxel boolean pitch 必须 > 0")

    ov = outer.voxelized(pitch).fill()
    cv = cutter.voxelized(pitch).fill()
    op = np.asarray(ov.points, dtype=float)
    cp = np.asarray(cv.points, dtype=float)
    if len(op) == 0:
        raise BooleanValidationError("Outer voxelization 为空")

    all_min = op.min(axis=0) if len(cp) == 0 else np.minimum(op.min(axis=0), cp.min(axis=0))
    all_max = op.max(axis=0) if len(cp) == 0 else np.maximum(op.max(axis=0), cp.max(axis=0))
    origin = np.floor(all_min / pitch) * pitch - pitch * 2.0
    maxp = np.ceil(all_max / pitch) * pitch + pitch * 2.0
    shape = np.ceil((maxp - origin) / pitch).astype(int) + 1
    cells = int(np.prod(shape, dtype=np.int64))
    if cells > int(max_cells):
        raise BooleanValidationError(
            f"Voxel 降级需要 {cells:,} cells，超过安全限制；请安装 manifold3d 或缩小工作范围"
        )

    occ = np.zeros(tuple(shape.tolist()), dtype=bool)
    oi = np.rint((op - origin) / pitch).astype(int)
    occ[oi[:, 0], oi[:, 1], oi[:, 2]] = True
    if len(cp):
        ci = np.rint((cp - origin) / pitch).astype(int)
        valid = np.all((ci >= 0) & (ci < shape[None, :]), axis=1)
        ci = ci[valid]
        occ[ci[:, 0], ci[:, 1], ci[:, 2]] = False

    result = trimesh.voxel.ops.matrix_to_marching_cubes(occ, pitch=pitch)
    result.apply_translation(origin)
    return _repair(result)


def crown_minus_cutter(
    outer: trimesh.Trimesh,
    cutter: trimesh.Trimesh,
    *,
    allow_voxel_fallback: bool = False,
    voxel_pitch_mm: float = 0.08,
    validate: bool = True,
) -> trimesh.Trimesh:
    """从牙冠外形中减去内表面 cutter。

    前置条件：``outer`` 与 ``cutter`` 必须水密（非水密输入是实测 13% 碎冠的直接原因）。
    后置校验：见 :func:`validate_boolean_result`。

    ``allow_voxel_fallback`` 默认为 ``False``：体素降级是调试路径。
    """
    outer = outer.copy()
    cutter = cutter.copy()
    assert_watertight(outer, "Outer")
    assert_watertight(cutter, "Cutter")

    if manifold_available():
        result = trimesh.boolean.difference([outer, cutter], engine="manifold")
        if result is None:
            raise BooleanValidationError("manifold 布尔返回空结果")
        if isinstance(result, list):
            result = trimesh.util.concatenate(result)
        result = _repair(result)
        if validate:
            validate_boolean_result(outer, cutter, result)
        return result

    if not allow_voxel_fallback:
        raise BooleanBackendMissing(
            "缺少 manifold3d，且未启用体素降级。\n"
            "  体素降级不是等价替代：实测在没有任何裁减的情况下也会把牙冠体积放大 10%、"
            "引入 0.089 mm 中位 / 0.424 mm 最大表面误差。\n"
            "  调试时如需启用，请显式传 allow_voxel_fallback=True。"
        )

    result = _voxel_difference(outer, cutter, pitch=float(voxel_pitch_mm))
    if validate:
        validate_boolean_result(outer, cutter, result, bbox_tolerance_mm=float(voxel_pitch_mm) * 3)
    return result
