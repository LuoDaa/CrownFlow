"""验收套件的 fixture 解析与 baseline 加载。

验收套件需要**仓库外**的真实病例数据（患者数据不进 Git）。路径按以下顺序解析：

1. 环境变量（CI / 其他机器）
2. 本项目所在工作区的已知绝对路径（当前开发机）
3. 找不到 -> 带明确理由 skip，而不是静默通过
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

BASELINE_PATH = Path(__file__).resolve().parent / "baselines" / "baseline.json"

_ENV_FDI36 = "CROWNFLOW_FDI36_DIR"
_ENV_EXOCAD = "CROWNFLOW_EXOCAD_CASES_DIR"

_DEFAULT_FDI36 = Path(r"D:\code\牙冠自动生成\数据\1629-王煜-STL")
_DEFAULT_EXOCAD = Path(r"D:\code\牙冠自动生成\数据\15个完整病例")


def _resolve(env_name: str, default: Path) -> Path | None:
    value = os.environ.get(env_name)
    if value:
        p = Path(value)
        return p if p.exists() else None
    return default if default.exists() else None


def fdi36_dir() -> Path | None:
    return _resolve(_ENV_FDI36, _DEFAULT_FDI36)


def exocad_cases_dir() -> Path | None:
    return _resolve(_ENV_EXOCAD, _DEFAULT_EXOCAD)


def require_fdi36() -> Path:
    p = fdi36_dir()
    if p is None:
        pytest.skip(
            f"FDI36 基准病例不可用。请设置 {_ENV_FDI36} 指向病例目录"
            f"（期望包含 01_主要病例模型/1629-王煜 0.stl 与 1.stl）。"
            f"默认查找位置 {_DEFAULT_FDI36} 不存在。"
            "患者数据不进入 Git，因此这里 skip 而不是失败 —— 但几何验收断言在"
            "该数据可用前无法生效，这一点是已知的。"
        )
    return p


def require_exocad_cases() -> Path:
    p = exocad_cases_dir()
    if p is None:
        pytest.skip(
            f"15 例 exocad 证据数据不可用。请设置 {_ENV_EXOCAD} 指向病例根目录。"
            f"默认查找位置 {_DEFAULT_EXOCAD} 不存在。"
        )
    return p


@pytest.fixture(scope="session")
def fdi36_case_dir() -> Path:
    return require_fdi36()


@pytest.fixture(scope="session")
def exocad_root() -> Path:
    return require_exocad_cases()


@pytest.fixture(scope="session")
def fdi36_meshes(fdi36_case_dir):
    from tests.acceptance import _metrics
    return _metrics.load_fdi36(fdi36_case_dir)


@pytest.fixture(scope="session")
def lab_profile():
    from app.parameters.profiles import load_lab_profile, profile_path
    path = profile_path("inhouse_76007")
    if not path.is_file():
        pytest.skip(
            f"Lab Profile 不存在: {path}。请先运行 "
            "`python -m app.parameters.build_lab_profile --raw <病例目录> --lab-id inhouse_76007`"
        )
    return load_lab_profile(path)


@pytest.fixture(scope="session")
def baseline() -> dict:
    if not BASELINE_PATH.is_file():
        pytest.skip(
            f"baseline.json 不存在: {BASELINE_PATH}。请运行 "
            "`python -m tests.acceptance.baselines.regenerate` 生成。"
        )
    return json.loads(BASELINE_PATH.read_text(encoding="utf-8"))


#: 测试函数名 -> baseline.json ``known_defects`` 里的键。
#:
#: 用集中映射而不是在每个测试上写 xfail 装饰器，是因为标记必须与 baseline 的
#: 文字理由保持同源；修好一项后 XPASS 会因 strict 模式报错，强制同时清掉两处。
KNOWN_DEFECT_TESTS: dict[str, str] = {
    # 原 "test_insertion_axis_error_within_limit" -> insertion_axis_error_deg 已移除：
    # 该判据（vs +Z < 10°）不成立，+Z 不是真值。取而代之的四项就位方向断言
    # 全部如实通过，不属已知缺陷。
    "test_margin_band_clearance_within_range": "margin_band_clearance_median_mm",
    "test_supra_margin_clearance_within_limit": "supra_margin_clearance_p95_mm",
    # test_isolated_chain_preserves_outer_anatomy 已于 2026-10-12 移除：
    # 该缺陷属于退役链路的 enforce_clearance（不再被任何链路调用），
    # live 链路的同一断言（test_live_chain_preserves_outer_anatomy）实测通过。
}


def pytest_collection_modifyitems(config, items):
    """按 baseline.known_defects 给已知缺陷项施加 ``xfail(strict=True)``。

    strict=True 是刻意的：任何一项意外通过都会让测试报错，从而强制把该项从
    ``known_defects`` 里删掉并转为正常断言。目标是 xfail 数量归零。
    """
    if not BASELINE_PATH.is_file():
        return
    try:
        known = json.loads(BASELINE_PATH.read_text(encoding="utf-8")).get("known_defects", {})
    except (OSError, ValueError):
        return

    for item in items:
        key = KNOWN_DEFECT_TESTS.get(item.name)
        if key and key in known:
            item.add_marker(pytest.mark.xfail(strict=True,
                                              reason=f"[已知缺陷] {key}: {known[key]}"))
