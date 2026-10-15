"""Lab Profile 的读写与 PHI 校验。

Profile 文件是**可提交进 Git 的构建产物**：只含参数名、数值、计数与依据规模，
不含病例目录名、患者姓名、订单号。写入前有强制校验。
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from .models import LabProfile

#: Profile 中不允许出现的字符类别 —— 出现即说明混入了 PHI。
_CJK_RE = re.compile(r"[\u4e00-\u9fff]")
_UUID_RE = re.compile(r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
                      r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b")
_UNC_RE = re.compile(r"\\\\\\\\")

PROFILE_DIR = Path(__file__).resolve().parent / "lab_profiles"


class LabProfilePhiError(ValueError):
    """Profile 中含有 PHI 迹象，拒绝写出。"""


def phi_scan_fields(profile: LabProfile) -> list[str]:
    """返回 Profile 中**数据派生**的、必须为 ASCII 无 PHI 的字段值。

    只校验数据派生字段（参数名、修复体类型名、lab_id、notes），不校验本模块自己
    撰写的中文 ``rationale`` —— 那是代码级文案，不是来源数据，把它算进 PHI 会
    让校验形同虚设（每次都被自己的说明文字触发）。
    """
    fields: list[str] = [profile.lab_id, profile.notes]
    for name, cls in profile.parameters.items():
        fields.append(name)
        fields.extend(cls.by_restoration_type)
    return fields


def assert_no_phi(profile: LabProfile) -> None:
    """写出前强制校验：数据派生字段必须无 PHI 迹象。"""
    problems: list[str] = []
    for value in phi_scan_fields(profile):
        if not value:
            continue
        if _CJK_RE.search(value):
            problems.append(f"含中日韩字符（可能是姓名/病例名）: {value!r}")
        if _UUID_RE.search(value):
            problems.append(f"含 UUID（可能是订单号/患者ID）: {value!r}")
        if _UNC_RE.search(value):
            problems.append(f"含 UNC 共享路径: {value!r}")
    if problems:
        raise LabProfilePhiError(
            "Lab Profile 拒绝写出，检测到 PHI 迹象：\n  - " + "\n  - ".join(problems[:10])
        )


def profile_path(lab_id: str) -> Path:
    return PROFILE_DIR / f"{lab_id}.json"


def save_lab_profile(profile: LabProfile, path: Path | str | None = None) -> Path:
    assert_no_phi(profile)
    path = Path(path) if path is not None else profile_path(profile.lab_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(profile.as_dict(), ensure_ascii=False, indent=2, sort_keys=False),
        encoding="utf-8",
    )
    return path


def load_lab_profile(path: Path | str) -> LabProfile:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"找不到 Lab Profile: {path}\n"
            "请先用 `python -m app.parameters.build_lab_profile --raw <病例目录> "
            "--lab-id <id>` 从真实 exocad 证据数据生成。"
        )
    return LabProfile.from_dict(json.loads(path.read_text(encoding="utf-8")))
