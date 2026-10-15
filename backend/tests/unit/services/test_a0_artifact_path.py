"""A0 回归：产物路径必须**限定在 run 目录内**（审计 P0-5，已复现）。

历史 bug：用字符串前缀判断 ``str(p).startswith(str(root))``——
``runs/r10/secret.txt`` 以 ``runs/r1`` 为前缀，于是 ``run_id="r1"`` 能读到**兄弟 run** 的文件。
现有测试只覆盖 ``../secret`` 与 ``sub/../../secret``，漏掉了前缀碰撞。
"""
from __future__ import annotations

from pathlib import Path

import pytest

from app.services.crown_outer_runner import PipelineError, artifact_path


@pytest.fixture()
def runs_root(tmp_path):
    root = tmp_path / "runs"
    (root / "r1").mkdir(parents=True)
    (root / "r10").mkdir(parents=True)
    (root / "r1" / "ok.txt").write_text("legit", encoding="utf-8")
    (root / "r10" / "secret.txt").write_text("SIBLING-RUN", encoding="utf-8")
    return root


def test_sibling_run_prefix_collision_is_rejected(runs_root):
    """r1 + ../r10/secret.txt 必须被拒（这是历史 bug 的复现用例）。"""
    with pytest.raises(PipelineError):
        artifact_path("r1", "../r10/secret.txt", runs_root=runs_root)


@pytest.mark.parametrize("name", ["../secret.txt", "sub/../../secret.txt", "../../etc/passwd"])
def test_plain_traversals_are_rejected(runs_root, name):
    with pytest.raises(PipelineError):
        artifact_path("r1", name, runs_root=runs_root)


def test_legit_artifact_still_readable(runs_root):
    p = artifact_path("r1", "ok.txt", runs_root=runs_root)
    assert Path(p).read_text(encoding="utf-8") == "legit"


def test_sibling_run_file_readable_only_from_its_own_run(runs_root):
    """从 r10 自己读是合法的 —— 说明拒绝的是"越界"，不是"文件不该存在"。"""
    p = artifact_path("r10", "secret.txt", runs_root=runs_root)
    assert Path(p).name == "secret.txt"
