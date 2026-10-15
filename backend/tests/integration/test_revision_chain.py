"""修订链的集成测试。

断言的是"人工结果不被静默覆盖"与"候选被采纳后不再重复要求决策"这两条不变量 ——
它们都是实测踩过坑之后才确立的（见各测试的 docstring）。
"""

from __future__ import annotations

from pathlib import Path
import math

import pytest
from fastapi.testclient import TestClient

from app.main import app

CASE_ROOT = Path(r"D:\code\牙冠自动生成")
_SUPPORT = CASE_ROOT / "数据" / "1629-王煜-STL" / "01_主要病例模型" / "1629-王煜 0.stl"
_TEMPLATE = CASE_ROOT / "32牙位基础解剖模板库" / "32位牙列-分" / "FDI_36.stl"

pytestmark = pytest.mark.skipif(
    not (_SUPPORT.is_file() and _TEMPLATE.is_file()),
    reason="需要 FDI36 基准病例与 FDI_36 模板（患者数据不进 Git）",
)


@pytest.fixture()
def design_id(isolated_db, monkeypatch):
    # 模板必须落在被隔离的 TEMPLATE_DIR 里：resolve_template_source 在调用时从 config 取路径，
    # 若只 patched config 而没 patched 模块别名，上传会写到真实 data/templates（污染运行数据）。
    from app.api import routes
    from app.services import design_dependencies
    monkeypatch.setattr(routes, 'TEMPLATE_DIR', isolated_db / 'templates')
    monkeypatch.setattr(design_dependencies, 'TEMPLATE_DIR', isolated_db / 'templates')
    client = TestClient(app)
    case = client.post("/api/cases", json={"name": "rev-it"}).json()["id"]
    with _SUPPORT.open("rb") as fh:
        client.post(f"/api/cases/{case}/assets/support",
                    files={"file": ("s.stl", fh, "model/stl")})
    with _TEMPLATE.open("rb") as fh:
        client.post("/api/templates/36", files={"file": ("t.stl", fh, "model/stl")})
    did = client.post("/api/designs",
                      json={"case_id": case, "target_fdi": 36,
                            "support_type": "abutment"}).json()["id"]
    return client, did


def _margin_points(n: int = 18):
    return [[22+2*math.cos(2*math.pi*i/n),3+2*math.sin(2*math.pi*i/n),5.9+.1*math.sin(4*math.pi*i/n)] for i in range(n)]


def test_manual_margin_persists_immediately(design_id):
    """手绘 Margin 保存即落库 —— 旧版前端从不调用该接口，
    手绘结果只有走到 finalize 才保存，等于丢失。"""
    client, did = design_id
    r = client.post(f"/api/designs/{did}/margin", json={"points": _margin_points()})
    assert r.status_code == 200
    assert r.json()["margin_source"] == "manual"
    assert r.json()["revision_no"] == 1

    chain = client.get(f"/api/designs/{did}/revisions").json()
    assert chain["n_revisions"] == 1
    assert chain["revisions"][0]["kind"] == "margin_manual"
    assert chain["revisions"][0]["is_current"] is True


def test_auto_suggestion_never_silently_overwrites_manual(design_id):
    """自动提案不得覆盖人工结果：只登记候选，当前生效版本不变。"""
    client, did = design_id
    client.post(f"/api/designs/{did}/margin", json={"points": _margin_points()})

    from app.revisions import current_margin, record_auto_margin_candidate
    revision, applied = record_auto_margin_candidate(did, [[1.0, 2.0, 3.0]] * 5,
                                                    algorithm_version="test")

    assert applied is False, "当前是人工结果，自动提案不该直接生效"
    assert revision.kind.value == "margin_candidate"

    current = current_margin(did)
    assert current.kind.value == "margin_manual"
    assert current.payload["point_count"] == 18, "人工点列被改动了"


def test_accepted_axis_records_its_basis_and_consumes_candidate(design_id):
    """采纳就位方向时必须记录"为什么接受它"，且该候选不再要求决策。

    两个坑都是实测踩到的：
    1. 求解结论原先只留在候选修订上，接受修订里没有 —— 审计回答不了"为什么接受"。
    2. 采纳后候选未被标记为已消费，``requires_decision`` 永久为 True，
       前端每次都弹决策框。
    """
    client, did = design_id
    client.post(f"/api/designs/{did}/margin", json={"points": _margin_points()})

    solved = client.post(f"/api/designs/{did}/suggest-axis", json={}).json()
    before = client.get(f"/api/designs/{did}/revisions").json()
    assert before["requires_decision"] is True
    candidate_no = before["pending_candidates"][0]

    accepted = client.post(f"/api/designs/{did}/axis",
                           json={"axis": solved["axis"], "approved_by": "tech_A"}).json()
    assert accepted["kind"] == "axis_accepted"
    assert accepted["based_on_candidate_revision_no"] == candidate_no

    after = client.get(f"/api/designs/{did}/revisions").json()
    assert after["requires_decision"] is False, "候选已被采纳却仍在要求决策"
    assert after["pending_candidates"] == []
    assert candidate_no in after["consumed_candidates"]

    accepted_rev = [r for r in after["revisions"] if r["kind"] == "axis_accepted"][-1]
    basis = accepted_rev["payload"]["basis"]
    assert basis["candidate_revision_no"] == candidate_no
    assert basis["solver"]["status"] == solved["status"], "审计未记录当时求解器状态"
    assert basis["solver_reasons"] == solved["diagnostics"]["reasons"]


def test_outer_does_not_create_spurious_axis_candidate(design_id):
    """``outer/auto`` 不得重复登记轴候选。

    传入的方向可能刚被人工确认过；再登记一条 axis_candidate 会让修订链冒出
    一个并不需要决策的待决定项，使 requires_decision 误报为 True。
    """
    client, did = design_id
    client.post(f"/api/designs/{did}/margin", json={"points": _margin_points()})
    solved = client.post(f"/api/designs/{did}/suggest-axis", json={}).json()
    client.post(f"/api/designs/{did}/axis",
                json={"axis": solved["axis"], "approved_by": "tech_A"})
    # A1：/outer/auto 是 live 路径，必须先确认坐标帧（无帧直接 409，不用世界轴推断）。
    # 帧绑定 Margin/轴/配置的确认版本，所以配置也必须先确认 —— 与界面顺序一致。
    assert client.put(f"/api/designs/{did}/profile",
                      json={"material": "unvalidated fixture", "coverage": "ENGINEERING_ONLY",
                            "bite_confirmed": True, "confirmed": True,
                            "parameters": {"machine_axes": 5}}).status_code == 200
    status = client.get(f"/api/designs/{did}/status").json()
    assert client.put(f"/api/designs/{did}/auto-fit/frame",
                      json={"mesial": [1, 0, 0], "buccal": [0, 1, 0], "units": "mm",
                            "confirmed": True,
                            "expected_input_digest": status["input_digest"]}).status_code == 200
    n_before = client.get(f"/api/designs/{did}/revisions").json()["n_revisions"]

    placed = client.post(f"/api/designs/{did}/outer/auto",
                json={"margin_points": _margin_points(), "insertion_axis": solved["axis"]})
    assert placed.status_code == 200, placed.text
    assert placed.json()["frame_source"] == "CONFIRMED_FRAME"
    assert placed.json()["placement_path"] == "INITIALIZED_ONCE"

    chain = client.get(f"/api/designs/{did}/revisions").json()
    kinds = [r["kind"] for r in chain["revisions"][n_before:]]
    assert "axis_candidate" not in kinds, f"outer 之后冒出了轴候选: {kinds}"
    assert chain["requires_decision"] is False
    assert kinds == ["outer"]


def test_solver_result_is_json_serializable_without_margin(design_id):
    """无 Margin 的降级路径必须可序列化（NaN 不是合法 JSON）。"""
    client, did = design_id
    r = client.post(f"/api/designs/{did}/suggest-axis", json={})
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "REVIEW_REQUIRED"
    assert body["score"] is None, "未知应表达为 null，而不是 NaN"
    assert body["diagnostics"]["degraded_without_margin"] is True
