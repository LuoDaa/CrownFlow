"""P1：真实语料上的 Prep Conditioning 报表（**只读、无副作用**）。

数据不在仓库里（患者数据），所以按验收套件的惯例解析路径 + 缺失即 skip 并给出理由。
断言的**不是**具体例数（那取决于数据可得性），而是：

1. 报表能在真实数据上跑完、字段齐全、可序列化；
2. 分诊口径自洽（各 indication 之和 = 单位总数；eligible 与 indication 一致）；
3. 契约尝试过的单位，其 readiness 必落在三档 + NO_SUPPORT_ASSET 内，且每条都有拒绝码；
4. 结果写入 `prep_conditioning_report.json` 供人工核对（本测试用 tmp 路径，不污染仓库）。
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

from app.services import prep_conditioning as PC

_DEFAULT_CASES = Path(r"D:\code\牙冠自动生成\数据\15个完整病例")
_DEFAULT_BUNDLES = Path(r"D:\code\牙冠自动生成\数据\crown_outer_cases")

_ALLOWED_READINESS = {r.value for r in PC.Readiness}


def _resolve(env_name: str, default: Path) -> Path | None:
    value = os.environ.get(env_name)
    if value:
        path = Path(value)
        return path if path.is_dir() else None
    return default if default.is_dir() else None


@pytest.fixture(scope="module")
def corpus_paths():
    cases = _resolve("CROWNFLOW_EXOCAD_CASES_DIR", _DEFAULT_CASES)
    bundles = _resolve("CROWNFLOW_OUTER_BUNDLE_DIR", _DEFAULT_BUNDLES)
    if cases is None and bundles is None:
        pytest.skip("真实病例语料不可用（设置 CROWNFLOW_EXOCAD_CASES_DIR / "
                    "CROWNFLOW_OUTER_BUNDLE_DIR）；报表无法在真实数据上生成")
    return cases, bundles


def test_real_corpus_report_is_generated_and_self_consistent(corpus_paths, tmp_path):
    cases, bundles = corpus_paths
    report = PC.build_corpus_report(cases_root=cases, bundles_root=bundles)

    out = tmp_path / "prep_conditioning_report.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    assert json.loads(out.read_text(encoding="utf-8"))["scope"] == PC.SCOPE_1_0

    assert report["total_units"] == sum(report["by_indication"].values()), \
        "indication 统计必须覆盖全部单位（不许漏项）"
    eligible = sum(count for name, count in report["by_indication"].items()
                   if name in PC.ELIGIBLE_INDICATIONS)
    assert report["eligible_units"] == eligible
    assert report["bridge_rule"], "桥体相邻规则必须写清楚（分母是上界这件事要说出来）"

    for name, count in report["readiness"].items():
        assert name in _ALLOWED_READINESS, name
        assert count >= 0
    for code, count in report["by_rejection_code"].items():
        assert code and count >= 0
    assert report["contract_attempts"] == sum(
        1 for entry in report["bundles"] if (entry.get("contract") or {}).get("geometry"))
    assert report["eligible_without_asset"] >= report["eligible_units"], \
        "元数据侧合格单位全部没有支撑资产，必须体现在分母里"


def test_real_contract_attempts_never_silently_pass(corpus_paths):
    """跑过契约的单位必须有明确结论：READY 或带码的拒绝，不许"未知"。"""
    cases, bundles = corpus_paths
    if bundles is None:
        pytest.skip("病例包目录不可用")
    attempts = [entry for entry in PC.assess_bundle_dir(bundles)
                if (entry.get("contract") or {}).get("geometry")]
    for entry in attempts:
        classification = entry["contract"]["classification"]
        assert classification["readiness"] in _ALLOWED_READINESS
        assert classification["rejection_code"], entry["bundle"]
        if classification["readiness"] != PC.Readiness.PREP_SURFACE_READY.value:
            assert classification["rejection_reason"], \
                "拒绝必须给出理由（fail loud，不许只给码）"


def test_contract_endpoint_is_read_only(isolated_db, monkeypatch):
    """端点是只读自检：不改设计、不落库、状态与 ready 语义一致。"""
    from fastapi.testclient import TestClient

    from app.api import designer, routes
    from app.services import design_dependencies, prep_conditioning

    monkeypatch.setattr(routes, "CASE_DIR", isolated_db / "cases")
    monkeypatch.setattr(designer, "CASE_DIR", isolated_db / "cases")
    monkeypatch.setattr(prep_conditioning, "CASE_DIR", isolated_db / "cases", raising=False)
    monkeypatch.setattr(design_dependencies, "TEMPLATE_DIR", isolated_db / "templates")
    from app.main import app

    client = TestClient(app)
    case = client.post("/api/cases", json={"name": "prep-contract"}).json()["id"]
    design = client.post("/api/designs", json={"draft": True, "case_id": case,
                                              "target_fdi": 46, "support_type": "prep"}).json()["id"]
    path = f"/api/designs/{design}"

    first = client.get(path + "/prep-surface-contract")
    assert first.status_code == 200, first.text
    payload = first.json()
    assert payload["classification"]["eligible_for_1_0"] is True
    assert payload["classification"]["readiness"] == "REVIEW_REQUIRED"
    assert payload["classification"]["rejection_code"] == "MARGIN_OR_AXIS_UNCONFIRMED"

    theta = np.arange(32) * np.pi / 16
    margin = np.c_[2 * np.cos(theta), 2 * np.sin(theta), np.zeros(32)].tolist()
    assert client.post(path + "/margin", json={"points": margin}).status_code == 200
    assert client.post(path + "/axis", json={"axis": [0, 0, 1]}).status_code == 200
    # 只读性：把"调端点前后"的设计状态对比（margin/axis 那两步是测试自己做的，会推进修订号）
    snapshot = client.get(path).json()
    second = client.get(path + "/prep-surface-contract").json()
    assert second["classification"]["readiness"] == "NO_SUPPORT_ASSET", "还没定位基牙就要这么说"
    after = client.get(path).json()
    assert after["design_revision"] == snapshot["design_revision"], "只读端点不得改设计"
    assert after["outer_path"] is None and after["final_path"] is None
    assert after["margin_json"] == snapshot["margin_json"]
