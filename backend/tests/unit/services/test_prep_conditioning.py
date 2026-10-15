"""P1：Prep Surface Contract 的编排与分类（**不测几何算法，只测契约与分诊**）。

这一批刻意没有新几何：`readiness` 由现成谓词 `fitting_support_above_margin` 决定，
诊断数字来自 `prep_margin_diagnostics.diagnose`（只读）。所以测试的重点是：

1. 每条拒绝码都能被**现有几何**触发（合成输入，逐条钉住分类）；
2. indication 与 readiness 是**两条正交轴**（不属于 1.0 范围 ≠ 输入不合格）；
3. 分诊规则（桥体直接相邻 ⇒ BRIDGE_UNIT；Coping/Pontic/Antagonist/implant 各自归类）；
4. 契约报告字段齐全、可直接 JSON 序列化、带 provenance；
5. **契约不做任何几何修复**（输入网格字节不变）。
"""

from __future__ import annotations

import json

import numpy as np
import trimesh

from app.services import prep_conditioning as PC


def _support_cap(radius: float = 3.0, *, sections: int = 48) -> trimesh.Trimesh:
    """闭合圆柱预备体（单一完整就位面）—— 契约应通过。"""
    return trimesh.creation.cylinder(radius=radius, height=5.0, sections=sections) \
        .apply_translation([0, 0, 0.5])


def _open_tube(radius: float = 3.0, *, z0: float = -2.0, z1: float = 3.0,
               sections: int = 48) -> trimesh.Trimesh:
    """**真正开放**的管壁（无上下盖）：两圈开放边界 —— 用来复现真实病例里
    "ROI/裁片边界跨过边缘线"的情形。

    注意：本机 trimesh 的 ``creation.cylinder`` **没有** ``cap`` 参数（传了会被忽略、
    结果仍然水密），所以这里手工只保留侧壁面。
    """
    full = trimesh.creation.cylinder(radius=radius, height=z1 - z0, sections=sections)
    full.apply_translation([0, 0, (z0 + z1) / 2])
    keep = np.abs(np.asarray(full.face_normals)[:, 2]) < 0.5      # 只留竖直侧壁
    return full.submesh([np.flatnonzero(keep)], append=True, repair=False)


def _margin(radius: float = 3.0, *, amplitude: float = 0.0, sections: int = 48) -> np.ndarray:
    theta = np.arange(sections) * 2 * np.pi / sections
    return np.c_[radius * np.cos(theta), radius * np.sin(theta),
                 amplitude * np.sin(2 * theta)]


AXIS = np.array([0.0, 0.0, 1.0])


# ---------------------------------------------------------------- 分类单元

def test_indication_and_readiness_are_orthogonal():
    """不是 1.0 范围（indication）与输入不合格（readiness）是两件事，不许互相短路。"""
    # 基台：indication 外 ⇒ 不跑几何，直接 UNSUPPORTED_INDICATION
    abutment = PC.assess_prep_surface(
        _support_cap(), _margin(), AXIS,
        indication=PC.Indication.IMPLANT_ABUTMENT.value, screw_channel_present=True)
    assert abutment.readiness == PC.Readiness.UNSUPPORTED_INDICATION.value
    assert abutment.rejection_code == "IMPLANT_ABUTMENT_WITH_SCREW_CHANNEL"
    assert abutment.rejection_class == PC.REJECTION_CLASS_INDICATION
    assert abutment.diagnostics["assessed"] is False, "范围外的病例不浪费在几何上"

    # indication 未判定：**仍然要给几何结论**（否则会掩盖真实拒绝码）
    unknown = PC.assess_prep_surface(_support_cap(), _margin(), AXIS,
                                     indication=PC.Indication.UNKNOWN_INDICATION.value)
    assert unknown.readiness == PC.Readiness.PREP_SURFACE_READY.value
    assert unknown.eligible is False, "未判定 ⇒ 不进 1.0 合格池，但几何结论照样给"


def test_ready_contract_on_clean_single_surface_prep():
    support = _support_cap()
    before = np.asarray(support.vertices).copy()
    contract = PC.assess_prep_surface(
        support, _margin(amplitude=0.6), AXIS,
        indication=PC.Indication.NATURAL_PREP.value, target_fdi=46,
        source_asset_sha256="deadbeef", margin_point_count=48,
        provenance={"source": "unit_test"})
    report = contract.as_report()

    assert contract.ready is True
    assert report["classification"]["readiness"] == "PREP_SURFACE_READY"
    assert report["classification"]["rejection_code"] == "READY"
    assert report["classification"]["eligible_for_1_0"] is True
    assert report["geometry"]["source_face_ids"], "只读源码面 id 必须记录"
    assert report["geometry"]["component_count"] == 1
    assert report["diagnostics"]["assessed"] is True
    assert report["diagnostics"]["conditioning_operations"].startswith("NONE")
    assert report["provenance"] == {"source": "unit_test"}
    json.dumps(report, ensure_ascii=False)          # 必须可直接序列化
    assert np.array_equal(np.asarray(support.vertices), before), "契约**不得**改输入网格"


def test_margin_crossing_open_boundary_is_asset_preparation_required():
    """Margin 挨着开放边界 ⇒ **B 类（资产准备）**，可人工重裁后再试。

    这正是真实病例里 `prep.stl` 的失败形态（FDI46 病例包实测）：ROI 边界跨过边缘线。
    """
    tube = _open_tube()                                  # z ∈ [-2, 3]，无盖
    margin = _margin(radius=3.03) + np.array([0.0, 0.0, -1.95])   # 紧贴下圈开放边界
    contract = PC.assess_prep_surface(tube, margin, AXIS,
                                      indication=PC.Indication.NATURAL_PREP.value)
    assert contract.ready is False
    assert contract.rejection_class == PC.REJECTION_CLASS_ASSET
    assert contract.rejection_code == "MARGIN_CROSSES_OPEN_BOUNDARY", contract.as_report()
    assert "开放边界" in contract.rejection_reason, "拒绝理由必须写出来（fail loud）"


def test_multi_loop_above_margin_is_asset_preparation_required():
    """边缘线以上出现**第二道边界环**（例如螺丝通道/额外开口）⇒ B 类。

    这一条用同一根开放管壁触发"单一完整就位面"判据：Margin 落在管壁中段时，
    切出来的就位面除了 Margin 环还带着上圈开放边界。
    """
    tube = _open_tube()
    contract = PC.assess_prep_surface(tube, _margin(radius=3.0), AXIS,
                                      indication=PC.Indication.NATURAL_PREP.value)
    assert contract.ready is False
    assert contract.rejection_class == PC.REJECTION_CLASS_ASSET
    assert contract.rejection_code in ("FITTING_SURFACE_NOT_SINGLE_COMPLETE",
                                       "EXTRA_BOUNDARY_LOOPS_ABOVE_MARGIN",
                                       "MULTIPLE_COMPONENTS_RAW_SUPPORT"), contract.as_report()


def test_multicomponent_support_is_asset_preparation_required():
    """多组件支撑（邻牙/底座被包进来）⇒ B 类；并如实标 island 污染嫌疑。"""
    first = trimesh.creation.cylinder(radius=3.0, height=5.0, sections=48).apply_translation([0, 0, 0.5])
    second = trimesh.creation.cylinder(radius=1.0, height=1.0, sections=24).apply_translation([8, 0, 0.5])
    support = trimesh.util.concatenate([first, second])
    contract = PC.assess_prep_surface(support, _margin(), AXIS,
                                      indication=PC.Indication.NATURAL_PREP.value)
    assert contract.ready is False
    assert contract.rejection_class == PC.REJECTION_CLASS_ASSET
    assert contract.diagnostics["cropped_component_count"] >= 1
    assert "PREP_MULTICOMPONENT" in contract.diagnostics["diagnose_codes"]
    assert "MULTIPLE_COMPONENTS_RAW_SUPPORT" in contract.diagnostics["corroborating_codes"]


def test_failure_message_mapping_is_total():
    """失败消息映射必须给出码与分类；未映射的也必须**明确**是未映射，不静默当成 READY。"""
    for message, expected_code in (
            ("Margin 跨扫描开放边界或缺口，不允许自动连接", "MARGIN_CROSSES_OPEN_BOUNDARY"),
            ("Margin 以上基牙不是单一完整就位面；存在缺口或多余组件，请人工复核来源",
             "FITTING_SURFACE_NOT_SINGLE_COMPLETE"),
            ("Margin 投影三角化退化", "MARGIN_CURTAIN_DEGENERATE")):
        code, category = PC.classify_failure_message(message)
        assert code == expected_code and category == PC.REJECTION_CLASS_ASSET
    code, category = PC.classify_failure_message("某种没见过的失败")
    assert code == "PREP_SURFACE_CONTRACT_UNMAPPED"
    assert category == PC.REJECTION_CLASS_REVIEW


# ---------------------------------------------------------------- indication 分诊

def test_tooth_metadata_classification():
    assert PC.classify_tooth_metadata("AnatomicCrown", "None") is \
        PC.Indication.NATURAL_TOOTH_SINGLE_ANATOMICAL_CROWN
    assert PC.classify_tooth_metadata("AnatomicPontic", None) is PC.Indication.ANATOMIC_PONTIC
    assert PC.classify_tooth_metadata("Coping", None) is PC.Indication.COPING_ONLY
    assert PC.classify_tooth_metadata("Antagonist", None) is PC.Indication.ANTAGONIST_DESIGN_ARTIFACT
    assert PC.classify_tooth_metadata("HealthyTooth", None) is PC.Indication.HEALTHY_TOOTH
    assert PC.classify_tooth_metadata("AnatomicCrown", "SIC_ace") is PC.Indication.IMPLANT_ABUTMENT
    assert PC.classify_tooth_metadata("SomethingNew", None) is PC.Indication.UNKNOWN_INDICATION


def test_bridge_adjacency_rule():
    """与桥体**直接相邻**的冠判为 BRIDGE_UNIT（分母不许把桥体单元算成游离单冠）。"""
    teeth = [{"fdi": "24", "recon": "AnatomicCrown", "implant": "None"},
             {"fdi": "25", "recon": "AnatomicCrown", "implant": "None"},
             {"fdi": "26", "recon": "AnatomicPontic", "implant": "None"},
             {"fdi": "27", "recon": "AnatomicCrown", "implant": "None"}]
    units = {u["fdi"]: u for u in PC.triage_units(teeth)}
    assert units[25]["indication"] == PC.Indication.BRIDGE_UNIT.value
    assert units[27]["indication"] == PC.Indication.BRIDGE_UNIT.value
    assert units[24]["indication"] == PC.Indication.NATURAL_TOOTH_SINGLE_ANATOMICAL_CROWN.value
    assert units[24]["eligible"] is True
    assert units[25]["notes"] and "桥体" in units[25]["notes"][0]

    # 不同象限的同号牙不算相邻
    cross = [{"fdi": "36", "recon": "AnatomicCrown", "implant": "None"},
             {"fdi": "46", "recon": "AnatomicCrown", "implant": "None"},
             {"fdi": "37", "recon": "AnatomicPontic", "implant": "None"}]
    cross_units = {u["fdi"]: u for u in PC.triage_units(cross)}
    assert cross_units[36]["indication"] == PC.Indication.BRIDGE_UNIT.value   # 37 同象限相邻
    assert cross_units[46]["indication"] == PC.Indication.NATURAL_TOOTH_SINGLE_ANATOMICAL_CROWN.value


def test_design_row_assessment_without_support_asset():
    """没有支撑资产必须明确说"没资产可判"，而不是"试试看"。"""
    design = {"case_id": "case-x", "target_fdi": 46, "support_type": "prep",
              "margin_json": json.dumps(_margin().tolist()),
              "axis_json": json.dumps(AXIS.tolist()), "design_revision": 3,
              "confirmed_margin_revision_id": "m1", "current_axis_revision_id": "a1"}
    contract = PC.assess_design_prep(design, support_path=None)
    assert contract.readiness == PC.Readiness.NO_SUPPORT_ASSET.value
    assert contract.indication == PC.Indication.NATURAL_PREP.value
    assert contract.eligible is True

    # 已有实体基台上的粘接冠支持，但没有资产不能进入链路。
    abutment = dict(design, support_type="abutment")
    out = PC.assess_design_prep(abutment, support_path=None)
    assert out.readiness == PC.Readiness.NO_SUPPORT_ASSET.value
    assert out.eligible
    custom=dict(abutment,indication='CUSTOM_IMPLANT_ABUTMENT')
    assert PC.assess_design_prep(custom,support_path=None).readiness==PC.Readiness.UNSUPPORTED_INDICATION.value

    # 没确认 Margin/轴：先让用户确认，别浪费在几何上
    pending = dict(design, margin_json="null")
    assert PC.assess_design_prep(pending, support_path=None).rejection_code == \
        "MARGIN_OR_AXIS_UNCONFIRMED"


def test_corpus_report_shape(tmp_path):
    """报表只做统计、无副作用；字段齐全且可直接序列化。"""
    cases = tmp_path / "cases"
    (cases / "caseA").mkdir(parents=True)
    (cases / "caseB").mkdir(parents=True)
    bundles = tmp_path / "bundles"
    (bundles / "bundle1").mkdir(parents=True)
    (bundles / "bundle1" / "case.json").write_text(json.dumps(
        {"fdi": 46, "margin_points": _margin().tolist(), "insertion_axis": AXIS.tolist()}),
        encoding="utf-8")

    report = PC.build_corpus_report(cases_root=cases, bundles_root=bundles)
    json.dumps(report, ensure_ascii=False)
    for key in ("generated_at", "scope", "bridge_rule", "total_cases", "total_units",
                "eligible_units", "by_indication", "contract_attempts",
                "by_rejection_code", "readiness", "eligible_without_asset", "assets"):
        assert key in report, key
    assert report["assets"]["bundles_total"] == 1
    assert report["assets"]["bundles_with_support_asset"] == 0
    assert report["eligible_without_asset"] >= 1
    assert report["scope"] == PC.SCOPE_1_0


def test_margin_in_a_different_frame_is_never_ready():
    """**同坐标系健全性门**：margin 离支撑表面 30mm ⇒ 绝不 READY。

    实测触发场景：exocad 的逐牙位代型 `*-modeldie.stl` 在**模型/打印系**，
    而 constructionInfo 的 margin 在设计系 —— 拓扑谓词会平凡通过（margin 离网格很远，
    "不落在开放边界上"必然成立），从而产出假 READY。这条测试把它钉死。
    """
    support = _support_cap()
    far_margin = _margin() + np.array([60.0, 0.0, 0.0])       # 整体平移 60mm ＝ 另一个坐标系
    contract = PC.assess_prep_surface(support, far_margin, AXIS,
                                      indication=PC.Indication.NATURAL_PREP.value)
    assert contract.ready is False
    assert contract.rejection_code == "MARGIN_NOT_ON_SUPPORT_SURFACE"
    assert contract.rejection_class == PC.REJECTION_CLASS_ASSET
    assert contract.diagnostics["coordinate_frame_sane"] is False
    assert "坐标系" in contract.rejection_reason

    # 同一对输入，只把 margin 拉回表面附近 ⇒ READY（证明门是"距离"判据，不是别的）
    contract_ok = PC.assess_prep_surface(support, _margin(), AXIS,
                                         indication=PC.Indication.NATURAL_PREP.value)
    assert contract_ok.ready is True
    assert contract_ok.diagnostics["coordinate_frame_sane"] is True
    assert contract_ok.diagnostics["margin_to_surface_max_mm"] <= PC.MARGIN_ON_SURFACE_MAX_MM
