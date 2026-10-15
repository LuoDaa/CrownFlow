"""参数系统回归验收。

这一组是**最便宜、拦截面最广**的断言：它拦住的是"又有人把某个值硬编码进代码"
这类回归。用户在验收套件里明确要求的第一条断言就在这里。

对应三条硬性要求：

1. 解析结果必须等于导入的 exocad 真值（而非代码里的猜测值）。
2. 未知参数必须保持 ``value is None``，**不得**被 ``value or 0`` 之类写法变成 0。
3. 语义未定的值（如 ``CementGapThickness = 0``）必须阻止自动设计，而不是被静默替换。
"""

from __future__ import annotations

import collections

import pytest

from app.importers.exocad.case_loader import discover_cases, load_case
from app.importers.exocad.models import FunctionalRole, ParameterState
from app.parameters.models import ParameterClass, ResolvedSource
from app.parameters.resolver import (
    SYSTEM_DEFAULT_FORBIDDEN,
    SYSTEM_DEFAULTS,
    ParameterResolver,
    ParameterResolutionError,
)

# --------------------------------------------------------------------------- 解析链

def test_min_thickness_explicit_from_tooth(lab_profile, exocad_root):
    """有 .constructionInfo 的牙位：牙位显式值胜出，值为 0.6（不是代码里的 0.8）。"""
    case = load_case(exocad_root / "82404060702张哲森")
    resolved = ParameterResolver(lab_profile).resolve(tooth=case.tooth(12))

    p = resolved["MinThickness"]
    assert p.value == 0.6, f"MinThickness 应为 exocad 真值 0.6，实际 {p.value}"
    assert p.source is ResolvedSource.TOOTH_EXPLICIT
    assert p.state is ParameterState.EXPLICIT
    assert p.is_resolved


def test_min_thickness_falls_back_to_restoration_preset(lab_profile):
    """没有 .constructionInfo 的病例（15 例中有 6 例）：restoration preset 生效。"""
    resolved = ParameterResolver(lab_profile).resolve(
        tooth=None, restoration_type="AnatomicCrown")

    p = resolved["MinThickness"]
    assert p.value == 0.6
    assert p.source is ResolvedSource.RESTORATION_PRESET
    assert p.provenance["basis_records"] >= 3


def test_min_thickness_differs_for_reduced_pontic(lab_profile):
    """按修复体类型区分：减量桥体是 0.4，不是 0.6。"""
    resolved = ParameterResolver(lab_profile).resolve(
        tooth=None, restoration_type="ReducedPontic")
    assert resolved["MinThickness"].value == 0.4


def test_insertion_angle_limit_matches_lab_preset(lab_profile):
    """±20° 是实验室全局 preset，且与算法里锥采样角度一致。"""
    resolved = ParameterResolver(lab_profile).resolve(
        tooth=None, restoration_type="AnatomicCrown")
    p = resolved["CrownBottomInsertionAngleBetweenTeethLimit"]
    assert p.value == 20.0
    assert p.source is ResolvedSource.LAB_PRESET


# ---------------------------------------------------------------- 未知不得变成 0

def test_missing_parameter_stays_none(lab_profile):
    """核心安全性质：未知参数的 value 必须是 None，而不是 0。"""
    resolved = ParameterResolver(lab_profile).resolve(tooth=None, names=["CementGapThickness"])
    p = resolved["CementGapThickness"]

    assert p.value is None
    assert p.state is ParameterState.MISSING_INTERNAL_DEFAULT_UNKNOWN
    assert p.source is ResolvedSource.UNRESOLVED
    assert p.blocks_auto_design


def test_missing_parameter_never_falls_back_to_zero(lab_profile):
    """``value or 0`` 这类写法会在 require() 处直接失败，而不是得到 0。"""
    resolved = ParameterResolver(lab_profile).resolve(tooth=None, names=["CementGapThickness"])
    with pytest.raises(ParameterResolutionError):
        resolved.require("CementGapThickness")


def test_case_variable_has_no_default(lab_profile):
    """真变量（CASE_VARIABLE）在无证据时必须保持未知，不得由 preset 补齐。"""
    cls = lab_profile.get("CementGapThickness")
    assert cls.parameter_class is ParameterClass.CASE_VARIABLE
    assert len(cls.distinct_values) >= 4
    assert cls.preset_value() is None

    resolved = ParameterResolver(lab_profile).resolve(
        tooth=None, restoration_type="AnatomicCrown", names=["CementGapThickness"])
    assert resolved["CementGapThickness"].value is None


def test_system_defaults_are_empty_by_design(lab_profile):
    """系统级默认值故意为空：没有可引用的临床来源时，留空比猜一个数安全。"""
    assert SYSTEM_DEFAULTS == {}
    assert "CementGapThickness" in SYSTEM_DEFAULT_FORBIDDEN

    resolved = ParameterResolver(lab_profile).resolve(
        tooth=None, names=["CementGapThickness"], allow_system_defaults=True)
    # 即使开启 system default，歧义参数也不允许被补齐
    assert resolved["CementGapThickness"].value is None


# ------------------------------------------------------ 语义未定的零值必须阻塞

def test_ambiguous_zero_is_flagged_and_blocks_design(exocad_root, lab_profile):
    """``CementGapThickness = 0`` 必须被标为 EXPLICIT_BUT_UNRESOLVED 并阻止自动设计。

    实测：该值出现在 2 个病例（张哲森 FDI 12/13、82404140726 FDI 35/36），共 4 条记录。
    0 粘接间隙意味着冠与预备体直接贴合、无法就位，不可能是设计意图。
    """
    flagged = []
    for case_dir in sorted(p for p in exocad_root.iterdir() if p.is_dir()):
        case = load_case(case_dir)
        for tooth in case.teeth:
            if tooth.parameter("CementGapThickness").state is ParameterState.EXPLICIT_BUT_UNRESOLVED:
                flagged.append((case_dir.name, tooth.tooth_fdi))

    assert len(flagged) == 4, f"期望 4 条歧义零值记录，实际 {len(flagged)}: {flagged}"
    assert {f[1] for f in flagged} == {12, 13, 35, 36}

    case = load_case(exocad_root / "82404060702张哲森")
    resolved = ParameterResolver(lab_profile).resolve(tooth=case.tooth(12))
    p = resolved["CementGapThickness"]
    assert p.state is ParameterState.EXPLICIT_BUT_UNRESOLVED
    assert p.value == 0.0
    assert p.unresolved_reason
    with pytest.raises(ParameterResolutionError):
        resolved.require("CementGapThickness")


def test_no_four_state_is_lost(exocad_root):
    """四态必须都能在真实数据上命中 —— 防止某种状态被实现成不可达的死代码。"""
    from collections import Counter

    counts: Counter = Counter()
    for case_dir in sorted(p for p in exocad_root.iterdir() if p.is_dir()):
        case = load_case(case_dir)
        for tooth in case.teeth:
            for cell in tooth.parameters.values():
                counts[cell.state] += 1
            # MISSING 通过 parameter() 查询获得
            counts[tooth.parameter("__absent__").state] += 1

    assert counts[ParameterState.EXPLICIT] > 2000
    assert counts[ParameterState.NO_CONSTRAINT] > 100
    assert counts[ParameterState.EXPLICIT_BUT_UNRESOLVED] == 4
    assert counts[ParameterState.MISSING_INTERNAL_DEFAULT_UNKNOWN] > 0


# ------------------------------------------------------------ 分类与作用域归位

def test_classification_matches_evidence(lab_profile):
    """分类必须由数据推导，且与手工推导一致。"""
    expected = {
        "MinThickness": ParameterClass.RESTORATION_PRESET,
        "CementGapThickness": ParameterClass.CASE_VARIABLE,
        "MillingDiameter": ParameterClass.CASE_VARIABLE,
        "MillingMachineAxes": ParameterClass.LAB_PRESET_VARIANT,
        "UntouchableZoneNearPreparationMargin": ParameterClass.LAB_PRESET_VARIANT,
        "CrownBottomInsertionAngleBetweenTeethLimit": ParameterClass.LAB_GLOBAL_PRESET,
        "CrownAdaptationShapePreservation": ParameterClass.LAB_GLOBAL_PRESET,
    }
    for name, cls in expected.items():
        actual = lab_profile.get(name)
        assert actual is not None, f"{name} 未出现在 Lab Profile 中"
        assert actual.parameter_class is cls, (
            f"{name} 分类为 {actual.parameter_class.value}，期望 {cls.value}")


def test_manufacturing_params_are_order_scoped_not_material(lab_profile):
    """刀具与机台参数属于 ORDER/CASE 维度，**不能**落进材料 profile。

    实测：MillingDiameter 取 5 种值（0.5–1.2，2.4 倍跨度），MillingMachineAxes 取 3/5。
    把它们写进 materials/zirconia.json 是错的键位。
    """
    from app.importers.exocad.models import ParameterScope

    for name in ("MillingDiameter", "MillingMachineAxes"):
        cls = lab_profile.get(name)
        assert cls is not None
        assert cls.scope is ParameterScope.ORDER, (
            f"{name} 作用域应为 ORDER（加工/订单维度），实际 {cls.scope.value}")

    dia = lab_profile.get("MillingDiameter")
    assert len(dia.distinct_values) >= 4, "MillingDiameter 应为逐病例变量"
    axes = lab_profile.get("MillingMachineAxes")
    assert set(axes.distinct_values) == {3.0, 5.0}, "机轴数只应出现 3 与 5"


def test_lab_profile_contains_no_phi(lab_profile):
    """Lab Profile 是可提交进 Git 的产物，必须无 PHI。"""
    from app.parameters.profiles import assert_no_phi, phi_scan_fields

    assert_no_phi(lab_profile)  # 不抛异常即通过
    for field in phi_scan_fields(lab_profile):
        assert field.isascii(), f"数据派生字段含非 ASCII（可能混入姓名）: {field!r}"


# ------------------------------------------------- 病例身份 / 防训练-验证泄露

def test_case_identity_and_leakage_grouping(exocad_root):
    """病例身份只认 source_order_id；同一订单的多个目录必须归为一组。

    实测：``82404020415薛文爱`` 与 ``薛文爱`` 是同一订单 ``03c3329e``。
    按目录随机切分会让同一口扫描同时出现在训练集与验证集。
    """
    groups = discover_cases(exocad_root)

    assert len(groups) == 14, "15 个目录应归为 14 个订单组"
    multi = {oid: dirs for oid, dirs in groups.items() if len(dirs) > 1}
    assert len(multi) == 1, f"应恰好有 1 个多目录订单，实际 {len(multi)}"

    order_id, dirs = next(iter(multi.items()))
    assert order_id == "03c3329e-87f8-43f3-93d5-c5c9537ee4d1"
    assert {d.name for d in dirs} == {"82404020415薛文爱", "薛文爱"}

    case = load_case(dirs[0])
    assert case.source_order_id == order_id


def test_matrix4_is_not_a_bite_transform(exocad_root):
    """.matrix4 必须是 FIXED_COORDINATE_TRANSFORM 且不得驱动 Occlusion。

    实测：15 个文件内容逐字节一致（1 个内容 hash），旋转 176.5°、平移固定。
    逐病例的咬合配准不可能在不同患者间完全一致。
    """
    from app.importers.exocad.matrix4 import Matrix4Role, classify_matrix4_role, parse_matrix4

    matrices = []
    for case_dir in sorted(p for p in exocad_root.iterdir() if p.is_dir()):
        matrices += [parse_matrix4(p) for p in case_dir.rglob("*.matrix4")]

    verdict = classify_matrix4_role(matrices)
    assert len(matrices) == 15
    assert verdict.distinct_content_hashes == 1, "15 个 .matrix4 应内容完全一致"
    assert verdict.role is Matrix4Role.FIXED_COORDINATE_TRANSFORM
    assert verdict.usable_for_occlusion is False, (
        ".matrix4 不得用于 Occlusion —— 它不是逐病例咬合配准")


def test_antagonist_labels_available(exocad_root):
    """exocad 记录的 Antagonist 牙位可作为对颌识别真值使用。

    注意这是"哪颗牙是对颌"的**身份真值**，与"上下颌咬合空间变换"是两件不同的事；
    后者在现有数据里没有真值（见 .matrix4 的角色判定）。
    """
    total = 0
    for case_dir in sorted(p for p in exocad_root.iterdir() if p.is_dir()):
        case = load_case(case_dir)
        total += len(case.antagonist_teeth)
    assert total >= 1, "应至少存在一条 Antagonist 记录"


# ------------------------------------------------------------------ 证据数据集状态

def test_evidence_dataset_scale(exocad_root, baseline):
    """锁定证据数据集规模，避免数据被悄悄替换后指标失去可比性。"""
    ds = baseline["measurements"]["_evidence_dataset"]
    groups = discover_cases(exocad_root)
    assert ds["n_order_groups"] == len(groups)
    assert ds["n_case_dirs"] == 15
    assert ds["n_duplicate_dir_groups"] == 1


# --------------------------------------------------- 派生参数（功能依赖，非 preset）

def test_derived_rules_match_evidence_data(exocad_root):
    """派生规则必须与真实证据数据完全一致 —— 有例外就失败。

    实测（25 条同时写出这两个参数的记录，相关度 100%）：
    ``AdditionalBlockOutAngleAfterCementGap`` 与
    ``UntouchableZoneNearPreparationMargin`` 都是 ``MillingMachineAxes`` 的确定性函数，
    不是独立的实验室 preset。

    这条断言的作用：规则表一旦与现实脱节（新数据出现例外）就立刻失败，
    而不是让一个过时的经验值悄悄留在解析链里继续被套用。
    """
    from app.parameters.derived import DERIVED_RULES

    observed: dict[tuple[str, float], collections.Counter] = collections.defaultdict(
        collections.Counter)
    for case_dir in sorted(p for p in exocad_root.iterdir() if p.is_dir()):
        case = load_case(case_dir)
        for tooth in case.teeth:
            axes = tooth.parameter("MillingMachineAxes")
            if axes.value is None:
                continue
            for rule in DERIVED_RULES:
                cell = tooth.parameter(rule.target)
                if cell.value is not None:
                    observed[(rule.target, float(axes.value))][cell.value] += 1

    assert observed, "证据数据里没有任何可供核验的派生参数记录"
    for (target, axes_value), counter in sorted(observed.items()):
        if axes_value not in (3.0, 5.0):
            pytest.fail(f"出现未预期的 MillingMachineAxes 取值 {axes_value}")
        rule = next(r for r in DERIVED_RULES if r.target == target)
        expected = rule.table[axes_value]
        distinct = set(counter)
        assert distinct == {expected}, (
            f"{target} 在 MillingMachineAxes={axes_value} 下实测取值 {sorted(distinct)}，"
            f"与规则表 {expected} 不符。规则表已过时，必须重新推导。"
        )


def test_resolver_derives_parameters_from_machine_axes(lab_profile):
    """机轴数确定后，派生参数必须被推导出来，而不是各自去查 preset。"""
    resolver = ParameterResolver(lab_profile)
    names = ["MillingMachineAxes", "AdditionalBlockOutAngleAfterCementGap",
             "UntouchableZoneNearPreparationMargin"]

    expected = {3.0: (0.0, 0.0), 5.0: (-1.0, 0.1)}
    for axes, (angle, untouchable) in expected.items():
        resolved = resolver.resolve(
            tooth=None, restoration_type="AnatomicCrown", names=names,
            case_overrides={"MillingMachineAxes": axes})

        for name, want in (("AdditionalBlockOutAngleAfterCementGap", angle),
                           ("UntouchableZoneNearPreparationMargin", untouchable)):
            param = resolved[name]
            assert param.value == want, f"axes={axes} 时 {name} 应为 {want}，实际 {param.value}"
            assert param.source is ResolvedSource.DERIVED_FROM_PARAMETER, (
                f"{name} 应由参数派生得到，实际来源 {param.source.value}；"
                "若它变成 preset 说明有人把功能依赖压平成了经验值"
            )
            assert param.provenance["derived_from"] == "MillingMachineAxes"
            assert param.provenance["rule_evidence"]


def test_derivation_refuses_when_rule_not_applicable(lab_profile):
    """规则不适用时必须保持未解析，不得用最近似的表项敷衍。

    否则"规则不适用于这个机台"会被伪装成"规则已应用"，后果是给一个 4 轴机
    套上 3 轴或 5 轴的参数。
    """
    resolver = ParameterResolver(lab_profile)
    resolved = resolver.resolve(
        tooth=None, restoration_type="AnatomicCrown",
        names=["MillingMachineAxes", "AdditionalBlockOutAngleAfterCementGap"],
        case_overrides={"MillingMachineAxes": 4.0})

    param = resolved["AdditionalBlockOutAngleAfterCementGap"]
    assert param.value is None
    assert param.state is ParameterState.MISSING_INTERNAL_DEFAULT_UNKNOWN
    assert param.provenance["derived_rule_not_applicable"]["dependency_value"] == 4.0
    assert "派生规则不适用" in param.unresolved_reason


def test_explicit_value_wins_over_derivation(lab_profile):
    """病例显式写出的值优先于派生值 —— 派生只补空缺，不覆盖。"""
    from app.importers.exocad.models import (
        ExocadToothRecord, FunctionalRole, ParameterCell, ParameterScope,
        ReconstructionType,
    )

    cell = ParameterCell(
        name="AdditionalBlockOutAngleAfterCementGap",
        state=ParameterState.EXPLICIT, value=2.5,
        source="constructionInfo", scope=ParameterScope.TOOTH,
    )
    tooth = ExocadToothRecord(
        source_file="synthetic", source_order_id=None, case_dir_name="synthetic",
        tooth_fdi=36, reconstruction_type=ReconstructionType.ANATOMIC_CROWN,
        functional_role=FunctionalRole.RESTORATION,
        parameters={cell.name: cell},
    )

    resolved = ParameterResolver(lab_profile).resolve(tooth=tooth)
    param = resolved["AdditionalBlockOutAngleAfterCementGap"]
    assert param.value == 2.5
    assert param.source is ResolvedSource.TOOTH_EXPLICIT
