"""B1：三个域的测量契约必须与**代码实际行为**一致（把"谁负责什么"钉成单一真相）。

`measurement_core.DOMAIN_CONTRACTS` 是跨域读数字时的唯一入口。本文件的职责是防止它
退化成"好看的文档"：每条契约都要在真实模块上被验一遍。
"""

from __future__ import annotations

import numpy as np
import pytest
import trimesh

from app.geometry import measurement_core as MC


def test_contract_registry_covers_three_domains():
    assert set(MC.DOMAIN_CONTRACTS) == {MC.DOMAIN_CONTACT, MC.DOMAIN_OCCLUSION, MC.DOMAIN_THICKNESS}
    for contract in MC.DOMAIN_CONTRACTS.values():
        report = contract.as_report()
        for key in ("domain", "sign_method", "sign_semantics", "validity_convention",
                    "penetration_meaning", "owns_crossing_audit", "input_contract", "implementation"):
            assert report[key] not in (None, ""), key


def test_crossing_audit_ownership_matches_the_modules():
    """契约说 thickness 负责相交审计、occlusion 不负责 —— 这条必须与代码一致。"""
    from app.geometry.crown_outer import occlusion, thickness

    assert MC.THICKNESS_CONTRACT.owns_crossing_audit is True
    assert hasattr(thickness, "self_intersections")
    assert MC.OCCLUSION_CONTRACT.owns_crossing_audit is False
    assert not hasattr(occlusion, "self_intersections")


def test_occlusion_and_contact_share_the_same_sign_vocabulary():
    """O1：符号词汇表只有一套 —— Contact 与 Occlusion 用同一组名字与同一个 primitive。"""
    from app.services import contact_measurement as CM

    assert CM.SIGN_SEMANTICS.startswith("positive=gap_to_reference")
    for name in (MC.SIGN_SOLID_CONTAINMENT, MC.SIGN_REVIEWED_NORMAL_LOCAL):
        assert name in MC.OCCLUSION_CONTRACT.sign_method
        assert name in MC.CONTACT_CONTRACT.sign_method or name in (MC.SIGN_REVIEWED_NORMAL_LOCAL,)
    assert MC.SIGN_GLOBAL_INSIDE_OUTSIDE == MC.SIGN_SOLID_CONTAINMENT
    assert MC.SIGN_REVIEWED_LOCAL_NORMAL == MC.SIGN_REVIEWED_NORMAL_LOCAL
    assert "CENTROID" not in MC.OCCLUSION_CONTRACT.sign_method, "质心投票已退役"


def test_contact_contract_matches_service_semantics():
    """Contact 契约的语义串必须与 `services.contact_measurement` 里写的完全一致。"""
    from app.services import contact_measurement as CM

    assert MC.CONTACT_CONTRACT.sign_semantics == CM.SIGN_SEMANTICS
    assert MC.CONTACT_CONTRACT.implementation.endswith("measure_surface")
    assert MC.SIGN_GLOBAL_INSIDE_OUTSIDE in MC.CONTACT_CONTRACT.sign_method
    assert MC.SIGN_REVIEWED_LOCAL_NORMAL in MC.CONTACT_CONTRACT.sign_method


def test_validity_conventions_are_not_accidentally_identical():
    """三域的**有效性约定**不该被悄悄改成同一种：Contact 有折痕掩码，另两域只有边界安全。"""
    assert MC.CONTACT_CONTRACT.validity == MC.VALIDITY_BOUNDARY_AND_CREASE
    assert MC.OCCLUSION_CONTRACT.validity == MC.VALIDITY_BOUNDARY_ONLY
    assert MC.THICKNESS_CONTRACT.validity == MC.VALIDITY_BOUNDARY_ONLY


def test_thickness_result_carries_its_contract():
    from app.geometry.crown_outer.patient_frame import PatientFrame
    from app.geometry.crown_outer.thickness import ThicknessParameters, audit_thickness

    frame = PatientFrame(np.zeros(3), np.array([0.0, 0.0, 1.0]),
                         np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]))

    def cap(radius: float) -> trimesh.Trimesh:
        mesh = trimesh.creation.icosphere(subdivisions=3, radius=radius)
        keep = np.asarray(mesh.triangles_center)[:, 2] > -radius * 0.55
        return mesh.submesh([np.flatnonzero(keep)], append=True, repair=False)

    result = audit_thickness(cap(3.0), cap(2.6), frame,
                             params=ThicknessParameters(min_thickness_mm=0.5,
                                                        boundary_safety_mm=0.0,
                                                        max_query_edge_mm=0.0))
    assert result.domain_report["domain"] == MC.DOMAIN_THICKNESS
    assert result.domain_report["sign_method"] == MC.SIGN_SMOOTHED_VERTEX_NORMAL
    assert result.domain_report["validity_convention"] == MC.VALIDITY_BOUNDARY_ONLY
    assert result.domain_report["owns_crossing_audit"] is True
    assert any("本域口径" in note for note in result.notes), result.notes


def test_occlusion_result_carries_its_contract_and_disclaims_crossings():
    from app.geometry.crown_outer.contact import NeighborPatch
    from app.geometry.crown_outer.occlusion import OcclusionParams, solve_occlusion_collision
    from app.geometry.crown_outer.patient_frame import PatientFrame

    frame = PatientFrame(np.zeros(3), np.array([0.0, 0.0, 1.0]),
                         np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]))
    theta = np.arange(48) * 2 * np.pi / 48
    ring = np.c_[2.0 * np.cos(theta), 2.0 * np.sin(theta), np.zeros(48)]
    apex = np.array([[0.0, 0.0, 5.0]])
    crown = trimesh.Trimesh(np.vstack((ring, apex)),
                            np.array([[i, (i + 1) % 48, 48] for i in range(48)], np.int64),
                            process=False)
    plate = trimesh.Trimesh(np.array([[-4.0, -4.0, 4.6], [4.0, -4.0, 4.6],
                                      [4.0, 4.0, 4.6], [-4.0, 4.0, 4.6]]),
                            np.array([[0, 1, 2], [0, 2, 3]], np.int64), process=False)
    result = solve_occlusion_collision(crown, frame, NeighborPatch(plate.vertices, plate.faces),
                                       params=OcclusionParams(target_gap_mm=0.05),
                                       antagonist_outward=[0.0, 0.0, -1.0])

    assert result.domain_report["domain"] == MC.DOMAIN_OCCLUSION
    # O1：闭合实体用实体内外判定；开放裁片用**已声明**外向法向；无声明则 UNSIGNED_ONLY。
    assert result.domain_report["active_sign_method"] == MC.SIGN_REVIEWED_NORMAL_LOCAL
    assert result.sign_method == MC.SIGN_REVIEWED_NORMAL_LOCAL
    assert MC.SIGN_SOLID_CONTAINMENT in MC.OCCLUSION_CONTRACT.sign_method
    assert result.domain_report["owns_crossing_audit"] is False
    assert result.history[0]["max"] == pytest.approx(0.45, abs=0.02)
    assert any("目标相对" in note or "target_gap" in note for note in result.notes), result.notes


def test_shared_stats_matches_contact_report_shape():
    """共用统计 primitive 的形状必须与 Contact 报告里既有的键名一致（不许分叉）。"""
    values = np.array([0.0, 1.0, 2.0])
    assert MC.stats(values) == {"p50_mm": 1.0, "p95_mm": pytest.approx(1.9), "max_mm": 2.0}
    assert MC.stats([]) is None
    assert MC.validity_counts(np.array([True, False, True])) == {"valid_count": 2, "ambiguous_count": 1}
