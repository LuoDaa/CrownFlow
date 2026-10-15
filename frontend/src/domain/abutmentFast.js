import { api, autoFitRequest } from "../api";
import adapter from "./adapter";

const IDENTITY = [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]];

async function inputDigest(designId) {
  const status = await api.designStatus(designId);
  if (!status?.input_digest) throw new Error("无法读取当前设计输入版本");
  return status.input_digest;
}

/**
 * Existing-abutment fast path.
 *
 * The operator supplies bite-aligned upper/lower scans + FDI.  Everything that
 * should not be a routine click is confirmed from deterministic geometry here:
 * abutment finish line, insertion axis, local patient frame, static bite pose and
 * adjacent contact patches.
 */
export async function prepareAbutmentOrder({ caseId, fdi }) {
  if (!caseId || !fdi) throw new Error("请先选择牙位并导入病例");

  const design = await adapter.setToothPosition({ caseId, fdi, supportType: "abutment" });
  const draft = await adapter.detectMargin({
    designId: design.id,
    axisConfirmed: false,
    supportType: "abutment",
    center: null,
    axis: [0, 0, 1],
  });
  const margin = (draft?.points || []).map((point) => Array.from(point));
  const axis = (draft?.axis || []).map(Number);
  if (margin.length < 12) throw new Error("基台自动 Margin 点数不足，无法进入快速流程");
  if (axis.length !== 3 || axis.some((value) => !Number.isFinite(value))) {
    throw new Error("基台自动插入方向无效");
  }

  // saveMargin / setAxis create the same authoritative revisions used by the
  // natural-prep workflow; the fast path merely removes routine confirmation UI.
  await adapter.saveMarginPoints(design.id, margin);
  await adapter.setInsertionAxis(design.id, axis, "CAD1 abutment automatic insertion axis");

  await autoFitRequest(design.id, "abutment-frame", "POST", {
    expected_input_digest: await inputDigest(design.id),
    confirmed: true,
  });

  // The product contract for this fast path is: uploaded jaws are already in the
  // bite relationship.  Record identity as REGISTERED_EXPORT, never run a hidden
  // automatic bite registration.
  await api.confirmBite(design.id, {
    expected_input_digest: await inputDigest(design.id),
    confirmed: true,
    transform: IDENTITY,
    source_method: "REGISTERED_EXPORT",
    units: "mm",
  });

  const neighbors = await autoFitRequest(design.id, "auto-neighbors", "POST", {
    expected_input_digest: await inputDigest(design.id),
    confirmed: true,
    patch_radius_mm: 3.0,
  });

  return { design, margin, axis, neighbors };
}

export async function selectAbutmentTemplate({ designId, library, params }) {
  if (!designId) throw new Error("请先创建基台订单");
  if (!library) throw new Error("请选择牙库");
  await adapter.saveProfile(designId, params);
  const candidate = await adapter.selectCrownTemplate(designId, library);
  await adapter.confirmTemplateReference(designId, candidate.id);
  return candidate;
}

function stageFailure(candidate) {
  const stages = candidate?.payload?.stages || [];
  return stages.find((stage) => ["CONSTRAINT_CONFLICT", "BLOCKED", "INPUT_UNVERIFIED"].includes(stage.status));
}

/** Generate, accept and finalize one abutment crown without intermediate clicks. */
export async function generateAbutmentOrder({ designId, params }) {
  if (!designId) throw new Error("请先创建基台订单");
  await adapter.saveProfile(designId, params);
  const eligibility = await api.generationEligibility(designId);
  const neighborStates = Object.values(eligibility?.neighbors || {});
  if (!neighborStates.length || !neighborStates.every((state) => ["CONFIRMED", "ABSENT"].includes(state))) {
    throw new Error("邻牙自动接触面未完整识别，快速流程已停止；请进入高级复核而不是忽略邻接形态");
  }
  if (!eligibility?.bite_confirmed) throw new Error("上下颌咬合位置未确认");
  if (!eligibility?.automatic_mode) throw new Error("当前病例没有可执行的邻牙/对颌自动适配");

  const candidate = await adapter.generateInitialCrown(designId, { kind: "AUTO" });
  const failed = stageFailure(candidate);
  if (failed) {
    throw new Error(`${failed.stage || "自动设计"} 未通过：${failed.reason || failed.status}`);
  }
  if (candidate?.payload?.status !== "PASS_ENGINEERING") {
    throw new Error(`自动牙冠未达到可自动采纳状态：${candidate?.payload?.status || "UNKNOWN"}`);
  }

  await adapter.confirmCrownCandidate(designId, candidate.id);
  const final = await adapter.finalize(designId, { preserveCervical: true });
  return { candidate, final, eligibility };
}
