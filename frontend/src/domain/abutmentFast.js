import { api, autoFitRequest } from "../api";
import adapter from "./adapter";

const IDENTITY = [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]];

async function inputDigest(designId) {
  const status = await api.designStatus(designId);
  if (!status?.input_digest) throw new Error("无法读取当前设计输入版本");
  return status.input_digest;
}

function neighborStates(eligibility) {
  return Object.values(eligibility?.neighbors || {});
}

function referencesReady(eligibility) {
  const neighbors = neighborStates(eligibility);
  return Boolean(
    eligibility?.frame_confirmed
    && eligibility?.bite_confirmed
    && neighbors.length === 2
    && neighbors.every((state) => ["CONFIRMED", "ABSENT"].includes(state))
  );
}

/**
 * Existing-abutment fast path.
 *
 * No routine Margin/Axis/Neighbor clicking is required.  The incoming upper/lower
 * scans are assumed to already be bite aligned; the current relative pose is
 * recorded as REGISTERED_EXPORT (identity transform).
 */
export async function prepareAbutmentOrder({ caseId, fdi, designId = null }) {
  if (!caseId || !fdi) throw new Error("请先选择牙位并导入病例");

  const design = designId
    ? await api.design(designId)
    : await adapter.setToothPosition({ caseId, fdi, supportType: "abutment" });
  if (design.support_type !== "abutment") throw new Error("当前设计不是基台冠，不能进入快速流程");

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

  await adapter.saveMarginPoints(design.id, margin);
  await adapter.setInsertionAxis(design.id, axis, "CAD1 abutment automatic insertion axis");

  await autoFitRequest(design.id, "abutment-frame", "POST", {
    expected_input_digest: await inputDigest(design.id),
    confirmed: true,
  });

  await api.confirmBite(design.id, {
    expected_input_digest: await inputDigest(design.id),
    confirmed: true,
    transform: IDENTITY,
    source_method: "REGISTERED_EXPORT",
    units: "mm",
  });

  // Wider v2 search: coronal-band + mesiodistal percentile + connected patch.
  const neighbors = await autoFitRequest(design.id, "auto-neighbors", "POST", {
    expected_input_digest: await inputDigest(design.id),
    confirmed: true,
    patch_radius_mm: 4.25,
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

async function ensureAutomaticReferences(designId) {
  let eligibility = await api.generationEligibility(designId);
  if (referencesReady(eligibility)) return eligibility;

  // Resumed/older cases can have stale frame/bite/neighbor revisions.  Rebuild
  // them automatically instead of asking the user to click confirmation pages.
  const design = await api.design(designId);
  await prepareAbutmentOrder({
    caseId: design.case_id,
    fdi: String(design.target_fdi),
    designId,
  });
  eligibility = await api.generationEligibility(designId);
  return eligibility;
}

export async function generateAbutmentOrder({ designId, params }) {
  if (!designId) throw new Error("请先创建基台订单");

  // Reference preparation comes first because a rebuilt Margin/Axis revision may
  // invalidate downstream digests.  Save the user's profile only after geometry
  // references are current.
  let eligibility = await ensureAutomaticReferences(designId);
  await adapter.saveProfile(designId, params);
  eligibility = await api.generationEligibility(designId);

  // One last automatic neighbor retry after profile persistence.  No user click.
  let states = neighborStates(eligibility);
  if (states.length !== 2 || !states.every((state) => ["CONFIRMED", "ABSENT"].includes(state))) {
    await autoFitRequest(designId, "auto-neighbors", "POST", {
      expected_input_digest: await inputDigest(designId),
      confirmed: true,
      patch_radius_mm: 5.0,
    });
    eligibility = await api.generationEligibility(designId);
    states = neighborStates(eligibility);
  }

  if (!eligibility?.bite_confirmed) {
    await api.confirmBite(designId, {
      expected_input_digest: await inputDigest(designId),
      confirmed: true,
      transform: IDENTITY,
      source_method: "REGISTERED_EXPORT",
      units: "mm",
    });
    eligibility = await api.generationEligibility(designId);
  }

  if (states.length !== 2 || !states.every((state) => ["CONFIRMED", "ABSENT"].includes(state))) {
    const text = Object.entries(eligibility?.neighbors || {})
      .map(([side, state]) => `${side}:${state}`)
      .join(" / ");
    throw new Error(`自动邻牙识别未收敛（${text || "无结果"}）。请检查工作颌 STL 是否包含目标牙两侧区域后重新生成。`);
  }

  if (!eligibility?.automatic_mode) {
    const blockers = (eligibility?.blockers || []).map((item) => item.message || item.code).filter(Boolean).join("；");
    throw new Error(blockers || "自动参考已建立，但当前参数没有形成可执行的邻接/对颌约束");
  }

  const candidate = await adapter.generateInitialCrown(designId, { kind: "AUTO" });
  const failed = stageFailure(candidate);
  if (failed) throw new Error(`${failed.stage || "自动设计"} 未通过：${failed.reason || failed.status}`);
  if (candidate?.payload?.status !== "PASS_ENGINEERING") {
    throw new Error(`自动牙冠未达到可自动采纳状态：${candidate?.payload?.status || "UNKNOWN"}`);
  }

  await adapter.confirmCrownCandidate(designId, candidate.id);
  const final = await adapter.finalize(designId, { preserveCervical: true });
  return { candidate, final, eligibility };
}
