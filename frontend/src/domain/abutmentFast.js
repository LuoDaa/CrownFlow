import { api, autoFitRequest } from "../api";
import adapter from "./adapter";

const IDENTITY = [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]];

const emit = (onProgress, id, label, progress, detail = "") => {
  onProgress?.({ id, label, progress, detail, at: Date.now() });
};

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

export async function prepareAbutmentOrder({ caseId, fdi, designId = null, onProgress = null }) {
  if (!caseId || !fdi) throw new Error("请先选择牙位并导入病例");

  emit(onProgress, "DESIGN", "创建/恢复设计", 4);
  const design = designId
    ? await api.design(designId)
    : await adapter.setToothPosition({ caseId, fdi, supportType: "abutment" });
  if (design.support_type !== "abutment") throw new Error("当前设计不是基台冠，不能进入快速流程");

  emit(onProgress, "MARGIN", "识别基台与 Margin", 10);
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

  emit(onProgress, "AXIS", "确认 Margin 与插入方向", 18);
  await adapter.saveMarginPoints(design.id, margin);
  await adapter.setInsertionAxis(design.id, axis, "CAD1 abutment automatic insertion axis");

  emit(onProgress, "FRAME", "建立患者坐标系", 24);
  await autoFitRequest(design.id, "abutment-frame", "POST", {
    expected_input_digest: await inputDigest(design.id),
    confirmed: true,
  });

  emit(onProgress, "BITE", "锁定当前上下颌咬合位姿", 30);
  await api.confirmBite(design.id, {
    expected_input_digest: await inputDigest(design.id),
    confirmed: true,
    transform: IDENTITY,
    source_method: "REGISTERED_EXPORT",
    units: "mm",
  });

  emit(onProgress, "NEIGHBORS", "自动识别近中/远中邻牙", 36);
  const neighbors = await autoFitRequest(design.id, "auto-neighbors", "POST", {
    expected_input_digest: await inputDigest(design.id),
    confirmed: true,
    patch_radius_mm: 4.25,
  });

  emit(onProgress, "READY", "自动参考建立完成", 40);
  return { design, margin, axis, neighbors };
}

export async function selectAbutmentTemplate({ designId, library, params, onProgress = null }) {
  if (!designId) throw new Error("请先创建基台订单");
  if (!library) throw new Error("请选择牙库");
  emit(onProgress, "PROFILE", "保存设计参数", 42);
  await adapter.saveProfile(designId, params);
  emit(onProgress, "TEMPLATE", `载入牙库 ${library}`, 46);
  const candidate = await adapter.selectCrownTemplate(designId, library);
  await adapter.confirmTemplateReference(designId, candidate.id);
  emit(onProgress, "TEMPLATE_READY", "模板参考已确认", 48);
  return candidate;
}

async function ensureAutomaticReferences(designId, onProgress = null) {
  emit(onProgress, "SYNC", "同步服务端设计状态", 3);
  let eligibility = await api.generationEligibility(designId);
  if (referencesReady(eligibility)) return eligibility;

  const design = await api.design(designId);
  await prepareAbutmentOrder({
    caseId: design.case_id,
    fdi: String(design.target_fdi),
    designId,
    onProgress,
  });
  eligibility = await api.generationEligibility(designId);
  return eligibility;
}

export async function generateAbutmentOrder({ designId, params, onProgress = null }) {
  if (!designId) throw new Error("请先创建基台订单");

  emit(onProgress, "SYNC", "检查自动设计输入", 2);
  let eligibility = await ensureAutomaticReferences(designId, onProgress);

  emit(onProgress, "PROFILE", "同步牙冠参数", 8);
  await adapter.saveProfile(designId, params);
  eligibility = await api.generationEligibility(designId);

  // Always refresh automatic neighbor references before AUTO generation.  V3
  // stores a broad connected reference halo around the compact contact core; old
  // V2 cases may still be CONFIRMED but have an edge-bound patch that produces
  // zero valid signed samples.  Manual HUMAN_* references are preserved server-side.
  emit(onProgress, "NEIGHBORS", "刷新邻牙参考面", 14, "扩展接触核心周围安全参考面");
  await autoFitRequest(designId, "auto-neighbors", "POST", {
    expected_input_digest: await inputDigest(designId),
    confirmed: true,
    patch_radius_mm: 5.0,
  });
  eligibility = await api.generationEligibility(designId);
  let states = neighborStates(eligibility);

  if (!eligibility?.bite_confirmed) {
    emit(onProgress, "BITE", "重新确认静态咬合", 18);
    await api.confirmBite(designId, {
      expected_input_digest: await inputDigest(designId),
      confirmed: true,
      transform: IDENTITY,
      source_method: "REGISTERED_EXPORT",
      units: "mm",
    });
    eligibility = await api.generationEligibility(designId);
    states = neighborStates(eligibility);
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

  // The backend candidate call performs the patient-driven CAD chain:
  // patient-space prefit -> Margin cervical fit -> proximal envelope -> antagonist
  // height/occlusal refinement -> intaglio preflight.
  emit(onProgress, "PREFIT", "患者空间整体适配", 28, "按邻牙宽度与咬合空间调整模板整体尺寸");
  emit(onProgress, "CERVICAL", "颈缘贴合 Margin", 36);
  emit(onProgress, "PROXIMAL", "生成近远中邻接形态", 48);
  emit(onProgress, "OCCLUSAL", "生成冠高与咬合面适配", 62);
  const candidate = await adapter.generateInitialCrown(designId, { kind: "AUTO" });

  const payloadStages = candidate?.payload?.stages || [];
  const stageNames = new Set(payloadStages.map((stage) => stage?.stage));
  if (stageNames.has("GLOBAL_PREFIT")) emit(onProgress, "PREFIT_DONE", "患者空间整体适配完成", 68);
  if (stageNames.has("CERVICAL_FIT")) emit(onProgress, "CERVICAL_DONE", "Margin 颈缘约束完成", 72);
  if (stageNames.has("PROXIMAL")) emit(onProgress, "PROXIMAL_DONE", "邻接形态完成", 76);
  if (stageNames.has("OCCLUSAL") || stageNames.has("OCCLUSAL_HEIGHT_PREFIT")) emit(onProgress, "OCCLUSAL_DONE", "咬合适配完成", 82);

  emit(onProgress, "PUBLISH", "发布患者化外冠", 86);
  await adapter.confirmCrownCandidate(designId, candidate.id);

  let final = null;
  let finalizeWarning = null;
  try {
    emit(onProgress, "INTAGLIO", "根据基台生成内冠", 90);
    final = await adapter.finalize(designId, { preserveCervical: true });
    emit(onProgress, "FINAL", "连接内外冠并生成最终 STL", 96);
  } catch (error) {
    finalizeWarning = error?.message || "最终内冠/QC 尚未完成";
  }

  const stageWarnings = (candidate?.payload?.stages || [])
    .filter((stage) => stage?.status && stage.status !== "PASS_ENGINEERING" && stage.status !== "PLACED")
    .map((stage) => `${stage.stage || "阶段"}:${stage.reason || stage.status}`);
  (candidate?.payload?.warnings || []).forEach((warning) => {
    const text = `${warning.stage || "检查"}:${warning.reason || "REVIEW_REQUIRED"}`;
    if (!stageWarnings.includes(text)) stageWarnings.push(text);
  });
  if (candidate?.payload?.status && candidate.payload.status !== "PASS_ENGINEERING") {
    stageWarnings.push(`candidate:${candidate.payload.status}`);
  }
  if (finalizeWarning) stageWarnings.push(`finalize:${finalizeWarning}`);

  emit(onProgress, "DONE", final ? "自动牙冠设计完成" : "外冠完成，可进入 3D 检查", 100,
    stageWarnings.length ? `${stageWarnings.length} 项工程提示` : "");

  return { candidate, final, eligibility, warnings: stageWarnings };
}