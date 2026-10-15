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

function stageOf(stages, name) {
  return (stages || []).find((stage) => stage?.stage === name) || null;
}

function latestStageOf(stages, names) {
  const accepted = new Set(names);
  return [...(stages || [])].reverse().find((stage) => accepted.has(stage?.stage)) || null;
}

function threeHardConstraintFailures(candidate, eligibility) {
  const stages = candidate?.payload?.stages || [];
  const failures = [];

  // 1) Margin：最终外冠必须保留整圈颈缘硬边界，优先检查所有后续 refit 的最后一次结果。
  const cervical = latestStageOf(stages, [
    "CERVICAL_REFIT_FINAL",
    "CERVICAL_REFIT_AFTER_SHELL",
    "CERVICAL_REFIT_AFTER_HEIGHT",
    "CERVICAL_FIT",
  ]);
  if (!cervical || cervical.status !== "PASS_ENGINEERING") {
    failures.push(`Margin:${cervical?.reason || cervical?.status || "NOT_ASSESSED"}`);
  } else {
    const coverage = cervical.full_loop_coverage;
    if (coverage && (
      Number(coverage.within_0_05mm_fraction ?? 1) < 0.98
      || Number(coverage.uncovered_samples ?? 0) > 0
    )) {
      failures.push("Margin:FULL_LOOP_COVERAGE_NOT_REACHED");
    }
    if (Number(cervical.after_mm?.p95 ?? 0) > 0.05 + 1e-9) {
      failures.push("Margin:P95_EXCEEDS_0.05MM");
    }
  }

  // 2) 邻接：只有真实存在且已确认的邻牙侧才要求接触；ABSENT 侧明确豁免。
  const confirmedSides = Object.entries(eligibility?.neighbors || {})
    .filter(([, state]) => state === "CONFIRMED")
    .map(([side]) => side);
  if (confirmedSides.length) {
    const proximal = latestStageOf(stages, ["PROXIMAL_RECHECK_AFTER_SHELL", "PROXIMAL"]);
    if (!proximal || proximal.status !== "PASS_ENGINEERING") {
      failures.push(`Proximal:${proximal?.reason || proximal?.status || "NOT_ASSESSED"}`);
    } else {
      const physical = proximal.physical_after || {};
      confirmedSides.forEach((side) => {
        if (physical?.[side]?.target_reached !== true) {
          failures.push(`Proximal:${side}:TARGET_NOT_REACHED`);
        }
      });
    }
  }

  // 3) 咬合：快速基台流程始终要求确认 Bite + 咬合阶段工程通过；不能只生成“看起来放上去”的模板。
  const occlusal = latestStageOf(stages, ["OCCLUSAL_RECHECK_AFTER_SHELL", "OCCLUSAL"]);
  if (!eligibility?.bite_confirmed) {
    failures.push("Occlusion:BITE_NOT_CONFIRMED");
  } else if (!occlusal || occlusal.status !== "PASS_ENGINEERING") {
    failures.push(`Occlusion:${occlusal?.reason || occlusal?.status || "NOT_ASSESSED"}`);
  }

  return failures;
}

function emitStageResult(onProgress, stage, id, okLabel, progress) {
  if (!stage) return;
  const ok = ["PASS_ENGINEERING", "PLACED"].includes(stage.status);
  emit(
    onProgress,
    id,
    ok ? okLabel : `${okLabel} · 需复核`,
    progress,
    ok ? "" : (stage.reason || stage.status || "REVIEW_REQUIRED"),
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

  // Refresh every automatic reference before generation so older V2 cases cannot
  // keep a CONFIRMED-but-edge-dominated neighbor patch. HUMAN_* input is preserved.
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

  emit(onProgress, "PREFIT", "患者空间整体适配", 28, "按邻牙宽度与咬合空间调整模板整体尺寸");
  emit(onProgress, "CERVICAL", "颈缘贴合 Margin", 36);
  emit(onProgress, "PROXIMAL", "生成近远中邻接形态", 48);
  emit(onProgress, "OCCLUSAL", "生成冠高与咬合面适配", 62);
  const candidate = await adapter.generateInitialCrown(designId, { kind: "AUTO" });

  const payloadStages = candidate?.payload?.stages || [];
  emitStageResult(onProgress, stageOf(payloadStages, "GLOBAL_PREFIT"), "PREFIT_DONE", "患者空间整体适配完成", 68);
  emitStageResult(onProgress, latestStageOf(payloadStages, ["CERVICAL_REFIT_FINAL", "CERVICAL_REFIT_AFTER_SHELL", "CERVICAL_REFIT_AFTER_HEIGHT", "CERVICAL_FIT"]), "CERVICAL_DONE", "Margin 颈缘约束完成", 72);
  emitStageResult(onProgress, latestStageOf(payloadStages, ["PROXIMAL_RECHECK_AFTER_SHELL", "PROXIMAL"]), "PROXIMAL_DONE", "邻接形态完成", 76);
  const occlusalStage = latestStageOf(payloadStages, ["OCCLUSAL_RECHECK_AFTER_SHELL", "OCCLUSAL"])
    || stageOf(payloadStages, "OCCLUSAL_HEIGHT_PREFIT");
  emitStageResult(onProgress, occlusalStage, "OCCLUSAL_DONE", "咬合适配完成", 82);

  const hardFailures = threeHardConstraintFailures(candidate, eligibility);
  if (hardFailures.length) {
    emit(onProgress, "HARD_CONSTRAINT_BLOCKED", "三个硬约束未全部通过", 84, hardFailures.join("；"));
    throw new Error(`三个硬约束未全部通过：${hardFailures.join("；")}。候选未发布，请修正病例输入或参数后重新生成。`);
  }
  emit(onProgress, "HARD_CONSTRAINTS_PASS", "Margin / 邻接 / 咬合全部通过", 85);

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
    emit(onProgress, "FINAL_REVIEW", "最终冠需要复核", 96, finalizeWarning);
  }

  const stageWarnings = payloadStages
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

  emit(
    onProgress,
    "DONE",
    final ? "自动牙冠设计完成" : "患者化外冠完成，可进入 3D 检查",
    100,
    stageWarnings.length ? `${stageWarnings.length} 项工程提示` : "",
  );

  return { candidate, final, eligibility, warnings: stageWarnings };
}