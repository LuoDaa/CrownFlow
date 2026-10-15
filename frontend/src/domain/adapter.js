/**
 * CrownDesignAdapter —— 新版 UI 与现有算法能力之间的**唯一**接缝层。
 *
 * 前端页面只调用本文件的方法，不直接拼 URL，也不自己判断"能不能做"。
 * 每个方法要么调用**已存在**的后端端点，要么抛 `MissingCapability`（fail-loud），
 * 绝不返回假数据冒充已实现（例如不编造 penetration / contact area）。
 *
 * 方法名沿用原型 adapter.js 的命名（loadCase / detectMargin / adaptToAntagonist / …），
 * 便于与设计基准逐条对照。
 */
import { api, base } from "../api";
import { suggestPatientFrame } from "./frameDirection";

export class MissingCapability extends Error {
  constructor(capability, note) {
    super(note || `后端暂未提供该能力：${capability}`);
    this.name = "MissingCapability";
    this.code = "MISSING_BACKEND_CAPABILITY";
    this.capability = capability;
  }
}

/** 能力清单：UI 用它决定显示入口还是显示"暂不可用"，不靠函数是否抛错去猜。 */
export const CAPABILITIES = {
  caseFiles: { available: true, detail: "GET/POST /api/cases, POST /api/cases/{id}/assets/{kind}" },
  orderImport: { available: true, detail: "POST /api/importers/three-shape/import" },
  marginDetect: { available: true, detail: "POST /api/designs/{id}/margin/detect（会话）· draft-preparation（粗候选）" },
  marginEdit: { available: true, detail: "margin/move · margin/add-anchor · margin/smooth · margin/replace-segment" },
  insertionAxis: { available: true, detail: "POST /api/designs/{id}/suggest-axis · /axis" },
  crownTemplates: { available: true, detail: "GET /auto-fit/templates · POST /auto-fit/template-preview · accept" },
  generateCrown: { available: true, detail: "POST /api/designs/{id}/workflow/generate" },
  adaptNeighbors: { available: true, detail: "POST /edit-outer (adapt_mesial / adapt_distal)" },
  adaptAntagonist: { available: true, detail: "POST /edit-outer (adapt_antagonist)" },
  contactMap: { available: true, detail: "GET /outer-analysis?kind=neighbors|antagonist" },
  /** 语义区域（MESIAL / DISTAL / OCCLUSAL / …）随冠形**自动建立**，无需人工标注。 */
  automaticSemanticRegions: { available: true, detail: "GET /sculpt-session 的 region_map；构建见 geometry/crown_regions" },
  freeformEdit: { available: true, detail: "POST /edit-outer (add/remove/smooth/flatten/drag/transform)" },
  thickness: { available: true, detail: "GET /outer-analysis?kind=thickness · manufacturing-qc" },
  undoRedo: { available: true, detail: "GET /revisions · POST /restore-outer" },
  finalize: { available: true, detail: "POST /api/designs/{id}/finalize" },
  export: { available: true, detail: "GET /final.stl · /package.zip" },
  /**
   * 对颌适配**不依赖**自动咬合配准：本产品输入的上/下颌已经咬好，
   * 直接使用病例里已确认的静态咬合位姿（bite transform）。
   */
  antagonistFromCaseBite: { available: true, detail: "使用已确认的 bite transform + 对侧颌扫描；无需自动配准" },
  /** 未来能力：不阻塞 1.0 的 Crown Design 流程。 */
  autoBiteRegistration: { available: false, note: "未来能力（病例自带咬合位姿已足够），不阻塞对颌适配" },
  dynamicOcclusion: { available: false, note: "本版本只做静态咬合间隙调整，不做动态咬合运动" },
  toolCompensation: { available: false, note: "刀具体补偿未接入设计导出链" },
  certifiedManufacturingQc: { available: false, note: "制造 QC 未取得认证口径，报告仅供工程复核" },
};

function require(capability) {
  const item = CAPABILITIES[capability];
  if (item && !item.available) throw new MissingCapability(capability, item.note);
}

const SCAN_KINDS = { upper: "upper", lower: "lower" };

/** 从服务端设计行里取人工确认过的帧/输入摘要（所有写操作都要带，服务端会校验）。 */
async function designContext(designId) {
  const status = await api.designStatus(designId);
  return { status, digest: status.input_digest, revisions: status.confirmed_revisions || {} };
}

export const adapter = {
  // ── 病例 ────────────────────────────────────────────────────────────────
  listCases: () => api.cases(),
  loadCase: (caseId) => api.case(caseId),
  createCase: (name) => api.createCase(name),
  deleteCase: (caseId) => api.deleteCase(caseId),
  caseDesigns: (caseId) => api.caseDesigns(caseId),

  /** 上传 Upper / Lower 扫描（STL / OBJ / PLY，后端 trimesh 读入，**保留原咬合坐标**）。 */
  async loadJaw(caseId, kind, file) {
    const slot = SCAN_KINDS[kind];
    if (!slot) throw new Error(`未知颌别：${kind}`);
    await api.upload(caseId, slot, file);
    return api.case(caseId);
  },
  loadUpperJaw: (caseId, file) => adapter.loadJaw(caseId, "upper", file),
  loadLowerJaw: (caseId, file) => adapter.loadJaw(caseId, "lower", file),
  importOrderPackage: (file, name) => api.threeShapeImport(file, name),

  /** 目标牙位 + 修复类型 → 建设计（draft）。 */
  setToothPosition: ({ caseId, fdi, supportType }) =>
    api.createDesign({ case_id: caseId, target_fdi: Number(fdi), support_type: supportType, draft: true }),

  /** 从种植定位件里找可用基台候选（只读）。 */
  listAbutmentCandidates: (caseId, source) => api.implantCandidates(caseId, source),

  // ── Margin / Axis / 坐标帧 ─────────────────────────────────────────────
  /** 自动识别边缘线：轴未确认时走粗准备；已确认轴时进入会话候选（Top-K + 磁吸）。 */
  detectMargin: ({ designId, axisConfirmed, supportType, center, axis }) =>
    axisConfirmed
      ? api.marginDetect(designId, { feature_mode: "AUTO", radius_mm: 1.5, top_k: 3, apply_snap: true })
      : api.draftPreparation(designId, {
          center: center || null,
          axis: supportType === "abutment" ? undefined : axis,
          radius_mm: 7,
          restorative_abutment_confirmed: supportType === "abutment",
        }),
  marginSession: (designId) => api.design(designId).then((d) => JSON.parse(d.margin_session_json || "null")),
  addMarginAnchor: (designId, anchors, featureMode) =>
    api.marginAddAnchor(designId, anchors, { feature_mode: featureMode || "AUTO", radius_mm: 1.5 }),
  moveMargin: (designId, deltaMm) => api.marginMove(designId, deltaMm),
  smoothMargin: (designId, strength, locked) => api.marginSmooth(designId, strength, locked || []),
  replaceMarginSegment: (designId, stroke, featureMode) =>
    api.marginReplaceSegment(designId, stroke, { feature_mode: featureMode || "AUTO", radius_mm: 1.5 }),
  /** 用户手工改过的曲线直接写成正式边缘线（**唯一权威曲线**）。 */
  saveMarginPoints: (designId, points) => api.saveMargin(designId, points),
  confirmMargin: (designId, proposalId, operator) => api.marginConfirm(designId, proposalId, operator),

  /**
   * 近中/颊侧方向候选（几何推导，不是世界轴）：边缘线投影到 ⊥就位轴平面后的 PCA 长主轴做近中候选，
   * 颊侧 = 就位轴 × 近中（右手系）。**正负号仍需人工确认**，所以界面给"翻转/确认"而不是让用户输向量。
   */
  suggestFrame: suggestPatientFrame,
  /** 确认患者坐标帧（后端要求三件（边缘线/轴/参数）都已确认）。 */
  confirmFrame: (designId, { mesial, buccal }) =>
    designContext(designId).then(({ digest }) =>
      api.confirmAutoFit(designId, "frame", { mesial, buccal, units: "mm", expected_input_digest: digest, confirmed: true })),
  confirmCurrentBite: (designId) => designContext(designId).then(({ digest }) =>
    api.confirmBite(designId, { transform: [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],
      source_method: "MANUAL_REGISTRATION", units: "mm", expected_input_digest: digest, confirmed: true })),
  confirmNeighbor: (designId, side, input) => designContext(designId).then(({ digest }) =>
    api.confirmAutoFit(designId, `neighbors/${side}`, {
      ...input, expected_input_digest: digest, confirmed: true,
    })),

  getInsertionAxis: async (designId) => {
    const [axis, report] = await Promise.all([
      api.getAxis(designId).catch(() => null),
      api.design(designId),
    ]);
    const stored = JSON.parse(report.axis_json || "[0,0,1]");
    return { axis: stored, history: axis, confirmed: !!report.current_axis_revision_id };
  },
  suggestInsertionAxis: (designId, marginPoints) =>
    api.suggestAxis(designId, { margin_points: marginPoints?.length ? marginPoints : undefined }),
  setInsertionAxis: (designId, axis, note) => api.setAxis(designId, axis, note),

  // ── 参数 / 模板 ─────────────────────────────────────────────────────────
  saveProfile: (designId, parameters) =>
    api.saveProfile(designId, {
      material: "1.0 设计配置",
      bite_confirmed: true,
      coverage: "ENGINEERING_ONLY",
      confirmed: true,
      parameters,
    }),

  /** 牙冠模板候选（按 FDI 由后端筛选）。 */
  getCrownTemplates: async (designId) => {
    const result = await api.autoFitTemplates(designId);
    const libraries = (result && result.libraries) || [];
    const eligibility = await api.generationEligibility(designId).catch(() => null);
    return { libraries, eligibility, fdi: result?.fdi ?? null };
  },
  /** 模板预览：后端只登记候选修订，**不**改当前外形。返回的候选 `id` 就是 accept 用的 id。 */
  selectCrownTemplate: (designId, libraryName) =>
    designContext(designId).then(({ digest }) =>
      api.templatePreview(designId, { expected_input_digest: digest, confirmed: true, library: libraryName })),
  confirmTemplateReference: (designId, candidateId) =>
    designContext(designId).then(({ digest }) =>
      api.selectTemplate(designId, candidateId, { expected_input_digest: digest, confirmed: true })),
  /** 采纳候选：这一步才会把候选写进当前外形（outer_current 更新）。 */
  confirmCrownCandidate: (designId, candidateId, { templateAnatomy = false } = {}) =>
    designContext(designId).then(({ digest }) =>
      api.acceptCandidate(designId, candidateId, {
        expected_input_digest: digest, confirmed: true,
        ...(templateAnatomy ? { confirmed_template_anatomy: true } : {}),
      })),
  candidateUrl: (designId, candidateId, label = "candidate") =>
    `${base}/api/designs/${designId}/auto-fit/${candidateId}/${label}.stl?v=${Date.now()}`,

  /**
   * 不选模板时的自动生成。请求体合同是 ``kind: AUTO | TEMPLATE_ONLY``（不是 mode），
   * 返回的是**候选修订**：候选要到"采纳"那一步才成为当前外形。
   */
  generateInitialCrown: async (designId, { kind = "AUTO" } = {}) => {
    const eligibility = await api.generationEligibility(designId);
    if (!eligibility.template_allowed) {
      throw new Error((eligibility.blockers || []).map((b) => b.message).join("；") || "当前输入还不能自动生成");
    }
    if (kind === "AUTO" && !eligibility.automatic_mode) {
      throw new Error("尚无可执行的自动适配：请先确认咬合位姿或邻牙参考及目标间隙");
    }
    const candidate = await api.generationCandidate(designId, {
      kind, expected_input_digest: eligibility.input_digest, confirmed: true,
    });
    return candidate;
  },

  // ── 设计页 ──────────────────────────────────────────────────────────────
  sculptSession: (designId) => api.sculptSession(designId),
  /** 开发态叠加用：带逐顶点语义权重（正式工作流不需要）。 */
  sculptSessionRegions: (designId) => api.sculptSessionRegions(designId),

  /**
   * 语义区域在服务端**自动建立**（冠几何 + 就位轴 + 牙位朝向），前端不需要人工标注。
   * 这里只把状态读出来给界面显示；拿不到时如实报原因。
   */
  getRegionMap: async (designId, { withWeights = false } = {}) => {
    const session = withWeights ? await api.sculptSessionRegions(designId) : await api.sculptSession(designId);
    return { regionMap: session.region_map, weights: session.region_weights,
             protection: session.region_protection, manual: session.manual_region_names || [],
             automatic: session.automatic_region_names || [] };
  },

  /** 邻牙（近中 / 远中）适配：语义区域由服务端自动给出，这里只负责提交请求。 */
  adaptToNeighbors: (designId, { side, targetMm }) =>
    adapter._adapt(designId, side === "mesial" ? "adapt_mesial" : "adapt_distal", targetMm),
  /**
   * 咬合适配：唯一入口 `POST /adapt-occlusal`（服务端按对颌天花板给咬合区减材）。
   * 失败时服务端**不改冠、不发布**，这里如实把理由与 before/after 交给界面显示。
   */
  adaptToAntagonist: async (designId, { targetMm } = {}) => {
    const report = await api.adaptOcclusal(designId, targetMm ? { target_mm: targetMm } : {});
    return {
      status: report.status,
      reason: report.reason || null,
      reasons: report.reasons || [],
      occlusion: report.occlusion || null,
      requiredDrop: report.required_drop || null,
      occlusalSpace: report.occlusal_space || null,
      protected: report.protected || null,
      meshUnchanged: report.mesh_unchanged === true,
      published: report.published === true,
      changed: report.published === true,
      skipped: [],
      regionMap: null,
      marginDisplacementMm: report.protected ? report.protected.margin_band_displacement_max_mm : null,
      verdict: report.verdict || null,
      result: report,
    };
  },

  async _adapt(designId, operation, targetMm) {
    const design = await api.design(designId);
    if (!design.outer_path) throw new Error("请先生成牙冠外形");
    const result = await api.editOuter(designId, {
      operation, target_distance_mm: targetMm ?? null, base_outer_path: design.outer_path,
    });
    return {
      status: result?.status || "APPLIED",
      skipped: result?.edit?.skipped_sides || [],
      regionMap: result?.edit?.region_report || null,
      marginDisplacementMm: result?.edit?.margin_displacement_mm ?? null,
      changed: result?.final_invalidated === true,
      result,
    };
  },

  /** Free-forming：笔刷（add/remove/smooth/flatten/drag）与整体变换（transform/region_transform）。 */
  applyFreeformEdit: async (designId, payload) => {
    const design = await api.design(designId);
    if (!design.outer_path) throw new Error("请先生成牙冠外形");
    return api.editOuter(designId, { ...payload, base_outer_path: design.outer_path });
  },

  /** 真实的接触/咬合逐顶点有符号距离（Contact Map 的唯一数据来源）。 */
  getContactMap: (designId, kind) => api.outerAnalysis(designId, kind),
  /** 壁厚（内外表面距离）与制造复核报告。 */
  getThickness: (designId) => api.outerAnalysis(designId, "thickness"),
  getManufacturingQc: (designId) => api.manufacturingQc(designId),

  // ── 历史 ────────────────────────────────────────────────────────────────
  /** 外形修订游标（P1-9）：**只用后端显式字段**，不靠 is_current 推断。 */
  revisionCursor: async (designId) => {
    const chain = await api.revisions(designId);
    return {
      history: chain.outer_history || [],
      current: chain.current_outer_revision_no ?? null,
      previous: chain.previous_outer_revision_no ?? null,
      next: chain.next_outer_revision_no ?? null,
      canUndo: chain.can_undo_outer === true,
      canRedo: chain.can_redo_outer === true,
    };
  },
  /** 撤销 = 线性回退到上一个外形修订。 */
  undo: async (designId) => {
    const cursor = await adapter.revisionCursor(designId);
    if (cursor.previous == null) throw new Error("没有可撤销的外形修订");
    return api.restoreOuter(designId, cursor.previous);
  },
  /** 重做 = 线性前进到下一个外形修订。 */
  redo: async (designId) => {
    const cursor = await adapter.revisionCursor(designId);
    if (cursor.next == null) throw new Error("已经在最新的外形修订上");
    return api.restoreOuter(designId, cursor.next);
  },
  /** 重置到本次设计最早的外形修订。 */
  resetCrown: async (designId) => {
    const cursor = await adapter.revisionCursor(designId);
    const first = cursor.history[0];
    if (first == null) throw new Error("还没有外形修订可重置");
    return api.restoreOuter(designId, first);
  },

  // ── 检查 / 导出 ─────────────────────────────────────────────────────────
  /**
   * 测量状态（P2-11）：只有"有符号 + 有有效样本 + 已测量"才允许下结论。
   * 无符号幅值（UNSIGNED_ONLY）不能判穿透；缺数据一律 NOT_ASSESSED —— 绝不当 PASS。
   */
  measurementState: (payload) => {
    if (!payload) return "NOT_ASSESSED";
    const hasValid = Array.isArray(payload.valid) ? payload.valid.some(Boolean) : true;
    const signed = payload.signed_mm;
    const hasSigned = Array.isArray(signed) && signed.length > 0;
    if (payload.sign_method === "UNSIGNED_ONLY") return "UNSIGNED_ONLY";
    if (!hasSigned) return Array.isArray(payload.distance_mm) ? "UNSIGNED_ONLY" : "NOT_ASSESSED";
    if (!hasValid) return "INVALID";
    return "ASSESSED";
  },
  /**
   * Final QC 归一化（P2-12）：后端真实结构是 ``qc.manufacturing_qc.FinalCrownThicknessAudit``。
   * UI 只读归一化后的字段（不读 qc.thickness / qc.minimum_thickness_mm，也不猜深层结构）。
   */
  normalizeFinalQc: (qc) => {
    if (!qc) return null;
    const manufacturing = qc.manufacturing_qc || {};
    const audit = manufacturing.FinalCrownThicknessAudit || qc.FinalCrownThicknessAudit || null;
    const conservative = (audit && audit.conservative) || {};
    const samples = audit ? Number(audit.samples || 0) : 0;
    return {
      thickness: audit ? {
        state: samples > 0 ? "ASSESSED" : "NOT_ASSESSED",
        status: audit.status || "REVIEW_REQUIRED",
        min_mm: typeof conservative.min_mm === "number" ? conservative.min_mm : null,
        p05_mm: typeof conservative.p05_mm === "number" ? conservative.p05_mm : null,
        configured_minimum_mm: audit.configured_minimum_mm ?? null,
        thin_area_mm2: audit.body_thin_face_sample_area_mm2 ?? audit.thin_face_sample_area_mm2 ?? null,
        material_limit_validated: audit.material_limit_validated === true,
        coverage_note: audit.coverage_note || null,
        source: "qc.manufacturing_qc.FinalCrownThicknessAudit",
      } : { state: "NOT_ASSESSED", status: "NOT_ASSESSED", source: null },
      margin_build: manufacturing.NonplanarMarginBuild || null,
      intaglio_fit: manufacturing.IntaglioFitAudit || null,
      release_status: qc.release_status || "NOT_PRODUCTION_APPROVED",
    };
  },
  validateCrown: async (designId) => {
    const status = await api.designStatus(designId);
    const qc = await api.design(designId).then((d) => JSON.parse(d.qc_json || "null"));
    const gate = await api.supportContract(designId).catch(() => null);
    return { status, qc, finalQc: adapter.normalizeFinalQc(qc), gate };
  },
  finalize: (designId, { preserveCervical = false } = {}) =>
    designContext(designId).then(({ digest, revisions }) => {
      const missing = ["margin_revision_id", "axis_revision_id", "profile_revision_id"].filter((k) => !revisions[k]);
      if (missing.length) throw new Error("请先确认边缘线、插入轴和设计参数");
      const request = { expected_input_digest: digest, ...revisions };
      return preserveCervical ? api.rebuildCervical(designId, request) : api.finalize(designId, request);
    }),
  finalCrownUrl: (designId, version) => api.finalUrl(designId) + (version ? `?v=${version}` : ""),
  outerUrl: (designId, version) => api.outerUrl(designId) + (version ? `?v=${version}` : ""),
  innerCrownUrl: (designId) => api.cutterUrl(designId),
  exportPackageUrl: (designId) => api.packageUrl(designId),

  // ── 明确缺失的能力（不 fake） ─────────────────────────────────────────
  autoBiteRegistration: () => { require("autoBiteRegistration"); },
  dynamicOcclusion: () => { require("dynamicOcclusion"); },
  toolCompensation: () => { require("toolCompensation"); },
};

export default adapter;
