/**
 * 一次 Crown Design 会话的全部编排（步骤、数据、动作）。
 *
 * 页面只读这里的字段、只调这里的动作；后端调用一律走 `domain/adapter`。
 * 原则：服务端状态优先（confirmed revisions / 当前外形 / QC 都由 `/status` 与设计行给出），
 * 前端不自己判断"算不算已确认"。
 */
import { useCallback, useEffect, useRef, useState } from "react";
import { api } from "../api";
import adapter from "./adapter";
import { S, factsFrom, gateFor } from "./steps";
import useDesignStatus from "../workflow/useDesignStatus";
import { MIN_MARGIN_POINTS } from "../marginReview";

export const DEFAULT_PARAMS = {
  cement_gap_mm: 0.06,
  margin_gap_mm: 0,
  margin_band_mm: 0.8,
  minimum_thickness_mm: 0.8,
  margin_extension_mm: 0,
  enforce_minimum_thickness: false,
  machine_axes: 5,
  cement_gap_start_mm: 0,
  proximal_target_mm: -0.02,
  occlusal_target_mm: 0.02,
};

/** 第 4 页暴露给用户的 5 个核心参数（其余走"更多参数"）。 */
export const CORE_PARAMS = [
  { key: "cement_gap_mm", label: "Cement Gap", hint: "粘接间隙", step: 0.005, min: 0, max: 0.3 },
  { key: "margin_gap_mm", label: "Margin Gap", hint: "边缘区域（当前版本固定）", step: 0.005, min: 0, max: 0.2, fixed: true },
  { key: "minimum_thickness_mm", label: "Minimum Thickness", hint: "最小壁厚", step: 0.05, min: 0.3, max: 2.5 },
  { key: "proximal_target_mm", label: "Proximal Contact", hint: "邻牙接触：0=刚好接触，负值=轻微压接", step: 0.01, min: -0.1, max: 0.5 },
  { key: "occlusal_target_mm", label: "Occlusal Clearance", hint: "静态对颌目标间隙，默认近接触", step: 0.01, min: 0, max: 0.5 },
];

export const ADVANCED_PARAMS = [
  { key: "margin_band_mm", label: "边缘带宽度", step: 0.1, min: 0, max: 3 },
  { key: "margin_extension_mm", label: "边缘延伸", step: 0.05, min: -0.5, max: 1, fixed: true },
  { key: "cement_gap_start_mm", label: "起隙位置", step: 0.01, min: 0, max: 1 },
  { key: "machine_axes", label: "加工轴数", step: 1, min: 3, max: 5 },
];

export default function useWorkflow() {
  const [step, setStep] = useState(S.HOME);
  const [cases, setCases] = useState([]);
  const [designs, setDesigns] = useState({});
  const [caseId, setCaseId] = useState("");
  const [detail, setDetail] = useState(null);
  const [fdi, setFdi] = useState("");
  const [supportType, setSupportType] = useState("");
  const [design, setDesign] = useState(null);
  const [busy, setBusy] = useState(false);
  const [busyLabel, setBusyLabel] = useState("");
  const [msg, setMsg] = useState("");
  const [margin, setMarginRaw] = useState([]);
  const [marginSession, setMarginSession] = useState(null);
  const [marginTool, setMarginTool] = useState("edit");
  const [featureMode, setFeatureMode] = useState("AUTO");
  const [smoothStrength, setSmoothStrength] = useState(0.5);
  const [axis, setAxis] = useState([0, 0, 1]);
  const [axisInfo, setAxisInfo] = useState(null);
  const [params, setParams] = useState({ ...DEFAULT_PARAMS });
  const [templates, setTemplates] = useState({ libraries: [], eligibility: null });
  const [library, setLibrary] = useState("");
  const [templateCandidate, setTemplateCandidate] = useState(null);
  const [generatedCandidate, setGeneratedCandidate] = useState(null);
  const [sculpt, setSculpt] = useState(null);
  const [regionMap, setRegionMap] = useState(null);
  const [regionWeights, setRegionWeights] = useState(null);
  const [regionProtection, setRegionProtection] = useState(null);
  const [contacts, setContacts] = useState(null);
  const [contactKind, setContactKind] = useState("antagonist");
  const [contactUpdating, setContactUpdating] = useState(false);
  const contactRequest = useRef(0);
  useEffect(() => {
    return () => { contactRequest.current++; setContacts(null); setSculpt(null); setThickness(null); setContactUpdating(false); };
  }, [design?.id]);
  const [thickness, setThickness] = useState(null);
  const [adaptStatus, setAdaptStatus] = useState(null);
  const [qc, setQc] = useState(null);
  const [meshVersion, setMeshVersion] = useState(0);
  const [view, setView] = useState({ upper: true, lower: true, support: true, margin: true, wireframe: false, grid: false, scanOpacity: 0.62 });
  const [designTool, setDesignTool] = useState("select");
  const [brush, setBrush] = useState({ radius: 1.5, strength: 0.3, falloff: 2 });
  const [editState, setEditState] = useState("IDLE");
  const [editFailure, setEditFailure] = useState(null);
  const [pickRole, setPickRole] = useState(null);
  const [neighborSelection, setNeighborSelection] = useState(null);
  const [supportCenter, setSupportCenter] = useState(null);
  const [detection, setDetection] = useState("");
  const [marginUndo, setMarginUndo] = useState([]);
  const [marginRedo, setMarginRedo] = useState([]);
  const [marginDirty, setMarginDirty] = useState(false);   // 用户手工改过曲线 ⇒ 以用户曲线为权威
  const [frame, setFrame] = useState(null);
  // ⚠️ useDesignStatus 返回的是 [state, updater]；解构成数组再读字段会把状态读成数组（P0-1）。
  const [serverStatus] = useDesignStatus(design, busy, setMsg);
  const frameConfirmed = !!serverStatus?.frame_confirmed;
  const selectedTemplate = design?.landmarks_json
    ? JSON.parse(design.landmarks_json)._selected_template || null : null;
  const probeAt = useRef(0);

  const setMargin = useCallback((next) => {
    setMarginRaw((current) => {
      setMarginUndo((history) => [...history.slice(-49), current]);
      return typeof next === "function" ? next(current) : next;
    });
    setMarginRedo([]);
  }, []);
  /** 用户手工改曲线：本地预览 + 标记"以用户曲线为权威"（确认时直接写这份）。 */
  const editMargin = useCallback((next) => { setMargin(next); setMarginDirty(true); }, [setMargin]);

  /**
   * 统一的长任务包装：`label` 会显示在全屏遮罩上（别的电脑上等的就是这一步，
   * 不能让人以为"生成失败"）。
   */
  /** 重新拉取所有网格（换 `?v=` 绕过 three 对失败 promise 的缓存）。 */
  const reloadMeshes = () => setMeshVersion(Date.now());

  const work = async (fn, label, failureLabel = "操作失败") => {
    if (busy) return false;
    setBusy(true);
    setBusyLabel(label || "正在处理…");
    setMsg("");
    try {
      return await fn();
    } catch (error) {
      if (error?.payload?.status === "REJECTED_AND_ROLLED_BACK") {
        setEditState("REJECTED");
        setEditFailure(error.payload);
      } else if (error?.status === 409 && /编辑|会话|输入/.test(error?.message || "")) {
        setEditState("STALE");
      }
      setMsg(error?.network && failureLabel !== "操作失败"
        ? "连接中断，无法确认是否生成成功；请重新打开病例核对结果，避免重复提交"
        : `${failureLabel}：${error?.message || "请稍后重试"}`);
      return false;
    } finally {
      if (design?.id) {
        try {
          setDesign(await api.design(design.id));
        } catch { /* 刷新失败不覆盖已有错误提示 */ }
      }
      setBusy(false);
      setBusyLabel("");
    }
  };

  const refresh = useCallback(async () => {
    const list = await api.cases();
    setCases(list);
    const map = {};
    await Promise.all(list.map(async (item) => {
      try { map[item.id] = await api.caseDesigns(item.id); } catch { map[item.id] = []; }
    }));
    setDesigns(map);
  }, []);

  useEffect(() => { refresh().catch((error) => setMsg(error.message)); }, [refresh]);

  // 步骤准入：不允许跳过前置确认直接往前跳（回退始终允许）。
  const facts = factsFrom({
    caseId,
    design,
    assets: { upper: has(detail, "upper"), lower: has(detail, "lower"), support: has(detail, "support") },
    confirmed: {
      margin: !!serverStatus?.confirmed_revisions?.margin_revision_id,
      axis: !!serverStatus?.confirmed_revisions?.axis_revision_id,
      profile: !!serverStatus?.confirmed_revisions?.profile_revision_id,
      frame: frameConfirmed,
    },
    hasOuter: !!serverStatus?.outer_current,
    hasFinal: !!serverStatus?.final_current,
  });

  const go = (target) => {
    if (busy) return false;
    if (target === step) return true;
    const gate = gateFor(target, facts);
    const forward = ["case", "margin", "setup", "design", "review"].indexOf(target) > ["case", "margin", "setup", "design", "review"].indexOf(step);
    if (forward && !gate.ok) { setMsg(gate.reason); return false; }
    setPickRole(null);
    setNeighborSelection(null);
    setMsg("");
    setStep(target);
    return true;
  };

  // ── 病例 ────────────────────────────────────────────────────────────────
  const openCase = (id) => work(async () => {
    const row = await adapter.loadCase(id);
    setDetail(row); setCaseId(id); setFdi(""); setSupportType("");
    setNeighborSelection(null);
    setDesign(null); setQc(null); setMarginRaw([]); setMarginSession(null); setTemplates({ libraries: [], eligibility: null });
    setStep(S.CASE);
    setMsg("请确认牙位与修复类型，再导入上颌 / 下颌文件");
  });

  const createCase = () => work(async () => {
    const name = `病例 ${new Date().toISOString().slice(0, 10)}-${cases.length + 1}`;
    const created = await adapter.createCase(name);
    await refresh();
    await openCase(created.id);
  });

  const removeCase = (id) => work(async () => {
    await adapter.deleteCase(id);
    await refresh();
    if (caseId === id) { setCaseId(""); setDetail(null); setDesign(null); }
    setMsg("病例已删除");
  });

  const upload = (kind, file) => work(async () => {
    const row = await adapter.loadJaw(caseId, kind, file);
    setDetail(row);
    setMeshVersion(Date.now());
    setMsg(`已导入${kind === "upper" ? "上颌" : "下颌"}模型（保留原始咬合坐标）`);
  });

  /** 任何会新增/替换资产的操作之后都要刷新资产表，否则 Viewer 还在用旧地址（P0-5）。 */
  const refreshDetail = async () => {
    if (!caseId) return null;
    const row = await adapter.loadCase(caseId);
    setDetail(row);
    setMeshVersion(Date.now());
    return row;
  };

  const importOrder = (file) => work(async () => {
    setBusyLabel("正在解析订单包…");
    const result = await adapter.importOrderPackage(file, `导入 ${new Date().toISOString().slice(0, 10)}`);
    await refresh();
    await openCase(result.case_id || result.case?.id || result.id);
    setMsg("订单包已导入");
  });

  const prepare = () => work(async () => {
    setBusyLabel("正在准备基牙与边缘线（提取支撑面、生成预备体）…");
    if (design) { setStep(S.MARGIN); return true; }
    const created = await adapter.setToothPosition({ caseId, fdi, supportType });
    setDesign(created);
    setFdi(String(created.target_fdi ?? fdi));
    setSupportType(created.support_type || supportType);
    setDesigns((map) => ({ ...map, [caseId]: [...(map[caseId] || []), created] }));
    setStep(S.MARGIN);
    setMsg("设计已创建，请点“自动识别边缘线”");
    return true;
  });

  const resumeDesign = (designId, targetCaseId) => work(async () => {
    setBusyLabel("正在载入设计…");
    const [row, status] = await Promise.all([api.design(designId), api.designStatus(designId)]);
    setDesign(row);
    setNeighborSelection(null);
    setCaseId(targetCaseId);
    setDetail(await adapter.loadCase(targetCaseId));
    setFdi(String(row.target_fdi));
    setSupportType(row.support_type);
    setParams({ ...DEFAULT_PARAMS, ...JSON.parse(row.parameters_json || "{}") });
    setMarginRaw(JSON.parse(row.margin_json || "[]"));
    setAxis(JSON.parse(row.axis_json || "[0,0,1]"));
    setQc(JSON.parse(row.qc_json || "null"));
    setMeshVersion(Date.now());
    setStep(status.final_current ? S.REVIEW : status.outer_current ? S.DESIGN
      : status.confirmed_revisions?.margin_revision_id && status.confirmed_revisions?.axis_revision_id && status.frame_confirmed
        ? S.SETUP : S.MARGIN);
    setMsg("已载入设计");
  });

  // ── 边缘线 ──────────────────────────────────────────────────────────────
  const adoptSession = (session) => {
    if (!session) return session;
    setMarginSession(session);
    const active = (session.proposals || []).find((p) => p.proposal_id === session.active_proposal_id) || (session.proposals || [])[0];
    if (active?.points?.length) setMargin(active.points);
    const last = (session.operations || []).at(-1);
    if (last && (last.rejected || last.action?.endsWith("_rejected"))) {
      throw new Error(`本次调整未应用：${last.reason || "边缘线会变得无效"}。已保留原线。`);
    }
    return session;
  };

  const detectMargin = () => work(async () => {
    if (!design) throw new Error("请先创建设计");
    const axisConfirmed = !!serverStatus?.confirmed_revisions?.axis_revision_id;
    if (!axisConfirmed) {
      const result = await adapter.detectMargin({ designId: design.id, axisConfirmed, supportType, center: supportCenter, axis });
      if (result.axis) setAxis(result.axis.map(Number));
      setMargin((result.points || []).map((p) => Array.from(p)));
      setMarginDirty(false);
      setDesign(await api.design(design.id));
      // 基台提取会**新增/替换 support 资产**，不刷新资产表 Viewer 就还指着旧地址（P0-5）。
      await refreshDetail();
      setDetection(supportType === "abutment" ? "自动肩线（凸外缘＋整圈路径）" : "粗定位候选（尚未绑定已确认轴）");
      setMsg(supportType === "abutment" ? "已自动提取基台肩线，请旋转核对后确认" : "已生成粗候选与临时插入轴，请逐面复核后确认");
      return;
    }
    adoptSession(await adapter.detectMargin({ designId: design.id, axisConfirmed }));
    setDetection("候选已按已确认插入轴生成");
    setMsg("已生成候选，可在 3D 中复核；不准则加纠偏点");
  });

  const startAnchorPick = () => { setPickRole("marginAnchor"); setMsg("在模型上点选该处边缘应经过的位置（1–4 个纠偏点）"); };
  const addAnchor = (point) => work(async () => {
    const anchors = [...((marginSession || {}).anchors || []), point].slice(0, 4);
    adoptSession(await adapter.addMarginAnchor(design.id, anchors, featureMode));
    setPickRole(null);
    setMsg(`已加入第 ${anchors.length} 个纠偏点`);
  });

  const moveMargin = (delta) => work(async () => {
    if (!marginSession) throw new Error("请先运行自动识别");
    adoptSession(await adapter.moveMargin(design.id, delta));
    setMsg(`整圈${delta > 0 ? "上移" : "下移"}并重新贴合表面`);
  });

  const smoothMargin = () => work(async () => {
    if (!marginSession) throw new Error("请先运行自动识别");
    adoptSession(await adapter.smoothMargin(design.id, smoothStrength, marginSession.anchors));
    setMsg("已平滑（纠偏点保持不动）");
  });

  /**
   * 确认边缘线。**只允许一条权威曲线**（P1-7）：
   * 用户手工改过（``marginDirty``）⇒ 直接把用户最后看到的曲线写成正式边缘线；
   * 否则才确认服务端会话候选。禁止"UI 显示已改、后端确认旧候选"。
   */
  const confirmMargin = () => work(async () => {
    setBusyLabel("正在保存边缘线…");
    if (marginDirty) {
      await adapter.saveMarginPoints(design.id, margin);
      setMsg("已按你在 3D 中最后看到的曲线保存并确认边缘线");
    } else if (marginSession) {
      await adapter.confirmMargin(design.id, marginSession.active_proposal_id, undefined);
      setMsg("边缘线已确认（服务端候选）");
    } else {
      await adapter.saveMarginPoints(design.id, margin);
      setMsg("边缘线已确认");
    }
    setMarginSession(null);
    setMarginDirty(false);
    setFrame(null);
    setQc(null);
    setDesign(await api.design(design.id));
    setStep(S.MARGIN);
  });

  /** 插入轴确认后，自动推导近中/颊侧候选并确认患者帧（P0-2）。 */
  const suggestFrame = () => work(async () => {
    if (!design) throw new Error("请先创建设计");
    const candidate = adapter.suggestFrame(margin, axis);
    setFrame(candidate);
    setMsg("已按边缘线与就位轴推导近中/颊侧方向，请核对后确认（必要时翻转近中/远中）");
    return candidate;
  });
  const flipFrame = () => setFrame((current) => (current
    ? { ...current, mesial: current.mesial.map((v) => -v), buccal: current.buccal.map((v) => -v) }
    : current));
  const confirmFrame = () => work(async () => {
    if (!design) throw new Error("请先创建设计");
    const candidate = frame || adapter.suggestFrame(margin, axis);
    await adapter.confirmFrame(design.id, { mesial: candidate.mesial, buccal: candidate.buccal });
    setFrame(candidate);
    setMsg("近中/颊侧方向已确认（模板放置将按这个坐标帧进行）");
  });
  const confirmCurrentBite = () => work(async () => {
    if (!has(detail, "upper") || !has(detail, "lower")) throw new Error("请先导入上下颌扫描");
    await adapter.confirmCurrentBite(design.id);
    setMsg("已确认上下颌扫描使用当前相对位置；后续只按这一静态咬合位姿测量空间");
  });
  const selectNeighbor = (side) => {
    setNeighborSelection(null);
    setPickRole(`neighbor:${side}`);
    setMsg(`请在工作颌的${side === "mesial" ? "近中" : "远中"}邻牙接触面点击；随后检查高亮区域`);
  };
  const confirmNeighbor = (side, absent = false) => work(async () => {
    const source = Number(fdi) >= 31 ? "lower" : "upper";
    if (!absent && (neighborSelection?.side !== side || neighborSelection.source !== source || neighborSelection.face_ids.length < 3)) {
      throw new Error("请先在工作颌邻牙表面点选有效区域");
    }
    await adapter.confirmNeighbor(design.id, side, {
      status: absent ? "ABSENT" : "CONFIRMED", source,
      instance_name: `${side === "mesial" ? "近中" : "远中"}邻牙${absent ? "缺失" : "人工确认面"}`,
      face_ids: absent ? [] : neighborSelection.face_ids,
      pollution_reviewed: !absent,
    });
    setNeighborSelection(null); setPickRole(null);
    setMsg(`${side === "mesial" ? "近中" : "远中"}侧已${absent ? "标记无邻牙" : "确认邻牙参考面"}`);
  });

  const undoMargin = () => {
    if (!marginUndo.length) return;
    setMarginRedo((redo) => [...redo, margin]);
    setMarginRaw(marginUndo.at(-1));
    setMarginUndo((history) => history.slice(0, -1));
  };
  const redoMargin = () => {
    if (!marginRedo.length) return;
    setMarginUndo((history) => [...history, margin]);
    setMarginRaw(marginRedo.at(-1));
    setMarginRedo((redo) => redo.slice(0, -1));
  };
  const drawMargin = () => { setPickRole("margin"); setMsg("沿真实边缘逐点绕一圈"); };
  const addMarginPoint = (point) => setMargin((current) => [...current, point]);

  // ── 插入轴 ──────────────────────────────────────────────────────────────
  const suggestAxis = () => work(async () => {
    if (!design) throw new Error("请先创建设计");
    const result = await adapter.suggestInsertionAxis(design.id, margin);
    setAxisInfo(result);
    if (Array.isArray(result.axis) && result.axis.length === 3) setAxis(result.axis.map(Number));
    setMsg("已自动计算插入轴，可拖动箭头复核");
  });
  const resetAxis = () => { setAxis([0, 0, 1]); setAxisInfo(null); setMsg("已恢复默认方向，请重新自动计算"); };
  const confirmAxis = () => work(async () => {
    if (!design) throw new Error("请先创建设计");
    await adapter.setInsertionAxis(design.id, axis);
    setFrame(null);
    setQc(null);
    setDesign(await api.design(design.id));
    setMsg("插入轴已确认");
  });

  // ── 参数 / 模板 ─────────────────────────────────────────────────────────
  const applyParams = () => work(async () => {
    setBusyLabel("正在应用参数…");
    if (!design) throw new Error("请先创建设计");
    await adapter.saveProfile(design.id, params);
    setTemplateCandidate(null);
    setGeneratedCandidate(null);
    setDesign(await api.design(design.id));
    setQc(null);
    setMsg("设计参数已应用");
    api.generationProfile(design.id).catch(() => {});
  });

  const loadTemplates = () => work(async () => {
    const result = await adapter.getCrownTemplates(design.id);
    setTemplates(result);
    if (!library && result.libraries.length) setLibrary(result.libraries[0].name);
    return result;
  });
  useEffect(() => {
    if (design && step === S.SETUP) loadTemplates();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [design?.id, design?.design_revision, step]);

  const previewTemplate = (name) => work(async () => {
    setTemplateCandidate(null);
    await adapter.saveProfile(design.id, params);
    const candidate = await adapter.selectCrownTemplate(design.id, name);
    setTemplateCandidate(candidate);
    setMsg("已按该牙型生成就位候选，请在 3D 中检查后点“应用并确认模板”");
  });
  /** 固定牙库来源后继续自动设计；模板初放置不是最终牙冠。 */
  const acceptTemplate = () => work(async () => {
    setBusyLabel("正在确认牙库参考形态…");
    if (!templateCandidate?.id) throw new Error("请先预览一个牙型");
    await adapter.confirmTemplateReference(design.id, templateCandidate.id);
    setTemplateCandidate(null);
    setGeneratedCandidate(null);
    setMsg("已选定牙库参考形态。请继续自动生成；当前放置结果不是设计完成的牙冠。");
  });
  /**
   * Step5 冠生成/采纳之后**立即**做颈缘重建（调用既有 rebuild-cervical：
   * margin curve → cervical ring → crown boundary reconstruction），
   * 让第 5 步显示的是 **margin-constrained crown**，而不是模板壳。
   */
  const cervicalRebuildInline = async () => {
    await adapter.finalize(design.id, { preserveCervical: true });
    setDesign(await api.design(design.id));
    setMeshVersion(Date.now());
    setMsg("颈缘重建完成：冠颈缘已贴合确认边缘线");
  };

  /** 自动生成：后端只登记候选，必须再"采纳"才成为当前外形（P0-4）。 */
  const generateCrown = (kind = "AUTO") => work(async () => {
    setBusyLabel("正在生成牙冠候选（可能需要 10–60 秒）…");
    await adapter.saveProfile(design.id, params);
    const candidate = await adapter.generateInitialCrown(design.id, { kind });
    setGeneratedCandidate(candidate);
    setMsg(candidate.payload?.status === "PASS_ENGINEERING"
      ? "已生成牙冠候选，请在 3D 中检查后点“采纳候选”"
      : `已生成粗糙候选，可继续设计；工程检查提示：${candidate.payload?.w3_preflight?.reason || candidate.payload?.status || "待检查"}`);
    return candidate;
  }, undefined, "牙冠候选生成失败");
  const acceptGenerated = () => work(async () => {
    setBusyLabel("正在采纳生成结果，并重建颈缘与内冠…");
    if (!generatedCandidate?.id) throw new Error("请先生成候选");
    await adapter.confirmCrownCandidate(design.id, generatedCandidate.id);
    setGeneratedCandidate(null);
    setTemplateCandidate(null);
    setMsg("候选已采纳，正在做颈缘重建（把模板壳变成贴合边缘线的冠体）…");
    await cervicalRebuildInline();
    setStep(S.DESIGN);
  });

  // ── 设计页 ──────────────────────────────────────────────────────────────
  const loadSculpt = () => work(async () => {
    const session = await adapter.sculptSession(design.id);
    setSculpt(session);
    setRegionMap(session.region_map || null);
    setEditState("READY");
    setEditFailure(null);
    if (!session.region_map || session.region_map.status === "NOT_AVAILABLE") {
      setMsg("无法自动建立牙冠语义区域：" + ((session.region_map || {}).message || "原因未知"));
    }
    return session;
  });
  /** 开发态叠加：带逐顶点权重。 */
  const loadRegionOverlay = () => work(async () => {
    const result = await adapter.getRegionMap(design.id, { withWeights: true });
    setRegionMap(result.regionMap);
    setRegionWeights(result.weights);
    setRegionProtection(result.protection);
    return result;
  });
  /**
   * 外形一变就作废派生状态（P1-8）：重新读设计行 + **重新载入 sculpt 会话**
   * （顶点/法向/拓扑/会话 sha 全部换新），并丢弃本地预览。
   */
  const reloadAfterOuterChange = async (kind = contactKind) => {
    if (!design?.id) return;
    setRegionWeights(null);
    setContacts(null);
    setThickness(null);
    setDesign(await api.design(design.id));
    setMeshVersion(Date.now());
    await adapter.sculptSession(design.id).then((session) => {
      setSculpt(session);
      setRegionMap(session.region_map || null);
      setEditState("READY");
      setEditFailure(null);
    });
    void loadContacts(kind);
  };

  const commitEdit = (payload) => work(async () => {
    if (!design?.id) throw new Error("请先创建设计");
    setEditState("VALIDATING");
    setEditFailure(null);
    const result = await adapter.applyFreeformEdit(design.id, payload);
    await reloadAfterOuterChange();
    if (result?.thickness_audit) setThickness(result.thickness_audit);
    if (result?.contacts?.status) setAdaptStatus({ status: result.contacts.status });
    setEditState("SAVED");
    // 1.0：微调一律保存；质量类问题只在消息里如实提示，不再回滚。
    const warnings = result?.edit?.quality_warnings || [];
    setMsg(warnings.length
      ? `已保存到外形修订（${payload.operation}）· 质量提示 ${warnings.length} 项：${warnings.slice(0, 2).join("；")}${warnings.length > 2 ? " …" : ""}`
      : `已应用并保存到外形修订（${payload.operation}）`);
    return result;
  }, "正在验证并保存本次微调…", "微调未保存");
  const markEditPreview = () => { if (!busy) setEditState("PREVIEW"); };
  const clearEditFailure = () => { setEditFailure(null); setEditState(s => s === "REJECTED" ? "READY" : s); };

  const loadContacts = async (kind = contactKind) => {
    if (!design?.id) return;
    const request = ++contactRequest.current;
    setContactKind(kind);
    setContacts(null);
    setContactUpdating(true);
    try {
      const data = await adapter.getContactMap(design.id, kind);
      if (request === contactRequest.current) setContacts(data);
      return data;
    } catch (error) {
      if (request === contactRequest.current) setMsg(`接触分析不可用：${error.message}`);
    } finally {
      if (request === contactRequest.current) setContactUpdating(false);
    }
  };
  const loadThickness = () => work(async () => {
    setThickness(await adapter.getThickness(design.id));
  });
  const adaptToAntagonist = () => work(async () => {
    setBusyLabel("正在做咬合适配…");
    const outcome = await adapter.adaptToAntagonist(design.id, { targetMm: params.occlusal_target_mm });
    setAdaptStatus(outcome);
    setRegionMap(outcome.regionMap || regionMap);
    if (outcome.changed) {
      await reloadAfterOuterChange("antagonist");
    }
    if (outcome.status === "OCCLUSAL_ADAPTED") {
      setMsg("已按对颌消除咬合穿透，接触图已刷新");
    } else if (outcome.status === "OCCLUSAL_ADAPTATION_FAILED") {
      // 服务端没有改冠：照实说清楚为什么，不让用户以为已适配。
      const space = outcome.occlusalSpace || {};
      const drop = outcome.requiredDrop || {};
      const detail = outcome.reasons && outcome.reasons.includes("OCCLUSAL_SPACE_INSUFFICIENT_FOR_MIN_WALL") && space.detail
        ? `咬合空间不足：${space.detail}`
        : outcome.reasons && outcome.reasons.includes("REQUIRED_DROP_EXCEEDS_OCCLUSAL_DOMAIN")
          ? `需要下压 ${(drop.max_mm ?? 0).toFixed(2)}mm，超过咬合区自身高度 ${(drop.occlusal_domain_extent_mm ?? 0).toFixed(2)}mm`
          : (outcome.verdict || "咬合适配未能消除穿透");
      setMsg(`无法完成咬合适配（冠未改动）：${detail}`);
    } else {
      setMsg("对颌适配未执行");
    }
  });
  const adaptToNeighbors = (side) => work(async () => {
    setBusyLabel("正在做邻牙适配…");
    const label = side === "mesial" ? "近中" : "远中";
    const outcome = await adapter.adaptToNeighbors(design.id, { side, targetMm: params.proximal_target_mm });
    setAdaptStatus(outcome);
    setRegionMap(outcome.regionMap || regionMap);
    if (outcome.changed) {
      await reloadAfterOuterChange("neighbors");
    }
    const skipped = outcome.skipped.find((item) => item.side === side);
    setMsg(outcome.status === "NO_APPLICABLE_SIDE"
      ? (skipped
        ? `${label}侧没有可用的邻牙（${skipped.status}），该侧已跳过，未改动几何`
        : "没有可用的邻牙区域，已跳过")
      : `${label}邻牙适配已完成，接触图已刷新`);
  });
  const undoEdit = () => work(async () => { await adapter.undo(design.id); await reloadAfterOuterChange(); setMsg("已回退到上一个外形修订"); });
  const redoEdit = () => work(async () => { await adapter.redo(design.id); await reloadAfterOuterChange(); setMsg("已恢复到下一个外形修订"); });
  const resetCrown = () => work(async () => { await adapter.resetCrown(design.id); await reloadAfterOuterChange(); setMsg("已重置到本次设计最早的外形"); });

  // ── 检查 / 导出 ─────────────────────────────────────────────────────────
  const finalize = (preserveCervical = false) => work(async () => {
    setBusyLabel("正在生成内冠与质控（可能需要 10–60 秒）…");
    const result = await adapter.finalize(design.id, { preserveCervical });
    setQc(result.qc);
    setMeshVersion(Date.now());
    try {
      setDesign(await api.design(design.id));
      setMsg("最终牙冠已生成 · 工程口径，未做制造放行");
    } catch {
      setMsg("最终牙冠已生成，但页面状态刷新失败；请重新打开病例核对结果");
    }
  }, undefined, "最终牙冠生成失败");

  const probeSection = (point, normal) => {
    if (!design || !view.margin) return;
    const now = Date.now();
    if (now - probeAt.current < 150) return;
    probeAt.current = now;
    api.sectionPreview(design.id, { point, normal, span_mm: 8 }).catch(() => {});
  };

  const pick = (role, point, source, hit) => {
    if (pickRole === "supportCenter" && role === "patient") { setSupportCenter(point); setPickRole(null); setMsg("已定位预备体中心"); }
    else if (pickRole === "marginAnchor" && role === "support") addAnchor(point);
    else if (pickRole === "margin" && role === "support") addMarginPoint(point);
    else if (pickRole?.startsWith("neighbor:") && role === "patient") {
      const expected = Number(fdi) >= 31 ? "lower" : "upper";
      if (source !== expected) { setMsg("请点击目标牙所在颌的真实邻牙表面"); return; }
      const face_ids = hit?.face_ids || [];
      setNeighborSelection({ side: pickRole.split(":")[1], source, face_ids });
      setMsg(face_ids.length >= 3 ? `已高亮 ${face_ids.length} 个扫描面；请检查只包含该侧邻牙接触面后确认` : "选区太小，请在邻牙接触面重新点击");
    }
  };

  const url = (kind) => {
    const meta = (detail?.assets || []).find((a) => a.kind === kind);
    return caseId && meta ? `${api.assetUrl(caseId, kind)}?v=${meta.id}` : null;
  };

  return {
    // 数据
    cases, designs, caseId, detail, fdi, supportType, design, serverStatus, facts,
    margin, marginSession, marginUndo, marginRedo, detection, axis, axisInfo,
    params, templates, library, selectedTemplate, templateCandidate, generatedCandidate, sculpt, contacts, contactKind, contactUpdating, thickness,
    regionMap, regionWeights, regionProtection,
    adaptStatus, qc, meshVersion, view, designTool, brush, editState, editFailure, marginTool, featureMode, smoothStrength,
    pickRole, supportCenter, neighborSelection,
    // 派生
    upper: url("upper"), lower: url("lower"), support: url("support"),
    outer: design?.outer_path ? adapter.outerUrl(design.id, meshVersion) : null,
    candidatePreview: (templateCandidate || generatedCandidate)?.id
      ? adapter.candidateUrl(design.id, (templateCandidate || generatedCandidate).id) : null,
    finalCrown: serverStatus?.final_current ? adapter.finalCrownUrl(design.id, meshVersion) : null,
    innerCrown: design?.cutter_path ? adapter.innerCrownUrl(design.id) : null,
    marginConfirmed: facts.marginConfirmed, axisConfirmed: facts.axisConfirmed, profileConfirmed: facts.profileConfirmed,
    frame, frameConfirmed, marginDirty,
    hasOuter: facts.hasOuter, hasFinal: facts.hasFinal,
    canConfirmMargin: margin.length >= MIN_MARGIN_POINTS,
    // 动作
    step, go, setStep, busy, busyLabel, msg, setMsg, reloadMeshes,
    setFdi, setSupportType, setParams, setLibrary, setView, setDesignTool, setBrush, setContactKind,
    setMarginTool, setFeatureMode, setSmoothStrength, setAxis, setPickRole, setSupportCenter,
    refresh, openCase, createCase, removeCase, upload, importOrder, prepare, resumeDesign,
    detectMargin, startAnchorPick, moveMargin, smoothMargin, confirmMargin, undoMargin, redoMargin, drawMargin,
    suggestAxis, resetAxis, confirmAxis,
    applyParams, loadTemplates, previewTemplate, acceptTemplate, generateCrown, acceptGenerated,
    suggestFrame, flipFrame, confirmFrame, confirmCurrentBite, selectNeighbor, confirmNeighbor, refreshDetail, editMargin,
    loadSculpt, loadRegionOverlay, commitEdit, markEditPreview, clearEditFailure, loadContacts, loadThickness, adaptToAntagonist, adaptToNeighbors,
    undoEdit, redoEdit, resetCrown, finalize, probeSection, pick,
  };
}

function has(detail, kind) {
  return !!(detail?.assets || []).some((asset) => asset.kind === kind);
}
