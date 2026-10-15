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
  proximal_target_mm: 0.00,
  occlusal_target_mm: 0.05,
};

/** 第 4 页暴露给用户的 5 个核心参数（其余走"更多参数"）。 */
export const CORE_PARAMS = [
  { key: "cement_gap_mm", label: "Cement Gap", hint: "粘接间隙", step: 0.005, min: 0, max: 0.3 },
  { key: "margin_gap_mm", label: "Margin Gap", hint: "边缘区域（当前版本固定）", step: 0.005, min: 0, max: 0.2, fixed: true },
  { key: "minimum_thickness_mm", label: "Minimum Thickness", hint: "最小壁厚", step: 0.05, min: 0.3, max: 2.5 },
  { key: "proximal_target_mm", label: "Proximal Contact", hint: "邻牙接触目标：0=接触，负值=轻微压接", step: 0.01, min: -0.10, max: 0.30, optional: true },
  { key: "occlusal_target_mm", label: "Occlusal Clearance", hint: "对颌间隙目标", step: 0.01, min: 0, max: 0.5, optional: true },
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
    setBusy(true); setBusyLabel(label); setMsg("");
    try { return await fn(); }
    catch (error) { setMsg(`${failureLabel}：${error.message}`); throw error; }
    finally { setBusy(false); setBusyLabel(""); }
  };

  const loadCases = useCallback(async () => {
    const rows = await api.cases();
    setCases(rows || []);
    return rows || [];
  }, []);

  const openCase = useCallback(async (id) => {
    setCaseId(id); setDesign(null); setDetail(null); setFdi(""); setSupportType("");
    setLibrary(""); setTemplateCandidate(null); setGeneratedCandidate(null); setTemplates({ libraries: [], eligibility: null });
    setMarginRaw([]); setMarginSession(null); setMarginDirty(false); setAxis([0,0,1]); setAxisInfo(null); setFrame(null);
    setMsg(""); setStep(S.CASE);
    const d = await api.caseDetail(id); setDetail(d); return d;
  }, []);

  const newCase = useCallback(async () => {
    const created = await api.createCase();
    await loadCases();
    return openCase(created.id);
  }, [loadCases, openCase]);

  const refreshDetail = useCallback(async () => {
    if (!caseId) return null;
    const d = await api.caseDetail(caseId); setDetail(d); return d;
  }, [caseId]);

  const upload = async (kind, file) => work(async () => {
    const result = await api.upload(caseId, kind, file);
    await refreshDetail();
    return result;
  }, `正在上传 ${kind === "upper" ? "上颌" : "下颌"}…`, "上传失败");

  const prepare = () => work(async () => {
    const created = await adapter.prepare({ caseId, fdi, supportType, params });
    await refreshDetail();
    await resumeDesign(created.id, caseId);
    return created;
  }, "正在准备设计…", "准备设计失败");

  const resumeDesign = useCallback(async (designId, explicitCaseId = caseId) => {
    const d = await adapter.resume(designId);
    setDesign(d.design); setFdi(String(d.design.target_fdi)); setSupportType(d.design.support_type);
    setParams((current) => ({ ...current, ...(d.parameters || {}) }));
    setMarginRaw(d.margin || []); setAxis(d.axis || [0,0,1]); setAxisInfo(d.axis_info || null); setFrame(d.frame || null);
    setTemplateCandidate(d.template || null); setGeneratedCandidate(d.generated || null); setSculpt(d.sculpt || null);
    setRegionMap(d.region_map || null); setRegionWeights(d.region_weights || null); setRegionProtection(d.region_protection || null);
    setAdaptStatus(d.adapt_status || null); setQc(d.qc || null); setLibrary(d.library || "");
    if (explicitCaseId && explicitCaseId !== caseId) setCaseId(explicitCaseId);
    reloadMeshes();
    return d;
  }, [caseId]);

  const loadTemplates = useCallback(async () => {
    if (!design?.id) return null;
    const data = await adapter.templates(design.id);
    setTemplates(data || { libraries: [], eligibility: null });
    return data;
  }, [design?.id]);

  const saveMargin = (points, source = "USER") => work(async () => {
    const response = await adapter.saveMargin(design.id, points, source);
    setMarginRaw(points); setMarginDirty(false); await resumeDesign(design.id);
    return response;
  }, "正在保存 Margin…", "Margin 保存失败");

  const setAxisRevision = (value, source = "USER") => work(async () => {
    const response = await adapter.setAxis(design.id, value, source);
    setAxis(value); await resumeDesign(design.id); return response;
  }, "正在保存插入方向…", "插入方向保存失败");

  const confirmFrame = () => work(async () => {
    const response = await adapter.confirmFrame(design.id);
    await resumeDesign(design.id); return response;
  }, "正在确认患者坐标系…", "患者坐标系确认失败");

  const go = (next) => setStep(next);
  const setStepSafe = (next) => setStep(next);

  const reset = useCallback(() => {
    setCaseId(""); setDetail(null); setFdi(""); setSupportType(""); setDesign(null); setMarginRaw([]); setAxis([0,0,1]);
    setAxisInfo(null); setFrame(null); setTemplates({ libraries: [], eligibility: null }); setLibrary("");
    setTemplateCandidate(null); setGeneratedCandidate(null); setSculpt(null); setRegionMap(null); setRegionWeights(null); setRegionProtection(null);
    setContacts(null); setQc(null); setAdaptStatus(null); setMsg(""); setStep(S.HOME);
  }, []);

  const facts = factsFrom({ design, detail, margin, axis, frameConfirmed, serverStatus, selectedTemplate, generatedCandidate });
  const gate = gateFor(step, facts);

  return {
    step, setStep: setStepSafe, go, reset,
    cases, designs, caseId, detail, loadCases, openCase, newCase, refreshDetail, upload,
    fdi, setFdi, supportType, setSupportType, design, prepare, resumeDesign,
    busy, busyLabel, msg, setMsg,
    margin, setMargin, editMargin, marginSession, setMarginSession, marginTool, setMarginTool, marginDirty,
    axis, setAxis, axisInfo, setAxisRevision, frame, frameConfirmed, confirmFrame,
    params, setParams,
    templates, loadTemplates, library, setLibrary, selectedTemplate, templateCandidate, setTemplateCandidate,
    generatedCandidate, setGeneratedCandidate,
    sculpt, setSculpt, regionMap, setRegionMap, regionWeights, setRegionWeights, regionProtection, setRegionProtection,
    contacts, setContacts, contactKind, setContactKind, contactUpdating, setContactUpdating,
    thickness, setThickness, adaptStatus, setAdaptStatus, qc, setQc,
    meshVersion, reloadMeshes, view, setView, designTool, setDesignTool, brush, setBrush,
    editState, setEditState, editFailure, setEditFailure, pickRole, setPickRole,
    neighborSelection, setNeighborSelection, supportCenter, setSupportCenter, detection, setDetection,
    marginUndo, setMarginUndo, marginRedo, setMarginRedo,
    serverStatus, facts, gate,
  };
}
