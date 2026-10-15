import React, { useEffect, useRef, useState } from "react";
import { Field, Notice, PageHead, Panel, Segmented } from "../ui";
import { ADVANCED_PARAMS, CORE_PARAMS, DEFAULT_PARAMS } from "../../domain/useWorkflow";
import { generateAbutmentOrder, prepareAbutmentOrder, selectAbutmentTemplate } from "../../domain/abutmentFast";
import "../../styles/generation-overlay.css";

const FDI_OPTIONS = ["16", "15", "14", "13", "12", "11", "21", "22", "23", "24", "25", "26",
  "36", "35", "34", "33", "32", "31", "41", "42", "43", "44", "45", "46", "37", "47"];

const MODEL_PARAM_KEYS = new Set([
  "cement_gap_mm",
  "cement_gap_start_mm",
  "minimum_thickness_mm",
  "proximal_target_mm",
  "occlusal_target_mm",
  "margin_band_mm",
  "machine_axes",
]);

const MODEL_PARAMS = [...CORE_PARAMS, ...ADVANCED_PARAMS]
  .filter((spec) => MODEL_PARAM_KEYS.has(spec.key) && !spec.fixed);

function Upload({ label, mark, hint, name, disabled, onPick }) {
  const input = useRef(null);
  return <label className={`cd-upload cd-order-upload ${name ? "done" : ""}`}>
    <input ref={input} type="file" accept=".stl,.STL,.dcm,.DCM" disabled={disabled}
      onChange={(event) => { const file = event.target.files?.[0]; if (file) onPick(file); }} />
    <div className={`cd-upload-mark ${mark === "L" ? "lower" : ""}`}>{mark}</div>
    <div className="cd-order-upload-copy">
      <b>{label}</b>
      <span>{name || `选择${label}文件`}</span>
      <small>{hint}</small>
    </div>
    <div className="cd-order-upload-state">{name ? "已导入" : "上传"}</div>
  </label>;
}

function ParamCard({ spec, value, disabled, onChange }) {
  const preset = DEFAULT_PARAMS[spec.key];
  const unit = spec.key === "machine_axes" ? "轴" : "mm";
  return <label className="cd-model-param">
    <div className="cd-model-param-head">
      <span><b>{spec.label}</b><small>{spec.hint || "模型生成参数"}</small></span>
      <em>预设 {preset}{unit === "mm" ? " mm" : ""}</em>
    </div>
    <div className="cd-model-param-control">
      <input type="number" min={spec.min} max={spec.max} step={spec.step} disabled={disabled}
        value={value ?? ""} placeholder={spec.optional ? "不设置" : String(preset ?? "")}
        onChange={(event) => onChange(event.target.value === "" ? null : Number(event.target.value))} />
      <span>{unit}</span>
    </div>
  </label>;
}

function GenerationProgress({ value }) {
  if (!value) return null;
  const progress = Math.max(0, Math.min(100, Number(value.progress || 0)));
  return <div className="cd-generation-progress" role="status" aria-live="polite">
    <div className="cd-generation-progress-head"><b>{value.label}</b><span>{Math.round(progress)}%</span></div>
    <div className="cd-generation-progress-track"><div style={{ width:`${progress}%` }} /></div>
    {value.detail && <small>{value.detail}</small>}
  </div>;
}

function GenerationOverlay({ progress, message, fdi, library }) {
  const percent = Math.max(0, Math.min(100, Number(progress?.progress || 0)));
  const label = progress?.label || "正在自动生成牙冠";
  const detail = progress?.detail || message || "正在根据基台、邻牙与对颌空间计算患者化牙冠…";
  return <div className="cd-generation-overlay" role="dialog" aria-modal="true" aria-labelledby="generation-overlay-title">
    <div className="cd-generation-overlay-card">
      <div className="cd-generation-overlay-icon" aria-hidden="true">
        <span className="cd-generation-overlay-ring" />
        <strong>{Math.round(percent)}%</strong>
      </div>
      <div className="cd-generation-overlay-copy">
        <div className="cd-generation-overlay-eyebrow">CROWNFLOW AUTO DESIGN</div>
        <h2 id="generation-overlay-title">正在生成牙冠</h2>
        <div className="cd-generation-overlay-meta">
          {fdi && <span>FDI {fdi}</span>}
          {library && <span>{library}</span>}
        </div>
      </div>
      <div className="cd-generation-overlay-stage">
        <b>{label}</b><span>{Math.round(percent)}%</span>
      </div>
      <div className="cd-generation-overlay-track" aria-label={`生成进度 ${Math.round(percent)}%`}>
        <div style={{ width:`${percent}%` }} />
      </div>
      <p>{detail}</p>
      <div className="cd-generation-overlay-steps" aria-hidden="true">
        <span className={percent >= 5 ? "on" : ""}>空间适配</span>
        <i />
        <span className={percent >= 25 ? "on" : ""}>Margin</span>
        <i />
        <span className={percent >= 45 ? "on" : ""}>邻接</span>
        <i />
        <span className={percent >= 60 ? "on" : ""}>咬合</span>
        <i />
        <span className={percent >= 80 ? "on" : ""}>内冠 / Final</span>
      </div>
      <small className="cd-generation-overlay-note">生成过程中请勿刷新或关闭页面。完成后会自动进入检查结果。</small>
    </div>
  </div>;
}

function assetName(assets, kind) {
  const asset = assets.find((item) => item.kind === kind);
  if (!asset) return "";
  return asset.original_name || asset.filename || asset.name || "已导入";
}

export default function CasePage({ wf }) {
  const assets = wf.detail?.assets || [];
  const fdi = wf.design ? String(wf.design.target_fdi) : wf.fdi;
  const supportType = wf.design?.support_type || wf.supportType;
  const has = (kind) => assets.some((asset) => asset.kind === kind);
  const fastAbutment = supportType === "abutment";
  const [fastBusy, setFastBusy] = useState(false);
  const [fastMsg, setFastMsg] = useState("");
  const [fastProgress, setFastProgress] = useState(null);
  const [templatesRequested, setTemplatesRequested] = useState(false);
  const [generationOpen, setGenerationOpen] = useState(false);

  const bothJaws = has("upper") && has("lower");
  const fastPrepared = Boolean(wf.design && wf.marginConfirmed && wf.axisConfirmed && wf.frameConfirmed && wf.serverStatus?.bite?.pose_confirmed);
  const neighbors = wf.serverStatus?.neighbors || {};
  const selectedLibrary = wf.selectedTemplate?.library || wf.library || "";
  const orderReady = Boolean(bothJaws && fdi && supportType);

  useEffect(() => {
    setFastBusy(false);
    setFastMsg("");
    setFastProgress(null);
    setTemplatesRequested(false);
    setGenerationOpen(false);
    wf.setLibrary("");
    // 病例相关的页面缓存必须随 caseId 清空；参数故意保留，作为用户偏好跨病例复用。
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [wf.caseId]);

  useEffect(() => {
    if (!generationOpen) return undefined;
    const previousOverflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    return () => { document.body.style.overflow = previousOverflow; };
  }, [generationOpen]);

  useEffect(() => { setTemplatesRequested(false); }, [wf.design?.id]);
  useEffect(() => {
    if (!fastAbutment || !fastPrepared || templatesRequested || wf.busy || fastBusy) return;
    setTemplatesRequested(true);
    wf.loadTemplates();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [fastAbutment, fastPrepared, wf.design?.id, templatesRequested, wf.busy, fastBusy]);

  const onProgress = (stage) => {
    setFastProgress(stage);
    if (stage?.label) setFastMsg(stage.detail ? `${stage.label}：${stage.detail}` : stage.label);
  };

  const prepareFast = async () => {
    setFastBusy(true); setFastProgress(null); setFastMsg("正在自动识别基台 Margin、插入方向、咬合和邻牙参考…");
    try {
      const result = await prepareAbutmentOrder({
        caseId: wf.caseId, fdi, designId: wf.design?.id || null, onProgress,
      });
      await wf.refreshDetail();
      await wf.resumeDesign(result.design.id, wf.caseId);
      wf.setStep("case");
      const detected = Object.values(result.neighbors?.neighbors || {}).filter((item) => item.confirmation_status === "CONFIRMED").length;
      setFastProgress({ id:"READY", label:"自动准备完成", progress:100, detail:`找到 ${detected} 个邻牙参考面` });
      setFastMsg(`自动参考已建立：Margin / Axis / Bite 完成，找到 ${detected} 个邻牙接触面。`);
    } catch (error) {
      setFastMsg(`自动准备未完成：${error.message}`);
    } finally { setFastBusy(false); }
  };

  const chooseLibrary = async (name) => {
    setFastBusy(true); setFastProgress(null); setFastMsg(`正在应用牙库 ${name}…`);
    try {
      wf.setLibrary(name);
      await selectAbutmentTemplate({ designId: wf.design.id, library: name, params: wf.params, onProgress });
      await wf.resumeDesign(wf.design.id, wf.caseId);
      wf.setStep("case");
      setFastProgress({ id:"TEMPLATE_READY", label:"牙库模板已确认", progress:100 });
      setFastMsg(`已选定 ${name}。生成时会按当前病例邻牙、对颌和基台空间变形。`);
    } catch (error) {
      setFastMsg(`牙库选择失败：${error.message}`);
    } finally { setFastBusy(false); }
  };

  const runFastGeneration = async () => {
    setGenerationOpen(true);
    setFastBusy(true); setFastProgress({ id:"START", label:"启动自动牙冠设计", progress:1 });
    setFastMsg("正在执行患者驱动 CAD：整体适配 → Margin → 邻接 → 咬合 → 内冠…");
    try {
      const result = await generateAbutmentOrder({ designId: wf.design.id, params: wf.params, onProgress });
      await wf.resumeDesign(wf.design.id, wf.caseId);
      if (result?.final) {
        setFastMsg(result.warnings?.length
          ? `牙冠已生成；有 ${result.warnings.length} 项工程提示，可在 3D 中检查。`
          : "基台冠已自动生成并完成内冠，正在打开最终检查。");
      } else {
        setFastMsg(`患者化外冠已生成并进入 3D。内冠/Final 尚有提示${result?.warnings?.length ? `（${result.warnings.length} 项）` : ""}。`);
      }
    } catch (error) {
      setFastProgress((current) => current ? { ...current, detail:`失败：${error.message}` } : null);
      setFastMsg(`自动生成停止：${error.message}`);
    } finally {
      setGenerationOpen(false);
      setFastBusy(false);
    }
  };

  const primaryAction = () => {
    if (!supportType) return { label:"请选择修复类型", disabled:true };
    if (!orderReady) return { label:"请先上传上下颌并选择牙位", disabled:true };
    if (!fastAbutment) return {
      label: wf.design ? "继续天然牙完整流程" : "创建天然牙设计并进入 Margin / Axis",
      disabled: wf.busy || fastBusy,
      run: () => (wf.design ? wf.go("margin") : wf.prepare()),
    };
    if (!fastPrepared) return {
      label: wf.design ? "继续自动准备病例" : "准备病例并读取牙模板",
      disabled: wf.busy || fastBusy,
      run: prepareFast,
    };
    if (!wf.selectedTemplate) return { label:"请选择牙模板", disabled:true };
    return {
      label: fastBusy ? "正在自动生成…" : "生成牙冠",
      disabled: wf.busy || fastBusy,
      run: runFastGeneration,
    };
  };

  const action = primaryAction();

  return <div className="cd-page cd-order-page">
    <div className="cd-order-topline">
      <button type="button" className="cd-order-back" disabled={wf.busy || fastBusy} onClick={() => wf.go("home")}>← 返回首页</button>
      <span>病例资料只属于当前病例 · 设计参数会保留为下一病例的偏好</span>
    </div>

    <PageHead eyebrow="NEW CROWN ORDER" title="新建牙冠病例"
      hint="上传病例、选择牙位与修复类型、设置实际参与生成的参数并选择牙模板。所有输入都保留在这一页，不再把参数或牙库藏到下一步。" />

    <div className="cd-order-flowline" aria-label="当前页面包含的输入">
      <span><i>1</i>上传病例</span><b>→</b><span><i>2</i>牙位 / 修复</span><b>→</b><span><i>3</i>生成参数</span><b>→</b><span><i>4</i>牙模板</span>
    </div>

    <Panel title="病例与设计对象" subtitle="Upper / Lower 使用原始相对坐标；快速基台流程要求上传文件已经 Bite 对齐">
      <div className="cd-order-intake-grid">
        <div className="cd-order-scans">
          <div className="cd-order-section-label"><b>上传病例</b><span>{bothJaws ? "2 / 2 已完成" : `${Number(has("upper")) + Number(has("lower"))} / 2 已完成`}</span></div>
          <div className="cd-uploads cd-order-uploads">
            <Upload label="Upper Jaw" mark="U" hint="STL / DCM · 保留原始坐标" name={assetName(assets, "upper")}
              disabled={wf.busy || fastBusy || !wf.caseId} onPick={(file) => wf.upload("upper", file)} />
            <Upload label="Lower Jaw" mark="L" hint="STL / DCM · 保留原始坐标" name={assetName(assets, "lower")}
              disabled={wf.busy || fastBusy || !wf.caseId} onPick={(file) => wf.upload("lower", file)} />
          </div>
        </div>

        <div className="cd-order-meta">
          <div className="cd-order-section-label"><b>设计对象</b><span>{wf.detail?.case?.name || wf.detail?.name || "当前病例"}</span></div>
          <Field label="牙位 FDI" hint={fdi ? "已选择" : "必填"}>
            <select value={fdi} disabled={!!wf.design || wf.busy || fastBusy} onChange={(event) => wf.setFdi(event.target.value)}>
              <option value="">请选择牙位</option>
              {FDI_OPTIONS.map((item) => <option key={item} value={item}>FDI {item}</option>)}
            </select>
          </Field>
          <Field label="修复类型" hint="必填">
            <Segmented value={supportType} disabled={!!wf.design || wf.busy || fastBusy}
              options={[{ value: "abutment", label: "种植基台冠" }, { value: "prep", label: "天然牙冠" }]}
              onChange={wf.setSupportType} />
          </Field>
          <div className="cd-order-type-note">
            {fastAbutment
              ? <><b>自动基台流程</b><span>系统自动建立 Margin、插入轴、邻牙与静态咬合参考，不跳转确认页。</span></>
              : supportType === "prep"
                ? <><b>天然牙完整流程</b><span>当前页收集病例与参数；创建后进入 Margin / Axis 专业复核。</span></>
                : <><b>请选择修复类型</b><span>不同类型会使用不同的后续几何流程。</span></>}
          </div>
        </div>
      </div>
    </Panel>

    <Panel title="模型生成参数" subtitle="只保留当前后端确实参与几何生成的参数；没有实现或当前固定的设置不显示">
      <div className="cd-parameter-toolbar">
        <div><b>标准预设</b><span>新病例沿用你上一次调整的参数；病例资料、牙位和牙模板不会沿用。</span></div>
        <button type="button" className="cd-btn ghost" disabled={wf.busy || fastBusy}
          onClick={() => wf.setParams({ ...DEFAULT_PARAMS })}>恢复标准预设</button>
      </div>
      <div className="cd-model-params">
        {MODEL_PARAMS.map((spec) => <ParamCard key={spec.key} spec={spec} value={wf.params[spec.key]}
          disabled={wf.busy || fastBusy}
          onChange={(value) => wf.setParams((params) => ({ ...params, [spec.key]: value }))} />)}
      </div>
      <div className="cd-order-param-legend">
        <span><b>Cement Gap / 起隙位置</b> 控制内冠空间</span>
        <span><b>Minimum Thickness</b> 控制外冠最小材料空间</span>
        <span><b>Proximal / Occlusal</b> 控制邻接与对颌目标</span>
        <span><b>边缘带 / 加工轴</b> 参与最终边缘过渡与插入方向 Blockout</span>
      </div>
    </Panel>

    <Panel title="牙模板" subtitle="模板只提供牙体解剖身份；最终尺寸与接触由当前病例空间约束决定">
      {!supportType && <Notice>先选择修复类型和牙位。牙模板区域不会消失，满足当前流程前置条件后会在这里直接加载。</Notice>}

      {fastAbutment && !fastPrepared && <div className="cd-template-empty-state">
        <div className="cd-template-empty-icon">◇</div>
        <div><b>牙库将在本页原位加载</b><p>上传上下颌并选择牙位后，点击下方“准备病例并读取牙模板”。系统完成自动参考后不会跳页。</p></div>
      </div>}

      {supportType === "prep" && !wf.design && <div className="cd-template-empty-state">
        <div className="cd-template-empty-icon">◇</div>
        <div><b>天然牙模板入口保留在当前病例页</b><p>天然牙仍需 Margin / Axis 复核后才能确认模板几何；当前参数会直接带入后续生成，不再另设参数页。</p></div>
      </div>}

      {fastAbutment && fastPrepared && ((wf.templates.libraries || []).length > 0
        ? <div className="cd-templates cd-order-templates">
          {wf.templates.libraries.map((library) => {
            const selected = selectedLibrary === library.name;
            const unavailable = library.status === "UNAVAILABLE";
            return <button key={library.name} type="button" className={`cd-template ${selected ? "active" : ""}`}
              disabled={wf.busy || fastBusy || unavailable} onClick={() => chooseLibrary(library.name)}>
              <div className="cd-template-shape"><span>FDI</span><strong>{fdi}</strong></div>
              <b>{library.label || library.name}</b>
              <small>{selected ? "当前模板" : unavailable ? library.reason || "不可用" : "选择此模板"}</small>
            </button>;
          })}
        </div>
        : <div className="cd-template-empty-state"><div className="cd-template-empty-icon">…</div><div><b>正在读取牙库</b><p>如果长时间没有可选模板，请检查当前 FDI 是否配置了对应牙库。</p></div></div>)}

      {wf.selectedTemplate && <div className="cd-selected-template-strip">
        <span>当前模板</span><b>{wf.selectedTemplate.library}</b><small>只作为解剖胚子，生成时会按病例空间变形</small>
      </div>}
    </Panel>

    {!generationOpen && (fastProgress || fastMsg) && <div className="cd-order-status-card">
      {fastProgress && <GenerationProgress value={fastProgress} />}
      {fastMsg && <Notice tone={/失败|未完成|停止/.test(fastMsg) ? "warn" : "ok"}>{fastMsg}</Notice>}
    </div>}

    <div className="cd-order-actionbar">
      <div className="cd-order-readiness">
        <span className={bothJaws ? "ok" : ""}>{bothJaws ? "✓" : "1"} 上下颌</span>
        <span className={fdi ? "ok" : ""}>{fdi ? "✓" : "2"} 牙位</span>
        <span className={supportType ? "ok" : ""}>{supportType ? "✓" : "3"} 修复类型</span>
        {fastAbutment && <span className={wf.selectedTemplate ? "ok" : ""}>{wf.selectedTemplate ? "✓" : "4"} 牙模板</span>}
      </div>
      <div className="cd-order-actioncopy">
        <b>{fastAbutment ? (fastPrepared ? "病例参考已建立" : "准备后仍停留在本页") : "天然牙进入专业复核"}</b>
        <span>{fastAbutment ? "参数已就绪后直接生成" : "参数会随设计保存，不需要再次填写"}</span>
      </div>
      <button type="button" className="cd-btn primary cd-order-primary" disabled={action.disabled} onClick={action.run}>{action.label}</button>
    </div>

    {generationOpen && <GenerationOverlay progress={fastProgress} message={fastMsg} fdi={fdi} library={selectedLibrary} />}
  </div>;
}
