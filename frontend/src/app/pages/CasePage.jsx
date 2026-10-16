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
        value={value ?? ""} placeholder={String(preset ?? "")}
        onChange={(event) => onChange(event.target.value === "" ? preset : Number(event.target.value))} />
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
        <span className={percent >= 5 ? "on" : ""}>Margin / Axis</span>
        <i />
        <span className={percent >= 25 ? "on" : ""}>空间适配</span>
        <i />
        <span className={percent >= 45 ? "on" : ""}>邻接</span>
        <i />
        <span className={percent >= 60 ? "on" : ""}>咬合</span>
        <i />
        <span className={percent >= 80 ? "on" : ""}>内冠 / Final</span>
      </div>
      <small className="cd-generation-overlay-note">Margin、插入方向、邻牙和咬合参考都已封装在自动流程中，不需要另外确认。</small>
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
  const [autoPrepareError, setAutoPrepareError] = useState("");
  const autoPrepareKey = useRef("");
  const latestParams = useRef(wf.params);

  const bothJaws = has("upper") && has("lower");
  const fastPrepared = Boolean(wf.design && wf.marginConfirmed && wf.axisConfirmed && wf.frameConfirmed && wf.serverStatus?.bite?.pose_confirmed);
  const neighbors = wf.serverStatus?.neighbors || {};
  const selectedLibrary = wf.selectedTemplate?.library || wf.library || "";
  const orderReady = Boolean(bothJaws && fdi && supportType);

  useEffect(() => { latestParams.current = wf.params; }, [wf.params]);

  useEffect(() => {
    setFastBusy(false);
    setFastMsg("");
    setFastProgress(null);
    setTemplatesRequested(false);
    setGenerationOpen(false);
    setAutoPrepareError("");
    autoPrepareKey.current = "";
    wf.setLibrary("");
    // 病例资料/牙位/模板不跨病例；参数继续作为用户偏好保留。
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

  const prepareFast = async ({ retry = false } = {}) => {
    if (!orderReady || !fastAbutment || fastBusy || generationOpen) return;
    if (retry) autoPrepareKey.current = "";
    const key = `${wf.caseId}:${fdi}:${supportType}`;
    if (!retry && autoPrepareKey.current === key) return;
    autoPrepareKey.current = key;
    setAutoPrepareError("");
    setFastBusy(true);
    setFastProgress({ id:"AUTO_PREP", label:"正在后台识别病例空间", progress:5, detail:"Margin / 插入方向 / 邻牙 / Bite 自动完成" });
    setFastMsg("正在后台自动识别，不需要另外点击 Margin 或插入轴；你可以继续调整参数。");
    try {
      const startParams = { ...latestParams.current };
      const result = await prepareAbutmentOrder({
        caseId: wf.caseId,
        fdi,
        designId: wf.design?.id || null,
        params: startParams,
        onProgress,
      });
      await wf.refreshDetail();
      // resumeDesign 会读回服务端参数；把准备期间用户继续调整的最新值重新放回页面，避免“参数又变回去”。
      const currentParams = { ...latestParams.current };
      await wf.resumeDesign(result.design.id, wf.caseId);
      wf.setParams(currentParams);
      wf.setStep("case");
      const detected = Object.values(result.neighbors?.neighbors || {}).filter((item) => item.confirmation_status === "CONFIRMED").length;
      setFastProgress({ id:"READY", label:"病例空间已自动识别", progress:100, detail:`Margin / Axis / Bite 完成 · ${detected} 个邻牙参考面` });
      setFastMsg("病例空间参考已建立。现在只需确认参数并选择牙模板；选择模板后会自动生成牙冠。");
    } catch (error) {
      setAutoPrepareError(error.message || "自动识别失败");
      setFastMsg(`自动识别未完成：${error.message}`);
    } finally {
      setFastBusy(false);
    }
  };

  useEffect(() => {
    if (!fastAbutment || !orderReady || fastPrepared || wf.busy || fastBusy || generationOpen || autoPrepareError) return;
    void prepareFast();
    // 自动准备只由病例输入状态驱动，不把函数 identity 放进依赖，避免重复触发。
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [fastAbutment, orderReady, fastPrepared, wf.busy, fastBusy, generationOpen, autoPrepareError, wf.caseId, fdi, supportType]);

  const chooseLibrary = async (name) => {
    if (!wf.design?.id || generationOpen || fastBusy) return;
    const paramsForGeneration = { ...latestParams.current };
    setGenerationOpen(true);
    setFastBusy(true);
    setFastProgress({ id:"TEMPLATE", label:`选择牙模板 ${name}`, progress:2, detail:"模板仅作为解剖胚子" });
    setFastMsg(`正在应用牙库 ${name}，随后直接自动生成牙冠…`);
    try {
      wf.setLibrary(name);
      await selectAbutmentTemplate({ designId: wf.design.id, library: name, params: paramsForGeneration, onProgress });
      setFastProgress({ id:"START", label:"模板已选定，开始自动生成", progress:4, detail:"Margin / 左右邻牙 / 上下咬合联合求解" });
      const result = await generateAbutmentOrder({ designId: wf.design.id, params: paramsForGeneration, onProgress });
      await wf.resumeDesign(wf.design.id, wf.caseId);
      if (result?.final) {
        setFastMsg(result.warnings?.length
          ? `牙冠已生成；有 ${result.warnings.length} 项工程提示，可在 3D 中检查。`
          : "基台冠已自动生成并完成内冠，正在打开最终检查。");
      } else {
        setFastMsg(`患者化外冠已生成并进入 3D。${result?.hardFailures?.length ? `仍有 ${result.hardFailures.length} 项空间约束需要检查。` : ""}`);
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
    if (!orderReady) return { label:"上传上下颌并选择牙位后自动继续", disabled:true };
    if (!fastAbutment) return {
      label: wf.design ? "继续天然牙完整流程" : "创建天然牙设计并进入 Margin / Axis",
      disabled: wf.busy || fastBusy,
      run: () => (wf.design ? wf.go("margin") : wf.prepare()),
    };
    if (autoPrepareError) return {
      label:"重试自动识别",
      disabled: wf.busy || fastBusy,
      run: () => prepareFast({ retry:true }),
    };
    if (!fastPrepared) return { label:"正在自动识别 Margin / Axis / 邻牙 / Bite…", disabled:true };
    if (!(wf.templates.libraries || []).length) return { label:"正在读取牙模板…", disabled:true };
    return { label:"选择牙模板后自动生成", disabled:true };
  };

  const action = primaryAction();

  return <div className="cd-page cd-order-page">
    <div className="cd-order-topline">
      <button type="button" className="cd-order-back" disabled={wf.busy || generationOpen} onClick={() => wf.go("home")}>← 返回首页</button>
      <span>病例资料不跨病例 · 参数作为个人偏好保留</span>
    </div>

    <PageHead eyebrow="NEW CROWN ORDER" title="新建牙冠病例"
      hint="上传上下颌、选择牙位与修复类型、微调参数、选择牙模板。Margin、插入方向、邻牙和静态咬合参考由系统在后台自动完成。" />

    <div className="cd-order-flowline" aria-label="当前页面包含的输入">
      <span><i>1</i>上传病例</span><b>→</b><span><i>2</i>牙位 / 修复</span><b>→</b><span><i>3</i>微调参数</span><b>→</b><span><i>4</i>选择模板即生成</span>
    </div>

    <Panel title="病例与设计对象" subtitle="Upper / Lower 保留原始咬合坐标；上传完成后系统会在后台自动准备，不需要额外点 Margin 或 Axis">
      <div className="cd-order-intake-grid">
        <div className="cd-order-scans">
          <div className="cd-order-section-label"><b>上传病例</b><span>{bothJaws ? "2 / 2 已完成" : `${Number(has("upper")) + Number(has("lower"))} / 2 已完成`}</span></div>
          <div className="cd-uploads cd-order-uploads">
            <Upload label="Upper Jaw" mark="U" hint="STL / DCM · 保留原始坐标" name={assetName(assets, "upper")}
              disabled={wf.busy || generationOpen || !wf.caseId} onPick={(file) => wf.upload("upper", file)} />
            <Upload label="Lower Jaw" mark="L" hint="STL / DCM · 保留原始坐标" name={assetName(assets, "lower")}
              disabled={wf.busy || generationOpen || !wf.caseId} onPick={(file) => wf.upload("lower", file)} />
          </div>
        </div>

        <div className="cd-order-meta">
          <div className="cd-order-section-label"><b>设计对象</b><span>{wf.detail?.case?.name || wf.detail?.name || "当前病例"}</span></div>
          <Field label="牙位 FDI" hint={fdi ? "已选择" : "必填"}>
            <select value={fdi} disabled={!!wf.design || wf.busy || generationOpen} onChange={(event) => wf.setFdi(event.target.value)}>
              <option value="">请选择牙位</option>
              {FDI_OPTIONS.map((item) => <option key={item} value={item}>FDI {item}</option>)}
            </select>
          </Field>
          <Field label="修复类型" hint="必填">
            <Segmented value={supportType} disabled={!!wf.design || wf.busy || generationOpen}
              options={[{ value: "abutment", label: "种植基台冠" }, { value: "prep", label: "天然牙冠" }]}
              onChange={wf.setSupportType} />
          </Field>
          <div className="cd-order-type-note">
            {fastAbutment
              ? <><b>自动基台流程</b><span>病例资料齐全后自动寻找 Margin、插入方向、邻牙和 Bite；这些步骤不再要求单独点击。</span></>
              : supportType === "prep"
                ? <><b>天然牙完整流程</b><span>天然牙仍保留专业 Margin / Axis 复核流程。</span></>
                : <><b>请选择修复类型</b><span>不同类型会使用不同的后续几何流程。</span></>}
          </div>
        </div>
      </div>
    </Panel>

    <Panel title="模型生成参数" subtitle="全部都有可直接使用的默认值；只调整你确实想改变的参数，不需要为了“生效”再确认一次">
      <div className="cd-parameter-toolbar">
        <div><b>标准预设已生效</b><span>邻接默认 -0.02 mm 轻压接；咬合默认 0.02 mm 近接触。新病例沿用你上一次调整的参数。</span></div>
        <button type="button" className="cd-btn ghost" disabled={wf.busy || generationOpen}
          onClick={() => wf.setParams({ ...DEFAULT_PARAMS })}>恢复标准预设</button>
      </div>
      <div className="cd-model-params">
        {MODEL_PARAMS.map((spec) => <ParamCard key={spec.key} spec={spec} value={wf.params[spec.key]}
          disabled={wf.busy || generationOpen}
          onChange={(value) => wf.setParams((params) => ({ ...params, [spec.key]: value }))} />)}
      </div>
      <div className="cd-order-param-legend">
        <span><b>Cement Gap / 起隙位置</b> 控制内冠空间</span>
        <span><b>Minimum Thickness</b> 控制外冠最小材料空间</span>
        <span><b>Proximal Contact</b> 0 为接触，负值为轻微压接</span>
        <span><b>Occlusal Clearance</b> 控制静态对颌目标间隙</span>
      </div>
    </Panel>

    <Panel title="牙模板" subtitle="模板只提供牙体解剖身份；选择模板后立即自动生成，不再需要第二次点击">
      {!supportType && <Notice>先完成病例上传、牙位和修复类型。后续准备会在后台自动执行。</Notice>}

      {fastAbutment && !fastPrepared && <div className="cd-template-empty-state">
        <div className="cd-template-empty-icon">◇</div>
        <div><b>{autoPrepareError ? "自动病例准备需要重试" : "系统正在后台准备病例"}</b>
          <p>{autoPrepareError
            ? `原因：${autoPrepareError}`
            : "自动寻找 Margin、插入方向、邻牙和静态咬合参考。你可以继续微调上面的参数，不需要等待后再重新填写。"}</p></div>
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
              disabled={wf.busy || fastBusy || generationOpen || unavailable} onClick={() => chooseLibrary(library.name)}>
              <div className="cd-template-shape"><span>FDI</span><strong>{fdi}</strong></div>
              <b>{library.label || library.name}</b>
              <small>{selected ? "当前模板" : unavailable ? library.reason || "不可用" : "选择并自动生成"}</small>
            </button>;
          })}
        </div>
        : <div className="cd-template-empty-state"><div className="cd-template-empty-icon">…</div><div><b>正在读取牙库</b><p>牙库准备完成后直接在这里选择；选择后立即自动生成。</p></div></div>)}

      {wf.selectedTemplate && <div className="cd-selected-template-strip">
        <span>当前模板</span><b>{wf.selectedTemplate.library}</b><small>只作为解剖胚子，最终尺寸由患者空间决定</small>
      </div>}
    </Panel>

    {!generationOpen && (fastProgress || fastMsg) && <div className="cd-order-status-card">
      {fastProgress && <GenerationProgress value={fastProgress} />}
      {fastMsg && <Notice tone={/失败|未完成|停止|重试/.test(fastMsg) ? "warn" : "ok"}>{fastMsg}</Notice>}
    </div>}

    <div className="cd-order-actionbar">
      <div className="cd-order-readiness">
        <span className={bothJaws ? "ok" : ""}>{bothJaws ? "✓" : "1"} 上下颌</span>
        <span className={fdi ? "ok" : ""}>{fdi ? "✓" : "2"} 牙位</span>
        <span className={supportType ? "ok" : ""}>{supportType ? "✓" : "3"} 修复类型</span>
        {fastAbutment && <span className={fastPrepared ? "ok" : ""}>{fastPrepared ? "✓" : "4"} 自动病例准备</span>}
        {fastAbutment && <span className={wf.selectedTemplate ? "ok" : ""}>{wf.selectedTemplate ? "✓" : "5"} 牙模板</span>}
      </div>
      <div className="cd-order-actioncopy">
        <b>{fastAbutment
          ? autoPrepareError
            ? "自动识别遇到问题"
            : fastPrepared
              ? "选择任一牙模板即可开始生成"
              : "正在后台自动识别病例空间"
          : "天然牙进入专业复核"}</b>
        <span>{fastAbutment
          ? "不需要再点 Margin、Axis、应用参数或生成按钮"
          : "参数会随设计保存，不需要再次填写"}</span>
      </div>
      {!fastAbutment && <button type="button" className="cd-btn primary cd-order-primary" disabled={action.disabled} onClick={action.run}>{action.label}</button>}
      {fastAbutment && autoPrepareError && <button type="button" className="cd-btn primary cd-order-primary" disabled={action.disabled} onClick={action.run}>{action.label}</button>}
    </div>

    {generationOpen && <GenerationOverlay progress={fastProgress} message={fastMsg} fdi={fdi} library={selectedLibrary} />}
  </div>;
}
