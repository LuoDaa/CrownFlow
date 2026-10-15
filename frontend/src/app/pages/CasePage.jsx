import React, { useEffect, useRef, useState } from "react";
import { Field, Notice, PageHead, Panel, Segmented } from "../ui";
import { ADVANCED_PARAMS, CORE_PARAMS, DEFAULT_PARAMS } from "../../domain/useWorkflow";
import { generateAbutmentOrder, prepareAbutmentOrder, selectAbutmentTemplate } from "../../domain/abutmentFast";

const FDI_OPTIONS = ["16", "15", "14", "13", "12", "11", "21", "22", "23", "24", "25", "26",
  "36", "35", "34", "33", "32", "31", "41", "42", "43", "44", "45", "46", "37", "47"];

function Upload({ label, mark, hint, name, disabled, onPick }) {
  const input = useRef(null);
  return <label className={`cd-upload ${name ? "done" : ""}`}>
    <input ref={input} type="file" accept=".stl,.STL,.dcm,.DCM" disabled={disabled}
      onChange={(event) => { const file = event.target.files?.[0]; if (file) onPick(file); }} />
    <div className={`cd-upload-mark ${mark === "L" ? "lower" : ""}`}>{mark}</div>
    <b>{label}</b>
    <span>{name || `选择${label}文件`}</span>
    <small>{hint}</small>
  </label>;
}

function ParamRow({ spec, value, disabled, onChange }) {
  const locked = disabled || spec.fixed === true;
  return <div className={`cd-param ${spec.fixed ? "fixed" : ""}`}>
    <span className="cd-param-label"><b>{spec.label}</b><small>{spec.hint}</small></span>
    <div className="cd-param-input">
      <input type="number" min={spec.min} max={spec.max} step={spec.step} disabled={locked}
        value={value ?? (spec.optional ? "" : 0)} placeholder={spec.optional ? "不设置" : ""}
        onChange={(event) => onChange(event.target.value === "" ? null : Number(event.target.value))} />
      <em>{spec.key === "machine_axes" ? "轴" : "mm"}</em>
    </div>
  </div>;
}

export default function CasePage({ wf }) {
  const assets = wf.detail?.assets || [];
  const fdi = wf.design ? String(wf.design.target_fdi) : wf.fdi;
  const supportType = wf.design?.support_type || wf.supportType;
  const has = (kind) => assets.some((asset) => asset.kind === kind);
  const fastAbutment = supportType === "abutment";
  const [fastBusy, setFastBusy] = useState(false);
  const [fastMsg, setFastMsg] = useState("");
  const [advanced, setAdvanced] = useState(false);
  const [templatesRequested, setTemplatesRequested] = useState(false);

  const bothJaws = has("upper") && has("lower");
  const fastPrepared = Boolean(wf.design && wf.marginConfirmed && wf.axisConfirmed && wf.frameConfirmed && wf.serverStatus?.bite?.pose_confirmed);
  const neighbors = wf.serverStatus?.neighbors || {};

  useEffect(() => { setTemplatesRequested(false); }, [wf.design?.id]);
  useEffect(() => {
    if (!fastAbutment || !fastPrepared || templatesRequested || wf.busy || fastBusy) return;
    setTemplatesRequested(true);
    wf.loadTemplates();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [fastAbutment, fastPrepared, wf.design?.id, templatesRequested, wf.busy, fastBusy]);

  const prepareFast = async () => {
    setFastBusy(true); setFastMsg("正在自动识别基台 Margin、插入方向、咬合和邻牙参考…");
    try {
      const result = await prepareAbutmentOrder({ caseId: wf.caseId, fdi, designId: wf.design?.id || null });
      await wf.refreshDetail();
      await wf.resumeDesign(result.design.id, wf.caseId);
      wf.setStep("case");
      const detected = Object.values(result.neighbors?.neighbors || {}).filter((item) => item.confirmation_status === "CONFIRMED").length;
      setFastMsg(`基台自动准备完成：Margin / Axis / Bite 已确认，自动找到 ${detected} 个邻牙接触面。`);
    } catch (error) {
      setFastMsg(`自动准备未完成：${error.message}`);
    } finally { setFastBusy(false); }
  };

  const chooseLibrary = async (name) => {
    setFastBusy(true); setFastMsg(`正在应用牙库 ${name}…`);
    try {
      wf.setLibrary(name);
      await selectAbutmentTemplate({ designId: wf.design.id, library: name, params: wf.params });
      await wf.resumeDesign(wf.design.id, wf.caseId);
      wf.setStep("case");
      setFastMsg(`已选定 ${name} 作为解剖模板。模板只是起始形态，生成时会按邻牙和对颌真实变形。`);
    } catch (error) {
      setFastMsg(`牙库选择失败：${error.message}`);
    } finally { setFastBusy(false); }
  };

  const runFastGeneration = async () => {
    setFastBusy(true); setFastMsg("正在执行 CAD 1.0：邻接形态 → 咬合高度/凹凸 → 内冠 → QC…");
    try {
      await generateAbutmentOrder({ designId: wf.design.id, params: wf.params });
      setFastMsg("基台冠已自动生成并完成内冠/QC，正在打开最终检查。");
      await wf.resumeDesign(wf.design.id, wf.caseId);
    } catch (error) {
      setFastMsg(`自动生成停止：${error.message}`);
    } finally { setFastBusy(false); }
  };

  return <div className="cd-page">
    <PageHead eyebrow={fastAbutment ? "ABUTMENT FAST FLOW" : "STEP 2"}
      title={fastAbutment ? "创建基台冠订单" : "病例信息"}
      hint={fastAbutment
        ? "上传已经 Bite 好的上下颌 STL，选择牙位、牙库和参数后直接生成。Margin、插入方向、静态咬合与邻牙参考由系统自动建立。"
        : "导入上颌与下颌文件，并确认目标牙位与修复类型。上下颌使用病例文件中的原始相对位置。"} />

    <div className="cd-grid two">
      <Panel title={fastAbutment ? "1 · 上传已咬合模型" : "病例文件"} subtitle="Upper / Lower 保留输入相对坐标">
        <div className="cd-uploads">
          <Upload label="Upper Jaw" mark="U" hint="STL / DCM · 已 Bite" name={has("upper") ? "已导入" : ""}
            disabled={wf.busy || fastBusy || !wf.caseId} onPick={(file) => wf.upload("upper", file)} />
          <Upload label="Lower Jaw" mark="L" hint="STL / DCM · 已 Bite" name={has("lower") ? "已导入" : ""}
            disabled={wf.busy || fastBusy || !wf.caseId} onPick={(file) => wf.upload("lower", file)} />
        </div>
        <Notice>{fastAbutment
          ? "快速流程不会重新做 Bite 配准：上传的 Upper / Lower 必须已经处于正确咬合位置。"
          : "Upper / Lower 直接使用病例文件里的相对位置，前端不做配准、不改变两者相对姿态。"}</Notice>
      </Panel>

      <Panel title={fastAbutment ? "2 · 订单信息" : "设计对象"}>
        <Field label="牙位 FDI" hint={fdi ? "" : "必填"}>
          <select value={fdi} disabled={!!wf.design || wf.busy || fastBusy} onChange={(event) => wf.setFdi(event.target.value)}>
            <option value="">请选择</option>
            {FDI_OPTIONS.map((item) => <option key={item} value={item}>{item}</option>)}
          </select>
        </Field>
        <Field label="修复类型">
          <Segmented value={supportType} disabled={!!wf.design || wf.busy || fastBusy}
            options={[{ value: "abutment", label: "种植基台 · 快速" }, { value: "prep", label: "天然牙 · 完整流程" }]}
            onChange={wf.setSupportType} />
        </Field>
        <Field label="订单编号">
          <input type="text" readOnly value={wf.detail?.case?.name || wf.detail?.name || "—"} />
        </Field>

        {fastAbutment ? <>
          {!fastPrepared && <button type="button" className="cd-btn primary full"
            disabled={wf.busy || fastBusy || !bothJaws || !fdi} onClick={prepareFast}>
            {fastBusy ? "正在自动准备…" : wf.design ? "继续自动准备基台" : "创建订单并自动识别基台"}
          </button>}
          {fastPrepared && <Notice tone="ok">
            自动准备完成：Margin、插入方向、PatientFrame、Bite 已确认；近中 {neighbors.mesial?.confirmation_status || "—"} / 远中 {neighbors.distal?.confirmation_status || "—"}。
          </Notice>}
          {fastMsg && <Notice tone={/失败|未完成|停止/.test(fastMsg) ? "warn" : "ok"}>{fastMsg}</Notice>}
        </> : <>
          {wf.design && <Notice tone="ok">设计已创建：FDI {wf.design.target_fdi} · 天然牙预备体</Notice>}
          {!wf.design && <Notice>选择牙位与修复类型后，用底部“创建设计”进入 Margin / Axis 复核。</Notice>}
        </>}
      </Panel>
    </div>

    {fastAbutment && fastPrepared && <div className="cd-grid two">
      <Panel title="3 · 选择牙库" subtitle="牙库只提供解剖初始形态；不是最终患者牙冠">
        {(wf.templates.libraries || []).length > 0 ? <div className="cd-templates">
          {wf.templates.libraries.map((library) => {
            const selected = wf.selectedTemplate?.library === library.name;
            const unavailable = library.status === "UNAVAILABLE";
            return <button key={library.name} type="button" className={`cd-template ${selected ? "active" : ""}`}
              disabled={wf.busy || fastBusy || unavailable} onClick={() => chooseLibrary(library.name)}>
              <div className="cd-template-shape">{fdi}</div>
              <b>{library.label || library.name}</b>
              <small>{selected ? "已选定" : unavailable ? library.reason || "不可用" : "点击选定"}</small>
            </button>;
          })}
        </div> : <Notice>{wf.busy || fastBusy ? "正在读取牙库…" : "当前牙位没有可用模板牙库。"}</Notice>}
        {wf.selectedTemplate && <Notice tone="ok">当前参考牙型：{wf.selectedTemplate.library}</Notice>}
      </Panel>

      <Panel title="4 · 设计参数" subtitle="确认后直接自动生成，不再经过 Margin / Axis 页面">
        {CORE_PARAMS.filter((spec) => !spec.fixed).map((spec) => <ParamRow key={spec.key} spec={spec} value={wf.params[spec.key]}
          disabled={wf.busy || fastBusy} onChange={(value) => wf.setParams((params) => ({ ...params, [spec.key]: value }))} />)}
        <details className="cd-advanced" open={advanced} onToggle={(event) => setAdvanced(event.target.open)}>
          <summary>高级参数</summary>
          {ADVANCED_PARAMS.filter((spec) => !spec.fixed).map((spec) => <ParamRow key={spec.key} spec={spec} value={wf.params[spec.key]}
            disabled={wf.busy || fastBusy} onChange={(value) => wf.setParams((params) => ({ ...params, [spec.key]: value }))} />)}
        </details>
        <div className="cd-btn-row">
          <button type="button" className="cd-btn ghost" disabled={wf.busy || fastBusy}
            onClick={() => wf.setParams({ ...DEFAULT_PARAMS })}>恢复默认</button>
          <button type="button" className="cd-btn primary" disabled={wf.busy || fastBusy || !wf.selectedTemplate}
            onClick={runFastGeneration}>{fastBusy ? "正在自动生成…" : "直接生成牙冠"}</button>
        </div>
        <p className="cd-hint">生成顺序：牙库模板定位 → 左右邻牙邻接形态/接触 → 对颌冠高与咬合面空间 → 基台内冠 → 厚度/碰撞/QC。</p>
      </Panel>
    </div>}
  </div>;
}
