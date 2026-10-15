import React, { useEffect, useRef, useState } from "react";
import { Field, Notice, PageHead, Panel, Segmented } from "../ui";
import { ADVANCED_PARAMS, CORE_PARAMS, DEFAULT_PARAMS } from "../../domain/useWorkflow";
import { generateAbutmentOrder, prepareAbutmentOrder, selectAbutmentTemplate } from "../../domain/abutmentFast";

const FDI_OPTIONS = ["16", "15", "14", "13", "12", "11", "21", "22", "23", "24", "25", "26",
  "36", "35", "34", "33", "32", "31", "41", "42", "43", "44", "45", "46", "37", "47"];

function Upload({ label, mark, hint, name, disabled, onPick }) {
  return <label className={`cd-upload automate-upload ${name ? "done" : ""}`}>
    <input type="file" accept=".stl,.STL,.dcm,.DCM" disabled={disabled}
      onChange={(event) => { const file = event.target.files?.[0]; if (file) onPick(file); }} />
    <div className={`cd-upload-mark ${mark === "L" ? "lower" : ""}`}>{mark}</div>
    <b>{label}</b>
    <span>{name || `拖放或点击选择 ${label}`}</span>
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

function NaturalOrder({ wf, fdi, supportType }) {
  const assets = wf.detail?.assets || [];
  const has = (kind) => assets.some((asset) => asset.kind === kind);
  return <div className="cd-page">
    <PageHead eyebrow="NATURAL PREP" title="天然牙订单" hint="天然牙继续使用完整 Margin / Axis 复核流程。" />
    <div className="cd-grid two">
      <Panel title="扫描文件">
        <div className="cd-uploads">
          <Upload label="Upper Jaw" mark="U" hint="STL / DCM" name={has("upper") ? "已导入" : ""}
            disabled={wf.busy || !wf.caseId} onPick={(file) => wf.upload("upper", file)} />
          <Upload label="Lower Jaw" mark="L" hint="STL / DCM" name={has("lower") ? "已导入" : ""}
            disabled={wf.busy || !wf.caseId} onPick={(file) => wf.upload("lower", file)} />
        </div>
      </Panel>
      <Panel title="设计对象">
        <Field label="牙位 FDI"><select value={fdi} disabled={!!wf.design || wf.busy} onChange={(e) => wf.setFdi(e.target.value)}>
          <option value="">请选择</option>{FDI_OPTIONS.map((item) => <option key={item}>{item}</option>)}
        </select></Field>
        <Field label="修复类型"><Segmented value={supportType} disabled={!!wf.design || wf.busy}
          options={[{ value: "abutment", label: "种植基台 · 自动" }, { value: "prep", label: "天然牙 · 完整流程" }]}
          onChange={wf.setSupportType} /></Field>
        <Notice>确认后使用底部“创建设计”进入 Margin / Axis 页面。</Notice>
      </Panel>
    </div>
  </div>;
}

export default function CasePage({ wf }) {
  const assets = wf.detail?.assets || [];
  const has = (kind) => assets.some((asset) => asset.kind === kind);
  const fdi = wf.design ? String(wf.design.target_fdi) : wf.fdi;
  const supportType = wf.design?.support_type || wf.supportType || "abutment";
  const [fastBusy, setFastBusy] = useState(false);
  const [fastMsg, setFastMsg] = useState("");
  const [advanced, setAdvanced] = useState(false);
  const [templatesRequested, setTemplatesRequested] = useState(false);
  const autoAttempt = useRef("");
  const bothJaws = has("upper") && has("lower");
  const fastAbutment = supportType === "abutment";
  const fastPrepared = Boolean(wf.design && wf.marginConfirmed && wf.axisConfirmed && wf.frameConfirmed && wf.serverStatus?.bite?.pose_confirmed);

  useEffect(() => {
    if (!wf.supportType && !wf.design) wf.setSupportType("abutment");
  }, [wf.supportType, wf.design, wf.setSupportType]);

  useEffect(() => { setTemplatesRequested(false); }, [wf.design?.id]);
  useEffect(() => {
    if (!fastAbutment || !fastPrepared || templatesRequested || wf.busy || fastBusy) return;
    setTemplatesRequested(true);
    wf.loadTemplates();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [fastAbutment, fastPrepared, wf.design?.id, templatesRequested, wf.busy, fastBusy]);

  const prepareFast = async (force = false) => {
    const key = `${wf.caseId}:${fdi}:${assets.map((a) => `${a.kind}:${a.id || a.path || "x"}`).join("|")}`;
    if (!force && autoAttempt.current === key) return;
    autoAttempt.current = key;
    setFastBusy(true); setFastMsg("正在验证扫描并自动识别设计输入…");
    try {
      const result = await prepareAbutmentOrder({ caseId: wf.caseId, fdi, designId: wf.design?.id || null });
      await wf.refreshDetail();
      await wf.resumeDesign(result.design.id, wf.caseId);
      wf.setStep("case");
      setFastMsg("订单验证通过，可以选择牙库并开始自动设计。系统已自动完成基台、边缘、方向、邻牙与对颌参考准备。");
    } catch (error) {
      setFastMsg(`订单验证未通过：${error.message}`);
    } finally { setFastBusy(false); }
  };

  useEffect(() => {
    if (!fastAbutment || !bothJaws || !fdi || fastPrepared || wf.busy || fastBusy) return;
    const timer = setTimeout(() => prepareFast(false), 180);
    return () => clearTimeout(timer);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [fastAbutment, bothJaws, fdi, fastPrepared, wf.caseId, assets.length]);

  const chooseLibrary = async (name) => {
    setFastBusy(true); setFastMsg(`正在应用牙库 ${name}…`);
    try {
      wf.setLibrary(name);
      await selectAbutmentTemplate({ designId: wf.design.id, library: name, params: wf.params });
      await wf.resumeDesign(wf.design.id, wf.caseId);
      wf.setStep("case");
      setFastMsg(`已选择 ${name}。生成时会依据邻牙与对颌对模板进行患者化变形。`);
    } catch (error) {
      setFastMsg(`牙库选择失败：${error.message}`);
    } finally { setFastBusy(false); }
  };

  const runFastGeneration = async () => {
    setFastBusy(true); setFastMsg("订单已提交，正在自动设计牙冠…");
    try {
      await generateAbutmentOrder({ designId: wf.design.id, params: wf.params });
      await wf.resumeDesign(wf.design.id, wf.caseId);
      setFastMsg("设计完成，已进入检查页面。");
      wf.setStep("review");
    } catch (error) {
      setFastMsg(`自动设计停止：${error.message}`);
    } finally { setFastBusy(false); }
  };

  if (!fastAbutment) return <NaturalOrder wf={wf} fdi={fdi} supportType={supportType} />;

  const libraries = wf.templates.libraries || [];
  const selectedLibrary = wf.selectedTemplate?.library || "";
  const readyToDesign = fastPrepared && selectedLibrary && !fastBusy && !wf.busy;
  const warning = /失败|未通过|停止|无法|错误/.test(fastMsg);

  return <div className="cd-page automate-upload-page">
    <PageHead eyebrow="UPLOAD" title="上传订单" hint="上传已处于正确 Bite 位置的上下颌扫描，设置牙位和设计首选项；订单验证通过后直接自动设计。" />

    <section className="automate-indication-tabs">
      <button className="active">牙冠</button><button disabled>内冠</button><button disabled>嵌体/高嵌体</button><button disabled>牙桥</button>
    </section>

    <Panel title="1 · 订单信息" subtitle="选择目标牙位和修复类型">
      <div className="automate-form-grid">
        <Field label="牙位 FDI"><select value={fdi} disabled={!!wf.design || wf.busy || fastBusy} onChange={(e) => wf.setFdi(e.target.value)}>
          <option value="">请选择牙位</option>{FDI_OPTIONS.map((item) => <option key={item}>{item}</option>)}
        </select></Field>
        <Field label="修复类型"><Segmented value={supportType} disabled={!!wf.design || wf.busy || fastBusy}
          options={[{ value: "abutment", label: "种植基台 · 自动" }, { value: "prep", label: "天然牙 · 完整流程" }]}
          onChange={wf.setSupportType} /></Field>
      </div>
    </Panel>

    <Panel title="2 · 设计首选项" subtitle="这些值用于本订单；以后可以再做全局默认首选项">
      <div className="automate-preference-grid">
        {CORE_PARAMS.filter((spec) => !spec.fixed).map((spec) => <ParamRow key={spec.key} spec={spec} value={wf.params[spec.key]}
          disabled={wf.busy || fastBusy} onChange={(value) => wf.setParams((params) => ({ ...params, [spec.key]: value }))} />)}
      </div>
      <div className="cd-sep" />
      <div className="automate-library-row">
        <div><b>牙库</b><small>牙库提供解剖初始形态，最终形态由邻牙和对颌约束调整。</small></div>
        <div className="cd-templates automate-templates">
          {libraries.length ? libraries.map((library) => {
            const selected = selectedLibrary === library.name;
            const unavailable = library.status === "UNAVAILABLE";
            return <button key={library.name} type="button" className={`cd-template ${selected ? "active" : ""}`}
              disabled={!fastPrepared || fastBusy || unavailable} onClick={() => chooseLibrary(library.name)}>
              <b>{library.label || library.name}</b><small>{selected ? "已选" : unavailable ? "不可用" : "选择"}</small>
            </button>;
          }) : <span className="cd-hint">{fastPrepared ? "正在读取可用牙库…" : "扫描验证后自动加载牙库"}</span>}
        </div>
      </div>
      <details className="cd-advanced" open={advanced} onToggle={(event) => setAdvanced(event.target.open)}>
        <summary>更多设计参数</summary>
        <div className="automate-preference-grid">
          {ADVANCED_PARAMS.filter((spec) => !spec.fixed).map((spec) => <ParamRow key={spec.key} spec={spec} value={wf.params[spec.key]}
            disabled={wf.busy || fastBusy} onChange={(value) => wf.setParams((params) => ({ ...params, [spec.key]: value }))} />)}
        </div>
      </details>
    </Panel>

    <Panel title="3 · 上传扫描" subtitle="Upper / Lower 必须已经处于正确静态咬合关系；CrownFlow 不会重新移动两者">
      <div className="cd-uploads automate-upload-grid">
        <Upload label="Upper Jaw" mark="U" hint="STL / DCM · 已 Bite" name={has("upper") ? "已上传" : ""}
          disabled={wf.busy || fastBusy || !wf.caseId} onPick={(file) => wf.upload("upper", file)} />
        <Upload label="Lower Jaw" mark="L" hint="STL / DCM · 已 Bite" name={has("lower") ? "已上传" : ""}
          disabled={wf.busy || fastBusy || !wf.caseId} onPick={(file) => wf.upload("lower", file)} />
      </div>
      <div className={`automate-validation ${fastPrepared ? "ok" : warning ? "warn" : ""}`}>
        <b>{fastPrepared ? "✓ 订单验证通过" : fastBusy ? "正在验证订单…" : bothJaws && fdi ? "等待自动验证" : "请上传上下颌并选择牙位"}</b>
        <span>{fastPrepared ? "必要扫描和自动设计输入已准备完成。" : fastMsg || "文件上传完成后自动验证，不需要手动确认 Margin、插入方向、邻牙或对颌。"}</span>
        {warning && <button className="cd-btn ghost" disabled={fastBusy} onClick={() => prepareFast(true)}>重新验证</button>}
      </div>
    </Panel>

    <section className="automate-submit-bar">
      <div>
        <b>{readyToDesign ? "订单已准备好" : "完成上方订单信息"}</b>
        <span>{readyToDesign ? `FDI ${fdi} · ${selectedLibrary} · 自动牙冠设计` : "上传扫描、选择牙位并等待验证，再选择牙库。"}</span>
      </div>
      <button type="button" className="cd-btn primary automate-submit" disabled={!readyToDesign} onClick={runFastGeneration}>
        {fastBusy ? "正在处理…" : "开始自动设计"}
      </button>
    </section>

    {fastMsg && !warning && <Notice tone={fastPrepared ? "ok" : undefined}>{fastMsg}</Notice>}
    <button type="button" className="cd-link-reset" disabled={wf.busy || fastBusy} onClick={() => wf.setParams({ ...DEFAULT_PARAMS })}>恢复默认设计首选项</button>
  </div>;
}
