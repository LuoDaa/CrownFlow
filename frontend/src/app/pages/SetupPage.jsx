import React, { useState } from "react";
import { Notice, PageHead, Panel } from "../ui";
import { ADVANCED_PARAMS, CORE_PARAMS, DEFAULT_PARAMS } from "../../domain/useWorkflow";

function ParamRow({ spec, value, disabled, onChange }) {
  // P2-13：后端当前不消费的参数**不给能输入却必然无效的控件**，直接标明固定。
  const locked = disabled || spec.fixed === true;
  return <div className={`cd-param ${spec.fixed ? "fixed" : ""}`}>
    <span className="cd-param-label">
      <b>{spec.label}</b>
      <small>{spec.fixed ? "当前版本固定，不参与内冠链" : spec.hint}</small>
    </span>
    <div className="cd-param-input">
      <input type="number" min={spec.min} max={spec.max} step={spec.step} disabled={locked}
        value={value ?? (spec.optional ? "" : 0)} placeholder={spec.optional ? "不设置" : ""}
        onChange={(event) => onChange(event.target.value === "" ? null : Number(event.target.value))} />
      <em>{spec.fixed ? "固定" : "mm"}</em>
    </div>
  </div>;
}

export default function SetupPage({ wf }) {
  const [advanced, setAdvanced] = useState(false);
  const libraries = wf.templates.libraries || [];
  const eligibility = wf.templates.eligibility;

  return <div className="cd-page">
    <PageHead eyebrow="STEP 4" title="参数与牙冠模板"
      hint="选定参考牙型并确认设计参数，再根据已确认的对颌与邻牙自动调整冠形。" />

    <div className="cd-grid setup">
      <Panel title={`牙冠模板 · FDI ${wf.fdi || "—"}`} subtitle="按牙位从牙库筛选">
        {libraries.length > 0
          ? <div className="cd-templates">
            {libraries.map((library) => <button key={library.name} type="button"
              className={`cd-template ${wf.library === library.name ? "active" : ""}`} disabled={wf.busy}
              onClick={() => { wf.setLibrary(library.name); wf.previewTemplate(library.name); }}>
              <div className="cd-template-shape">{wf.fdi || "—"}</div>
              <b>{library.label || library.name}</b>
              <small>{library.name}</small>
            </button>)}
          </div>
          : <Notice>{wf.busy ? "正在读取牙库…" : "当前牙位没有可用牙冠模板，无法开始自动设计。"}</Notice>}

        <div className="cd-btn-row">
          <button type="button" className="cd-btn ghost" disabled={wf.busy || !libraries.length} onClick={wf.loadTemplates}>刷新牙库</button>
          <button type="button" className="cd-btn secondary" disabled={wf.busy || !wf.templateCandidate?.id} onClick={wf.acceptTemplate}>
            选定参考牙型
          </button>
          <button type="button" className="cd-btn primary" disabled={wf.busy || !eligibility?.automatic_mode || !wf.selectedTemplate || !!wf.templateCandidate} onClick={() => wf.generateCrown("AUTO")}>
            自动生成牙冠
          </button>
        </div>
        {eligibility && !eligibility.template_allowed && <Notice tone="warn">
          {(eligibility.blockers || []).map((blocker) => blocker.message).join("；")}
        </Notice>}
        {eligibility?.automatic_mode && <Notice>
          自动适配模式：{{ BOTH: "近远中 + 咬合", PROXIMAL_ONLY: "仅近远中", OCCLUSAL_ONLY: "仅咬合" }[eligibility.automatic_mode]}
        </Notice>}
        {wf.templateCandidate?.id && <Notice tone="ok">
          已生成参考牙型预览（<b>{wf.templateCandidate.id.slice(0, 8)}</b>）。检查后点“选定参考牙型”，再运行自动设计。
        </Notice>}
        {wf.selectedTemplate && !wf.templateCandidate && <Notice tone="ok">已选定牙库：{wf.selectedTemplate.library}。模板只提供起始解剖形态。</Notice>}
        {wf.generatedCandidate?.id && <Notice tone={wf.generatedCandidate.payload?.status === "PASS_ENGINEERING" ? "ok" : "warn"}>
          已生成{wf.generatedCandidate.payload?.status === "PASS_ENGINEERING" ? "自动" : "粗糙"}候选（<b>{wf.generatedCandidate.id.slice(0, 8)}</b>）。
          <button type="button" className="cd-link" disabled={wf.busy} onClick={wf.acceptGenerated}>采纳候选</button>
        </Notice>}
        {!wf.frameConfirmed && <Notice tone="warn">请先在第 3 页确认近中/颊侧方向：模板就位必须绑定已确认坐标帧。</Notice>}
        {wf.selectedTemplate && eligibility?.template_allowed && !eligibility?.automatic_mode && <Notice tone="warn">尚无可执行的自动适配。请确认咬合位姿或真实邻牙，并设置相应目标间隙。</Notice>}
      </Panel>

      <Panel title="设计参数" subtitle="核心参数（其余见高级）">
        {CORE_PARAMS.filter((spec) => !spec.fixed).map((spec) => <ParamRow key={spec.key} spec={spec} value={wf.params[spec.key]}
          disabled={wf.busy} onChange={(value) => wf.setParams((params) => ({ ...params, [spec.key]: value }))} />)}

        <details className="cd-advanced" open={advanced} onToggle={(event) => setAdvanced(event.target.open)}>
          <summary>高级参数</summary>
          {ADVANCED_PARAMS.filter((spec) => !spec.fixed).map((spec) => <ParamRow key={spec.key} spec={spec} value={wf.params[spec.key]}
            disabled={wf.busy} onChange={(value) => wf.setParams((params) => ({ ...params, [spec.key]: value }))} />)}
        </details>

        <div className="cd-btn-row">
          <button type="button" className="cd-btn ghost" disabled={wf.busy} onClick={() => wf.setParams({ ...DEFAULT_PARAMS })}>恢复默认</button>
          <button type="button" className="cd-btn primary" disabled={wf.busy} onClick={wf.applyParams}>应用参数</button>
        </div>
        {wf.profileConfirmed && <Notice tone="ok">参数已写入设计修订，后续塑形都会引用这一版。</Notice>}
      </Panel>
    </div>
  </div>;
}
