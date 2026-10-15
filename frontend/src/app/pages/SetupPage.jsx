import React, { useMemo, useState } from "react";
import { Notice, PageHead, Panel, StatusPill } from "../ui";
import { ADVANCED_PARAMS, CORE_PARAMS, DEFAULT_PARAMS } from "../../domain/useWorkflow";

function ParamRow({ spec, value, disabled, onChange }) {
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

const STAGE_LABEL = { TEMPLATE:"牙库初始形态", PROXIMAL:"邻牙形态 / 接触", OCCLUSAL:"对颌高度 / 咬合面", RECHECK:"联合复测" };

function stageChanged(stage) {
  if (!stage) return false;
  if ((stage.trials || []).some((trial) => trial.accepted && Number(trial.maximum_displacement_mm || 0) > 1e-4)) return true;
  return (stage.passes || []).some((pass) => pass.status === "PASS_ENGINEERING" && Number(pass.seed_count || 0) > 0);
}

function stageMaxMove(stage) {
  return Math.max(0, ...(stage?.trials || []).filter((trial) => trial.accepted)
    .map((trial) => Number(trial.maximum_displacement_mm || 0)));
}

function StageReport({ candidate }) {
  const payload = candidate?.payload || {};
  const stages = (payload.stages || []).filter((stage) => ["PROXIMAL", "OCCLUSAL", "RECHECK"].includes(stage.stage));
  if (!candidate?.id || !stages.length) return null;

  const changed = stages.some(stageChanged);
  const assessed = stages.filter((stage) => stage.status && !["NOT_ASSESSED", "INPUT_UNVERIFIED", "SKIPPED"].includes(stage.status));
  const conflicts = stages.filter((stage) => ["CONSTRAINT_CONFLICT", "BLOCKED"].includes(stage.status));

  return <div className="cd-edit-state ready" style={{ marginTop: 12 }}>
    <b>CAD 1.0 自动设计执行结果</b>
    <span>{changed ? "已发生真实网格形变" : "未检测到有效网格形变"} · 候选 {candidate.id.slice(0, 8)}</span>
    <div style={{ display:"grid", gap:6, marginTop:8 }}>
      {stages.map((stage, index) => {
        const moved = stageMaxMove(stage);
        const didChange = stageChanged(stage);
        return <div key={`${stage.stage}-${index}`} className="cd-qc">
          <div>
            <span>{STAGE_LABEL[stage.stage] || stage.stage}</span>
            <small>{didChange ? `已形变${moved > 0 ? ` · 本轮最大 ${moved.toFixed(3)} mm` : ""}`
              : stage.reason === "NO_CORRECTION_NEEDED" ? "测量已满足目标，无需变形"
              : stage.reason || "未产生有效变形"}</small>
          </div>
          <StatusPill state={stage.status === "PASS_ENGINEERING" ? "ok" : ""}>{stage.status || "—"}</StatusPill>
        </div>;
      })}
    </div>
    {!changed && assessed.length > 0 && conflicts.length === 0 && <small>
      本候选没有移动网格。只有当各约束原本已满足目标时这才是正常结果；否则不要把“模板已放到基台上”当作自动设计完成。
    </small>}
    {conflicts.length > 0 && <small>存在约束冲突：当前候选只可用于诊断，不应当作完成的自动牙冠。</small>}
  </div>;
}

export default function SetupPage({ wf }) {
  const [advanced, setAdvanced] = useState(false);
  const libraries = wf.templates.libraries || [];
  const eligibility = wf.templates.eligibility;
  const candidate = wf.generatedCandidate;
  const stageFacts = useMemo(() => {
    const stages = candidate?.payload?.stages || [];
    const required = stages.filter((stage) => ["PROXIMAL", "OCCLUSAL"].includes(stage.stage));
    return {
      changed: required.some(stageChanged),
      conflicts: required.some((stage) => ["CONSTRAINT_CONFLICT", "BLOCKED", "INPUT_UNVERIFIED"].includes(stage.status)),
    };
  }, [candidate]);

  return <div className="cd-page">
    <PageHead eyebrow="STEP 4" title="CAD 1.0 自动牙冠设计"
      hint="牙库只提供初始解剖形态；邻牙控制近远中形态与接触，对颌控制冠高与咬合面空间，基台在最终阶段生成内冠。" />

    <div className="cd-grid setup">
      <Panel title={`参考牙库 · FDI ${wf.fdi || "—"}`} subtitle="模板不是结果，只是可变形的解剖先验">
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
          : <Notice>{wf.busy ? "正在读取牙库…" : "当前牙位没有可用牙冠模板，无法开始 CAD 1.0 自动设计。"}</Notice>}

        <div className="cd-btn-row">
          <button type="button" className="cd-btn ghost" disabled={wf.busy || !libraries.length} onClick={wf.loadTemplates}>刷新牙库</button>
          <button type="button" className="cd-btn secondary" disabled={wf.busy || !wf.templateCandidate?.id} onClick={wf.acceptTemplate}>
            选定参考牙型
          </button>
          <button type="button" className="cd-btn primary"
            disabled={wf.busy || !eligibility?.automatic_mode || !wf.selectedTemplate || !!wf.templateCandidate}
            onClick={() => wf.generateCrown("AUTO")}>
            运行 CAD 1.0 自动设计
          </button>
        </div>

        {eligibility && !eligibility.template_allowed && <Notice tone="warn">
          {(eligibility.blockers || []).map((blocker) => blocker.message).join("；")}
        </Notice>}
        {eligibility?.automatic_mode && <Notice>
          本次自动设计范围：{{ BOTH: "邻牙 + 对颌", PROXIMAL_ONLY: "仅邻牙形态 / 接触", OCCLUSAL_ONLY: "仅对颌高度 / 咬合面" }[eligibility.automatic_mode]}。
          算法会保留牙库牙尖、窝沟与轴面解剖，只对受约束区域做局部几何形变。
        </Notice>}
        {wf.templateCandidate?.id && <Notice tone="ok">
          已生成参考牙型就位预览（<b>{wf.templateCandidate.id.slice(0, 8)}</b>）。这里只检查牙位、方向和基础尺寸；它还不是自动设计结果。
        </Notice>}
        {wf.selectedTemplate && !wf.templateCandidate && <Notice tone="ok">
          已选定牙库：{wf.selectedTemplate.library}。下一步会真正读取邻牙和对颌几何并改变模板网格。
        </Notice>}

        <StageReport candidate={candidate} />
        {candidate?.id && <Notice tone={stageFacts.conflicts || !stageFacts.changed ? "warn" : "ok"}>
          {stageFacts.conflicts ? "自动约束没有完整求解，请先检查执行结果，不建议直接采纳。"
            : stageFacts.changed ? "候选已经发生邻牙 / 对颌驱动的真实几何形变，请在 3D 中复核后采纳。"
            : "候选目前看起来仍接近模板。只有执行结果明确显示各目标原本已满足时才建议采纳。"}
          <button type="button" className="cd-link" disabled={wf.busy} onClick={wf.acceptGenerated}>采纳候选</button>
        </Notice>}

        {!wf.frameConfirmed && <Notice tone="warn">请先确认近中 / 颊侧方向：邻接面形变必须绑定真实牙位坐标帧。</Notice>}
        {wf.selectedTemplate && eligibility?.template_allowed && !eligibility?.automatic_mode && <Notice tone="warn">
          尚无可执行的自动设计。至少需要确认真实邻牙或静态咬合位姿，并设置相应目标距离。
        </Notice>}
      </Panel>

      <Panel title="CAD 1.0 设计参数" subtitle="这些目标用于约束形变，而不是缩放一颗模板牙">
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
        {wf.profileConfirmed && <Notice tone="ok">参数已写入设计修订，自动设计和最终内冠都会引用这一版。</Notice>}
      </Panel>
    </div>
  </div>;
}
