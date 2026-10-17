import React, { useMemo, useState } from "react";
import adapter from "../../domain/adapter";
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
    {conflicts.length > 0 && <small>存在约束冲突：当前候选仍会保留并进入微调画布，但需要人工复核。</small>}
  </div>;
}

export default function SetupPage({ wf }) {
  const [advanced, setAdvanced] = useState(false);
  const [generating, setGenerating] = useState(false);
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

  const generateCompleteCrown = async () => {
    if (generating || wf.busy) return;
    if (!wf.design?.id) { wf.setMsg("请先创建设计"); return; }
    if (!wf.library) { wf.setMsg("请先选择一个牙冠模板"); return; }
    if (eligibility?.template_allowed === false) {
      wf.setMsg((eligibility.blockers || []).map((blocker) => blocker.message).join("；") || "当前输入还不能生成牙冠");
      return;
    }

    setGenerating(true);
    wf.setMsg("正在生成完整牙冠：外冠、邻接、咬合与患者基台内冠会在一次操作中完成…");
    try {
      // 参数和牙库选择只在用户明确点击“牙冠生成”后写入/计算。
      await adapter.saveProfile(wf.design.id, wf.params);

      if (wf.selectedTemplate?.library !== wf.library) {
        const templateCandidate = await adapter.selectCrownTemplate(wf.design.id, wf.library);
        await adapter.confirmTemplateReference(wf.design.id, templateCandidate.id);
      }

      const generated = await adapter.generateInitialCrown(wf.design.id, { kind: "AUTO" });
      // Accept endpoint is the CrownFlow 1.0 transaction boundary: it publishes
      // the solved outer and persists (or regenerates) the patient-owned open
      // intaglio. Do NOT immediately run the legacy manufacturing finalizer here;
      // that second geometry pass can be slow and can replace the review geometry
      // with a stitched/Boolean result. Hollow stitching is deferred to export.
      const accepted = await adapter.confirmCrownCandidate(wf.design.id, generated.id);

      await wf.resumeDesign(wf.design.id, wf.caseId);
      wf.setStep("design");
      wf.reloadMeshes();
      wf.setMsg(accepted?.inner_geometry_ready
        ? "牙冠已生成：外冠、患者基台内冠、邻接和咬合已固定，可直接微调或导出空心 STL。"
        : "外冠已生成并进入微调；内冠会由画布按当前基台继续生成，不会触发第二次外冠求解。");
    } catch (error) {
      wf.setMsg(`牙冠生成失败：${error?.message || "请检查输入后重试"}`);
    } finally {
      setGenerating(false);
    }
  };

  return <div className="cd-page">
    <PageHead eyebrow="STEP 4" title="CAD 1.0 自动牙冠设计"
      hint="先选牙位参数和参考牙库；只有点击右下角“牙冠生成”才开始计算。一次生成同时完成外冠、邻接/咬合适配与患者基台内冠。" />

    <div className="cd-grid setup">
      <Panel title={`参考牙库 · FDI ${wf.fdi || "—"}`} subtitle="点击模板只选择牙型，不会立即开始生成">
        {libraries.length > 0
          ? <div className="cd-templates">
            {libraries.map((library) => <button key={library.name} type="button"
              className={`cd-template ${wf.library === library.name ? "active" : ""}`} disabled={wf.busy || generating}
              onClick={() => wf.setLibrary(library.name)}>
              <div className="cd-template-shape">{wf.fdi || "—"}</div>
              <b>{library.label || library.name}</b>
              <small>{library.name}</small>
            </button>)}
          </div>
          : <Notice>{wf.busy ? "正在读取牙库…" : "当前牙位没有可用牙冠模板，无法开始 CAD 1.0 自动设计。"}</Notice>}

        <div className="cd-btn-row">
          <button type="button" className="cd-btn ghost" disabled={wf.busy || generating || !libraries.length} onClick={wf.loadTemplates}>刷新牙库</button>
        </div>

        {eligibility && !eligibility.template_allowed && <Notice tone="warn">
          {(eligibility.blockers || []).map((blocker) => blocker.message).join("；")}
        </Notice>}
        {eligibility?.automatic_mode && <Notice>
          本次自动设计范围：{{ BOTH: "邻牙 + 对颌", PROXIMAL_ONLY: "仅邻牙形态 / 接触", OCCLUSAL_ONLY: "仅对颌高度 / 咬合面", SPACE_ONLY: "患者空间 + 牙库形态" }[eligibility.automatic_mode] || eligibility.automatic_mode}。
          已成功的 Margin、另一侧邻接和咬合形态会继续作为保护区，不因其它方向微调被重新破坏。
        </Notice>}
        {wf.library && <Notice tone="ok">
          已选择牙库：{wf.library}。当前只记录选择；点击“牙冠生成”后才会完成模板确认和几何求解。
        </Notice>}

        <StageReport candidate={candidate} />
        {candidate?.id && <Notice tone={stageFacts.conflicts || !stageFacts.changed ? "warn" : "ok"}>
          {stageFacts.conflicts ? "自动约束存在提示，但候选仍会保留并显示。"
            : stageFacts.changed ? "候选已经发生邻牙 / 对颌驱动的真实几何形变。"
            : "候选目前接近模板，建议在微调画布继续检查。"}
        </Notice>}

        {!wf.frameConfirmed && <Notice tone="warn">请先确认近中 / 颊侧方向：邻接面形变必须绑定真实牙位坐标帧。</Notice>}
      </Panel>

      <Panel title="CAD 1.0 设计参数" subtitle="这些目标用于约束形变，而不是缩放一颗模板牙">
        {CORE_PARAMS.filter((spec) => !spec.fixed).map((spec) => <ParamRow key={spec.key} spec={spec} value={wf.params[spec.key]}
          disabled={wf.busy || generating} onChange={(value) => wf.setParams((params) => ({ ...params, [spec.key]: value }))} />)}

        <details className="cd-advanced" open={advanced} onToggle={(event) => setAdvanced(event.target.open)}>
          <summary>高级参数</summary>
          {ADVANCED_PARAMS.filter((spec) => !spec.fixed).map((spec) => <ParamRow key={spec.key} spec={spec} value={wf.params[spec.key]}
            disabled={wf.busy || generating} onChange={(value) => wf.setParams((params) => ({ ...params, [spec.key]: value }))} />)}
        </details>

        <div className="cd-btn-row">
          <button type="button" className="cd-btn ghost" disabled={wf.busy || generating} onClick={() => wf.setParams({ ...DEFAULT_PARAMS })}>恢复默认</button>
          <button type="button" className="cd-btn secondary" disabled={wf.busy || generating} onClick={wf.applyParams}>保存参数</button>
        </div>
        {wf.profileConfirmed && <Notice tone="ok">参数已写入设计修订；点击“牙冠生成”会以当前参数重新计算完整牙冠。</Notice>}
      </Panel>
    </div>

    <div className="cd-btn-row end" style={{ justifyContent:"flex-end", marginTop:16 }}>
      <button type="button" className="cd-btn primary"
        disabled={wf.busy || generating || !libraries.length || !wf.library || eligibility?.template_allowed === false}
        onClick={generateCompleteCrown}>
        {generating ? "正在生成完整牙冠…" : "牙冠生成"}
      </button>
    </div>
  </div>;
}
