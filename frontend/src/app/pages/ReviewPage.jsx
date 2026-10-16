import React, { useEffect, useState } from "react";
import adapter from "../../domain/adapter";
import { Notice, PageHead, Panel, StatusPill } from "../ui";
import ViewerStage from "../ViewerStage";

/** 每项检查只用真实数据判读；数据缺失/无符号/无效样本一律"未评估"，绝不默认 PASS。 */
function verdict(state, ok, note) {
  if (state !== "ASSESSED") {
    const label = { NOT_ASSESSED: "未评估", UNSIGNED_ONLY: "无符号（未评估）", INVALID: "样本无效（未评估）" }[state] || "未评估";
    return { label, tone: "warn", note: note || "缺少可判读的测量数据" };
  }
  return { label: ok ? "PASS" : "Review", tone: ok ? "ok" : "warn", note };
}

function minOf(payload) {
  if (!payload || !Array.isArray(payload.signed_mm) || !payload.signed_mm.length) return null;
  return Math.min(...payload.signed_mm);
}

export default function ReviewPage({ wf }) {
  const [measurements, setMeasurements] = useState({});
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (!wf.design?.id || !wf.hasOuter) return;
    let alive = true;
    setLoading(true);
    Promise.all([
      adapter.getContactMap(wf.design.id, "neighbors").catch(() => null),
      adapter.getContactMap(wf.design.id, "antagonist").catch(() => null),
      adapter.validateCrown(wf.design.id).catch(() => null),
    ]).then(([neighbors, antagonist, validation]) => {
      if (alive) setMeasurements({ neighbors, antagonist, validation });
    }).finally(() => { if (alive) setLoading(false); });
    return () => { alive = false; };
  }, [wf.design?.id, wf.hasOuter, wf.meshVersion]);

  const neighbors = measurements.neighbors;
  const antagonist = measurements.antagonist;
  const finalQc = measurements.validation?.finalQc || adapter.normalizeFinalQc(wf.qc);
  const proximalState = adapter.measurementState(neighbors);
  const occlusalState = adapter.measurementState(antagonist);
  const proximal = minOf(neighbors);
  const occlusal = minOf(antagonist);
  const thickness = finalQc?.thickness || { state: "NOT_ASSESSED" };
  const thicknessOk = thickness.state === "ASSESSED" && thickness.min_mm != null
    && thickness.min_mm >= (thickness.configured_minimum_mm ?? wf.params.minimum_thickness_mm ?? 0.8);

  const rows = [
    { name: "Margin", state: wf.marginConfirmed && wf.axisConfirmed ? "ASSESSED" : "NOT_ASSESSED",
      ok: true, note: wf.marginConfirmed ? "已确认边缘线与插入轴" : "尚未确认边缘线" },
    { name: "Minimum Thickness", state: thickness.state, ok: thicknessOk,
      note: thickness.state === "ASSESSED"
        ? `最薄 ${thickness.min_mm?.toFixed(3)} mm（配置下限 ${thickness.configured_minimum_mm ?? "—"} mm）`
        : "尚无最终厚度审计" },
    { name: "Proximal Contact", state: proximalState, ok: proximal != null && proximal >= -0.001,
      note: proximal != null ? `最小有符号距离 ${proximal.toFixed(3)} mm` : "缺少邻牙参考" },
    { name: "Occlusion", state: occlusalState, ok: occlusal != null && occlusal >= -0.001,
      note: occlusal != null ? `最小有符号距离 ${occlusal.toFixed(3)} mm` : "缺少对颌参考" },
    { name: "Major Collision", state: occlusalState, ok: occlusal == null || occlusal >= -0.2,
      note: occlusal != null && occlusal < -0.2 ? "存在明显穿插，请返回微调处理" : "未见明显穿插" },
    { name: "Inner Crown", state: wf.innerCrown ? "ASSESSED" : "NOT_ASSESSED", ok: Boolean(wf.innerCrown),
      note: wf.innerCrown ? "已在自动生成阶段完成内冠" : "完整牙冠未包含内冠，请返回自动生成" },
  ];
  const evaluatedRows = rows.map((row) => ({ ...row, ...verdict(row.state, row.ok, row.note) }));
  const blocking = evaluatedRows.filter((row) => row.state === "ASSESSED" && row.tone !== "ok" && row.name !== "Inner Crown");

  const acceptance = measurements.validation?.qc?.final_acceptance || wf.qc?.final_acceptance || null;
  const designReady = acceptance?.status === "DESIGN_READY";

  return <div className="cd-page">
    <PageHead eyebrow="FINAL" title="检查与导出" hint="自动生成阶段已经完成外冠、内冠和颈缘闭合；这里仅检查并导出。" />

    <div className="cd-grid finalize">
      <Panel title="Final Crown Preview" subtitle={`FDI ${wf.fdi || "—"} · ${wf.supportType === "abutment" ? "种植基台" : "天然牙预备体"}`}>
        <ViewerStage wf={wf} mode="final" renderUpper={false} renderLower={false} renderSupport={false}
          showMargin={!wf.finalCrown} />
        <div className="cd-btn-row end">
          <button type="button" className="cd-btn secondary" disabled={wf.busy} onClick={() => wf.setStep("design")}>返回微调</button>
          <button type="button" className="cd-btn primary" disabled={!wf.hasFinal}
            onClick={() => window.open(adapter.finalCrownUrl(wf.design.id), "_blank")}>导出 Final Crown STL</button>
          <button type="button" className="cd-btn ghost" disabled={!wf.hasFinal}
            onClick={() => window.open(adapter.exportPackageUrl(wf.design.id), "_blank")}>导出交付包</button>
        </div>
        {!wf.hasFinal && <Notice tone="warn">当前完整冠已失效或尚未生成。请返回微调页；离开微调时系统会自动同步闭合冠，不再单独执行“生成内冠”步骤。</Notice>}
      </Panel>

      <aside className="cd-rail">
        <Panel title="自动检查" subtitle="逐项来自真实测量结果">
          {evaluatedRows.map((row) => <div key={row.name} className="cd-qc">
            <div><span>{row.name}</span><small>{row.note}</small></div>
            <b className={row.tone}>{row.label}</b>
          </div>)}
        </Panel>
        {designReady && <Notice tone="ok">
          <b>设计完成</b> · 完整冠已在自动设计 / 微调同步阶段构建
          {(acceptance.warnings || []).length > 0 && <span>（{acceptance.warnings.join("；")}）</span>}
          <br /><small>本结果用于设计确认，不用于制造放行（manufacturing_check: false）。</small>
        </Notice>}
        {loading && <Notice>正在读取接触与咬合测量…</Notice>}
        {blocking.length > 0
          ? <Notice tone="warn">以下项目需要复核：{blocking.map((row) => row.name).join("、")}。可返回微调继续处理。</Notice>
          : <Notice tone="ok">所有可用检查项均已通过。导出仅代表工程口径完成，未做制造放行。</Notice>}
        <StatusPill state={wf.hasFinal ? "ok" : ""}>{wf.hasFinal ? "完整牙冠可导出" : "完整牙冠未就绪"}</StatusPill>
      </aside>
    </div>
  </div>;
}
