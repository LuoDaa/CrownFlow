import React, { useEffect, useState } from "react";
import adapter from "../../domain/adapter";
import { Notice, PageHead, Panel, StatusPill } from "../ui";
import ViewerStage from "../ViewerStage";

function minOf(payload) {
  if (!payload || !Array.isArray(payload.signed_mm) || !payload.signed_mm.length) return null;
  return Math.min(...payload.signed_mm);
}

export default function ReviewPage({ wf }) {
  const [measurements, setMeasurements] = useState({});
  const [loading, setLoading] = useState(false);
  const fastAbutment = wf.design?.support_type === "abutment";

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
  const proximal = minOf(neighbors);
  const occlusal = minOf(antagonist);
  const thickness = finalQc?.thickness || { state: "NOT_ASSESSED" };
  const thicknessOk = thickness.state === "ASSESSED" && thickness.min_mm != null
    && thickness.min_mm >= (thickness.configured_minimum_mm ?? wf.params.minimum_thickness_mm ?? 0.8);

  const checks = [
    { name: "边缘", ok: wf.marginConfirmed && wf.axisConfirmed, value: wf.marginConfirmed ? "已建立" : "未评估" },
    { name: "邻接", ok: proximal != null && proximal >= -0.001, value: proximal != null ? `${proximal.toFixed(3)} mm` : "未评估" },
    { name: "咬合", ok: occlusal != null && occlusal >= -0.001, value: occlusal != null ? `${occlusal.toFixed(3)} mm` : "未评估" },
    { name: "最小厚度", ok: thicknessOk, value: thickness.state === "ASSESSED" ? `${thickness.min_mm?.toFixed(3)} mm` : "未评估" },
    { name: "内冠", ok: !!wf.innerCrown || wf.hasFinal, value: wf.innerCrown || wf.hasFinal ? "已生成" : "未生成" },
  ];
  const needsReview = checks.filter((item) => !item.ok);

  return <div className="cd-page automate-review-page">
    <PageHead eyebrow={fastAbutment ? "REVIEW" : "STEP 6"} title="检查设计"
      hint={fastAbutment ? "检查自动生成结果；满意可直接下载 STL，需要时再进入 3D 编辑器。" : "确认设计质量并输出最终牙冠。"} />

    <div className="cd-grid finalize">
      <Panel title="设计预览" subtitle={`FDI ${wf.fdi || "—"} · ${fastAbutment ? "自动基台冠" : "天然牙预备体"}`}>
        <ViewerStage wf={wf} mode={wf.hasFinal ? "final" : "outer"} renderSupport={false} showMargin={!wf.hasFinal} />
        <div className="cd-btn-row end automate-review-actions">
          {fastAbutment && <button type="button" className="cd-btn secondary" disabled={wf.busy || !wf.hasOuter}
            onClick={() => wf.setStep("design")}>编辑 / 查看 3D</button>}
          {!fastAbutment && <button type="button" className="cd-btn secondary" disabled={wf.busy} onClick={() => wf.setStep("design")}>返回微调</button>}
          {!wf.hasFinal && <button type="button" className="cd-btn ghost" disabled={wf.busy || !wf.hasOuter} onClick={() => wf.finalize(false)}>
            {wf.busy ? "正在生成…" : "生成最终牙冠"}
          </button>}
          <button type="button" className="cd-btn primary" disabled={!wf.hasFinal}
            onClick={() => window.open(adapter.finalCrownUrl(wf.design.id), "_blank")}>下载 STL</button>
          <button type="button" className="cd-btn ghost" disabled={!wf.hasFinal}
            onClick={() => window.open(adapter.exportPackageUrl(wf.design.id), "_blank")}>下载交付包</button>
        </div>
      </Panel>

      <aside className="cd-rail">
        <Panel title="自动检查" subtitle="所有结果来自实际几何测量">
          {checks.map((item) => <div key={item.name} className="cd-qc">
            <div><span>{item.name}</span><small>{item.value}</small></div>
            <b className={item.ok ? "ok" : "warn"}>{item.ok ? "PASS" : "检查"}</b>
          </div>)}
        </Panel>
        {loading && <Notice>正在读取邻接、咬合与厚度测量…</Notice>}
        {needsReview.length
          ? <Notice tone="warn">建议复核：{needsReview.map((item) => item.name).join("、")}。可进入 3D 编辑器检查。</Notice>
          : <Notice tone="ok">自动检查通过，可以下载设计结果。</Notice>}
        <StatusPill state={wf.hasFinal ? "ok" : ""}>{wf.hasFinal ? "设计已完成" : "正在等待最终牙冠"}</StatusPill>
        {fastAbutment && <button type="button" className="cd-link-reset" onClick={() => wf.setStep("home")}>返回结果列表</button>}
      </aside>
    </div>
  </div>;
}
