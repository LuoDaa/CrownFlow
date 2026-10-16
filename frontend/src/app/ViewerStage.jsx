import React from "react";
import Viewer from "../components/Viewer";
import ViewTools, { useViewTools } from "../components/ViewTools";

/**
 * 3D 视口外壳：Viewer / ViewTools（相机与显示工具）在这里接一次，
 * 各页面只决定"画什么 + 显示开关"，不重复写渲染代码。
 *
 * 1.0 预览策略：渲染与审核彻底解耦。只要本轮生成出了候选几何，画布
 * 就优先展示 candidatePreview；是否 PASS/REVIEW_REQUIRED/是否已采纳都
 * 不能阻止用户先看到结果。final 模式在最终冠尚未生成时也回退到最新
 * candidate/outer，而不是显示空画布。
 */
export default function ViewerStage({
  wf, mode = "patient", renderUpper = true, renderLower = true, renderSupport = true,
  showMargin = true, editorScene = null, headlight = false, legend = null, strip = null,
  extraTools = null, endNote = null,
}) {
  const tools = useViewTools((action) => {
    if (["upper", "lower", "support"].includes(action)) {
      wf.setView((view) => ({ ...view, [action]: !view[action] }));
      return true;
    }
    if (action === "margin") { wf.setView((view) => ({ ...view, margin: !view.margin })); return true; }
    if (action === "escape") { wf.setPickRole(null); return true; }
    return false;
  }, true, wf.step === "design");

  const visible = wf.view;
  const latestOuter = wf.candidatePreview || wf.outer || null;
  const hasFinal = mode === "final" && !!wf.finalCrown;
  const activeOuter = hasFinal ? null : latestOuter;
  const activeFinal = hasFinal ? wf.finalCrown : null;
  const previewSource = hasFinal ? "final"
    : wf.candidatePreview ? "candidate"
      : wf.outer ? "accepted" : "none";

  return <section className="cd-viewport-card">
    <div className="cd-viewport-head">
      <div className="cd-viewport-tabs">
        <button type="button" className="cd-mini active">3D</button>
      </div>
      <small>
        右键旋转 · 滚轮缩放 · 左右键平移
        {previewSource === "candidate" ? " · 正在显示本次生成候选（审核不阻止预览）" : ""}
        {mode === "final" && previewSource !== "final" && previewSource !== "none"
          ? " · 最终冠未完成，显示最新可用外冠" : ""}
      </small>
    </div>
    <div className="cd-toolbar">
      <ViewTools tools={tools} />
      {extraTools}
    </div>
    <div className="cd-viewport">
      <Viewer
        mode={hasFinal ? "final" : mode === "final" ? "patient" : mode}
        upper={renderUpper && visible.upper ? wf.upper : null}
        lower={renderLower && visible.lower ? wf.lower : null}
        support={renderSupport && visible.support ? wf.support : null}
        outer={activeOuter}
        finalMesh={activeFinal}
        cutter={wf.innerCrown}
        insertionAxis={wf.axis}
        showAxis={wf.step === "margin"}
        onAxisChange={wf.setAxis}
        marginPoints={wf.step === "case" ? [] : wf.margin}
        marginConfirmed={wf.marginConfirmed}
        showMargin={showMargin && visible.margin}
        marginMode="move"
        marginPatch={[]}
        marginRadius={1.5}
        onMarginCurveChange={mode === "patient" && wf.margin.length >= 12 && !wf.busy && wf.marginTool !== "patch"
          ? (points) => wf.editMargin(points) : undefined}
        onMarginError={wf.setMsg}
        pickRole={wf.pickRole === "marginAnchor" || wf.pickRole === "margin" ? "support"
          : wf.pickRole === "supportCenter" || wf.pickRole?.startsWith("neighbor:") ? "patient" : wf.pickRole}
        onPick={wf.pick}
        selection={wf.neighborSelection || (wf.pickRole?.startsWith("neighbor:")
          ? { source: Number(wf.fdi) >= 31 ? "lower" : "upper", face_ids: [] } : null)}
        pickRadius={2}
        onSectionProbe={wf.step === "margin" && !wf.busy ? wf.probeSection : undefined}
        tools={tools}
        scanOpacity={visible.scanOpacity}
        headlight={headlight}
        editorScene={editorScene}
        onRetryMesh={wf.reloadMeshes}
        sceneKey={`${wf.caseId}:${mode}:${wf.meshVersion}:${wf.candidatePreview || ""}:${wf.finalCrown || ""}`}
      />
      {!wf.upper && !wf.lower && !wf.support && !activeOuter && !activeFinal
        && <div className="cd-empty">导入病例文件后，这里显示牙列与牙冠</div>}
    </div>
    {legend}
    {strip && <div className="cd-strip">{strip}</div>}
    {endNote}
  </section>;
}
