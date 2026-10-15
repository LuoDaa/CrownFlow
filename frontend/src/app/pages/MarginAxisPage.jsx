import React from "react";
import { Field, Metric, Notice, PageHead, Panel, StatusPill, ToolGrid } from "../ui";
import ViewerStage from "../ViewerStage";

const MARGIN_TOOLS = [
  { value: "edit", label: "选择", hint: "拖动控制点局部修正" },
  { value: "move", label: "局部拖动", hint: "拖动控制点，周围曲线平顺跟随" },
  { value: "smooth", label: "平滑", hint: "平滑折线，纠偏点锁死" },
  { value: "reset", label: "重置", hint: "重新自动识别" },
];

export default function MarginAxisPage({ wf }) {
  const session = wf.marginSession;
  const proposals = session?.proposals || [];
  const boundaryLabel = session?.boundary_label || (wf.supportType === "abutment" ? "基台冠终止线" : "天然牙边缘线");
  const closed = wf.margin.length >= 12;

  const onTool = (tool) => {
    if (tool === "reset") return wf.detectMargin();
    if (tool === "smooth") { wf.setMarginTool("edit"); return wf.smoothMargin(); }
    return wf.setMarginTool(tool);
  };

  return <div className="cd-page">
    <PageHead eyebrow="STEP 3" title="Margin / Axis"
      hint="自动识别边缘线与插入轴，逐面复核后确认。边缘线只在这里确认，几何由服务端算法给出。" />

    <div className="cd-cad">
      <aside className="cd-rail">
        <Panel title="边缘线">
          <button type="button" className="cd-btn primary full" disabled={wf.busy || !wf.design} onClick={wf.detectMargin}>
            {wf.busy ? "正在识别…" : "自动识别 Margin"}
          </button>
          {wf.detection && <Notice>{wf.detection}</Notice>}
          <button type="button" className="cd-btn primary full" disabled={wf.busy || !wf.canConfirmMargin || (wf.marginConfirmed && !wf.marginDirty && !session)} onClick={wf.confirmMargin}>
            {wf.marginConfirmed ? "重新确认 Margin" : "确认当前 Margin"}
          </button>
          {!wf.marginConfirmed && closed && <p className="cd-hint">识别出的闭合线只是候选。检查位置正确后，点击上方按钮保存为正式边缘线。</p>}
          <div className="cd-sep" />
          <span className="cd-sub">Margin 编辑</span>
          <ToolGrid value={wf.marginTool} options={MARGIN_TOOLS}
            disabled={wf.busy || !closed} onChange={onTool} />
          <div className="cd-btn-row">
            <button type="button" className="cd-btn ghost" disabled={wf.busy || !closed || !wf.marginUndo.length} onClick={wf.undoMargin}>撤销</button>
            <button type="button" className="cd-btn ghost" disabled={wf.busy || !closed || !wf.marginRedo.length} onClick={wf.redoMargin}>重做</button>
          </div>
          <Field label="平滑强度">
            <input type="range" min="0" max="1" step="0.05" value={wf.smoothStrength} disabled={wf.busy || !closed}
              onChange={(event) => wf.setSmoothStrength(Number(event.target.value))} />
          </Field>
          <Field label="整圈微调" hint="按当前就位方向">
            <div className="cd-btn-row">
              <button type="button" className="cd-btn ghost" disabled={wf.busy || !session} onClick={() => wf.moveMargin(0.1)}>上移 +0.1</button>
              <button type="button" className="cd-btn ghost" disabled={wf.busy || !session} onClick={() => wf.moveMargin(-0.1)}>下移 −0.1</button>
            </div>
          </Field>
          <div className="cd-btn-row">
            <button type="button" className="cd-btn ghost" disabled={wf.busy || !session} onClick={wf.startAnchorPick}>加纠偏点</button>
            <button type="button" className="cd-btn ghost" disabled={wf.busy} onClick={wf.drawMargin}>手动绘制</button>
          </div>
        </Panel>

        <Panel title="插入轴 / 方向">
          <div className="cd-axis-card">
            <strong>↑</strong>
            <div>
              <b>{wf.axisConfirmed ? "已确认插入轴" : "待确认插入轴"}</b>
              <small>{[wf.axis[0].toFixed(2), wf.axis[1].toFixed(2), wf.axis[2].toFixed(2)].join(", ")}</small>
            </div>
          </div>
          {(wf.supportType !== "abutment" || !wf.axisConfirmed) && <>
            <button type="button" className="cd-btn secondary full" disabled={wf.busy || !wf.design} onClick={wf.suggestAxis}>自动计算插入轴</button>
            <button type="button" className="cd-btn ghost full" disabled={wf.busy} onClick={wf.resetAxis}>恢复自动方向</button>
            <button type="button" className="cd-btn primary full" disabled={wf.busy || !closed} onClick={wf.confirmAxis}>
              {wf.axisConfirmed ? "重新确认插入轴" : "确认插入轴"}
            </button>
          </>}
          <div className="cd-sep" />
          <span className="cd-sub">近中 / 颊侧方向（模板放置的坐标帧）</span>
          <div className="cd-axis-card">
            <strong>↔</strong>
            <div>
              <b>{wf.frameConfirmed ? "方向已确认" : wf.frame ? "方向待确认" : "尚未推导"}</b>
              <small>{wf.frame
                ? `近中 ${wf.frame.mesial.map((v) => v.toFixed(2)).join(", ")}`
                : "由边缘线与就位轴自动推导"}</small>
            </div>
          </div>
          <div className="cd-btn-row">
            <button type="button" className="cd-btn secondary" disabled={wf.busy || !closed} onClick={wf.suggestFrame}>推导方向</button>
            <button type="button" className="cd-btn ghost" disabled={wf.busy || !wf.frame} onClick={wf.flipFrame}>翻转近中/远中</button>
          </div>
          <button type="button" className="cd-btn primary full" disabled={wf.busy || !wf.marginConfirmed || !wf.axisConfirmed} onClick={wf.confirmFrame}>
            {wf.frameConfirmed ? "重新确认方向" : "确认方向"}
          </button>
          <p className="cd-hint">{wf.supportType === "abutment" ? "基台插入轴已由几何确定；近中/颊侧正负方向仍需结合牙列核对。" : "在 3D 中拖动蓝色箭头的端点手柄即可调整插入轴。"}确认方向后模板才能就位。</p>
        </Panel>
      </aside>

      <ViewerStage wf={wf} mode="patient" />

      <aside className="cd-rail">
        <Panel title="当前结果">
          <Metric label="边缘线" value={wf.marginConfirmed ? "已确认" : closed ? "候选已闭合，待确认" : `待补齐（${wf.margin.length}/12）`} tone={wf.marginConfirmed ? "ok" : "warn"} />
          <Metric label="边缘线点数" value={session ? `${wf.margin.length}` : `${wf.margin.length}`} />
          <Metric label="自动候选" value={proposals.length ? `${proposals.length} 条` : "—"} />
          <Metric label="插入轴" value={wf.axisConfirmed ? "已确认" : closed ? "待确认" : "待边缘线"} tone={wf.axisConfirmed ? "ok" : "warn"} />
          <Metric label="方向帧" value={wf.frameConfirmed ? "已确认" : wf.frame ? "待确认" : "未推导"} tone={wf.frameConfirmed ? "ok" : "warn"} />
          <Metric label="边界语义" value={boundaryLabel} />
          {wf.marginDirty && <Notice tone="warn">你已手工修改边缘线：确认时会以**你在 3D 中看到的这条曲线**为准写入正式修订。</Notice>}
          {session?.ambiguity_flags?.length > 0 && <Notice tone="warn">候选存在并列：{session.ambiguity_flags.join("、")}，请先显式选定一条候选再确认。</Notice>}
          {wf.axisInfo?.note && <Notice>{wf.axisInfo.note}</Notice>}
          <StatusPill state={wf.marginConfirmed ? "ok" : ""}>{wf.marginConfirmed ? "边缘线已确认" : "边缘线待确认"}</StatusPill>
        </Panel>
        <Panel title="保护区域">
          <p className="cd-hint">颈缘带在后续塑形中锁定，自由塑形只改外表面。确认后的边缘线是内冠与外形共同引用的唯一边界。</p>
        </Panel>
        <Panel title="静态咬合位置">
          <p className="cd-hint">请在 3D 中检查上下颌已按实际咬合关系对齐。此确认沿用文件现有坐标，不会自动配准。</p>
          <button type="button" className="cd-btn secondary full" disabled={wf.busy || !wf.frameConfirmed || !wf.upper || !wf.lower}
            onClick={wf.confirmCurrentBite}>确认当前上下颌位置</button>
          {wf.serverStatus?.bite?.pose_confirmed && <Notice tone="ok">已记录静态咬合位置。若重新导入扫描或修改方向，需再次确认。</Notice>}
        </Panel>
        <Panel title="邻牙参考面">
          <p className="cd-hint">分别在真实邻牙的接触面点选，高亮区只能包含该邻牙。没有邻牙的一侧标记缺失。</p>
          {[["mesial", "近中"], ["distal", "远中"]].map(([side, label]) => <div key={side}>
            <div className="cd-btn-row">
              <button type="button" className="cd-btn secondary" disabled={wf.busy || !wf.frameConfirmed}
                onClick={() => wf.selectNeighbor(side)}>点选{label}</button>
              <button type="button" className="cd-btn ghost" disabled={wf.busy || !wf.frameConfirmed}
                onClick={() => wf.confirmNeighbor(side, true)}>该侧无邻牙</button>
            </div>
            {wf.neighborSelection?.side === side && <button type="button" className="cd-btn primary full"
              disabled={wf.busy || wf.neighborSelection.face_ids.length < 3}
              onClick={() => wf.confirmNeighbor(side)}>确认高亮区域属于{label}邻牙</button>}
            <small>{label}：{wf.serverStatus?.neighbors?.[side]?.confirmation_status === "CONFIRMED" ? "已确认" :
              wf.serverStatus?.neighbors?.[side]?.confirmation_status === "ABSENT" ? "无邻牙" : "待确认"}</small>
          </div>)}
        </Panel>
      </aside>
    </div>
  </div>;
}
