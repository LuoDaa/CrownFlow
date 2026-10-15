import React from "react";
import useWorkflow from "./domain/useWorkflow";
import { S, prevStep, stepOf } from "./domain/steps";
import { Stepper, BusyOverlay } from "./app/ui";
import HomePage from "./app/pages/HomePage";
import CasePage from "./app/pages/CasePage";
import MarginAxisPage from "./app/pages/MarginAxisPage";
import SetupPage from "./app/pages/SetupPage";
import DesignPage from "./app/pages/DesignPage";
import ReviewPage from "./app/pages/ReviewPage";
import "./styles/app.css";

/** 每一步底部的主动作：只有这一步真的做完了，才允许往下走。 */
function primaryOf(wf) {
  switch (wf.step) {
    case S.CASE:
      return { label: wf.design ? "下一步：Margin / Axis" : "创建设计", disabled: !wf.caseId || (!wf.design && (!wf.fdi || !wf.supportType)),
        run: () => (wf.design ? wf.go(S.MARGIN) : wf.prepare()) };
    case S.MARGIN:
      if (!wf.marginConfirmed) return { label: "确认当前 Margin", disabled: !wf.canConfirmMargin, run: wf.confirmMargin };
      if (!wf.axisConfirmed) return { label: "确认插入轴", disabled: !wf.canConfirmMargin, run: wf.confirmAxis };
      if (!wf.frameConfirmed) return { label: "确认近中 / 颊侧方向", disabled: !wf.canConfirmMargin, run: wf.confirmFrame };
      return { label: "下一步：选择牙型", disabled: false, run: () => wf.go(S.SETUP) };
    case S.SETUP:
      return { label: wf.hasOuter ? "下一步：牙冠设计" : "请先自动生成并采纳牙冠", disabled: !wf.hasOuter,
        run: () => wf.go(S.DESIGN) };
    case S.DESIGN:
      return { label: "下一步：检查 / 导出", disabled: !wf.hasOuter, run: () => wf.go(S.REVIEW) };
    case S.REVIEW:
      return { label: wf.hasFinal ? "导出 Final Crown STL" : "生成内冠与 QC", disabled: false,
        run: () => (wf.hasFinal ? window.open(`/api/designs/${wf.design.id}/final.stl`, "_blank") : wf.finalize(false)) };
    default:
      return null;
  }
}

function Page({ wf }) {
  switch (wf.step) {
    case S.CASE: return <CasePage wf={wf} />;
    case S.MARGIN: return <MarginAxisPage wf={wf} />;
    case S.SETUP: return <SetupPage wf={wf} />;
    case S.DESIGN: return <DesignPage wf={wf} />;
    case S.REVIEW: return <ReviewPage wf={wf} />;
    default: return <HomePage wf={wf} />;
  }
}

export default function App() {
  const wf = useWorkflow();
  const primary = primaryOf(wf);
  const previous = prevStep(wf.step);
  const title = stepOf(wf.step).title;

  return <div className="cd-app">
    <header className="cd-topbar">
      <button type="button" className="cd-brand" onClick={() => wf.go(S.HOME)}>
        <div className="cd-logo">W</div>
        <div><b>WelooAI 牙冠自动生成</b></div>
      </button>
      <div className="cd-crumb">
        {wf.caseId ? `${wf.detail?.name || "病例"}${wf.fdi ? ` / FDI ${wf.fdi}` : ""}${wf.design ? ` / ${wf.supportType === "abutment" ? "种植基台" : "天然牙预备体"}` : ""}` : "请选择病例"}
      </div>
      <div className="cd-top-actions">
        <span className="cd-state" role="status">{wf.busy ? "处理中…" : "就绪"}</span>
        <button type="button" className="cd-btn ghost" disabled={wf.busy || !wf.caseId}
          onClick={() => wf.setMsg("病例文件与设计修订会自动保存在服务端，无需手动保存")}>保存病例</button>
      </div>
    </header>

    <Stepper step={wf.step} facts={wf.facts} busy={wf.busy} onGo={wf.go} />

    <main className="cd-main">
      <Page wf={wf} />
    </main>

    <footer className="cd-bottom">
      <button type="button" className="cd-btn secondary" disabled={!previous || wf.busy} onClick={() => previous && wf.go(previous)}>
        ← 上一步
      </button>
      <b role="status">{wf.busy ? "正在处理…" : title}</b>
      {primary
        ? <button type="button" className="cd-btn primary" disabled={wf.busy || primary.disabled} onClick={primary.run}>{primary.label}</button>
        : <span className="cd-bottom-spacer" />}
    </footer>

    {wf.msg && <div className="cd-toast" role="status">
      <span>{wf.msg}</span>
      <button type="button" aria-label="关闭提示" onClick={() => wf.setMsg("")}>×</button>
    </div>}

    <BusyOverlay busy={wf.busy} label={wf.busyLabel} />
  </div>;
}
