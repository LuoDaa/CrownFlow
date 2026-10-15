import React from "react";
import useWorkflow from "./domain/useWorkflow";
import { FLOW, S, prevStep, stepOf } from "./domain/steps";
import { Stepper, BusyOverlay } from "./app/ui";
import HomePage from "./app/pages/HomePage";
import CasePage from "./app/pages/CasePage";
import MarginAxisPage from "./app/pages/MarginAxisPage";
import SetupPage from "./app/pages/SetupPage";
import DesignPage from "./app/pages/DesignPage";
import ReviewPage from "./app/pages/ReviewPage";
import "./styles/app.css";

/** Natural-prep still uses the explicit CAD workflow. Abutment orders use the
 * Automate-style product shell: Upload -> Results -> Review/Edit on demand. */
function primaryOf(wf, step) {
  switch (step) {
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

function Page({ wf, step }) {
  switch (step) {
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
  const fastAbutment = (wf.design?.support_type || wf.supportType) === "abutment";
  const simpleShell = fastAbutment || wf.step === S.HOME || wf.step === S.CASE;
  const displayStep = fastAbutment && [S.MARGIN, S.SETUP].includes(wf.step) ? S.CASE : wf.step;
  const primary = simpleShell ? null : primaryOf(wf, displayStep);
  const previous = simpleShell ? null : prevStep(displayStep);
  const title = stepOf(displayStep).title;

  return <div className="cd-app">
    <header className="cd-topbar automate-topbar">
      <button type="button" className="cd-brand" onClick={() => wf.go(S.HOME)}>
        <div className="cd-logo">W</div>
        <div><b>WelooAI Automate</b><small>CrownFlow 1.0</small></div>
      </button>

      {simpleShell ? <nav className="cd-product-nav" aria-label="订单导航">
        <button type="button" className={displayStep === S.CASE ? "active" : ""} disabled={wf.busy}
          onClick={() => wf.caseId ? wf.setStep(S.CASE) : wf.createCase()}>上传</button>
        <button type="button" className={displayStep === S.HOME ? "active" : ""} disabled={wf.busy}
          onClick={() => wf.setStep(S.HOME)}>结果</button>
      </nav> : <div className="cd-crumb">
        {wf.caseId ? `${wf.detail?.name || "订单"}${wf.fdi ? ` / FDI ${wf.fdi}` : ""}` : ""}
      </div>}

      <div className="cd-top-actions">
        {wf.caseId && <span className="cd-order-chip">{wf.detail?.name || "当前订单"}{wf.fdi ? ` · FDI ${wf.fdi}` : ""}</span>}
        <span className="cd-state" role="status">{wf.busy ? "处理中…" : "就绪"}</span>
      </div>
    </header>

    {!simpleShell && <Stepper step={displayStep} facts={wf.facts} busy={wf.busy} onGo={wf.go} flow={FLOW} />}

    <main className={`cd-main ${simpleShell ? "automate-main" : ""}`}>
      <Page wf={wf} step={displayStep} />
    </main>

    {!simpleShell && <footer className="cd-bottom">
      <button type="button" className="cd-btn secondary" disabled={!previous || wf.busy} onClick={() => previous && wf.go(previous)}>
        ← 上一步
      </button>
      <b role="status">{wf.busy ? "正在处理…" : title}</b>
      {primary
        ? <button type="button" className="cd-btn primary" disabled={wf.busy || primary.disabled} onClick={primary.run}>{primary.label}</button>
        : <span className="cd-bottom-spacer" />}
    </footer>}

    {wf.msg && <div className="cd-toast" role="status">
      <span>{wf.msg}</span>
      <button type="button" aria-label="关闭提示" onClick={() => wf.setMsg("")}>×</button>
    </div>}

    <BusyOverlay busy={wf.busy} label={wf.busyLabel} />
  </div>;
}
