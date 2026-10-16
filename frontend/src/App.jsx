import React from "react";
import useWorkflow from "./domain/useWorkflow";
import { FLOW, S, prevStep, stepOf } from "./domain/steps";
import { Stepper, BusyOverlay } from "./app/ui";
import HomePage from "./app/pages/HomePage";
import CasePage from "./app/pages/CasePage";
import MarginAxisPage from "./app/pages/MarginAxisPage";
import SetupPage from "./app/pages/SetupPage";
import DesignPage from "./app/pages/DesignPage";
import "./styles/app.css";
import "./styles/order-page.css";

const ABUTMENT_FLOW = [
  { ...FLOW.find((item) => item.id === S.HOME), n: 1, label: "首页" },
  { ...FLOW.find((item) => item.id === S.CASE), n: 2, label: "订单" },
  { ...FLOW.find((item) => item.id === S.DESIGN), n: 3, label: "自动设计 / 导出" },
];

function exportCurrentCrown(wf) {
  if (!wf.design?.id || !wf.hasOuter) return false;
  const url = `/api/designs/${wf.design.id}/final.stl`;
  const popup = window.open(url, "_blank");
  if (!popup) window.location.href = url;
  return true;
}

/** 每一步底部的主动作：生成页自己拥有“牙冠生成”，微调页直接拥有导出。 */
function primaryOf(wf, step) {
  switch (step) {
    case S.CASE:
      return null;
    case S.MARGIN:
      if (!wf.marginConfirmed) return { label: "确认当前 Margin", disabled: !wf.canConfirmMargin, run: wf.confirmMargin };
      if (!wf.axisConfirmed) return { label: "确认插入轴", disabled: !wf.canConfirmMargin, run: wf.confirmAxis };
      if (!wf.frameConfirmed) return { label: "确认近中 / 颊侧方向", disabled: !wf.canConfirmMargin, run: wf.confirmFrame };
      return { label: "下一步：选择牙型", disabled: false, run: () => wf.go(S.SETUP) };
    case S.SETUP:
      return null;
    case S.DESIGN:
      return {
        label: "导出当前牙冠 STL",
        disabled: !wf.hasOuter,
        run: () => exportCurrentCrown(wf),
      };
    default:
      return null;
  }
}

function Page({ wf, step }) {
  switch (step) {
    case S.CASE: return <CasePage wf={wf} />;
    case S.MARGIN: return <MarginAxisPage wf={wf} />;
    case S.SETUP: return <SetupPage wf={wf} />;
    case S.DESIGN:
    case S.REVIEW:
      return <DesignPage wf={wf} />;
    default: return <HomePage wf={wf} />;
  }
}

function previousOf(step, fastAbutment) {
  if (!fastAbutment) return prevStep(step);
  if (step === S.CASE) return S.HOME;
  if (step === S.DESIGN) return S.CASE;
  return null;
}

export default function App() {
  const wf = useWorkflow();
  const fastAbutment = (wf.design?.support_type || wf.supportType) === "abutment";
  const legacyNormalizedStep = wf.step === S.REVIEW ? S.DESIGN : wf.step;
  const displayStep = fastAbutment && [S.MARGIN, S.SETUP].includes(legacyNormalizedStep) ? S.CASE : legacyNormalizedStep;
  const primary = primaryOf(wf, displayStep);
  const previous = previousOf(displayStep, fastAbutment);
  const title = stepOf(displayStep).title;
  const flow = fastAbutment ? ABUTMENT_FLOW : FLOW;
  const intakeWorkspace = displayStep === S.HOME || displayStep === S.CASE;

  return <div className={`cd-app ${intakeWorkspace ? "cd-app-intake" : ""}`}>
    <header className="cd-topbar">
      <button type="button" className="cd-brand" onClick={() => wf.go(S.HOME)}>
        <div className="cd-logo">W</div>
        <div><b>WelooAI 牙冠自动生成</b></div>
      </button>
      <div className="cd-crumb">
        {wf.caseId ? `${wf.detail?.name || "订单"}${wf.fdi ? ` / FDI ${wf.fdi}` : ""}${wf.design ? ` / ${wf.supportType === "abutment" ? "种植基台" : "天然牙预备体"}` : ""}` : "请选择订单"}
      </div>
      <div className="cd-top-actions">
        <span className="cd-state" role="status">{wf.busy ? "处理中…" : "就绪"}</span>
        <button type="button" className="cd-btn ghost" disabled={wf.busy || !wf.caseId}
          onClick={() => wf.setMsg("订单文件与设计修订会自动保存在服务端，无需手动保存")}>保存订单</button>
      </div>
    </header>

    {!intakeWorkspace && <Stepper step={displayStep} facts={wf.facts} busy={wf.busy} onGo={wf.go} flow={flow} />}

    <main className="cd-main">
      <Page wf={wf} step={displayStep} />
    </main>

    {!intakeWorkspace && <footer className="cd-bottom">
      <button type="button" className="cd-btn secondary" disabled={!previous || wf.busy} onClick={() => previous && wf.go(previous)}>
        ← 上一步
      </button>
      <b role="status">{wf.busy ? "正在处理…" : fastAbutment && displayStep === S.CASE ? "基台自动化订单" : title}</b>
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
