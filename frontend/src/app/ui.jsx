/** 新 UI 的基础部件：步进条、页面骨架、面板、字段、指标行。全部无状态。 */
import React from "react";
import { FLOW, progressPercent, stepState } from "../domain/steps";

export function Stepper({ step, facts, onGo, busy }) {
  return <nav className="cd-steps" aria-label="设计流程">
    {FLOW.map((item, index) => {
      const state = stepState(item.id, step);
      return <React.Fragment key={item.id}>
        {index > 0 && <em className={state === "todo" ? "" : "on"} />}
        <button type="button" className={`cd-step ${state}`} aria-current={state === "active" ? "step" : undefined}
          disabled={!!busy} onClick={() => onGo(item.id)}>
          <i>{state === "done" ? "✓" : item.n}</i><span>{item.label}</span>
        </button>
      </React.Fragment>;
    })}
  </nav>;
}

export function PageHead({ eyebrow, title, hint, children }) {
  return <header className="cd-pagehead">
    <div>
      {eyebrow && <div className="cd-eyebrow">{eyebrow}</div>}
      <h1>{title}</h1>
      {hint && <p>{hint}</p>}
    </div>
    {children && <div className="cd-pagehead-actions">{children}</div>}
  </header>;
}

export function Panel({ title, subtitle, actions, className = "", children }) {
  return <section className={`cd-panel ${className}`}>
    {(title || actions) && <div className="cd-panelhead">
      <div><h3>{title}</h3>{subtitle && <small>{subtitle}</small>}</div>
      {actions && <div className="cd-panelhead-actions">{actions}</div>}
    </div>}
    {children}
  </section>;
}

export function Field({ label, hint, children }) {
  return <label className="cd-field">
    <span className="cd-field-label">{label}{hint && <small>{hint}</small>}</span>
    {children}
  </label>;
}

export function Segmented({ value, options, onChange, disabled }) {
  return <div className="cd-segmented" role="group">
    {options.map((option) => <button key={option.value} type="button" disabled={disabled}
      aria-pressed={value === option.value} className={value === option.value ? "active" : ""}
      onClick={() => onChange(option.value)}>{option.label}</button>)}
  </div>;
}

export function ToolGrid({ value, options, onChange, disabled, columns = 2 }) {
  return <div className="cd-tools" style={{ gridTemplateColumns: `repeat(${columns},1fr)` }} role="group">
    {options.map((option) => <button key={option.value} type="button" disabled={disabled || option.disabled}
      title={option.hint} aria-pressed={value === option.value} className={value === option.value ? "active" : ""}
      onClick={() => onChange(option.value)}>{option.label}</button>)}
  </div>;
}

/**
 * 长任务遮罩：后端生成牙冠 / 内冠 / 咬合适配要 10–60 秒，这期间视口是空的。
 * 不加遮罩，别的电脑上就会把"正在生成"误判成"生成失败"。
 * 延迟 350ms 才出现，短操作（撤销、改参数）不会闪。
 */
export function BusyOverlay({ busy, label, delayMs = 350 }) {
  const [visible, setVisible] = React.useState(false);
  React.useEffect(() => {
    if (!busy) { setVisible(false); return undefined; }
    const timer = setTimeout(() => setVisible(true), delayMs);
    return () => clearTimeout(timer);
  }, [busy, delayMs]);
  if (!busy || !visible) return null;
  return <div className="cd-busy-overlay" role="status" aria-live="assertive">
    <div className="cd-busy-card">
      <span className="cd-spinner" aria-hidden="true" />
      <b>{label || "正在处理…"}</b>
      <small>请保持页面打开，不要重复点击；完成后会自动刷新。</small>
    </div>
  </div>;
}

export function Metric({ label, value, note, tone }) {
  return <div className="cd-metric">
    <span>{label}</span>
    <b className={tone || ""}>{value}</b>
    {note && <small>{note}</small>}
  </div>;
}

export function Notice({ children, tone }) {
  return <div className={`cd-notice ${tone || ""}`} role="status">{children}</div>;
}

export function Progress({ step }) {
  return <div className="cd-progress" aria-hidden="true"><i style={{ width: `${progressPercent(step)}%` }} /></div>;
}

export function StatusPill({ state, children }) {
  return <span className={`cd-pill ${state || ""}`}>{children}</span>;
}
