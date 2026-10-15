import React, { useRef, useState } from "react";
import { Panel, PageHead } from "../ui";

/** 设计行 → 人话状态（正式产品文案，不暴露内部状态码）。 */
function stateOf(design) {
  if (design.final_path) return { label: "已完成", tone: "ok" };
  if (design.outer_path) return { label: "设计中", tone: "work" };
  return { label: "待开始", tone: "" };
}

export default function HomePage({ wf }) {
  const [deleting, setDeleting] = useState(null);
  const deleteDialog = useRef(null);
  const recent = wf.cases.slice(0, 6).map((item) => {
    const designs = wf.designs[item.id] || [];
    const design = designs[0];
    return { caseId: item.id, name: item.name, design, state: design ? stateOf(design) : null };
  });

  return <div className="cd-home">
    <section className="cd-hero">
      <div>
        <div className="cd-eyebrow">CROWNFLOW 1.0</div>
        <h1>自动牙冠设计</h1>
        <p>基台冠走快速订单：上传已咬合上下颌、选择牙位/牙库/参数后直接自动生成；天然牙预备体继续保留完整 Margin / Axis 复核流程。</p>
        <div className="cd-hero-actions">
          <button type="button" className="cd-btn primary" disabled={wf.busy} onClick={wf.createCase}>＋ 创建订单</button>
          <button type="button" className="cd-btn secondary" disabled={wf.busy}
            onClick={() => wf.setMsg("打开订单：从下面的最近订单里选择")}>打开订单</button>
        </div>
      </div>
      <div className="cd-hero-info">
        <div><span>基台冠</span><b>快速自动化</b></div>
        <div><span>天然牙</span><b>完整复核</b></div>
        <div><span>输出</span><b>Final Crown STL</b></div>
      </div>
    </section>

    <PageHead title="最近订单" hint="选择一条继续设计，或创建新订单。" />
    <div className="cd-cases">
      {recent.map((row) => <div key={row.caseId} className="cd-case-entry"><button type="button" className="cd-case" disabled={wf.busy}
        onClick={() => (row.design ? wf.resumeDesign(row.design.id, row.caseId) : wf.openCase(row.caseId))}>
        <div className="cd-case-icon">{row.design ? row.design.target_fdi : "＋"}</div>
        <div className="cd-case-body">
          <b>{row.name}</b>
          <p>{row.design ? `FDI ${row.design.target_fdi} · ${row.design.support_type === "abutment" ? "种植基台快速流程" : "天然牙预备体"}` : "尚未选择牙位"}</p>
          {row.state && <small className={row.state.tone}>{row.state.label}</small>}
        </div>
      </button>
        <button type="button" className="cd-case-delete" disabled={wf.busy} aria-label={`删除订单 ${row.name}`}
          onClick={() => { setDeleting(row); deleteDialog.current.showModal(); }}>删除</button>
      </div>)}
      <button type="button" className="cd-case cd-case-add" disabled={wf.busy} onClick={wf.createCase}>
        <strong>＋</strong><span>创建订单</span>
      </button>
    </div>

    <dialog ref={deleteDialog} className="cd-delete-dialog" aria-labelledby="delete-case-title" onCancel={() => setDeleting(null)}>
      <h3 id="delete-case-title">删除订单</h3>
      <p>确定永久删除“{deleting?.name}”？</p>
      <p className="cd-hint">该订单的设计记录和扫描文件将一并删除，无法恢复。</p>
      <div className="cd-btn-row">
        <button type="button" className="cd-btn secondary" autoFocus onClick={() => { deleteDialog.current.close(); setDeleting(null); }}>取消</button>
        <button type="button" className="cd-btn primary" disabled={wf.busy || !deleting} onClick={() => {
          const id = deleting.caseId; deleteDialog.current.close(); setDeleting(null); wf.removeCase(id);
        }}>确认删除</button>
      </div>
    </dialog>

    {wf.cases.length === 0 && !wf.busy && <Panel><p className="cd-hint">还没有订单。点“创建订单”开始，或导入已有订单包。</p></Panel>}
  </div>;
}
