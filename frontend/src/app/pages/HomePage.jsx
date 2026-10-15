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
        <div className="cd-eyebrow">CROWN DESIGN WORKSPACE</div>
        <h1>数字牙冠设计</h1>
        <p>病例导入、边缘线与插入轴确认、牙冠模板与参数、邻牙与对颌适配、最终检查与导出。</p>
        <div className="cd-hero-actions">
          <button type="button" className="cd-btn primary" disabled={wf.busy} onClick={wf.createCase}>＋ 新建病例</button>
          <button type="button" className="cd-btn secondary" disabled={wf.busy}
            onClick={() => wf.setMsg("打开病例：从下面的最近病例里选择")}>打开病例</button>
        </div>
      </div>
      <div className="cd-hero-info">
        <div><span>工作流</span><b>6 步</b></div>
        <div><span>修复类型</span><b>基台 / 天然牙</b></div>
        <div><span>输出</span><b>Final Crown STL</b></div>
      </div>
    </section>

    <PageHead title="最近病例" hint="选择一条继续设计，或新建病例。" />
    <div className="cd-cases">
      {recent.map((row) => <div key={row.caseId} className="cd-case-entry"><button type="button" className="cd-case" disabled={wf.busy}
        onClick={() => (row.design ? wf.resumeDesign(row.design.id, row.caseId) : wf.openCase(row.caseId))}>
        <div className="cd-case-icon">{row.design ? row.design.target_fdi : "＋"}</div>
        <div className="cd-case-body">
          <b>{row.name}</b>
          <p>{row.design ? `FDI ${row.design.target_fdi} · ${row.design.support_type === "abutment" ? "种植基台" : "天然牙预备体"}` : "尚未选择牙位"}</p>
          {row.state && <small className={row.state.tone}>{row.state.label}</small>}
        </div>
      </button>
        <button type="button" className="cd-case-delete" disabled={wf.busy} aria-label={`删除病例 ${row.name}`}
          onClick={() => { setDeleting(row); deleteDialog.current.showModal(); }}>删除</button>
      </div>)}
      <button type="button" className="cd-case cd-case-add" disabled={wf.busy} onClick={wf.createCase}>
        <strong>＋</strong><span>新建病例</span>
      </button>
    </div>

    <dialog ref={deleteDialog} className="cd-delete-dialog" aria-labelledby="delete-case-title" onCancel={() => setDeleting(null)}>
      <h3 id="delete-case-title">删除病例</h3>
      <p>确定永久删除“{deleting?.name}”？</p>
      <p className="cd-hint">该病例的设计记录和扫描文件将一并删除，无法恢复。</p>
      <div className="cd-btn-row">
        <button type="button" className="cd-btn secondary" autoFocus onClick={() => { deleteDialog.current.close(); setDeleting(null); }}>取消</button>
        <button type="button" className="cd-btn primary" disabled={wf.busy || !deleting} onClick={() => {
          const id = deleting.caseId; deleteDialog.current.close(); setDeleting(null); wf.removeCase(id);
        }}>确认删除</button>
      </div>
    </dialog>

    {wf.cases.length === 0 && !wf.busy && <Panel><p className="cd-hint">还没有病例。点“新建病例”开始，或在病例页导入一个订单包。</p></Panel>}
  </div>;
}
