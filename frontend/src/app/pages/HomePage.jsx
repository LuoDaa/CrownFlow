import React, { useRef, useState } from "react";
import { PageHead } from "../ui";

function stateOf(design) {
  if (design?.final_path) return { label: "可检查", tone: "ok" };
  if (design?.outer_path) return { label: "设计中", tone: "work" };
  return { label: "待上传/处理", tone: "" };
}

export default function HomePage({ wf }) {
  const [deleting, setDeleting] = useState(null);
  const [filter, setFilter] = useState("pending");
  const deleteDialog = useRef(null);
  const rows = wf.cases.map((item) => {
    const designs = wf.designs[item.id] || [];
    const design = designs[0] || null;
    return { caseId: item.id, name: item.name, design, state: stateOf(design) };
  }).filter((row) => filter === "all" || !row.design?.final_path);

  return <div className="cd-page automate-results-page">
    <PageHead eyebrow="RESULTS" title="订单结果" hint="查看正在处理、待检查和已完成的牙冠订单。">
      <button type="button" className="cd-btn primary" disabled={wf.busy} onClick={wf.createCase}>＋ 新建订单</button>
    </PageHead>

    <div className="cd-results-toolbar">
      <div className="cd-results-filters">
        <button className={filter === "pending" ? "active" : ""} onClick={() => setFilter("pending")}>未决定</button>
        <button className={filter === "all" ? "active" : ""} onClick={() => setFilter("all")}>全部订单</button>
      </div>
      <span>{rows.length} 个订单</span>
    </div>

    <section className="cd-results-table" aria-label="订单结果列表">
      <div className="cd-results-head">
        <span>订单</span><span>牙位</span><span>修复类型</span><span>状态</span><span>操作</span>
      </div>
      {rows.map((row) => <div key={row.caseId} className="cd-results-row">
        <div><b>{row.name}</b><small>{row.caseId.slice(0, 8)}</small></div>
        <div>{row.design?.target_fdi || "—"}</div>
        <div>{row.design ? (row.design.support_type === "abutment" ? "种植基台冠" : "天然牙冠") : "未设置"}</div>
        <div><span className={`cd-result-state ${row.state.tone}`}>{row.state.label}</span></div>
        <div className="cd-result-actions">
          <button type="button" className="cd-btn primary" disabled={wf.busy}
            onClick={() => row.design ? wf.resumeDesign(row.design.id, row.caseId) : wf.openCase(row.caseId)}>
            {row.design?.final_path ? "检查" : row.design ? "继续" : "上传"}
          </button>
          <button type="button" className="cd-btn ghost" disabled={wf.busy}
            onClick={() => { setDeleting(row); deleteDialog.current.showModal(); }}>删除</button>
        </div>
      </div>)}
      {rows.length === 0 && <div className="cd-results-empty">暂无订单。点击“新建订单”开始自动牙冠设计。</div>}
    </section>

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
  </div>;
}
