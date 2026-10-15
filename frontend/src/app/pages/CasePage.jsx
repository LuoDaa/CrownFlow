import React, { useRef } from "react";
import { Field, Notice, PageHead, Panel, Segmented } from "../ui";

const FDI_OPTIONS = ["16", "15", "14", "13", "12", "11", "21", "22", "23", "24", "25", "26",
  "36", "35", "34", "33", "32", "31", "41", "42", "43", "44", "45", "46", "37", "47"];

function Upload({ label, mark, hint, name, disabled, onPick }) {
  const input = useRef(null);
  return <label className={`cd-upload ${name ? "done" : ""}`}>
    {/* 只暴露后端**真正支持**的格式：STL / DCM（OBJ/PLY 以后再说，不先放在界面上） */}
    <input ref={input} type="file" accept=".stl,.STL,.dcm,.DCM" disabled={disabled}
      onChange={(event) => { const file = event.target.files?.[0]; if (file) onPick(file); }} />
    <div className={`cd-upload-mark ${mark === "L" ? "lower" : ""}`}>{mark}</div>
    <b>{label}</b>
    <span>{name || `选择${label}文件`}</span>
    <small>{hint}</small>
  </label>;
}

export default function CasePage({ wf }) {
  const assets = wf.detail?.assets || [];
  const fdi = wf.design ? String(wf.design.target_fdi) : wf.fdi;
  const supportType = wf.design?.support_type || wf.supportType;
  const has = (kind) => assets.some((asset) => asset.kind === kind);
  return <div className="cd-page">
    <PageHead eyebrow="STEP 2" title="病例信息"
      hint="导入上颌与下颌文件，并确认目标牙位与修复类型。上下颌使用病例文件中的原始相对位置。" />

    <div className="cd-grid two">
      <Panel title="病例文件" subtitle="上颌 / 下颌扫描">
        <div className="cd-uploads">
          <Upload label="Upper Jaw" mark="U" hint="STL / DCM" name={has("upper") ? "已导入" : ""}
            disabled={wf.busy || !wf.caseId} onPick={(file) => wf.upload("upper", file)} />
          <Upload label="Lower Jaw" mark="L" hint="保留原始咬合坐标" name={has("lower") ? "已导入" : ""}
            disabled={wf.busy || !wf.caseId} onPick={(file) => wf.upload("lower", file)} />
        </div>
        <Notice>Upper / Lower 直接使用病例文件里的相对位置，前端不做配准、不改变两者相对姿态。</Notice>
      </Panel>

      <Panel title="设计对象">
        <Field label="牙位 FDI" hint={wf.fdi ? "" : "必填"}>
          <select value={fdi} disabled={!!wf.design || wf.busy} onChange={(event) => wf.setFdi(event.target.value)}>
            <option value="">请选择</option>
            {FDI_OPTIONS.map((fdi) => <option key={fdi} value={fdi}>{fdi}</option>)}
          </select>
        </Field>
        <Field label="修复类型" hint="直接对应后端 support_type：prep / abutment">
          <Segmented value={supportType} disabled={!!wf.design || wf.busy}
            options={[{ value: "abutment", label: "种植基台" }, { value: "prep", label: "天然牙预备体" }]}
            onChange={wf.setSupportType} />
        </Field>
        <Field label="病例编号">
          <input type="text" readOnly value={wf.detail?.case?.name || wf.detail?.name || "—"} />
        </Field>
        {wf.design && <Notice tone="ok">设计已创建：FDI {wf.design.target_fdi} · {wf.design.support_type === "abutment" ? "种植基台" : "天然牙预备体"}</Notice>}
        {!wf.design && <Notice>选择牙位与修复类型后，用底部“创建设计”进入下一步。</Notice>}
      </Panel>
    </div>
  </div>;
}
