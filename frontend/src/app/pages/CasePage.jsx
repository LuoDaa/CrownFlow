import React, { useRef } from "react";
import { Field, PageHead, Panel, Segmented } from "../ui";

const FDI_OPTIONS = ["16", "15", "14", "13", "12", "11", "21", "22", "23", "24", "25", "26",
  "36", "35", "34", "33", "32", "31", "41", "42", "43", "44", "45", "46", "37", "47"];

function CompactUpload({ label, mark, fileName, disabled, onPick }) {
  const input = useRef(null);
  return <label className={`cd-upload-compact ${fileName ? "done" : ""}`}>
    <input ref={input} type="file" accept=".stl,.STL,.dcm,.DCM" disabled={disabled}
      onChange={(event) => { const file = event.target.files?.[0]; if (file) onPick(file); }} />
    <div className={`cd-upload-mark ${mark === "L" ? "lower" : ""}`}>{mark}</div>
    <div className="cd-upload-copy">
      <b>{label}</b>
      <span>{fileName || "STL / DCM"}</span>
    </div>
    <em>{fileName ? "已导入" : "选择文件"}</em>
  </label>;
}

export default function CasePage({ wf }) {
  const assets = wf.detail?.assets || [];
  const assetOf = (kind) => assets.find((asset) => asset.kind === kind) || null;
  const upper = assetOf("upper");
  const lower = assetOf("lower");
  const fdi = wf.design ? String(wf.design.target_fdi) : wf.fdi;
  const supportType = wf.design?.support_type || wf.supportType;
  const locked = !!wf.design || wf.busy;

  return <div className="cd-page cd-case-page">
    <PageHead eyebrow="STEP 2" title="病例信息"
      hint="把病例基本信息填完即可；不在填写过程中逐项审核，点击底部“提交并继续”时统一提示缺少的必填项。" />

    <Panel title="设计信息" subtitle="扫描文件 · 目标牙位 · 修复类型">
      <div className="cd-case-form">
        <div className="cd-case-files">
          <CompactUpload label="上颌扫描" mark="U" fileName={upper?.original_name || (upper ? "已导入" : "")}
            disabled={locked || !wf.caseId} onPick={(file) => wf.upload("upper", file)} />
          <CompactUpload label="下颌扫描" mark="L" fileName={lower?.original_name || (lower ? "已导入" : "")}
            disabled={locked || !wf.caseId} onPick={(file) => wf.upload("lower", file)} />
        </div>

        <div className="cd-case-fields">
          <Field label="目标牙位">
            <select value={fdi} disabled={locked} onChange={(event) => wf.setFdi(event.target.value)}>
              <option value="">请选择 FDI</option>
              {FDI_OPTIONS.map((item) => <option key={item} value={item}>{item}</option>)}
            </select>
          </Field>
          <Field label="修复类型">
            <Segmented value={supportType} disabled={locked}
              options={[{ value: "abutment", label: "种植基台" }, { value: "prep", label: "天然牙预备体" }]}
              onChange={wf.setSupportType} />
          </Field>
          <Field label="病例编号">
            <input type="text" readOnly value={wf.detail?.case?.name || wf.detail?.name || "—"} />
          </Field>
        </div>
      </div>

      <p className="cd-case-submit-hint">
        可按任意顺序填写。提交时统一检查：上颌扫描、下颌扫描、目标牙位、修复类型。
        {wf.design ? ` 当前设计：FDI ${wf.design.target_fdi} · ${wf.design.support_type === "abutment" ? "种植基台" : "天然牙预备体"}` : ""}
      </p>
    </Panel>
  </div>;
}
