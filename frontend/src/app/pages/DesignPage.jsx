import React, { useEffect, useMemo, useRef, useState } from "react";
import SculptCanvas from "../../components/SculptCanvas";
import { contactStops, contactMetrics, CONTACT_BAND_MM } from "../../contactMap";
import { TOOL_FROM_SHORTCUT, editorShortcut } from "../../editorShortcuts";
import { api } from "../../api";
import { Metric, Notice, PageHead, Panel, ToolGrid } from "../ui";
import ViewerStage from "../ViewerStage";

const DESIGN_TOOLS = [
  { value: "select", label: "选择" }, { value: "move", label: "移动" },
  { value: "rotate", label: "旋转" }, { value: "add", label: "加材" },
  { value: "remove", label: "减材" }, { value: "smooth", label: "平滑" },
  { value: "flatten", label: "压平" }, { value: "drag", label: "局部拖动" },
  { value: "contact", label: "接触" },
  { value: "measure", label: "测量" },
];

const EDIT_STATE_LABEL = {
  IDLE:"待载入", READY:"可编辑", PREVIEW:"局部预览", VALIDATING:"正在验证",
  SAVED:"已保存", REJECTED:"已拒绝并回滚", STALE:"会话过期",
};

function Legend({ target }) {
  if (!Number.isFinite(target)) return <div className="cd-contact-legend">接触目标未确认，色谱不可用</div>;
  const stops = contactStops(target), low = stops[0][0], high = stops.at(-1)[0];
  return <div className="cd-contact-legend" aria-label="有符号接触距离色谱">
    <div className="cd-contact-labels"><span>穿插</span><span>目标 {target.toFixed(2)} mm</span><span>间隙</span></div>
    <div className="cd-contact-gradient" style={{ background: `linear-gradient(to right, ${stops.map(([v,c]) => `${c} ${(v-low)/(high-low)*100}%`).join(',')})` }} />
    <div className="cd-contact-ticks">{[low, -.1, 0, target, high].filter((v,i,a)=>a.indexOf(v)===i).map(v=><span key={v} style={{left:`${(v-low)/(high-low)*100}%`,transform:`translateX(${v===low?0:v===high?-100:-50}%)`}}>{v.toFixed(2)}</span>)}</div>
    <small>有符号距离 · mm · 灰色为无法确定符号</small>
  </div>;
}

export default function DesignPage({ wf }) {
  const [paintVertices, setPaintVertices] = useState(null);
  const [showContact, setShowContact] = useState(true);
  const [previewContact, setPreviewContact] = useState(null);
  const [previewThickness, setPreviewThickness] = useState(null);
  const [previewError, setPreviewError] = useState("");
  const [probe, setProbe] = useState(null);
  const previewTimer = useRef(null), previewRequest = useRef(0);
  const sculptAttempt = useRef(null);
  const session = wf.sculpt, tool = wf.designTool;
  const target = wf.params[wf.contactKind === "antagonist" ? "occlusal_target_mm" : "proximal_target_mm"];
  const painting = ["add", "remove", "smooth", "flatten", "drag"].includes(tool);
  const thicknessReference = Boolean(session?.state?.thickness_reference);

  useEffect(() => { if (!wf.busy && wf.hasOuter && !session && sculptAttempt.current !== wf.design?.id) {
      sculptAttempt.current = wf.design.id; wf.loadSculpt();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [wf.design?.id, wf.busy, wf.hasOuter, session]);
  useEffect(() => { if (wf.design && wf.hasOuter) wf.loadContacts("antagonist");
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [wf.design?.id, wf.hasOuter]);
  useEffect(() => {
    setPaintVertices(null); setPreviewContact(null); setPreviewThickness(null); setProbe(null); setPreviewError("");
    previewRequest.current++; clearTimeout(previewTimer.current);
    return () => { previewRequest.current++; clearTimeout(previewTimer.current); };
  }, [session?.source_session_sha256, wf.contactKind]);
  useEffect(() => {
    const handler = event => { const mapped = TOOL_FROM_SHORTCUT[editorShortcut(event)];
      if (mapped && !wf.busy) { wf.setDesignTool(mapped); event.preventDefault(); } };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [wf.setDesignTool, wf.busy]);

  const clearPreview = (keepVertices = false) => {
    previewRequest.current++; clearTimeout(previewTimer.current);
    if (!keepVertices) setPaintVertices(null); setPreviewContact(null); setPreviewThickness(null); setProbe(null);
  };
  const preview = next => {
    setPaintVertices(next);
    wf.markEditPreview?.();
    const request = ++previewRequest.current;
    clearTimeout(previewTimer.current);
    const wantsContact = showContact && Boolean(wf.contacts?.signed_mm);
    if (!wantsContact && !thicknessReference) return;
    const changed = next.flatMap((p,i) => p.some((v,k)=>Math.abs(v-session.vertices[i][k])>1e-7)?[i]:[]);
    if (!changed.length) return;
    const ids = changed.length <= 512 ? changed : Array.from({length:512},(_,i)=>changed[Math.floor(i*changed.length/512)]);
    // Sample existing measurement services; never run full analysis on pointer movement.
    previewTimer.current = setTimeout(async () => {
      try {
        if (wantsContact) {
          const result = await api.sculptContactPreview(wf.design.id, { vertices: next, sample_ids: ids,
            expected_input_digest: session.input_digest, source_session_sha256: session.source_session_sha256 });
          if (request === previewRequest.current && result.map) {
            const sides = Object.entries(result.map.per_side || {}).filter(([side]) => wf.contactKind === "antagonist" ? side === "antagonist" : side !== "antagonist");
            const signed = [...wf.contacts.signed_mm], valid = [...wf.contacts.valid];
            ids.forEach((id,i) => {
              const values = sides.flatMap(([,map])=>map.valid[i] && Number.isFinite(map.signed_mm[i])?[map.signed_mm[i]]:[]);
              valid[id] = values.length > 0; signed[id] = values.length ? Math.min(...values) : null;
            });
            setPreviewContact({ ...wf.contacts, signed_mm: signed, valid });
          }
        }
        if (thicknessReference) {
          const wall = await api.sculptThicknessPreview(wf.design.id, { vertices: next, sample_ids: ids,
            expected_input_digest: session.input_digest, source_session_sha256: session.source_session_sha256 });
          if (request === previewRequest.current) setPreviewThickness(wall);
        }
        if (request === previewRequest.current) setPreviewError("");
      } catch (error) { if (request === previewRequest.current) setPreviewError(`局部测量预览不可用：${error.message}`); }
    }, 180);
  };
  const vertices = paintVertices || session?.vertices || [];
  const currentContact = previewContact || wf.contacts;
  const matched = useMemo(() => currentContact?.vertices?.length === session?.vertices?.length
    && JSON.stringify(currentContact?.faces) === JSON.stringify(session?.faces),
    [currentContact?.vertices, currentContact?.faces, session?.vertices, session?.faces]);
  const contact = showContact && Number.isFinite(target) && matched ? currentContact : null;
  const editorScene = useMemo(() => session ? ({
    content: <SculptCanvas key={session.source_session_sha256} session={session} vertices={vertices}
      tool={painting ? tool : ""} brush={{...wf.brush, operation:tool}} contact={contact} contactTarget={target}
      gizmo={tool === "move" ? "translate" : tool === "rotate" ? "rotate" : null}
      busy={wf.busy} allowRemove={thicknessReference} failureFaces={wf.editFailure?.failure_face_ids || []}
      thickness={previewThickness}
      onBlocked={() => wf.setMsg("减材需要已确认的内冠厚度参考，请先完成内冠生成与检查。")}
      onPreview={preview}
      onMeasure={face => { if (tool !== "measure" || !contact) return;
        const values = session.faces[face]?.flatMap(i=>contact.valid?.[i]&&Number.isFinite(contact.signed_mm?.[i])?[contact.signed_mm[i]]:[]) || [];
        setProbe(values.length ? Math.min(...values) : null); }}
      onStroke={(centers, keys) => { clearPreview(true); wf.commitEdit({operation: keys.operation, centers, translation:keys.translation,
        radius_mm:wf.brush.radius, strength_mm:wf.brush.strength, falloff:wf.brush.falloff,
        source_session_sha256:session.source_session_sha256, expected_input_digest:session.input_digest}).finally(()=>setPaintVertices(null)); }}
      onTransform={payload => { clearPreview(true); wf.commitEdit({operation:"transform", ...payload,
        source_session_sha256:session.source_session_sha256, expected_input_digest:session.input_digest}).finally(()=>setPaintVertices(null)); }} />,
    hideScans:false, painting, captureOnlyOnMesh:true,
  }) : null, [session, vertices, tool, painting, contact, target, wf.brush, wf.busy, thicknessReference, showContact, wf.contactKind, wf.editFailure]);

  const metrics = useMemo(() => contactMetrics(wf.contacts, target), [wf.contacts, target]);
  const wall = wf.thickness?.measurement;
  const minimum = wall?.body_conservative?.min_mm ?? wall?.conservative?.min_mm ?? null;
  const previewMinimum = previewThickness?.status === "PREVIEW_ONLY" && previewThickness.distance_mm?.length
    ? Math.min(...previewThickness.distance_mm.filter(Number.isFinite)) : null;
  const format = (value, unit="mm") => Number.isFinite(value) ? `${value.toFixed(unit==="mm²"?2:3)} ${unit}` : "不可用";
  const scanToggle = (key,label) => <button type="button" className={`cd-swatch ${wf.view[key]?"on":""}`}
    aria-pressed={wf.view[key]} onClick={()=>wf.setView(v=>({...v,[key]:!v[key]}))}>{label}</button>;

  return <div className="cd-page">
    <PageHead eyebrow="STEP 5" title="牙冠微调" hint="自动适配已在生成阶段完成；这里仅检查并按需微调外表面。" />
    <div className="cd-design">
      <aside className="cd-rail">
        <Panel title="牙冠塑形" subtitle="左键笔刷 · 1 移动 / 2 旋转 · A / R / S">
          {!session&&<button className="cd-btn ghost full" disabled={wf.busy||!wf.hasOuter} onClick={wf.loadSculpt}>载入编辑器</button>}
          {session&&<div className={`cd-edit-state ${(wf.editState||"READY").toLowerCase()}`}>
            <b>{EDIT_STATE_LABEL[wf.editState] || "可编辑"}</b>
            <span>外形 {session.source_session_sha256?.slice(0,8)} · 帧 {session.frame?.revision_id?.slice?.(0,8)||"已确认"}</span>
          </div>}
          {wf.editFailure&&<div className="cd-edit-failure">
            <strong>本次修改未保存</strong>
            <p>{wf.editFailure.message || "几何检查未通过，已恢复原冠。"}</p>
            <div className="cd-failure-grid">
              <span>区域 <b>{wf.editFailure.failure_region || "未知"}</b></span>
              <span>翻面 <b>{wf.editFailure.severe_flip_count ?? 0}</b></span>
              <span>新增自交 <b>{wf.editFailure.new_self_intersection_count ?? 0}</b></span>
              <span>最大位移 <b>{Number(wf.editFailure.max_displacement_mm||0).toFixed(3)} mm</b></span>
            </div>
            <small>{wf.editFailure.suggested_action}</small>
            <button className="cd-btn ghost full" onClick={wf.clearEditFailure}>清除失败高亮</button>
          </div>}
          <ToolGrid value={tool} options={DESIGN_TOOLS} columns={3} disabled={wf.busy||!session} onChange={wf.setDesignTool} />
          <div className="cd-brush">
            <label>笔刷半径 <b>{wf.brush.radius.toFixed(1)} mm</b><input aria-label="笔刷半径" type="range" min="0.5" max="4" step="0.1" value={wf.brush.radius} disabled={wf.busy} onChange={e=>wf.setBrush(b=>({...b,radius:Number(e.target.value)}))}/></label>
            <label>笔刷强度 <b>{wf.brush.strength.toFixed(2)} mm</b><input aria-label="笔刷强度" type="range" min="0.02" max="0.3" step="0.01" value={wf.brush.strength} disabled={wf.busy} onChange={e=>wf.setBrush(b=>({...b,strength:Number(e.target.value)}))}/></label>
            <label>边缘衰减 <b>{wf.brush.falloff.toFixed(1)}</b><input aria-label="边缘衰减" type="range" min="1" max="4" step="0.1" value={wf.brush.falloff} disabled={wf.busy} onChange={e=>wf.setBrush(b=>({...b,falloff:Number(e.target.value)}))}/></label>
          </div>
          <p className="cd-hint">D 局部拖动 · Shift 切换反向操作 · Ctrl+Shift 平滑。修饰键在落笔时确定，整笔保持一致。</p>
          <div className="cd-btn-row"><button className="cd-btn ghost" disabled={wf.busy||!session} onClick={()=>{clearPreview();wf.undoEdit();}}>↶ 撤销</button><button className="cd-btn ghost" disabled={wf.busy||!session} onClick={()=>{clearPreview();wf.redoEdit();}}>↷ 重做</button></div>
          <button className="cd-btn ghost full" disabled={wf.busy||!session} onClick={()=>{clearPreview();wf.resetCrown();}}>重置牙冠外形</button>
        </Panel>
      </aside>
      {/* 有颈缘重建后的最终冠时，第 5 步直接显示它（margin-constrained crown），否则显示外形初稿 */}
      <ViewerStage wf={wf} mode={wf.hasFinal ? "final" : "outer"} renderSupport={false} showMargin={wf.marginConfirmed} editorScene={editorScene}
        legend={showContact?<Legend target={target}/>:null}
        strip={<>{scanToggle("upper","上颌")}{scanToggle("lower","下颌")}<span className="cd-swatch on">牙冠</span>
          <label className="cd-check"><input type="checkbox" checked={wf.view.scanOpacity<.4} onChange={e=>wf.setView(v=>({...v,scanOpacity:e.target.checked?.25:.62}))}/>淡化扫描</label>
          <span role="status">{wf.contactUpdating?"接触分析更新中…":paintVertices?"局部预览 · 松开后精确测量":wf.busy?"正在保存外形…":""}</span></>} />
      <aside className="cd-rail">
        <Panel title="接触分析" subtitle="负值为穿插 · 正值为间隙">
          <div className="cd-btn-row">{[["antagonist","咬合"],["neighbors","邻牙"]].map(([kind,label])=><button key={kind} className={`cd-btn ${wf.contactKind===kind?"primary":"ghost"}`} disabled={wf.busy||!wf.hasOuter} onClick={()=>{clearPreview();wf.loadContacts(kind);}}>{label}</button>)}</div>
          <Metric label="测量状态" value={wf.contacts?.status || (wf.contactUpdating?"更新中":"未测得")} note={wf.contacts?.analysis_mesh_source?`来源 ${wf.contacts.analysis_mesh_source} · ${wf.contacts.vertex_count||0} 顶点`:"只有有效有符号样本才显示穿插结论"}/>
          <label className="cd-check"><input type="checkbox" checked={showContact} onChange={e=>setShowContact(e.target.checked)}/>显示连续接触色谱</label>
          <Metric label="接触目标" value={format(target)}/>
          <Metric label="最大穿插" value={format(metrics?.penetration)} tone={metrics?.penetration>0?"bad":""}/>
          <Metric label="接触区面积" value={format(metrics?.area,"mm²")} note={`目标 ±${CONTACT_BAND_MM.toFixed(2)} mm，按有效顶点分摊面积估计`}/>
          <Metric label="最小有符号距离" value={format(metrics?.minimum)}/>
          <Metric label="有效测量顶点" value={metrics?`${metrics.validCount} / ${metrics.totalCount}`:"不可用"}/>
          {tool==="measure"&&<Metric label="指针位置距离" value={format(probe)} note="命中三角形的有效顶点最小值"/>}
          {!wf.contactUpdating&&!metrics&&<Notice tone="warn">有符号接触测量不可用，请确认坐标帧、参考牙、咬合与目标间隙。</Notice>}
          {previewError&&<Notice tone="warn">{previewError}</Notice>}
          <button className="cd-btn ghost full" disabled={wf.busy||wf.contactUpdating||!wf.hasOuter} onClick={()=>wf.loadContacts(wf.contactKind)}>重新测量</button>
        </Panel>
        <Panel title="保护与厚度">
          <Metric label="边缘线 / 颈缘" value={session?"锁定保护":"待载入"}/>
          <Metric label="最小壁厚" value={format(minimum)} note="服务端正式厚度报告；不以外冠距离替代壁厚"/>
          <Metric label="厚度参考" value={thicknessReference?"已载入内冠参考":"未建立"}/>
          {Number.isFinite(previewMinimum)&&<Metric label="当前笔刷厚度预览" value={format(previewMinimum)} note="异步采样参考，不替代正式壁厚 QC"/>}
          {!thicknessReference&&<Notice tone="warn">尚无可用内冠厚度参考，减材暂不可用。其他塑形仍需完成内冠与厚度检查。</Notice>}
          <button className="cd-btn ghost full" disabled={wf.busy||!wf.hasOuter} onClick={wf.loadThickness}>刷新厚度状态</button>
          <p className="cd-hint">笔刷仅编辑外表面。服务端验证颈缘保护和厚度；不满足条件的修改会拒绝保存。</p>
        </Panel>
      </aside>
    </div>
  </div>;
}
