export const base=import.meta.env.VITE_API_BASE||"";
export function autoFitRequest(id,path,method='GET',data){return req(`/api/designs/${id}/auto-fit/${path}`,{method,headers:{'Content-Type':'application/json'},body:data?JSON.stringify(data):undefined});}
const BASE=base;
async function req(url,opt={}){
  let r;
  try{r=await fetch(BASE+url,opt)}
  catch{
    const failure=new Error("无法连接服务器，请检查网络后重新打开病例核对结果");
    failure.network=true;
    throw failure;
  }
  if(!r.ok){
    // 导入闸门失败时 detail 是一个**对象**（含 validation 明细），不是字符串。
    // 直接把对象塞进 Error 会得到 "[object Object]"，把失败原因整个丢掉。
    let m=`${r.status} ${r.statusText}`,payload=null;
    try{const j=await r.json();
      if(j&&j.detail&&typeof j.detail==="object"){payload=j.detail;m=j.detail.message||m}
      else if(j&&j.detail)m=j.detail}catch{}
    if(Array.isArray(payload))m=payload.map(x=>`${(x.loc||[]).join(".")}: ${x.msg}`).join("；");
    const e=new Error(m);e.status=r.status;e.payload=payload;throw e}
  const ct=r.headers.get("content-type")||"";
  return ct.includes("application/json")?r.json():r;
}
const formFile=(file)=>{const f=new FormData();f.append("file",file);return f}
const formFileNamed=(file,fields)=>{const f=new FormData();f.append("file",file);
  Object.entries(fields||{}).forEach(([k,v])=>{if(v!=null&&v!=="")f.append(k,v)});return f}
export const api={
  supportContract:id=>req(`/api/designs/${id}/prep-surface-contract`),
  confirmAbutmentSupport:(id,body)=>req(`/api/designs/${id}/abutment-support/confirm`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)}),
  cases:()=>req("/api/cases"),
  deleteCase:id=>req(`/api/cases/${encodeURIComponent(id)}`,{method:'DELETE'}),
  createCase:name=>req("/api/cases",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({name})}),
  case:id=>req(`/api/cases/${id}`),
  implantCandidates:(id,source)=>req(`/api/cases/${id}/implant-candidates?source=${encodeURIComponent(source)}`),
  upload:(id,kind,file)=>req(`/api/cases/${id}/assets/${kind}`,{method:"POST",body:formFile(file)}),
  assetUrl:(id,kind)=>`${BASE}/api/cases/${id}/assets/${kind}/file`,
  templates:()=>req("/api/templates"),
  anatomy:()=>req("/api/anatomy"),
  anatomyUrl:fdi=>`${BASE}/api/anatomy/${fdi}/file`,
  templateCandidateUrl:fdi=>`${BASE}/api/templates/candidates/${fdi}/file`,
  importZip:file=>req("/api/templates/import-universal-zip",{method:"POST",body:formFile(file)}),
  uploadTemplate:(fdi,file)=>req(`/api/templates/${fdi}`,{method:"POST",body:formFile(file)}),
  templateUrl:fdi=>`${BASE}/api/templates/${fdi}/file`,
  createDesign:p=>req("/api/designs",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(p)}),
  design:id=>req(`/api/designs/${id}`),
  saveProfile:(id,p)=>req(`/api/designs/${id}/profile`,{method:'PUT',headers:{'Content-Type':'application/json'},body:JSON.stringify(p)}),
  designStatus:id=>req(`/api/designs/${id}/status`),
  generationEligibility:id=>req(`/api/designs/${id}/workflow/eligibility`),
  generationCandidate:(id,p)=>req(`/api/designs/${id}/workflow/generate`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(p)}),
  neighborUrl:(id,side)=>`${BASE}/api/designs/${id}/neighbors/${side}/file`,
  caseDesigns:id=>req(`/api/cases/${id}/designs`),
  editOuter:(id,p)=>req(`/api/designs/${id}/edit-outer`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(p)}),
  adaptOcclusal:(id,p)=>req(`/api/designs/${id}/adapt-occlusal`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(p||{})}),
  sculptSession:id=>req(`/api/designs/${id}/sculpt-session`),
  // 开发态叠加用：额外下发逐顶点语义权重（正式工作流不需要它）
  sculptSessionRegions:id=>req(`/api/designs/${id}/sculpt-session?regions=1`),
  sculptContactPreview:(id,p)=>req(`/api/designs/${id}/sculpt-contact-preview`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(p)}),
  sculptThicknessPreview:(id,p)=>req(`/api/designs/${id}/sculpt-thickness-preview`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(p)}),
  saveSculptAnatomy:(id,p)=>req(`/api/designs/${id}/sculpt-anatomy`,{method:'PUT',headers:{'Content-Type':'application/json'},body:JSON.stringify(p)}),
  outerAnalysis:(id,kind)=>req(`/api/designs/${id}/outer-analysis?kind=${encodeURIComponent(kind)}`),
  restoreOuter:(id,revision_no)=>req(`/api/designs/${id}/restore-outer`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({revision_no})}),
  autoMarginM1:(id,body)=>req(`/api/designs/${id}/auto-margin-m1`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)}),
  marginSnap:(id,body)=>req(`/api/designs/${id}/margin-snap`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)}),
  sectionPreview:(id,body)=>req(`/api/designs/${id}/section-preview`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)}),
  marginDetect:(id,body)=>req(`/api/designs/${id}/margin/detect`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)}),
  marginAddAnchor:(id,anchors,body)=>req(`/api/designs/${id}/margin/add-anchor`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({...body,anchors})}),
  marginReplaceSegment:(id,stroke,body)=>req(`/api/designs/${id}/margin/replace-segment`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({...body,stroke})}),
  marginMove:(id,delta_mm)=>req(`/api/designs/${id}/margin/move`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({delta_mm})}),
  marginSmooth:(id,strength,locked_index)=>req(`/api/designs/${id}/margin/smooth`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({strength,locked_index:locked_index||[]})}),
  marginConfirm:(id,proposal_id,operator)=>req(`/api/designs/${id}/margin/confirm`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({proposal_id:proposal_id||null,operator:operator||null})}),
  roughMargin:(id,seeds,center_hint,guide_axis)=>req(`/api/designs/${id}/rough-margin`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({seeds,center_hint,guide_axis})}),
  draftPreparation:(id,p)=>req(`/api/designs/${id}/draft-preparation`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(p)}),
  saveMargin:(id,points,operator)=>req(`/api/designs/${id}/margin${operator?`?operator=${encodeURIComponent(operator)}`:""}`,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({points})}),
  revisions:id=>req(`/api/designs/${id}/revisions`),
  setAxis:(id,axis,note,approved_by)=>req(`/api/designs/${id}/axis`,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({axis,note,approved_by})}),
  // ── 8 步流程新增的薄封装：全部指向**已存在**的后端端点，没有新后端逻辑 ──
  suggestAxis:(id,body)=>req(`/api/designs/${id}/suggest-axis`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body||{})}),
  getAxis:(id)=>req(`/api/designs/${id}/axis`),
  autoFitInputs:(id)=>req(`/api/designs/${id}/auto-fit/inputs`),
  autoFitTemplates:(id)=>req(`/api/designs/${id}/auto-fit/templates`),
  templatePreview:(id,body)=>req(`/api/designs/${id}/auto-fit/template-preview`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)}),
  selectTemplate:(id,proposal_id,body)=>req(`/api/designs/${id}/auto-fit/${proposal_id}/select-template`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)}),
  acceptCandidate:(id,proposal_id,body)=>req(`/api/designs/${id}/auto-fit/${proposal_id}/accept`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)}),
  rejectCandidate:(id,proposal_id,body)=>req(`/api/designs/${id}/auto-fit/${proposal_id}/reject`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)}),
  confirmBite:(id,body)=>req(`/api/designs/${id}/auto-fit/bite`,{method:'PUT',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)}),
  // 坐标帧 / 邻牙确认走同一个 accept 风格端点（body 需带 expected_input_digest + confirmed）
  confirmAutoFit:(id,subpath,body)=>req(`/api/designs/${id}/auto-fit/${subpath}`,{method:'PUT',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)}),
  generationProfile:(id)=>req(`/api/designs/${id}/generation-profile`),
  outer:(id,p)=>req(`/api/designs/${id}/outer`,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(p)}),
  finalize:(id,p)=>req(`/api/designs/${id}/finalize`,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(p)}),
  rebuildCervical:(id,p)=>req(`/api/designs/${id}/rebuild-cervical`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(p)}),
  manufacturingQcUrl:id=>`${BASE}/api/designs/${id}/manufacturing-qc`,
  outerUrl:id=>`${BASE}/api/designs/${id}/outer.stl`,
  finalUrl:id=>`${BASE}/api/designs/${id}/final.stl`,
  cutterUrl:id=>`${BASE}/api/designs/${id}/intaglio-cutter.stl`,
  packageUrl:id=>`${BASE}/api/designs/${id}/package.zip`,
  // ---- CrownOuter 流水线（封装层）
  crownStages:()=>req("/api/crown/stages"),
  crownCases:()=>req("/api/crown/cases"),
  crownRun:p=>req("/api/crown/runs",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(p)}),
  crownRuns:()=>req("/api/crown/runs"),
  crownJob:rid=>req(`/api/crown/runs/${rid}`),
  threeShapePreview:file=>req("/api/importers/three-shape/preview",{method:"POST",body:formFile(file)}),
  threeShapeImport:(file,caseName)=>req("/api/importers/three-shape/import",
    {method:"POST",body:formFileNamed(file,{case_name:caseName})})
};
