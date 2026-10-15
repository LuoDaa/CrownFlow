from __future__ import annotations
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4
import json, shutil, zipfile, os

from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import FileResponse, Response

from ..config import CASE_DIR, TEMPLATE_DIR, ANATOMY_DIR, AI_CHECKPOINT
from ..db import db
from ..models import (
    CaseCreate, DesignCreate, RoiRequest, MarginRequest,
    AxisRequest, AxisSetRequest, LandmarksRequest, AutoOuterRequest, FinalizeRequest, OcclusalAdaptRequest,
    AutoMarginM1Request, MarginSnapRequest, SectionPreviewRequest,
)
from ..geometry.io import load_mesh, save_mesh, mesh_info
from ..geometry.roi import spherical_roi
from ..geometry.implant_candidate import component_at_point, implant_candidates, uploaded_support_candidate
from ..importers.three_shape.order import PHI_PROPERTY_NAMES
from ..services.pipeline import (
    generate_outer, generate_initialized_outer, finalize_crown, auto_margin, auto_margin_m1,
    auto_axis, qc_from_paths
)
from ..services.ai_runtime import ai_runtime_available, predict_outer
from ..geometry.boolean import boolean_backend_name, manifold_available, earcut_available
from ..revisions import (
    DesignRevision,
    RevisionKind,
    accept_revision,
    current_margin,
    get_revision,
    record_auto_margin_candidate,
    latest_axis_candidate,
    record_axis,
    record_manual_margin,
    revision_chain,
)

#: 写入每条修订的算法版本，供审计链追溯。
ALGORITHM_VERSION = "crownflow/0.2.0+commit1"

from ..services.design_dependencies import dependency_digest, file_digest, publish_artifacts, require_current_outer

router=APIRouter(prefix="/api")
UNIVERSAL_TO_FDI={
    1:18,2:17,3:16,4:15,5:14,6:13,7:12,8:11,
    9:21,10:22,11:23,12:24,13:25,14:26,15:27,16:28,
    17:38,18:37,19:36,20:35,21:34,22:33,23:32,24:31,
    25:41,26:42,27:43,28:44,29:45,30:46,31:47,32:48
}
VALID_FDI=set(UNIVERSAL_TO_FDI.values())
VALID_ASSET={"upper","lower","support"}

def now(): return datetime.now(timezone.utc).isoformat()

def case_exists(case_id):
    with db() as c:
        return c.execute("SELECT 1 FROM cases WHERE id=?",(case_id,)).fetchone() is not None

def case_has_designs(case_id):
    with db() as c:
        return c.execute("SELECT 1 FROM designs WHERE case_id=? LIMIT 1",(case_id,)).fetchone() is not None

def case_has_geometry(case_id):
    with db() as c:
        return c.execute('SELECT 1 FROM designs WHERE case_id=? AND (margin_json IS NOT NULL OR axis_json IS NOT NULL OR outer_path IS NOT NULL) LIMIT 1',(case_id,)).fetchone() is not None

def asset(case_id,kind):
    with db() as c:
        r=c.execute("""SELECT * FROM assets WHERE case_id=? AND kind=?
                       ORDER BY created_at DESC LIMIT 1""",(case_id,kind)).fetchone()
    return dict(r) if r else None

def design_row(design_id):
    with db() as c:
        r=c.execute("SELECT * FROM designs WHERE id=?",(design_id,)).fetchone()
    if not r: raise HTTPException(404,"设计不存在")
    return dict(r)

def require_draft_ready(d):
    from ..services.abutment_support import indication_of
    if indication_of(d) in ('CUSTOM_IMPLANT_ABUTMENT','SCREW_RETAINED_CROWN'):
        raise HTTPException(422,{'status':'UNSUPPORTED_INDICATION','rejection_code':'CUSTOM_ABUTMENT_NOT_SUPPORTED' if indication_of(d)=='CUSTOM_IMPLANT_ABUTMENT' else 'UNSUPPORTED_INDICATION'})
    if d['support_type']=='abutment':
        support=asset(d['case_id'],'support')
        details=json.loads(support['mesh_info_json'] or '{}') if support else {}
        if details.get('implant_candidate') and not details.get('restorative_abutment_confirmed'):
            raise HTTPException(409,'仅定位了种植组件候选；扫描杆不能直接构冠，请先确认最终修复基台')
    if json.loads(d['parameters_json']).get('draft'):
        if not d.get('profile_json'): raise HTTPException(409,'请先确认设计配置')
        if not asset(d['case_id'],'support'): raise HTTPException(409,'请先确认目标基牙并提取 support')

@router.get("/health")
def health():
    return {"ok":True}

@router.get("/system/capabilities")
def capabilities():
    ai_path=Path(AI_CHECKPOINT) if AI_CHECKPOINT else None
    return {
        "template_generator":True,
        "ai_generator_available":ai_runtime_available(ai_path),
        "ai_checkpoint":str(ai_path) if ai_path and ai_path.exists() else None,
        "final_crown_pipeline":True,
        "automatic_margin_from_open_boundary":True,
        "automatic_insertion_axis":True,
        "boolean_engine":boolean_backend_name(),
        # 精确布尔与裁切盖片是功能依赖。缺失时 finalize 会明确失败而不是静默降级。
        "exact_boolean_available":manifold_available(),
        "crown_cap_available":earcut_available(),
        "voxel_fallback_is_debug_only":True,
        "parameter_system":"lab_profile + resolver (PROVENANCE_REQUIRED)",
    }

@router.post("/cases")
def create_case(p:CaseCreate):
    cid=uuid4().hex
    (CASE_DIR/cid).mkdir(parents=True,exist_ok=True)
    with db() as c:
        c.execute("INSERT INTO cases(id,name,created_at) VALUES(?,?,?)",(cid,p.name,now()))
    return {"id":cid,"name":p.name}

@router.get("/cases")
def list_cases():
    with db() as c:
        rows=c.execute("SELECT * FROM cases ORDER BY created_at DESC").fetchall()
    return [dict(x) for x in rows]

@router.get("/cases/{case_id}")
def get_case(case_id:str):
    with db() as c:
        cr=c.execute("SELECT * FROM cases WHERE id=?",(case_id,)).fetchone()
        if not cr: raise HTTPException(404,"病例不存在")
        ars=c.execute("SELECT * FROM assets WHERE case_id=? ORDER BY created_at",(case_id,)).fetchall()
        drs=c.execute("SELECT * FROM designs WHERE case_id=? ORDER BY created_at DESC",(case_id,)).fetchall()
    return {"case":dict(cr),"assets":[dict(x) for x in ars],"designs":[dict(x) for x in drs]}

@router.delete("/cases/{case_id}")
def delete_case(case_id:str):
    root=CASE_DIR.resolve()
    directory=CASE_DIR/case_id
    if directory.resolve().parent!=root or directory.is_symlink() or directory.is_junction():
        raise HTTPException(400,"病例目录无效，未执行删除")
    with db() as c:
        c.execute('BEGIN IMMEDIATE')
        if not c.execute('SELECT 1 FROM cases WHERE id=?',(case_id,)).fetchone():
            raise HTTPException(404,"病例不存在")
        # Cascade removes designs and their revisions; file failures must never report success.
        c.execute("DELETE FROM cases WHERE id=?",(case_id,))
        try:
            if directory.exists(): shutil.rmtree(directory)
        except OSError as exc:
            c.rollback()
            raise HTTPException(500,"病例文件删除未完成，请关闭占用文件后重试") from exc
    return {"deleted":case_id}

@router.post("/cases/{case_id}/assets/{kind}")
async def upload_asset(case_id:str,kind:str,file:UploadFile=File(...)):
    if kind not in VALID_ASSET: raise HTTPException(400,"kind 必须为 upper/lower/support")
    if not case_exists(case_id): raise HTTPException(404,"病例不存在")
    if (case_has_geometry(case_id) if kind=='support' else case_has_designs(case_id)): raise HTTPException(409,"该病例已有设计几何；请新建病例后再更换扫描数据")
    suffix=Path(file.filename or "").suffix.lower()
    if suffix not in {".stl", ".dcm"}: raise HTTPException(400,"请上传 STL 或 3Shape DCM 网格")
    base=CASE_DIR/case_id/f"{kind}_{uuid4().hex}"
    incoming=base.with_suffix(suffix)
    path=base.with_suffix(".stl")
    with incoming.open("wb") as f: shutil.copyfileobj(file.file,f)
    try:
        mesh=load_mesh(incoming)
        info=mesh_info(mesh)
        if suffix==".dcm": save_mesh(mesh,path)
    except Exception as e:
        incoming.unlink(missing_ok=True); raise HTTPException(400,f"扫描文件无效: {e}")
    if suffix==".dcm": incoming.unlink(missing_ok=True)
    aid=uuid4().hex
    with db() as c:
        c.execute("""INSERT INTO assets(id,case_id,kind,path,original_name,mesh_info_json,created_at)
                     VALUES(?,?,?,?,?,?,?)""",
                  (aid,case_id,kind,str(path),file.filename or "",json.dumps(info),now()))
    return {"asset_id":aid,"kind":kind,"mesh":info}

@router.get("/cases/{case_id}/assets/{kind}/file")
def get_asset_file(case_id:str,kind:str):
    a=asset(case_id,kind)
    if not a: raise HTTPException(404,"资产不存在")
    return FileResponse(a["path"],media_type="model/stl",filename=f"{kind}.stl")

@router.post("/cases/{case_id}/extract-support")
def extract_support(case_id:str,p:RoiRequest):
    if case_has_geometry(case_id): raise HTTPException(409,"该病例已有设计几何；请新建病例后再更换 Support")
    return save_extracted_support(case_id,p)

@router.get('/cases/{case_id}/implant-candidates')
def list_implant_candidates(case_id:str,source:str='lower'):
    if not case_exists(case_id): raise HTTPException(404,'病例不存在')
    if source not in ('upper','lower'): raise HTTPException(400,'source 必须是 upper 或 lower')
    scan=asset(case_id,source)
    if not scan: raise HTTPException(400,'请先上传工作颌扫描')
    opposite=asset(case_id,'upper' if source=='lower' else 'lower')
    try:
        jaw=load_mesh(Path(scan['path']))
        opposing=load_mesh(Path(opposite['path'])) if opposite else None
        candidates=implant_candidates(jaw,opposing)
        supplied=asset(case_id,'support')
        if not candidates and supplied and not supplied['original_name'].startswith('extracted_'):
            candidates=[uploaded_support_candidate(load_mesh(Path(supplied['path'])),jaw,opposing)]
    except ValueError as exc:
        raise HTTPException(422,str(exc)) from exc
    status=('NOT_FOUND' if not candidates else
            'CANDIDATES_FOUND' if len(candidates)==1 and candidates[0][1]['confidence']=='high'
            else 'REVIEW_REQUIRED')
    return {'status':status,'source':source,'candidates':[summary for _,summary in candidates],
            'classification_scope':'geometric_implant_component_only',
            'restorative_abutment_verified':False,
            'margin_available':False}


def save_support_mesh(case_id,mesh,original_name,metadata=None):
    path=CASE_DIR/case_id/f"support_{uuid4().hex}.stl"
    save_mesh(mesh,path)
    info=mesh_info(mesh)
    if metadata: info.update(metadata)
    aid=uuid4().hex
    with db() as c:
        c.execute("""INSERT INTO assets(id,case_id,kind,path,original_name,mesh_info_json,created_at)
                     VALUES(?,?,?,?,?,?,?)""",
                  (aid,case_id,"support",str(path),original_name,json.dumps(info),now()))
    return {"asset_id":aid,"mesh":info}

def save_extracted_support(case_id:str,p:RoiRequest):
    src=asset(case_id,p.source)
    if not src: raise HTTPException(404,f"没有 {p.source} STL")
    try:
        picked=component_at_point(load_mesh(Path(src["path"])),p.center)
        roi=spherical_roi(picked,p.center,p.radius_mm)
        return save_support_mesh(case_id,roi,f"extracted_{p.source}_roi.stl",
                                 {'picked_component_faces':int(len(picked.faces))})
    except Exception as e: raise HTTPException(400,str(e))

@router.post("/templates/import-universal-zip")
async def import_universal_zip(file:UploadFile=File(...)):
    if not file.filename.lower().endswith(".zip"): raise HTTPException(400,"请上传 ZIP")
    tmp=TEMPLATE_DIR/f"_tmp_{uuid4().hex}.zip"
    with tmp.open("wb") as f: shutil.copyfileobj(file.file,f)
    imported=[];staged=[];invalid=[]
    try:
        with zipfile.ZipFile(tmp) as z:
            names=z.namelist()
            # P0-14（ZIP 同样）：**全部**条目先落临时文件并校验完，再逐个原子替换；
            # 某个牙位无效只跳过它，该牙位的原模板不动。
            for u,fdi in UNIVERSAL_TO_FDI.items():
                want=f"tooth_{u}.stl"
                candidates=[n for n in names if n.replace("\\","/").lower().endswith("/"+want) or n.lower()==want]
                if not candidates: continue
                staged_path=TEMPLATE_DIR/f"_tmp_zip_{uuid4().hex}.stl"
                staged_path.write_bytes(z.read(candidates[0]))
                try: info=mesh_info(load_mesh(staged_path))
                except Exception as e:
                    staged_path.unlink(missing_ok=True);invalid.append({"universal":u,"fdi":fdi,"reason":str(e)});continue
                staged.append((staged_path,u,fdi,info))
        for staged_path,u,fdi,info in staged:
            os.replace(staged_path,TEMPLATE_DIR/f"FDI_{fdi}.stl")
            imported.append({"universal":u,"fdi":fdi,"mesh":info})
    finally:
        tmp.unlink(missing_ok=True)
        for staged_path,_,_,_ in staged: staged_path.unlink(missing_ok=True)
    if not imported: raise HTTPException(400,{"message":"没有找到 Tooth_1.stl ... Tooth_32.stl","invalid":invalid})
    return {"count":len(imported),"imported":imported,"rejected":invalid}
    return {"count":len(imported),"imported":imported}


@router.post("/templates/{fdi}")
async def upload_template(fdi:int,file:UploadFile=File(...)):
    if fdi not in VALID_FDI: raise HTTPException(400,"FDI 不合法")
    if not file.filename.lower().endswith(".stl"): raise HTTPException(400,"模板必须是 STL")
    # P0-14：先落**临时文件**并校验，通过后才原子替换正式模板；
    # 校验失败只删临时文件，正式模板（及其字节）保持不动。
    tmp=TEMPLATE_DIR/f"_tmp_upload_{uuid4().hex}.stl"
    with tmp.open("wb") as f: shutil.copyfileobj(file.file,f)
    try: info=mesh_info(load_mesh(tmp))
    except Exception as e:
        tmp.unlink(missing_ok=True); raise HTTPException(400,f"模板无效: {e}")
    os.replace(tmp,TEMPLATE_DIR/f"FDI_{fdi}.stl")
    return {"fdi":fdi,"mesh":info}

@router.get("/templates")
def list_templates():
    out=[]
    for p in sorted(TEMPLATE_DIR.glob("FDI_*.stl")):
        try:
            fdi=int(p.stem.split("_")[1])
            out.append({"fdi":fdi,"file":p.name,"mesh":mesh_info(load_mesh(p))})
        except Exception: pass
    return out

@router.get("/anatomy")
def list_anatomy():
    return [{"fdi":fdi,"file":f"FDI_{fdi}.stl"} for fdi in sorted(VALID_FDI)
            if (ANATOMY_DIR/f"FDI_{fdi}.stl").is_file()]

@router.get("/anatomy/{fdi}/file")
def anatomy_file(fdi:int):
    if fdi not in VALID_FDI: raise HTTPException(400,"FDI 不合法")
    path=ANATOMY_DIR/f"FDI_{fdi}.stl"
    if not path.is_file(): raise HTTPException(404,"解剖参考不存在")
    return FileResponse(path,media_type="model/stl",filename=path.name)

@router.get("/templates/{fdi}/file")
def template_file(fdi:int):
    p=TEMPLATE_DIR/f"FDI_{fdi}.stl"
    if not p.exists(): raise HTTPException(404,"模板不存在")
    return FileResponse(p,media_type="model/stl",filename=p.name)

@router.post("/designs")
def create_design(p:DesignCreate):
    indication=p.indication or ('CROWN_ON_EXISTING_ABUTMENT' if p.support_type=='abutment' else 'NATURAL_PREP')
    if (indication in ('CROWN_ON_EXISTING_ABUTMENT','CUSTOM_IMPLANT_ABUTMENT','SCREW_RETAINED_CROWN'))!=(p.support_type=='abutment'):
        raise HTTPException(422,'indication 与 support_type 不一致')
    if p.target_fdi not in VALID_FDI: raise HTTPException(400,"FDI 不合法")
    if not case_exists(p.case_id): raise HTTPException(404,"病例不存在")
    if not p.draft and not asset(p.case_id,"support"): raise HTTPException(400,"必须先上传或提取 Prep/Abutment support STL")
    if not p.draft and not (TEMPLATE_DIR/f"FDI_{p.target_fdi}.stl").exists():
        raise HTTPException(400,"当前牙位没有模板；Template CAD 与当前 AI baseline 都需要对应 FDI 模板")
    if p.generator_mode=="ai" and not ai_runtime_available(Path(AI_CHECKPOINT) if AI_CHECKPOINT else None):
        raise HTTPException(400,"没有可用的真实 AI TorchScript checkpoint / PyTorch runtime")
    if p.generator_mode=="ai" and (not asset(p.case_id,"upper") or not asset(p.case_id,"lower")):
        raise HTTPException(400,"AI 模式需要已配准的上下颚 STL")
    did=uuid4().hex; ts=now()
    params=p.model_dump()
    params['indication']=indication
    with db() as c:
        c.execute("""INSERT INTO designs(id,case_id,target_fdi,support_type,generator_mode,
                     parameters_json,status,created_at,updated_at)
                     VALUES(?,?,?,?,?,?,?,?,?)""",
                  (did,p.case_id,p.target_fdi,p.support_type,p.generator_mode,
                   json.dumps(params),"created",ts,ts))
    with db() as c:
        source=c.execute('SELECT order_json FROM cases WHERE id=?',(p.case_id,)).fetchone()
        c.execute('UPDATE designs SET order_source_json=?,indication=? WHERE id=?',(source['order_json'],indication,did))
    return {"id":did,**params,'indication':indication}

#: HTTP 响应瘦身阈值：`qc_json` 里的**调试数组**（面 id / 顶点 id）与超长字符串
#: 会把单个 design JSON 撑到 1.6MB，客户端每步操作都要重拉它 ⇒ 别的电脑上看起来"卡住/没反应"。
#: 只把超过阈值的数组/字符串换成占位符（**保留全部判定与摘要字段**），并如实列出省略路径；
#: 完整内容仍可用 `?full=1` 或 `artifact_paths` 里的审计文件取回。
_QC_INLINE_ARRAY_LIMIT = 512
_QC_INLINE_STRING_LIMIT = 20000


#: 响应里**只下发摘要**的内部诊断子树（完整内容见 `?full=1` 与 `artifact_paths` 的审计文件）。
#: 前端一处都不读这些子树；它们是把响应撑到 1.6MB 的主因（`blockout` 里还内含一份完整 QC 副本）。
_RESPONSE_SUMMARY_SUBTREES = (
    "$qc.blockout",
    "$qc.provenance.prepared_support.geometry",
    "$qc.provenance.prepared_support.diagnostics",
    "$qc.provenance.prepared_support.report",
    "$qc.provenance.prepared_support.audit",
)


def _summarize_subtrees(parsed, omitted):
    """把上面列出的子树换成占位符（**保留键与体量**，不静默删除）。"""
    for path in _RESPONSE_SUMMARY_SUBTREES:
        keys = path.split(".")[1:]
        node = parsed
        for key in keys[:-1]:
            node = node.get(key) if isinstance(node, dict) else None
            if node is None:
                break
        if not isinstance(node, dict) or keys[-1] not in node:
            continue
        value = node[keys[-1]]
        size = len(json.dumps(value, ensure_ascii=False))
        node[keys[-1]] = {"_omitted": {"kind": "subtree", "bytes": size,
                                      "keys": sorted(value.keys())[:40] if isinstance(value, dict) else None}}
        omitted.append({"path": path, "kind": "subtree", "bytes": size})
    return parsed


def _trim_payload_value(value, path="$", omitted=None):
    """递归细筛：**只**替换超长数组/字符串，其余原样保留（不删字段、不改数值）。"""
    omitted = omitted if omitted is not None else []
    if isinstance(value, dict):
        return {key: _trim_payload_value(item, path + "." + str(key), omitted)
                for key, item in value.items()}
    if isinstance(value, list):
        if len(value) > _QC_INLINE_ARRAY_LIMIT:
            omitted.append({"path": path, "kind": "array", "length": len(value)})
            return {"_omitted": {"kind": "array", "length": len(value),
                                 "sample": _trim_payload_value(value[0], path + "[0]", omitted)}}
        return [_trim_payload_value(item, "%s[%d]" % (path, index), omitted)
                for index, item in enumerate(value)]
    if isinstance(value, str) and len(value) > _QC_INLINE_STRING_LIMIT:
        omitted.append({"path": path, "kind": "string", "length": len(value)})
        return "<omitted: %d chars>" % len(value)
    return value


def _trim_design_response(row):
    """给 HTTP 响应瘦身（只动 `qc_json`），并注明省略了什么。"""
    raw = row.get("qc_json")
    if not raw:
        return row
    try:
        parsed = json.loads(raw)
    except (TypeError, ValueError):
        return row
    omitted = []
    trimmed = _summarize_subtrees(parsed, omitted)
    trimmed = _trim_payload_value(trimmed, "$qc", omitted)
    if not omitted:
        return row
    trimmed["_response_trim"] = {
        "note": "HTTP 响应省略了超长的调试数组/字符串（判定与摘要字段完整）；"
                "完整内容见 ?full=1 与 artifact_paths 指向的审计文件",
        "omitted_count": len(omitted), "omitted": omitted[:50]}
    return {**row, "qc_json": json.dumps(trimmed, ensure_ascii=False)}


@router.get("/designs/{design_id}")
def get_design(design_id:str, full:int=0):
    row=design_row(design_id)
    return row if full else _trim_design_response(row)

@router.post("/designs/{design_id}/suggest-margin")
def suggest_margin(design_id:str):
    """生成自动 Margin 提案。

    **不会静默覆盖人工结果。** 若当前生效的 Margin 含人工成分，本接口只记录一条
    ``margin_candidate`` 修订，当前生效版本不变；响应里的 ``requires_decision`` 为
    ``True``，由调用方决定 Replace / Compare / Cancel。
    """
    d=design_row(design_id); s=asset(d["case_id"],"support")
    if not s: raise HTTPException(400,"请先选择并提取目标基牙")
    if s["original_name"].startswith("extracted_"):
        raise HTTPException(400,"局部截取边界不是真实基台边缘线；请在 3D 中手工标注 Margin")
    try: pts=auto_margin(Path(s["path"]))
    except Exception as e: raise HTTPException(400,str(e))
    revision, applied = record_auto_margin_candidate(
        design_id, pts, algorithm_version=ALGORITHM_VERSION, note="open-boundary loop suggestion")
    chain = revision_chain(design_id)
    return {
        "points": pts,
        "applied": applied,
        "requires_decision": chain["requires_decision"],
        "candidate_revision_no": revision.revision_no,
        "current_margin_revision_no": chain["current_margin_revision_no"],
        "message": (
            "自动提案已生效" if applied
            else "当前生效的 Margin 含人工修改，本次自动提案已记录为候选修订，"
                 "请选择替换 / 比较 / 取消"
        ),
    }

@router.post("/designs/{design_id}/auto-margin-m1")
def auto_margin_m1_candidate(design_id:str,p:AutoMarginM1Request):
    """M1 自动 Margin（工作台）：代型网格 + 就位轴 → 闭环，可选凸/凹特征吸附。

    与 ``/suggest-margin``（开放边界启发式）的区别：这里跑的是 ``geometry/auto_margin``
    冻结配置 v2 的完整引擎（候选带 → 扇区 Top-K → 环状 DP → 平滑 → 投影 → QC）。

    **fail-loud**：引擎状态非 ``AUTO_ACCEPTED`` 时不返回 points，响应里带
    ``status`` / ``reason``，调用方应回退截面候选或手动描绘；不会静默给一条曲线。
    与 ``/suggest-margin`` 一样，**不覆盖人工边缘**：人工结果存在时只记录候选修订。
    """
    d=design_row(design_id); s=asset(d["case_id"],"support")
    if not s: raise HTTPException(400,"请先选择并提取目标基牙")
    if s["original_name"].startswith("extracted_"):
        raise HTTPException(400,"局部截取边界不是真实基台边缘线；请在 3D 中手工标注 Margin")
    try:
        result=auto_margin_m1(Path(s["path"]), axis=p.axis, mode=p.mode,
                              radius_mm=p.radius_mm, apply_snap=p.apply_snap)
    except Exception as e:
        raise HTTPException(400,f"M1 自动边缘失败：{e}") from e
    applied=None; revision=None
    if result["points"]:
        note=("M1 v2 auto margin; status=%s; config=%s; axis=%s; snap=%s"
              % (result["status"], result["config_id"], result["axis_source"],
                 (result["snap"] or {}).get("status")))
        revision, applied = record_auto_margin_candidate(
            design_id, result["points"],
            algorithm_version="welooai/auto_margin_m1v2/%s" % (result["config_id"] or "unknown"),
            note=note)
    chain = revision_chain(design_id)
    return {
        "points": result["points"],
        "status": result["status"],
        "reason": result["reason"],
        "config_id": result["config_id"],
        "solver": result["solver"],
        "quality": result["quality"],
        "confidence": result["confidence"],
        "candidate_rank_stats": result["candidate_rank_stats"],
        "axis": result["axis"],
        "axis_source": result["axis_source"],
        "mode": result["mode"],
        "radius_mm": result["radius_mm"],
        "snap": result["snap"],
        "elapsed_s": result["elapsed_s"],
        "applied": applied,
        "requires_decision": chain["requires_decision"],
        "candidate_revision_no": (revision.revision_no if revision is not None else None),
        "current_margin_revision_no": chain["current_margin_revision_no"],
        "message": (
            "M1 自动提案已生效，请逐面复核" if applied
            else ("M1 未给出可信闭环（%s），请用截面候选或手动描绘" % result["reason"]
                  if not result["points"]
                  else "当前生效的 Margin 含人工修改，M1 提案已记录为候选修订")
        ),
    }

@router.post("/designs/{design_id}/margin-snap")
def margin_snap(design_id:str,p:MarginSnapRequest):
    """把边缘线按模式吸到**凸棱**(SUPRA) 或**凹沟**(SUB)。

    对应 exocad `Detect / edit margins` 的 Supragingival / Subgingival：
    磁性线走凸曲线还是凹曲线。纯局部后处理；特征面占比不足时返回
    ``REVIEW_REQUIRED`` 且**原样返回输入点**（不假装吸附过）。
    """
    d=design_row(design_id); s=asset(d["case_id"],"support")
    if not s: raise HTTPException(400,"请先选择并提取目标基牙")
    from ..geometry.margin_feature_snap import snap_to_feature
    mesh=load_mesh(Path(s["path"]))
    axis=p.axis
    if axis is None:
        chain_axis=latest_axis_candidate(design_id) or current_margin(design_id)
        axis=None
    try:
        res=snap_to_feature(mesh.triangles, p.points, axis, p.mode, p.radius_mm)
    except Exception as e:
        raise HTTPException(400,f"特征吸附失败：{e}") from e
    if res["status"]!="APPLIED":
        return {**res,"points":p.points,"applied":False}
    return {**res,"applied":True}

@router.post("/designs/{design_id}/section-preview")
def section_preview(design_id:str,p:SectionPreviewRequest):
    """悬停剖面气泡：过给定表面点、包含就位轴的一张剖面折线。

    返回 3D 折线（多段）与 2D 坐标（u=轴方向，v=径向），供前端小窗直接画。
    只做几何，不改任何状态。
    """
    import numpy as np
    d=design_row(design_id); s=asset(d["case_id"],"support")
    if not s: raise HTTPException(400,"请先选择并提取目标基牙")
    point=np.asarray(p.point,float)
    if point.shape!=(3,) or not np.isfinite(point).all():
        raise HTTPException(400,"无效的剖面点")
    axis=None
    try:
        from ..geometry.insertion import suggest_insertion_axis
        axis=np.asarray(suggest_insertion_axis(load_mesh(Path(s["path"])),None),float)
    except Exception:
        axis=None
    if axis is None or not np.isfinite(axis).all() or float(np.linalg.norm(axis))<1e-9:
        axis=np.array([0.0,0.0,1.0])
    axis=axis/float(np.linalg.norm(axis))
    normal=np.asarray(p.normal,float) if p.normal is not None else None
    if normal is None or float(np.linalg.norm(normal))<1e-9:
        # 平面法向 = 轴 × 径向 → 平面包含轴与该点（颈缘在这个剖面上最清楚）
        radial=point-float(point@axis)*axis
        if float(np.linalg.norm(radial))<1e-6:
            radial=np.cross(axis,np.array([1.0,0.0,0.0]))
        n=np.cross(axis,radial)
        if float(np.linalg.norm(n))<1e-9:
            n=np.cross(axis,np.array([0.0,1.0,0.0]))
        normal=n
    normal=normal/float(np.linalg.norm(normal))
    mesh=load_mesh(Path(s["path"]))
    try:
        section=mesh.section(plane_origin=point,plane_normal=normal)
    except Exception as e:
        raise HTTPException(400,f"剖面失败：{e}") from e
    if section is None:
        return {"polylines_3d":[],"polylines_uv":[],"plane":{"origin":point.tolist(),"normal":normal.tolist()},
                "basis":{"u":axis.tolist(),"v":np.cross(normal,axis).tolist()},"span_mm":p.span_mm}
    u=axis; v=np.cross(normal,axis); v=v/float(np.linalg.norm(v))
    polys3d, polys_uv = [], []
    for entity in section.discrete:
        arr=np.asarray(entity,float)
        if len(arr)<2: continue
        poly3d=[[float(a) for a in row] for row in arr]
        rel=arr-point
        uv=[[float(a@u),float(a@v)] for a in rel]
        polys3d.append(poly3d); polys_uv.append(uv)
    return {"polylines_3d":polys3d,"polylines_uv":polys_uv,
            "plane":{"origin":point.tolist(),"normal":normal.tolist()},
            "basis":{"u":u.tolist(),"v":v.tolist()},"span_mm":p.span_mm}

def _validated_margin_points(raw):
    """Basic geometry gate for new and re-confirmed loops; existing revisions remain readable."""
    import numpy as np
    try: points=np.asarray(raw,float)
    except (TypeError,ValueError) as e: raise HTTPException(400,"Margin 必须是有限 XYZ 点") from e
    if points.ndim!=2 or points.shape[1]!=3 or not np.isfinite(points).all():
        raise HTTPException(400,"Margin 必须是有限 XYZ 点")
    if len(points)>12 and np.linalg.norm(points[0]-points[-1])<1e-6:
        points=points[:-1]
    if len(points)<12: raise HTTPException(400,"Margin 至少需要 12 个有效边缘点，三点不能形成牙颈缘")
    if len(points)>512: raise HTTPException(400,"Margin 点数超过 512，请先简化边缘线")
    if len({tuple(np.round(point,5)) for point in points})!=len(points):
        raise HTTPException(400,"Margin 有重复点，请沿颈缘顺序描绘")
    if np.min(np.linalg.norm(np.roll(points,-1,axis=0)-points,axis=1))<.02:
        raise HTTPException(400,"Margin 有过近的相邻点，请沿颈缘均匀描绘")
    center=points.mean(axis=0)
    if np.linalg.matrix_rank(points-center)<2: raise HTTPException(400,"Margin 点列退化为直线")
    _,_,basis=np.linalg.svd(points-center,full_matrices=False)
    xy=(points-center)@basis[:2].T
    next_xy=np.roll(xy,-1,axis=0)
    area=abs(np.sum(xy[:,0]*next_xy[:,1]-next_xy[:,0]*xy[:,1]))/2
    if area<1: raise HTTPException(400,"Margin 围成范围过小，请沿完整牙颈缘描绘")
    def turn(a,b,c):return (b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0])
    def on_segment(a,b,c):return all(min(a[k],b[k])-1e-8<=c[k]<=max(a[k],b[k])+1e-8 for k in (0,1))
    for i in range(len(xy)):
        for j in range(i+2,len(xy)):
            if i==0 and j==len(xy)-1:continue
            a,b,c,d=xy[i],next_xy[i],xy[j],next_xy[j]
            t1,t2,t3,t4=turn(a,b,c),turn(a,b,d),turn(c,d,a),turn(c,d,b)
            if ((t1*t2<0 and t3*t4<0) or
                (abs(t1)<1e-8 and on_segment(a,b,c)) or
                (abs(t2)<1e-8 and on_segment(a,b,d)) or
                (abs(t3)<1e-8 and on_segment(c,d,a)) or
                (abs(t4)<1e-8 and on_segment(c,d,b))):
                raise HTTPException(400,"Margin 线段交叉，请按顺序沿颈缘闭合")
    return points.tolist()

@router.post("/designs/{design_id}/margin")
def save_margin(design_id:str,p:MarginRequest,operator:str|None=None):
    """保存按颈缘顺序排列的完整闭环；历史短线仍可读取，但不能重新确认。"""
    design=design_row(design_id)
    support=asset(design['case_id'],'support')
    if design['support_type']=='abutment' and support:
        details=json.loads(support['mesh_info_json'] or '{}')
        if details.get('implant_candidate') and not details.get('restorative_abutment_confirmed'):
            raise HTTPException(409,'仅定位了种植组件候选；请先确认是实际最终修复基台，扫描杆不能画作冠边缘')
    points=_validated_margin_points(p.points)
    prior = current_margin(design_id)
    prior_source = prior.margin_source if prior is not None else None
    revision = record_manual_margin(
        design_id, points, operator=operator,
        note="manual margin set via API", prior_source=prior_source)
    return {"points":points,"revision_no":revision.revision_no,
            "margin_source":revision.margin_source.value}

@router.get("/designs/{design_id}/revisions")
def list_design_revisions(design_id:str):
    design_row(design_id)
    return revision_chain(design_id)

@router.get("/designs/{design_id}/revisions/{revision_no}")
def get_design_revision(design_id:str,revision_no:int):
    design_row(design_id)
    r = get_revision(design_id, revision_no)
    if r is None: raise HTTPException(404,f"修订 #{revision_no} 不存在")
    return r.as_dict()

@router.post("/designs/{design_id}/revisions/{revision_no}/accept")
def accept_design_revision(design_id:str,revision_no:int,approved_by:str|None=None):
    """把某个修订（通常是候选提案）提升为当前生效版本。"""
    design_row(design_id)
    candidate=get_revision(design_id,revision_no)
    if candidate and candidate.kind is RevisionKind.MARGIN_CANDIDATE:
        _validated_margin_points(candidate.payload.get('points',[]))
    try:
        r = accept_revision(design_id, revision_no, approved_by=approved_by)
    except KeyError as e:
        raise HTTPException(404,str(e)) from e
    except ValueError as e:
        raise HTTPException(409,str(e)) from e
    return {"revision_no":r.revision_no,"kind":r.kind.value,
            "margin_source":r.margin_source.value if r.margin_source else None,
            "approved_by":r.approved_by}

@router.post("/designs/{design_id}/suggest-axis")
def suggest_axis(design_id:str,p:AxisRequest):
    """求解就位方向，返回完整结果（状态 / 置信度 / 倒凹指标 / 诊断）。

    状态为 REVIEW_REQUIRED 时调用方**必须**让用户确认，不得自动采纳 ——
    求解器能表达"没有好方向"，前端不能把它当成功。
    """
    d=design_row(design_id); s=asset(d["case_id"],"support")
    if not s: raise HTTPException(400,"请先选择并提取目标基牙")
    margin=p.margin_points
    if not margin and d.get("margin_json"): margin=json.loads(d["margin_json"])
    try: result=auto_axis(Path(s["path"]),margin)
    except Exception as e: raise HTTPException(400,str(e))
    payload=result.as_dict()
    # 自动求解只登记为候选；采纳与否由用户在 /axis 端点显式决定。
    record_axis(design_id, payload["axis"], source="auto",
                algorithm_version=ALGORITHM_VERSION, result_payload=payload)
    return payload

@router.post("/designs/{design_id}/axis")
def set_axis(design_id:str,p:AxisSetRequest):
    """人工确认或修改就位方向 —— 写入 DesignRevision（axis_accepted）。

    不允许只更新数据库里的一个向量：审计链必须记录谁在什么时候把方向改成了什么，
    以及来源是自动确认还是人工调整。
    """
    design_row(design_id)
    axis=[float(v) for v in p.axis]
    if len(axis)!=3 or not all(abs(v)<1e300 for v in axis) or not any(abs(v)>1e-9 for v in axis):
        raise HTTPException(400,"axis 必须是 3 个非零有限数")
    candidate = latest_axis_candidate(design_id)
    rev = record_axis(design_id, axis, source="manual",
                      algorithm_version=ALGORITHM_VERSION,
                      note=p.note, approved_by=p.approved_by,
                      based_on_candidate=(candidate.revision_no if candidate else None),
                      based_on_solver=(candidate.payload.get("solver") if candidate else None),
                      based_on_reasons=(candidate.payload.get("solver_reasons") if candidate else None))
    return {"axis":axis,"revision_no":rev.revision_no,"kind":rev.kind.value,
            "based_on_candidate_revision_no":(rev.payload.get("basis") or {}).get("candidate_revision_no")}

@router.get("/designs/{design_id}/axis")
def get_axis(design_id:str):
    d=design_row(design_id)
    return {"axis":json.loads(d["axis_json"]) if d.get("axis_json") else None}

@router.post("/designs/{design_id}/outer")
def outer(design_id:str,p:LandmarksRequest):
    """legacy/AI 外形入口（A1）。

    - 模板一律走 ``resolve_template_source``（与初稿/AutoFit/报告同一真相源）。
    - 地标分支（kabsch+RBF）**不再属于 live**：它不消费已确认患者帧，必须显式
      ``legacy_landmarks=true`` 才允许（研究/回归对比用），并如实标记。
    - 产物与修订同事务发布（原子修订）；响应/residual 记 ``template_source`` /
      ``frame_source`` / ``placement_path``。
    """
    d=design_row(design_id)
    require_draft_ready(d)
    expected=dependency_digest(d)
    from ..services.template_source import resolve_template_source, TemplateSourceError
    try: resolved=resolve_template_source(d)
    except TemplateSourceError as exc: raise HTTPException(400,str(exc)) from exc
    template=resolved.path
    out=CASE_DIR/d["case_id"]/f"{design_id}_outer_{uuid4().hex}.stl"
    legacy_flags=[];frame_source='NOT_FRAME_BASED';placement_path='AI_PREDICTED'
    try:
        if d["generator_mode"]=="template":
            if not p.legacy_landmarks:
                raise HTTPException(409,'地标驱动的冠形生成已停用：请使用"生成候选/初稿"（基于已确认坐标的模板放置）；'
                                        '确需研究对比时请在请求里显式带 legacy_landmarks=true')
            legacy_flags.append('LEGACY_LANDMARK_KABSCH_RBF')
            placement_path='LEGACY_LANDMARKS_RIGID_RBF'
            result=generate_outer(template,out,p.template_points,p.patient_points,
                                  p.morph_points_template,p.morph_points_patient,p.morph_strength)
            landmarks=p.model_dump()
        else:
            cp=Path(AI_CHECKPOINT) if AI_CHECKPOINT else None
            if not ai_runtime_available(cp):
                raise ValueError("AI checkpoint 或 PyTorch runtime 不可用")
            upper=asset(d["case_id"],"upper"); lower=asset(d["case_id"],"lower"); support=asset(d["case_id"],"support")
            m=predict_outer(cp,template,
                            Path(upper["path"]) if upper else None,
                            Path(lower["path"]) if lower else None,
                            Path(support["path"]),out,int(d["target_fdi"]))
            result={"mesh":mesh_info(m),"generator":"ai"}
            legacy_flags.append('RESEARCH_AI_CHECKPOINT')
            landmarks={}
    except HTTPException: raise
    except Exception as e: raise HTTPException(400,str(e))
    # landmarks_json 里还存着 _selected_template / _body_dependency_digest，只能合并，不能整表覆盖
    merged_landmarks=json.loads(d.get('landmarks_json') or '{}');merged_landmarks.update(landmarks)
    updates={'landmarks_json':json.dumps(merged_landmarks,ensure_ascii=False),'outer_path':str(out),
             'final_path':None,'cutter_path':None,'qc_json':None,'status':'outer_ready','updated_at':now()}
    result['template_source']=resolved.as_report()
    result['frame_source']=frame_source
    result['placement_path']=placement_path
    if legacy_flags: result['legacy_flags']=legacy_flags
    provenance={'template_source':result['template_source'],'patient_frame':{'revision_id':None,
                'source':frame_source},'placement':{'implementation':placement_path,'outer_path':str(out),
                'outer_sha256':file_digest(out)},'legacy_flags':legacy_flags,'algorithm_version':ALGORITHM_VERSION}
    provenance_path=out.with_name(out.stem+'_provenance.json')
    provenance_path.write_text(json.dumps(provenance,ensure_ascii=False,indent=1),encoding='utf-8')
    result['provenance']=provenance
    revision=DesignRevision(design_id=design_id,revision_no=0,kind=RevisionKind.OUTER,
        payload={"type":"outer","mode":"landmarks" if d["generator_mode"]=="template" else "ai",
                 "template_source":result['template_source'],"frame_source":frame_source,
                 "placement_path":placement_path,"legacy_flags":legacy_flags,
                 "outer_dependency_digest":dependency_digest({**d,**updates},stage='outer'),
                 "output_sha256":file_digest(out)},
        artifacts={"outer":str(out),"provenance":str(provenance_path)},algorithm_version=ALGORITHM_VERSION)
    publish_artifacts(d,expected,updates,stage='outer',revision=revision)
    result['revision_id']=revision.id
    return result

def initialize_draft_outer(d,p,*,allow_world_frame=False,allow_insufficient_occlusal_space=False):
    """草稿模板初稿：**唯一模板来源** + **唯一帧真相**（A0/A1）。

    - 模板：一律走 ``services.template_source.resolve_template_source``
      （用户确认的牙库优先，并校验磁盘真实字节；无确认才按回退链，且如实标注 source_kind）。
    - 帧：**必须**是已确认的 mesial/buccal/axis（``api.auto_fit.frame_from``）。
      A1 起 live 路径不再允许 world-X 推断；只有显式 ``allow_world_frame=True``（研究/legacy 调试）
      才回退，并标 ``frame_source='WORLD_X_FALLBACK_EXPLICIT'``。
    - 库件放置只做**一次**（``placed_library_crown`` 已内含颈缘 footprint 拟合）。
    - 返回里带 truth source（``template_source`` / ``patient_frame`` / ``placement``），供报告与产物 metadata 使用。
    """
    import numpy as np
    from ..services.template_source import (resolve_template_source, TemplateSourceError,
                                            FALLBACK_TEMPLATE_STUDIO)
    design_id=d['id']
    try:
        resolved=resolve_template_source(d)
    except TemplateSourceError as exc:
        raise HTTPException(400,str(exc)) from exc

    frame=None;frame_source='CONFIRMED_FRAME';frame_revision=None
    try:
        from .auto_fit import frame_from
        frame,frame_value=frame_from(d);frame_revision=frame_value.get('revision_id')
    except HTTPException as exc:
        if not allow_world_frame:
            raise HTTPException(409,'请先确认近中/颊侧方向与毫米单位（设计参数页的坐标确认），'
                                    '再生成初稿；live 生成不允许用世界轴推断放置') from exc
        frame=None;frame_source='WORLD_X_FALLBACK_EXPLICIT'

    placement='INITIALIZED_ONCE'
    if resolved.library_obj is not None:
        # 库件：在（确认或回退）帧里放置一次即得到 baseline
        from ..services.auto_fit_templates import placed_library_crown
        from types import SimpleNamespace
        if frame is not None:
            use_frame=frame
        else:
            z=np.asarray(p.insertion_axis,float);z/=np.linalg.norm(z)
            x=np.array([1.,0,0]);x-=z*np.dot(x,z)
            if np.linalg.norm(x)<.1:x=np.cross([0,1,0],z)
            x/=np.linalg.norm(x);basis=np.column_stack((x,np.cross(z,x),z))
            use_frame=SimpleNamespace(origin=np.zeros(3),matrix=lambda:basis)
        try:
            mesh,metadata=placed_library_crown(resolved.library_obj,d['target_fdi'],
                                               np.asarray(p.margin_points),use_frame)
        except (ValueError,FileNotFoundError) as exc:
            raise HTTPException(400,'牙库件放置失败：%s'%exc) from exc
        placement='LIBRARY_PLACED_ONCE'
    elif resolved.source_kind==FALLBACK_TEMPLATE_STUDIO:
        frame_data=json.loads((resolved.path.parent/'local_frame.json').read_text(encoding='utf-8'))
        mesh=load_mesh(resolved.path);x=np.asarray(frame_data['mesial']);z=np.asarray(frame_data['occlusal']);y=np.cross(z,x)
        vertices=np.asarray(mesh.vertices)@np.column_stack((x,y,z))
        vertices-=np.array([vertices[:,0].mean(),vertices[:,1].mean(),vertices[:,2].min()]);mesh.vertices=vertices
    else:
        mesh=None

    if mesh is not None:
        if not mesh.is_watertight:raise HTTPException(400,'候选冠形存在缺口，请导入完整冠形')
        template=CASE_DIR/d['case_id']/f"{design_id}_draft_template_{uuid4().hex}.stl";save_mesh(mesh,template)
        out=CASE_DIR/d["case_id"]/f"{design_id}_outer_{uuid4().hex}.stl"
        # 库件已带确认帧放置；其余模板走 initialize_from_margin（传确认帧）
        if placement=='LIBRARY_PLACED_ONCE':
            save_mesh(mesh,out)
            result={"mesh":mesh_info(mesh),"generator":"library_placement","review_required":True,
                    "frame_source":frame_source,"placement_path":placement}
        else:
            try: result=generate_initialized_outer(template,out,p.margin_points,p.insertion_axis,
                                                  patient_frame=frame)
            except Exception as e: raise HTTPException(400,str(e)) from e
            result["placement_path"]=placement
    else:
        out=CASE_DIR/d["case_id"]/f"{design_id}_outer_{uuid4().hex}.stl"
        try: result=generate_initialized_outer(resolved.path,out,p.margin_points,p.insertion_axis,
                                              patient_frame=frame)
        except Exception as e: raise HTTPException(400,str(e)) from e
        result["placement_path"]=placement
    result["template_source"]=resolved.as_report()
    result["frame_source"]=frame_source
    if not resolved.is_user_selected:
        result["template_source_warning"]='未找到人工确认的牙库；本次使用回退模板（%s）'%resolved.source_kind

    # ── 轴向初放置约束（上游修复：模板放置后先进入咬合可行区间）────────────────────
    # 横向拟合 + 颈缘对齐不使用对颌冠高；在当前病例的确认位姿下测空间，
    # 可用时只做颈缘锚定的有限轴向缩放，局部不可行时明确报告。
    axial_report=None
    if frame is not None and (d.get("cutter_path") or asset(d["case_id"],"support")):
        import numpy as np
        from ..services.axial_placement import occlusal_envelope, plan_axial_placement, apply_axial_placement, coarse_height_fit
        placed_mesh=load_mesh(Path(out))
        support_asset=asset(d["case_id"],"support")
        params=json.loads(d.get("parameters_json") or '{}')
        envelope=occlusal_envelope(d,np.asarray(p.margin_points,float),np.asarray(p.insertion_axis,float),
                                   frame,crown_mesh=placed_mesh,
                                   prep_path=support_asset['path'] if support_asset else None,
                                   min_thickness_mm=float(params.get('minimum_thickness_mm') or 0.5),
                                   target_clearance_mm=float(params.get('occlusal_target_mm') or 0.05))
        if envelope['status']=='NOT_ASSESSED' and d['support_type']=='abutment':
            raise HTTPException(422,{'status':'OCCLUSAL_SPACE_NOT_ASSESSED',
                                     'reason':envelope.get('reason'),'envelope':envelope})
        result['height_feasibility_status']=envelope['status']
        plan=(plan_axial_placement(envelope,crown_top_mm=envelope.get("crown_top_mm") or 0.0)
              if envelope.get("status")!="NOT_ASSESSED" else None)
        axial_report={"envelope":envelope,"plan":plan}
        if plan and plan.get("feasible") and plan.get("mechanism")=="AXIAL_SCALE":
            placed,applied=apply_axial_placement(placed_mesh,p.margin_points,p.insertion_axis,plan)
            axial_report["applied"]=applied
            if placed is None:
                raise HTTPException(422,{"status":"AXIAL_PLACEMENT_FAILED",
                                         "message":"模板轴向放置未通过形变硬门",**axial_report})
            save_mesh(placed,out)
            result["mesh"]=mesh_info(placed)
        elif plan and plan.get('mechanism')=='LOCAL_SHAPE_REQUIRED':
            placed,coarse=coarse_height_fit(d,placed_mesh,p.margin_points,p.insertion_axis,frame,envelope,
                                            target_mm=float(params.get('occlusal_target_mm') or 0.05))
            axial_report['coarse_fit']=coarse
            if placed is not None and coarse['status']=='APPLIED':
                save_mesh(placed,out)
                result['mesh']=mesh_info(placed)
        if (plan and not plan.get("feasible") and envelope['status']=='INSUFFICIENT_OCCLUSAL_SPACE'
                and not allow_insufficient_occlusal_space):
            raise HTTPException(422,{"status":"INSUFFICIENT_OCCLUSAL_SPACE",
                                     "message":envelope.get("verdict"),
                                     "detail":plan.get("reason"),**axial_report})
        result["axial_placement"]=axial_report
    # A1：产物旁边的 provenance sidecar —— 下载任何 STL 都能自证来源（模板字节 / 确认帧 / 放置实现）
    provenance={'template_source':result["template_source"],'patient_frame':{'revision_id':frame_revision,
                'source':frame_source},'placement':{'implementation':placement,'outer_path':str(out),
                'outer_sha256':file_digest(out)},'algorithm_version':ALGORITHM_VERSION}
    provenance_path=out.with_name(out.stem+'_provenance.json')
    provenance_path.write_text(json.dumps(provenance,ensure_ascii=False,indent=1),encoding='utf-8')
    result['provenance_path']=str(provenance_path)
    result['provenance']=provenance
    return result,out

@router.post("/designs/{design_id}/outer/auto")
def auto_outer(design_id:str,p:AutoOuterRequest):
    d=design_row(design_id)
    require_draft_ready(d)
    if not d.get('confirmed_margin_revision_id') or not d.get('current_axis_revision_id'):
        raise HTTPException(409,'请先保存并确认 Margin 和插入轴')
    if p.margin_points!=json.loads(d.get('margin_json') or 'null') or p.insertion_axis!=json.loads(d.get('axis_json') or 'null'):
        raise HTTPException(409,'初稿请求与已确认 Margin/轴不同，请先保存确认')
    expected=dependency_digest(d)
    if d["generator_mode"]!="template": raise HTTPException(400,"仅模板模式支持自动初始提案")
    # A1：live 初稿必须建立在已确认帧上（无帧直接 409，不再用世界轴推断放置）
    result,out=initialize_draft_outer(d,p)
    # landmarks_json 还存着 _selected_template / _body_dependency_digest：只更新 mode，不整表覆盖，
    # 否则"用户确认过的牙库件"会在下一次初稿时被静默丢弃。
    merged_landmarks=json.loads(d.get('landmarks_json') or '{}');merged_landmarks['mode']='auto_initialization'
    updates={'landmarks_json':json.dumps(merged_landmarks,ensure_ascii=False),'outer_path':str(out),'final_path':None,
             'cutter_path':None,'qc_json':None,'status':'outer_ready','updated_at':now()}
    # 不在这里登记轴候选：传入的就位方向可能是用户刚刚确认过的。若再登记一条
    # axis_candidate，修订链会冒出一个并不需要决策的"待决定候选"，
    # 使 revision_chain.requires_decision 误报为 True（实测踩到过）。
    # 修订与产物发布必须同事务（A1 原子修订）：否则"几何已更新、链上没记"。
    revision=DesignRevision(
        design_id=design_id, revision_no=0, kind=RevisionKind.OUTER,
        payload={"type": "outer", "mode": "margin_initialized_template",
                 "margin_point_count": len(p.margin_points),
                 "template_source":result.get('template_source'),
                 "frame_source":result.get('frame_source'),
                 "placement_path":result.get('placement_path'),
                 "outer_dependency_digest":dependency_digest({**d,**updates},stage='outer'),
                 "output_sha256":file_digest(out)},
        algorithm_version=ALGORITHM_VERSION,
        artifacts={"outer": str(out),"provenance":result.get('provenance_path')},
    )
    publish_artifacts(d,expected,updates,stage='outer',revision=revision)
    result['revision_id']=revision.id
    return result

@router.get("/designs/{design_id}/outer.stl")
def outer_file(design_id:str):
    d=design_row(design_id)
    if not d["outer_path"]: raise HTTPException(404,"Outer 尚未生成")
    require_current_outer(d)
    return FileResponse(d["outer_path"],media_type="model/stl",filename="crown_outer.stl")


@router.get("/designs/{design_id}/axial-envelope")
def axial_envelope(design_id:str):
    """**只读**：这位病例的轴向放置可行区间与逐柱可用咬合空间（不改任何几何）。

    上游放置要用的数字都在这里：模板原始高度 / 放置后冠体高度 / Margin→冠顶 / 预备体顶 /
    对颌天花板 / 逐柱可用空间 / lower·upper / 判定。放不下时明确给 ``INSUFFICIENT_OCCLUSAL_SPACE``，
    不留"让咬合求解器去削半颗牙"的余地。
    """
    import numpy as np
    from .auto_fit import frame_from
    from ..services.axial_placement import occlusal_envelope, plan_axial_placement
    d=design_row(design_id)
    margin=json.loads(d["margin_json"] or 'null')
    if not margin:
        raise HTTPException(409,'请先确认边缘线')
    frame,ref=frame_from(d)
    mesh=load_mesh(Path(d["outer_path"])) if d.get("outer_path") and Path(d["outer_path"]).is_file() else None
    support_asset=asset(d['case_id'],'support')
    params=json.loads(d.get('parameters_json') or '{}')
    envelope=occlusal_envelope(d,np.asarray(margin,float),np.asarray(frame.z,float),ref,crown_mesh=mesh,
        prep_path=support_asset['path'] if support_asset else None,
        min_thickness_mm=float(params.get('minimum_thickness_mm') or 0.5),
        target_clearance_mm=float(params.get('occlusal_target_mm') or 0.05))
    return {**envelope,
            "plan": plan_axial_placement(envelope,crown_top_mm=envelope.get("crown_top_mm") or 0.0)
                    if envelope.get("status")!="NOT_ASSESSED" else None}

@router.post("/designs/{design_id}/finalize")
def finalize(design_id:str,p:FinalizeRequest):
    return _finalize(design_id,p)


@router.post('/designs/{design_id}/rebuild-cervical')
def rebuild_cervical(design_id:str,p:FinalizeRequest):
    return _finalize(design_id,p,preserve_body=True)



@router.post('/designs/{design_id}/adapt-occlusal')
def adapt_occlusal(design_id:str,p:OcclusalAdaptRequest):
    """咬合适配：按对颌天花板给冠的**咬合区**减材。

    失败（``OCCLUSAL_ADAPTATION_FAILED``）时**不改冠、不发布**，只如实返回理由与 before/after；
    成功才发布新的 outer 修订（payload ``occlusal_adaptation``），并可接着重建内冠/颈缘
    （``rebuild_final``）——颈缘 / margin / 内冠全程锁定，位移精确为 0。
    """
    from ..services.design_dependencies import (confirmed_inputs, dependency_digest,
                                                publish_artifacts, require_current_outer)
    from ..services.occlusal_adaptation import adapt_occlusal as run_adaptation
    from .auto_fit import frame_from
    d=design_row(design_id)
    require_current_outer(d)
    frame,ref=frame_from(d)
    mesh=load_mesh(Path(d['outer_path']))
    adapted,report=run_adaptation(d,mesh,frame,ref,target_mm=p.target_mm,min_wall_mm=p.min_wall_mm,
                                  mesh_source='outer')
    report['analysis_mesh_source']=('final' if d.get('final_path') and Path(d['final_path']).is_file()
                                    else 'outer')
    if adapted is None:
        report['published']=False
        return report
    expected=dependency_digest(d)
    consumed=confirmed_inputs(d)
    out=CASE_DIR/d['case_id']/f"{design_id}_outer_occlusal_{uuid4().hex}.stl"
    save_mesh(adapted,out)
    revision=DesignRevision(design_id=design_id,revision_no=0,kind=RevisionKind.OUTER,
        payload={'type':'occlusal_adaptation','mechanism':report['mechanism'],
                 'occlusion':report['occlusion'],'required_drop':report['required_drop'],
                 'locked_displacement_max_mm':report['protected']['locked_displacement_max_mm'],
                 'mesh_sha256_before':report['mesh_sha256_before'],
                 'mesh_sha256_after':report['mesh_sha256_after']},
        artifacts={'outer':str(out)},algorithm_version=ALGORITHM_VERSION)
    publish_artifacts(d,expected,{'outer_path':str(out),'final_path':None,'cutter_path':None,
        'qc_json':None,'status':'outer_ready','updated_at':now(),'sculpt_state_json':None},
        stage='outer',revision=revision)
    report.update(published=True,outer_path=str(out),outer_revision_id=revision.id)
    report['rebuild_final']=False
    if p.rebuild_final:
        fresh=design_row(design_id)
        request=FinalizeRequest(expected_input_digest=dependency_digest(fresh),
            margin_revision_id=consumed['margin'].id,axis_revision_id=consumed['axis'].id,
            profile_revision_id=consumed['profile'].id)
        report['finalize']=_finalize(design_id,request,preserve_body=True)
        report['rebuild_final']=True
    return report


def _finalize(design_id,p,*,preserve_body=False):
    d=design_row(design_id)
    require_draft_ready(d)
    from ..services.design_dependencies import confirmed_inputs
    consumed=confirmed_inputs(d)
    if preserve_body:
        from ..services.design_dependencies import require_current_outer_body
        require_current_outer_body(d)
    else:
        require_current_outer(d)
    dependency_before=dependency_digest(d)
    if p.expected_input_digest!=dependency_before or any(getattr(p,name+'_revision_id')!=revision.id for name,revision in consumed.items()):
        raise HTTPException(409,'生成请求引用的确认版本已过期，请重新载入当前设计')
    if not d["outer_path"]: raise HTTPException(400,"请先生成 Crown Outer")
    s=asset(d["case_id"],"support")
    params=consumed['profile'].parameter_set
    margin=consumed['margin'].payload['points']; axis=consumed['axis'].payload['axis']
    from ..services.abutment_support import prepare_design_support,indication_of
    support_contract,prepared_support=prepare_design_support(d)
    from ..services.contact_measurement import thickness_protection_surfaces
    protection=thickness_protection_surfaces(d) if params.get('enforce_minimum_thickness') else ()
    final=CASE_DIR/d["case_id"]/f"{design_id}_final_{uuid4().hex}.stl"
    cutter=CASE_DIR/d["case_id"]/f"{design_id}_intaglio_cutter_{uuid4().hex}.stl"
    try:
        trimmed,cutter_mesh,final_mesh,blockout_diag=finalize_crown(
            Path(d["outer_path"]),Path(s["path"]),final,cutter,
            margin,axis,params,prepared_support=prepared_support,support_contract=support_contract,protected_surfaces=protection
        )
        target_lower=31 <= int(d["target_fdi"]) <= 48
        same=asset(d["case_id"],"lower" if target_lower else "upper")
        ant=asset(d["case_id"],"upper" if target_lower else "lower")
        if blockout_diag.get('w3'):
            from ..services.auto_fit import audit_final_context
            w3=blockout_diag['w3']
            audit_final_context(d,w3['artifact_paths']['outer_surface'],final,w3['qc'])
            audit_path=Path(w3['artifact_paths']['manufacturing_qc'])
            stored=json.loads(audit_path.read_text(encoding='utf-8'));stored['qc']=w3['qc']
            audit_path.write_text(json.dumps(stored,ensure_ascii=False),encoding='utf-8')
            qc={'final_mesh':mesh_info(final),'manufacturing_qc':blockout_diag['w3']['qc'],
                'artifact_paths':blockout_diag['w3']['artifact_paths']}
        else:
            qc=qc_from_paths(final,Path(d["outer_path"]),cutter,Path(s["path"]),
                             Path(same["path"]) if same else None,
                             Path(ant["path"]) if ant else None, outer_mesh=trimmed)

        # finalize 验收策略标签（DESIGN/STRICT）：无论走哪个 QC 分支都写进去，供检查页显示
        _mode=(json.loads(d.get("parameters_json") or "{}").get("quality_mode") or "DESIGN")
        _stitch_warning=None
        for _cand in (blockout_diag.get("w3",{}).get("qc") if isinstance(blockout_diag,dict) else None, qc.get("manufacturing_qc")):
            if not isinstance(_cand,dict): continue
            _exc=_cand.get("design_mode_exception") or (_cand.get("final_acceptance") or {}).get("cervical_stitch_exception")
            if _exc:
                _stitch_warning=_exc.get("warning"); break
        qc["final_acceptance"]={"status":"DESIGN_READY" if str(_mode).upper().startswith("DESIGN") else "STRICT_VERIFIED",
            "quality_mode":_mode,"warnings":([_stitch_warning] if _stitch_warning else []),
            "manufacturing_check":False,
            "note":"DESIGN 模式：仅降级颈缘缝合的轻微自交；margin/内冠/咬合与制造检查口径不变"}
    except Exception as e:
        from ..geometry.margin_seam import MarginReconstructionRequired
        from ..geometry.nonplanar_crown import MarginBuildError
        if isinstance(e,MarginBuildError):
            raise HTTPException(422,{'message':str(e),'geometry':{'status':'MARGIN_BUILD_BLOCKED'}}) from e
        if isinstance(e,MarginReconstructionRequired):
            raise HTTPException(422,{'message':'非平面 Margin 的真实颈部适配尚未完成，禁止用平面裁切生成最终冠','geometry':e.diagnostics}) from e
        raise HTTPException(400,f"Finalize 失败: {e}") from e
    qc["blockout"] = blockout_diag
    consumed_ids={name+'_revision_id':revision.id for name,revision in consumed.items()}
    qc['consumed_revisions']=consumed_ids
    qc['release_status']='ENGINEERING_ONLY'
    # A1：把 truth source 写进**产物 metadata**（任何 STL 都能自证来自哪颗模板/哪个确认帧/哪个实现）
    from ..services.design_dependencies import file_digest as _file_digest
    from ..services.template_source import resolve_template_source, TemplateSourceError
    try:
        template_source=resolve_template_source(d).as_report()
    except TemplateSourceError as exc:
        template_source={'error':str(exc)}
    frame_value=json.loads(d.get('autofit_frame_json') or 'null') or {}
    provenance={'template_source':template_source,
        'patient_frame':{'revision_id':frame_value.get('revision_id'),
                         'source':'CONFIRMED' if frame_value else 'NOT_CONFIRMED'},
        'placement':{'implementation':'patient_frame_v1','outer_path':d['outer_path'],
                     'outer_sha256':_file_digest(d['outer_path']) if d.get('outer_path') else None},
        'consumed_revisions':consumed_ids,'input_dependency_digest':dependency_before,
        'algorithm_version':ALGORITHM_VERSION}
    qc['provenance']=provenance
    provenance_path=Path(final).with_name(Path(final).stem+'_provenance.json')
    provenance.update(indication=indication_of(d),support_type=d['support_type'],margin_semantics='CROWN_FINISH_LINE' if d['support_type']=='abutment' else 'PREPARATION_MARGIN',
        intaglio_parameters=params,gap_profile={k:params.get(k) for k in ('cement_gap_mm','cement_gap_start_mm','cement_gap_top_mm')})
    if blockout_diag.get('w3'):
        provenance['manufacturing_geometry']={key:blockout_diag['w3']['qc'].get(key) for key in ('CrownBorderAudit','MinimumThicknessEnforcement')}
    if support_contract:
        provenance['prepared_support']=support_contract.as_report()
    provenance_path.write_text(json.dumps(provenance,ensure_ascii=False,indent=1),encoding='utf-8')
    updates={'cutter_path':str(cutter),'final_path':str(final),'qc_json':json.dumps(qc),'status':'final_ready','updated_at':now()}
    if preserve_body: updates['outer_dependency_digest']=dependency_digest(d,stage='outer')

    # finalize 也必须进修订链，且与产物发布**同一个事务**：
    # 否则进程死在两步之间会出现"current geometry 已变、修订链没记"（审计 P1）。
    current = consumed['margin']
    chain_revision = DesignRevision(
        design_id=design_id,
        revision_no=0,
        kind=RevisionKind.FINAL,
        payload={"type":"final","margin_point_count":len(margin),
                 "artifact_outer":str(d["outer_path"]),'consumed_revisions':consumed_ids,
                 'input_design_revision':d['design_revision'],'input_digest':dependency_before,
                 'rebuild_mode':'PRESERVE_EXISTING_BODY' if preserve_body else 'CURRENT_OUTER',
                 'provenance':provenance},
        algorithm_version=ALGORITHM_VERSION,
        parameter_set={k:params[k] for k in sorted(params)},
        margin_source=current.margin_source if current is not None else None,
        operator_actions=[{"action":"finalize","at":now()}],
        artifacts={"cutter":str(cutter),"final":str(final),"provenance":str(provenance_path)},
    )
    published=publish_artifacts(d,dependency_before,updates,revision=chain_revision)
    qc=json.loads(published['qc_json'])

    # blockout 诊断随 QC 一起返回，便于核对"倒凹是否真的被封闭"
    qc["blockout"] = blockout_diag
    return {"mesh":mesh_info(final_mesh),"qc":qc,
            "revision_no":chain_revision.revision_no,
            "margin_revision_no":(current.revision_no if current is not None else None),
            "provenance":provenance}

@router.get("/designs/{design_id}/final.stl")
def final_file(design_id:str):
    d=design_row(design_id)
    if not d["final_path"]: raise HTTPException(404,"Final Crown 尚未生成")
    from ..services.design_dependencies import require_current_final
    require_current_final(d)
    return FileResponse(d["final_path"],media_type="model/stl",filename=f"FDI_{d['target_fdi']}_final_crown.stl")

@router.get("/designs/{design_id}/intaglio-cutter.stl")
def cutter_file(design_id:str):
    d=design_row(design_id)
    if not d["cutter_path"]: raise HTTPException(404,"Intaglio cutter 尚未生成")
    from ..services.design_dependencies import require_current_final
    require_current_final(d)
    return FileResponse(d["cutter_path"],media_type="model/stl",filename="intaglio_cutter.stl")

@router.get("/designs/{design_id}/qc")
def qc(design_id:str):
    d=design_row(design_id)
    if d['qc_json']:
        from ..services.design_dependencies import require_current_final
        require_current_final(d)
    return json.loads(d["qc_json"]) if d["qc_json"] else None


@router.get("/designs/{design_id}/package.zip")
def design_package(design_id:str):
    """Export the real files and parameters already produced for this design."""
    import io
    d=design_row(design_id)
    if d.get('outer_path'): require_current_outer(d)
    if d.get('final_path'):
        from ..services.design_dependencies import require_current_final
        require_current_final(d)
    buf=io.BytesIO()
    with zipfile.ZipFile(buf,"w",zipfile.ZIP_DEFLATED) as z:
        meta={k:d.get(k) for k in (
            "id","case_id","target_fdi","support_type","generator_mode","status",
            "created_at","updated_at","final_dependency_digest"
        )}
        meta['export_type']='ENGINEERING_ONLY'
        meta['profile']=json.loads(d.get('profile_json') or 'null')
        meta['final_current']=bool(d.get('final_path'))
        for key in ("parameters_json","margin_json","axis_json","landmarks_json","qc_json"):
            raw=d.get(key)
            if raw:
                try: meta[key.removesuffix("_json")]=json.loads(raw)
                except Exception: meta[key.removesuffix("_json")]=raw
        z.writestr("design.json",json.dumps(meta,ensure_ascii=False,indent=2))
        # 审计链：每次 Margin/轴/外形/最终结果的修订与人工确认人都随包导出。
        chain=revision_chain(design_id)
        z.writestr("revisions.json",json.dumps(chain,ensure_ascii=False,indent=2))
        for revision in chain['revisions']:
            for label,fp in revision.get('artifacts',{}).items():
                source=Path(fp)
                if source.is_file() and source.resolve().is_relative_to((CASE_DIR/d['case_id']).resolve()):
                    z.write(source,arcname=f"history/{revision['revision_no']}/{label}{source.suffix}")
        template=TEMPLATE_DIR/f"FDI_{d['target_fdi']}.stl"
        if template.is_file():z.write(template,arcname='inputs/template.stl')
        for label,key in (("crown_outer.stl","outer_path"),("intaglio_cutter.stl","cutter_path"),(f"FDI_{d['target_fdi']}_final_crown.stl","final_path")):
            fp=d.get(key)
            if fp and Path(fp).exists(): z.write(fp,arcname=label)
        for label,fp in json.loads(d.get('qc_json') or '{}').get('artifact_paths',{}).items():
            z.write(fp,arcname=f'w3/{label}{Path(fp).suffix}')
        case_dir=CASE_DIR/d["case_id"]
        for kind in ("upper","lower","support"):
            a=asset(d["case_id"],kind)
            if a and Path(a["path"]).exists():
                # Inputs are included because the package is user-generated case data, not bundled demo data.
                z.write(a["path"],arcname=f"inputs/{kind}.stl")
    # G0：导出前过 PHI 闸门。导出包是**对外**产物（DEIDENTIFIED_EXPORT），
    # 结构规则（条目名 ASCII 安全、不得含 source_package.zip / 脱敏映射表 / PHI 字段名）
    # 与值扫描（该病例以 IDENTIFIED 导入时留下的原值）双保险；命中就拒绝导出，
    # 且错误消息只报"哪个条目命中哪个标记"，**不回显 PHI 值**。
    from ..services.phi_policy import PhiPolicyError, assert_export_clean, phi_needles_for_case
    payload=buf.getvalue()
    try:
        assert_export_clean(payload, needles=phi_needles_for_case(d["case_id"]),
                            phi_keys=PHI_PROPERTY_NAMES, where="工程包导出")
    except PhiPolicyError as exc:
        raise HTTPException(409,str(exc)) from exc
    return Response(
        content=payload,
        media_type="application/zip",
        headers={"Content-Disposition":f'attachment; filename="welooai_{design_id}.zip"'}
    )
