"""Dependency checks for the manual workbench; research runs remain separate."""
from functools import lru_cache
import hashlib
import json
from pathlib import Path
from fastapi import HTTPException
from ..db import db
from ..config import TEMPLATE_DIR


def dependency_digest(design, *, stage="final", conn=None):
    keys=['target_fdi','support_type','indication','generator_mode','axis_json','current_axis_revision_id']
    if stage!='body': keys += ['margin_json','confirmed_margin_revision_id']
    if stage not in ('outer','body'): keys += ['design_revision','current_profile_revision_id']
    if stage not in ('outer','body'): keys += ['parameters_json','outer_path']+[k for k in design if k.endswith('_json') and k not in ('qc_json','margin_json','axis_json','parameters_json')]
    data={k:design.get(k) for k in keys}
    selected_template=json.loads(design.get('landmarks_json') or '{}').get('_selected_template')
    data['selected_template']=selected_template
    # 确认帧参与放置（A1）：它一变，已发布的外形就不该再算"当前"。
    # 只取决定几何的字段（不含每次确认都会换的 revision_id / 时间戳），
    # 否则"原样再确认一次方向"也会把已发布外形误判为过期。
    frame_value=json.loads(design.get('autofit_frame_json') or 'null')
    if frame_value:
        data['template_frame']={key:frame_value.get(key) for key in
            ('x','y','z','buccal','units','axis_revision_id','margin_revision_id','prep_sha256')}
    if conn is None:
        with db() as source: rows=source.execute('SELECT id,kind,path FROM assets WHERE case_id=? ORDER BY created_at,id',(design['case_id'],)).fetchall()
    else: rows=conn.execute('SELECT id,kind,path FROM assets WHERE case_id=? ORDER BY created_at,id',(design['case_id'],)).fetchall()
    files=([design.get('outer_path')] if stage not in ('outer','body') else [])+[r['path'] for r in rows]
    files.append(str(TEMPLATE_DIR/f"FDI_{design['target_fdi']}.stl"))
    if selected_template:
        files.append(selected_template['source_path'])
        files.append(selected_template.get('metadata_path'))
    else:
        # 没有人工确认时，回退链可能落到 template_studio 候选或牙库件上：
        # 那些字节也必须进摘要，否则"牙库文件被换掉"不会让已发布外形失效。
        try:
            from .template_source import resolve_template_source, FALLBACK_DESIGN_DIR
            resolved=resolve_template_source(design)
            if resolved.source_kind!=FALLBACK_DESIGN_DIR:
                files.append(str(resolved.path))
        except Exception:
            pass                                            # 无可用模板：摘要只覆盖已有输入
    if stage not in ('outer','body'):
        files += [value['path'] for value in json.loads(design.get('neighbors_json') or '{}').values() if value.get('status')=='SELECTED' and value.get('path')]
    data['assets']=[dict(r) for r in rows]
    data['sha256']={}
    for p in files:
        if p and Path(p).is_file():
            data['sha256'][p]=file_digest(p)
    return hashlib.sha256(json.dumps(data,sort_keys=True,ensure_ascii=False).encode()).hexdigest()


def require_current_final(design):
    if not design.get('final_path'):
        raise HTTPException(409,'最终冠未生成或已过期，请重建内冠并重新检查')
    if not design.get('final_dependency_digest') or design['final_dependency_digest']!=dependency_digest(design):
        raise HTTPException(409,'最终冠依赖版本不一致，请重建；旧产物不可作为当前结果导出')
    report=json.loads(design.get('qc_json') or '{}')
    paths={**report.get('artifact_paths',{}),**{key:design.get(key) for key in ('final_path','cutter_path')}}
    for key,path in paths.items():
        if not path or not Path(path).is_file() or report.get('artifact_sha256',{}).get(key)!=file_digest(path):
            raise HTTPException(409,'产物文件缺失或已变化，请重新生成并检查')


@lru_cache(maxsize=512)
def _file_digest_cached(resolved_path, size, mtime_ns, ctime_ns):
    # size/mtime/ctime participate in the cache key.  Replacing or modifying an
    # artifact therefore invalidates the cached digest while repeated workflow
    # eligibility checks stop re-reading multi-megabyte STL files.
    with Path(resolved_path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def file_digest(path):
    p=Path(path).resolve()
    stat=p.stat()
    return _file_digest_cached(str(p),int(stat.st_size),int(stat.st_mtime_ns),int(stat.st_ctime_ns))


def confirmed_inputs(design, *, require_profile=True):
    from ..revisions.models import DesignRevision, RevisionKind
    from .abutment_support import indication_of
    if indication_of(design) in ('CUSTOM_IMPLANT_ABUTMENT','SCREW_RETAINED_CROWN'):
        raise HTTPException(422,{'status':'UNSUPPORTED_INDICATION','rejection_code':'CUSTOM_ABUTMENT_NOT_SUPPORTED' if indication_of(design)=='CUSTOM_IMPLANT_ABUTMENT' else 'UNSUPPORTED_INDICATION'})
    if design.get('support_type')=='abutment':
        with db() as conn:
            support=conn.execute("SELECT mesh_info_json FROM assets WHERE case_id=? AND kind='support' ORDER BY created_at DESC LIMIT 1",
                                 (design['case_id'],)).fetchone()
        details=json.loads(support['mesh_info_json'] or '{}') if support else {}
        if details.get('implant_candidate') and not details.get('restorative_abutment_confirmed'):
            raise HTTPException(409,'仅定位了种植组件候选；请确认实际最终修复基台，扫描杆不能直接构冠')
    ids={'margin':design.get('confirmed_margin_revision_id'),'axis':design.get('current_axis_revision_id'),
         'profile':design.get('current_profile_revision_id')}
    if not require_profile: ids.pop('profile')
    if not ids['margin'] or ids['margin']!=design.get('current_margin_revision_id'):
        raise HTTPException(409,'边缘线只是候选，请在第 3 步点击“确认当前 Margin”')
    if not ids['axis']:
        raise HTTPException(409,'请在第 3 步确认插入轴')
    if require_profile and not ids['profile']:
        raise HTTPException(409,'请在第 4 步应用设计参数')
    result={}
    with db() as conn:
        for name,identifier in ids.items():
            row=conn.execute('SELECT * FROM design_revisions WHERE id=? AND design_id=?',(identifier,design['id'])).fetchone()
            if row is None: raise HTTPException(409,'确认修订不存在或不属于当前设计')
            result[name]=DesignRevision.from_row(row)
    allowed={'margin':(RevisionKind.MARGIN_MANUAL,RevisionKind.MARGIN_CANDIDATE,RevisionKind.MARGIN_AUTO_ACCEPTED),
             'axis':(RevisionKind.AXIS_ACCEPTED,), 'profile':(RevisionKind.PROFILE,)}
    if any(revision.kind not in allowed[name] for name,revision in result.items()):
        raise HTTPException(409,'确认修订类型不一致')
    if (result['margin'].payload.get('points')!=json.loads(design.get('margin_json') or 'null')
        or result['axis'].payload.get('axis')!=json.loads(design.get('axis_json') or 'null')
        or (require_profile and (result['profile'].parameter_set!=json.loads(design['parameters_json'])
        or result['profile'].payload!=json.loads(design.get('profile_json') or 'null')))):
        raise HTTPException(409,'生效数据与确认修订不一致，请重新确认')
    return result


def require_current_outer(design):
    if not design.get('outer_path') or not design.get('outer_dependency_digest'):
        raise HTTPException(409,'外冠未生成或缺少版本记录，请重新生成')
    if not Path(design['outer_path']).is_file() or design['outer_dependency_digest']!=dependency_digest(design,stage='outer') or design.get('outer_output_sha256')!=file_digest(design['outer_path']):
        raise HTTPException(409,'外冠依赖或网格已变化，请重新生成')


def require_current_outer_body(design):
    stored=json.loads(design.get('landmarks_json') or '{}').get('_body_dependency_digest')
    path=design.get('outer_path')
    if (not path or not Path(path).is_file() or not stored or stored!=dependency_digest(design,stage='body')
        or design.get('outer_output_sha256')!=file_digest(path)):
        raise HTTPException(409,'现有冠形缺少可信主体版本，或基牙/轴/模板已改变；不能仅重建颈部，请重新生成外冠')


def _validate_published_candidate_artifacts(proposal, accepted_artifacts):
    """Only artifacts copied into the accepted design can invalidate acceptance."""
    artifacts=json.loads(proposal['artifacts_json'] or '{}')
    metadata=json.loads(proposal['payload_json'])
    expected_hashes=metadata.get('artifact_sha256',{})
    for label,path in artifacts.items():
        if label not in accepted_artifacts:
            continue
        accepted_path=accepted_artifacts[label]
        if (str(accepted_path)!=str(path) or not Path(path).is_file()
                or file_digest(path)!=expected_hashes.get(label)):
            raise HTTPException(409,'候选或预审产物已变化，不能采纳')


def publish_artifacts(baseline, expected, updates, *, stage='final', revision=None):
    """Compare and publish in one SQLite write transaction, including slow task completion."""
    with db() as conn:
        conn.execute('BEGIN IMMEDIATE')
        row=conn.execute('SELECT * FROM designs WHERE id=?',(baseline['id'],)).fetchone()
        if row is None: raise HTTPException(409,'设计已删除，本轮结果不生效')
        current=dict(row)
        if current['design_revision']!=baseline['design_revision'] or dependency_digest(current,conn=conn)!=expected:
            raise HTTPException(409,'任务运行期间设计已修改；本轮结果不生效，请重新生成')
        if revision is not None and revision.payload.get('type') in ('autofit_accepted','template_selected'):
            if conn.execute("SELECT 1 FROM design_revisions WHERE design_id=? AND kind='autofit_rejected' AND json_extract(payload_json,'$.proposal_id')=?",(baseline['id'],revision.payload['proposal_id'])).fetchone():
                raise HTTPException(409,'候选已退回，不能采纳')
            proposal=conn.execute('SELECT * FROM design_revisions WHERE design_id=? AND id=?',(baseline['id'],revision.payload['proposal_id'])).fetchone()
            metadata=json.loads(proposal['payload_json']) if proposal else {}
            source_path=metadata.get('source_template_path')
            if source_path and (not Path(source_path).is_file() or file_digest(source_path)!=metadata.get('template_sha256')):
                raise HTTPException(409,'牙库源模板已变化，请重新提案')
            metadata_path=metadata.get('source_metadata_path')
            if metadata_path and (not Path(metadata_path).is_file() or file_digest(metadata_path)!=metadata.get('metadata_sha256')):
                raise HTTPException(409,'牙库方向元数据已变化，请重新提案')
            if proposal:
                _validate_published_candidate_artifacts(proposal, revision.artifacts or {})
        updates['design_revision']=current['design_revision']+1
        merged={**current,**updates}
        if stage=='outer':
            landmarks=json.loads(merged.get('landmarks_json') or '{}')
            landmarks['_body_dependency_digest']=dependency_digest(merged,stage='body',conn=conn)
            updates['landmarks_json']=json.dumps(landmarks)
            merged.update(updates)
        column='outer_dependency_digest' if stage=='outer' else 'final_dependency_digest'
        if stage!='template': updates[column]=dependency_digest(merged,stage=stage,conn=conn)
        if stage=='outer': updates['outer_output_sha256']=file_digest(merged['outer_path'])
        elif stage!='template' and updates.get('qc_json'):
            report=json.loads(updates['qc_json'])
            report['input_dependency_digest']=updates[column]
            paths={**report.get('artifact_paths',{}),**{k:merged[k] for k in ('final_path','cutter_path')}}
            report['artifact_sha256']={k:file_digest(path) for k,path in paths.items()}
            updates['qc_json']=json.dumps(report)
        conn.execute('UPDATE designs SET '+','.join(k+'=?' for k in updates)+' WHERE id=?',(*updates.values(),baseline['id']))
        if revision is not None:
            from ..revisions.service import _next_revision_no, _now
            revision.revision_no=_next_revision_no(conn,baseline['id'])
            revision.created_at=_now()
            row=revision.to_row()
            conn.execute('INSERT INTO design_revisions('+','.join(row)+') VALUES('+','.join(':'+k for k in row)+')',row)
    return {**merged,**updates}
