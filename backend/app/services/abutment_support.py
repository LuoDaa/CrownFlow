"""One support truth shared by eligibility, intaglio construction and measured QC."""
import json
from pathlib import Path
from fastapi import HTTPException
from .prep_conditioning import PreparedSurfaceContract, Indication, Readiness
from .design_dependencies import file_digest
from ..geometry.abutment_support import extract_abutment_support, SupportExtractionError
from ..geometry.io import load_mesh
from ..geometry.nonplanar_crown import fitting_support_above_margin

def indication_of(design):
    return design.get('indication') or ('CROWN_ON_EXISTING_ABUTMENT' if design.get('support_type')=='abutment' else 'NATURAL_PREP')

def assess_abutment_surface(support,margin,axis,*,source_sha256=None,margin_revision_id=None,axis_revision_id=None,
                            patient_frame_revision_id=None,target_fdi=None,seed=None,confirmed_loop_ids=(),indication='CROWN_ON_EXISTING_ABUTMENT'):
    contract=PreparedSurfaceContract(target_fdi=target_fdi,indication=indication,
        source_asset_sha256=source_sha256,margin_revision_id=margin_revision_id,insertion_axis_revision_id=axis_revision_id,
        provenance={'support_type':'abutment','margin_semantics':'CROWN_FINISH_LINE',
                    'patient_frame_revision_id':patient_frame_revision_id,'source_abutment_sha256':source_sha256})
    def rejected(code,reason,readiness='REVIEW_REQUIRED'):
        contract.readiness=readiness;contract.rejection_code=code;contract.rejection_reason=reason
        contract.rejection_class='UNSUPPORTED_INDICATION' if readiness=='UNSUPPORTED_INDICATION' else 'REVIEW_REQUIRED'
        return contract,None
    if indication!='CROWN_ON_EXISTING_ABUTMENT':return rejected('CUSTOM_ABUTMENT_NOT_SUPPORTED' if indication=='CUSTOM_IMPLANT_ABUTMENT' else 'UNSUPPORTED_INDICATION',
        '当前流程只支持已有实体基台上的粘接冠','UNSUPPORTED_INDICATION')
    if support is None:return rejected('NO_SUPPORT_ASSET','缺少实体基台网格','NO_SUPPORT_ASSET')
    if not margin_revision_id:return rejected('MARGIN_NOT_CONFIRMED','请先确认基台冠终止线')
    if not axis_revision_id or not patient_frame_revision_id:return rejected('SUPPORT_EXTRACTION_AMBIGUOUS','请先确认插入轴和当前 PatientFrame')
    try:
        prepared=extract_abutment_support(support,margin,axis,seed=seed,confirmed_loop_ids=confirmed_loop_ids)
        report=prepared.report
        for loop in report['loops']:
            if loop['classification']=='CROWN_MARGIN':
                loop['confirmed']=True;loop['margin_revision_id']=margin_revision_id
        if report['readiness']=='ABUTMENT_SUPPORT_READY':
            # The existing input predicate remains authoritative after explicit conditioning.
            fitting_support_above_margin(prepared.mesh,margin,axis,prepared=True)
    except ValueError as exc:
        return rejected(getattr(exc,'code','MARGIN_NOT_ON_SUPPORT' if '缺口' in str(exc) else 'SUPPORT_EXTRACTION_AMBIGUOUS'),str(exc))
    contract.readiness=report['readiness'];contract.rejection_code=report['rejection_code']
    contract.rejection_reason={'SCREW_CHANNEL_CONFIRMATION_REQUIRED':'请核对并确认基台顶部螺丝通道开口',
        'UNKNOWN_SUPPORT_HOLE':'存在未分类孔洞，请检查扫描来源，禁止自动封闭'}.get(contract.rejection_code,'')
    contract.rejection_class='READY' if contract.readiness=='ABUTMENT_SUPPORT_READY' else 'REVIEW_REQUIRED'
    contract.component_count=report['component_count'];contract.boundary_loop_count=report['boundary_loop_count']
    contract.conditioned_mesh_sha256=report['conditioned_support_sha256'];contract.source_face_ids=report['source_face_ids']
    contract.operations=report['virtual_cap_operations'];contract.diagnostics=report
    contract.provenance.update(conditioned_support_sha256=report['conditioned_support_sha256'],
        algorithm_version=report['algorithm_version'],screw_channel_present=report['screw_channel_present'])
    return contract,prepared

def prepare_design_support(design,*,require_ready=True,lookup_asset=True,reference_override=None):
    from ..api.routes import asset
    from ..api.auto_fit import frame_from
    reference=asset(design['case_id'],'support') if lookup_asset else reference_override
    indication=indication_of(design)
    if indication not in ('CROWN_ON_EXISTING_ABUTMENT','CUSTOM_IMPLANT_ABUTMENT','SCREW_RETAINED_CROWN'):
        return None,None
    margin=json.loads(design.get('margin_json') or 'null');axis=json.loads(design.get('axis_json') or 'null')
    source_exists=bool(reference and Path(reference['path']).is_file())
    source_sha=file_digest(reference['path']) if source_exists else None
    frame=json.loads(design.get('autofit_frame_json') or 'null') or {}
    frame_id=frame.get('revision_id')
    try:frame_from(design)
    except (HTTPException,KeyError,ValueError,OSError):frame_id=None
    basis={'source_abutment_sha256':source_sha,'margin_revision_id':design.get('confirmed_margin_revision_id'),
           'axis_revision_id':design.get('current_axis_revision_id'),'patient_frame_revision_id':frame_id}
    confirmation=json.loads(design.get('support_conditioning_json') or 'null') or {}
    current=confirmation.get('basis')==basis
    source_mesh=None;load_error=None
    if source_exists:
        try:source_mesh=load_mesh(Path(reference['path']))
        except (ValueError,OSError) as exc:load_error=str(exc)
    contract,prepared=assess_abutment_surface(source_mesh,margin,axis,
        source_sha256=source_sha,margin_revision_id=design.get('confirmed_margin_revision_id') if design.get('confirmed_margin_revision_id')==design.get('current_margin_revision_id') else None,
        axis_revision_id=basis['axis_revision_id'],patient_frame_revision_id=frame_id,target_fdi=design['target_fdi'],
        seed=confirmation.get('seed') if current else None,confirmed_loop_ids=confirmation.get('confirmed_loop_ids',[]) if current else (),indication=indication)
    contract.case_id=design['case_id'];contract.provenance['confirmation_basis']=basis
    contract.source_revision_id=design.get('design_revision')
    if load_error:
        contract.readiness='REVIEW_REQUIRED';contract.rejection_code='SUPPORT_EXTRACTION_AMBIGUOUS';contract.rejection_reason=load_error
    if source_exists and file_digest(reference['path'])!=source_sha:
        raise HTTPException(409,'支撑提取期间源文件发生变化，请重新检查')
    metadata=json.loads(reference.get('mesh_info_json') or '{}') if reference else {}
    if metadata.get('implant_candidate') and not metadata.get('restorative_abutment_confirmed'):
        contract.readiness='REVIEW_REQUIRED';contract.rejection_code='SUPPORT_EXTRACTION_AMBIGUOUS'
        contract.rejection_reason='尚未确认这是实际修复基台，不能把扫描定位件作为支撑';prepared=None
    contract.provenance['support_confirmation_revision_id']=confirmation.get('revision_id') if current else None
    if require_ready and contract.readiness!='ABUTMENT_SUPPORT_READY':
        raise HTTPException(422,{'message':contract.rejection_reason or contract.rejection_code,'support_contract':contract.as_report(),'rejection_code':contract.rejection_code})
    return contract,prepared
