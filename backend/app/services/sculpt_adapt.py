"""Tooth-part adaptation delegates to the existing frozen-domain fit stage.

语义区域（MESIAL / DISTAL / OCCLUSAL）**自动建立**：人工标注优先，没有人工标注时用
`geometry.crown_regions` 从冠几何 + 就位轴 + 牙位朝向得到（不依赖世界 XYZ）。
缺一侧邻牙不是错误：该侧返回 ``NO_NEIGHBOR`` 并跳过，其余侧照常适配。
"""
import json
import numpy as np
from .contact_measurement import confirmed_neighbor_surfaces,confirmed_antagonist_surface,neighbor_patch_diagnostics
from .sculpt_commit import region_masks
from ..geometry.crown_regions import RegionMapError,build_region_map,region_weights
from ..geometry.crown_outer.auto_fit_adapter import fit_stage
from .proximal_coarse import proximal_coarse_fit, proximal_snapshot
from ..geometry.crown_shape_quality import deformation_shape_report, shape_gate


def adapt_sculpt(design,mesh,frame,ref,state,request):
    neighbors,neighbor_reports=confirmed_neighbor_surfaces(design,ref,strict=False)
    sides=['mesial','distal'] if request.operation=='cut_intersections' else ['antagonist'] if request.operation=='adapt_antagonist' else [request.operation.removeprefix('adapt_')]
    result=mesh.copy();steps=[];skipped=[];region_map=None;region_report={'status':'NOT_COMPUTED'};proximal=[]
    margin=json.loads(design['margin_json']);parameters=json.loads(design['parameters_json'])
    for side in sides:
        region='occlusal' if side=='antagonist' else side
        if side=='antagonist':
            surface,_,error=confirmed_antagonist_surface(design,ref,strict=True)
            if error or surface is None:raise ValueError('NOT_AVAILABLE：静态咬合输入尚未完成')
            stage='OCCLUSAL';key='occlusal'
        else:
            if side not in neighbors:
                # 缺牙不是算法错误：明确标记该侧跳过（前端显示“该侧无邻牙”）。
                skipped.append({'side':side,'status':'NO_NEIGHBOR','reason':neighbor_reports.get(side,{}).get('status','NOT_CONFIRMED')})
                continue
            surface=neighbors[side][0];stage='PROXIMAL';key=side
        if region_map is None:
            region_map=build_region_map(design,mesh,frame,ref,neighbors=neighbors,neighbor_reports=neighbor_reports)
            region_report=region_map.report()
        try:
            weights,_source=region_weights(state,region,region_map)
        except RegionMapError as exc:
            raise ValueError('REVIEW_REQUIRED：无法确定 %s 语义区域（%s）'%(region,getattr(exc,'code',exc))) from exc
        target=0 if request.operation=='cut_intersections' else request.target_distance_mm
        if target is None:target=parameters.get('occlusal_target_mm' if side=='antagonist' else 'proximal_target_mm')
        if target is None:raise ValueError('请明确目标间隙')
        lock=np.asarray(region_map.protection)
        masks=np.asarray(region_masks(result,weights,lock));domain=(weights>.5)&(lock>0)
        if not domain.any():
            skipped.append({'side':side,'status':'REGION_EMPTY','reason':'AUTO_REGION_EMPTY_'+region.upper()})
            continue
        if side in ('mesial','distal'):
            patch_report=neighbor_patch_diagnostics(result,region_map,{side:neighbors[side]})[side]
            if patch_report['status']!='VALID':
                skipped.append({'side':side,'status':'INVALID_NEIGHBOR_PATCH','validity':patch_report})
                continue
            if request.operation!='cut_intersections':
                antagonist,_,_=confirmed_antagonist_surface(design,ref,strict=False)
                opposite='distal' if side=='mesial' else 'mesial'
                opposite_domain=region_map.domain(opposite)
                opposite_surface=neighbors[opposite][0] if opposite in neighbors else None
                original_vertices=np.asarray(result.vertices).copy()
                opposite_distance_before=float(np.min(opposite_surface.query(original_vertices)[3])) if opposite_surface is not None else None
                reference={'vertices':original_vertices,'locked':np.asarray(region_map.protection)<=0}
                before=proximal_snapshot(result,margin,frame,surface,domain,reference=reference,antagonist=antagonist)
                coarse,coarse_report=proximal_coarse_fit(result,margin,frame,region_map,side,surface,domain,
                    protected=masks=='FIXED',antagonist=antagonist,opposite_domain=opposite_domain,
                    opposite_surface=opposite_surface)
                if coarse_report['status']=='CONSTRAINT_CONFLICT':
                    raise ValueError('近远中粗适配未达到安全距离，已回滚：'+str(coarse_report))
                result=coarse
                after_coarse=proximal_snapshot(result,margin,frame,surface,domain,reference=reference,antagonist=antagonist)
        fine_rounds=8 if side in ('mesial','distal') and request.operation!='cut_intersections' else 1
        for _ in range(fine_rounds):
            fitted,report,_=fit_stage(result,frame,np.asarray(margin),{key:surface},{key:float(target)},stage,
                protected=masks=='FIXED',domain_masks={key:domain},cut_only=request.operation=='cut_intersections',
                transition_rings=7 if stage=='PROXIMAL' else 3)
            if report['status']!='PASS_ENGINEERING':raise ValueError('适配约束冲突，已回滚：'+str(report))
            _,_,old,_,valid0=surface.query(result.vertices);_,_,new,_,valid1=surface.query(fitted.vertices)
            if not valid1[valid0].all() or np.any(np.maximum(-new[valid0],0)>np.maximum(-old[valid0],0)+1e-6):
                raise ValueError('适配增加了穿透或失去有效符号，已回滚')
            result=fitted;steps.append(report)
            if fine_rounds==1:break
            current=proximal_snapshot(result,margin,frame,surface,domain)
            if current['domain_signed_min_mm'] is not None and current['domain_signed_min_mm']<=float(target)+.05:break
        if side in ('mesial','distal') and request.operation!='cut_intersections':
            after_fine=proximal_snapshot(result,margin,frame,surface,domain,reference=reference,antagonist=antagonist)
            if after_fine['domain_signed_min_mm'] is None or after_fine['domain_signed_min_mm']>float(target)+.05:
                raise ValueError('近远中精适配未达到目标间隙，已回滚')
            ant0,ant1=before['antagonist_min_mm'],after_fine['antagonist_min_mm']
            opposite_drift=float(np.max(np.linalg.norm(result.vertices[opposite_domain]-original_vertices[opposite_domain],axis=1))) if opposite_domain.any() else 0.0
            opposite_distance_drift=(abs(float(np.min(opposite_surface.query(result.vertices)[3]))-opposite_distance_before)
                                     if opposite_surface is not None else 0.0)
            if (after_fine['margin_drift_mm']>=.01
                    or abs(after_fine['crown_height_mm']-before['crown_height_mm'])>=.05
                    or (ant0 is not None and ant1 is not None and abs(ant1-ant0)>=.10)
                    or opposite_drift>=.05 or opposite_distance_drift>=.05):
                raise ValueError('近远中适配违反颈缘/冠高/咬合/对侧保护约束，已回滚')
            total_displacement=np.linalg.norm(np.asarray(result.vertices)-original_vertices,axis=1)
            focus_faces=np.flatnonzero(np.any(total_displacement[np.asarray(result.faces)]>1e-9,axis=1))
            shape=deformation_shape_report(original_vertices,np.asarray(result.vertices),np.asarray(result.faces),focus_faces=focus_faces)
            shape_check=shape_gate(shape,max_edge_stretch=2.75,p95_edge_stretch=1.65,
                                   min_quality_floor=.025,relative_quality_floor=.40,
                                   min_area_ratio=.18,max_area_ratio=4.0)
            if not shape_check['ok']:
                raise ValueError('PROXIMAL_SHAPE_INVALID：邻牙距离虽改善，但冠体局部形态失真，已回滚：'+str({'shape_quality':shape,'shape_gate':shape_check}))
            proximal.append({'side':side,'patch_validity':patch_report,'before':before,
                             'after_coarse':after_coarse,'after_fine':after_fine,
                             'coarse':coarse_report,'shape_quality':shape,'shape_gate':shape_check,
                             'opposite_side_drift_mm':opposite_drift,
                             'opposite_distance_drift_mm':opposite_distance_drift})
    if not steps and skipped:
        # 请求的侧全部不可用：如实返回，不假装成功，也不报成“算法错误”。
        return mesh.copy(),{'affected_area_mm2':0,'max_displacement_mm':0,'p95_displacement_mm':0,
            'adapt_steps':[],'skipped_sides':skipped,'region_report':region_report,'status':'NO_APPLICABLE_SIDE',
            'margin_displacement_mm':0.0,'protected_vertex_count':int((np.asarray(region_map.protection)<=0).sum()) if region_map is not None else 0,
            'solver':'auto_fit_adapter.fit_stage'}
    displacement=np.linalg.norm(result.vertices-mesh.vertices,axis=1);faces=np.any(displacement[mesh.faces]>1e-9,axis=1)
    protection=np.asarray(region_map.protection)
    if any(step['stage']=='PROXIMAL' for step in steps):
        from ..geometry.nonplanar_crown import MarginCurtain
        curtain=MarginCurtain(np.asarray(margin,float),frame.z)
        height_change=abs(float(curtain.axial_above(result.vertices).max()
                                -curtain.axial_above(mesh.vertices).max()))
        if height_change>.05 or ((protection<=0).any() and displacement[protection<=0].max()>1e-9):
            raise ValueError('邻牙适配改变冠高或颈缘保护区，已回滚')
    return result,{'affected_area_mm2':float(mesh.area_faces[faces].sum()),'max_displacement_mm':float(displacement.max()),
        'p95_displacement_mm':float(np.quantile(displacement,.95)),'adapt_steps':steps,'solver':'auto_fit_adapter.fit_stage',
        'skipped_sides':skipped,'region_report':region_report,'proximal_coarse_fine':proximal,'status':'APPLIED',
        # 保护区的位移必须为 0：这是"只改外表面形态"的可验证口径。
        'margin_displacement_mm':float(displacement[protection<=0].max()) if (protection<=0).any() else 0.0,
        'protected_vertex_count':int((protection<=0).sum())}
