"""Read-only real input probe; imported source Margin is not a human confirmation."""
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
import trimesh

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from app.geometry.io import load_mesh
from app.geometry.nonplanar_crown import MarginCurtain, audit_margin_on_scan, fitting_support_above_margin
from app.geometry.crown_outer.collar_body_outer import _clip_by_scalar, _boundary_loops
from app.geometry.margin_seam import margin_plane_diagnostics


def main():
    source=Path(sys.argv[1]);output=Path(sys.argv[2])
    case=json.loads((source/'case.json').read_text(encoding='utf-8'))
    margin=np.asarray(case['margin_points']);axis=case['insertion_axis']
    curtain=MarginCurtain(margin,axis)
    report={'case_id':case['case_uuid'],'status':'ENGINEERING_INPUT_PROBE',
        'human_confirmed_in_workbench':False,'production_status':'NOT_PRODUCTION_APPROVED',
        'margin_diagnostics':margin_plane_diagnostics(margin),'inputs':{}}
    for name in ('prep.stl','die_prep_frame.stl'):
        path=source/name;mesh=load_mesh(path)
        entry={'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
            'components':len(mesh.split(only_watertight=False)),'watertight':bool(mesh.is_watertight)}
        try: entry['scan_audit']=audit_margin_on_scan(mesh,margin)
        except ValueError as error: entry['scan_audit']={'status':'BLOCKED','reason':str(error)}
        vertices,faces,_=_clip_by_scalar(mesh.vertices,mesh.faces,-curtain.axial_above(mesh.vertices))
        if len(faces): faces=faces[curtain.axial_above(vertices)[faces].max(axis=1)>1e-9]
        cropped=trimesh.Trimesh(vertices,faces,process=False)
        entry['clipped_components_faces']=sorted([len(part.faces) for part in cropped.split(only_watertight=False)],reverse=True)
        entry['clipped_boundary_loops']=len(_boundary_loops(faces))
        try: _,entry['fitting_domain']=fitting_support_above_margin(mesh,margin,axis)
        except ValueError as error: entry['fitting_domain']={'status':'BLOCKED','reason':str(error)}
        report['inputs'][name]=entry
    report['case_json_sha256']=hashlib.sha256((source/'case.json').read_bytes()).hexdigest()
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False))


if __name__=='__main__': main()
