from pathlib import Path

import numpy as np
import pytest
import trimesh

from app.geometry.implant_candidate import component_at_point, implant_candidates
from app.geometry.io import load_mesh


def ring_cylinder(radius=2.2, height=10, sections=128, layers=12):
    theta=np.linspace(0,2*np.pi,sections,endpoint=False)
    z=np.linspace(-height/2,height/2,layers)
    vertices=np.array([[radius*np.cos(t),radius*np.sin(t),level] for level in z for t in theta])
    faces=[]
    for layer in range(layers-1):
        for i in range(sections):
            a=layer*sections+i;b=layer*sections+(i+1)%sections
            c=(layer+1)*sections+i;d=(layer+1)*sections+(i+1)%sections
            faces.extend(((a,b,c),(b,d,c)))
    vertices=np.vstack((vertices,[[0,0,-height/2],[0,0,height/2]]))
    for i in range(sections):
        faces.append((len(vertices)-2,(i+1)%sections,i))
        faces.append((len(vertices)-1,(layers-1)*sections+i,(layers-1)*sections+(i+1)%sections))
    return trimesh.Trimesh(vertices=vertices,faces=faces,process=True)


def test_selected_component_does_not_mix_adjacent_gingiva():
    jaw=trimesh.creation.annulus(r_min=2.35,r_max=12,height=10,sections=128)
    implant=ring_cylinder()
    scan=trimesh.util.concatenate((jaw,implant))
    picked=component_at_point(scan,[2.2,0,0])
    assert len(picked.faces)==len(implant.faces)
    assert len(picked.split(only_watertight=False))==1


def test_synthetic_component_is_localized_but_ellipsoid_is_not():
    jaw=trimesh.creation.annulus(r_min=2.25,r_max=12,height=10,sections=512)
    implant=ring_cylinder()
    result=implant_candidates(trimesh.util.concatenate((jaw,implant)))
    assert len(result)==1
    assert result[0][1]['confidence']=='high'
    assert abs(np.dot(result[0][1]['axis'],[0,0,1]))>.95
    tooth=trimesh.creation.icosphere(subdivisions=3,radius=3)
    tooth.apply_scale([1,1,1.5])
    assert not any(row['confidence']=='high' for _,row in implant_candidates(trimesh.util.concatenate((jaw,tooth))))


def test_three_real_disconnected_components_and_crown_controls():
    data=Path(r'D:\code\牙冠自动生成\1.0测试数据')
    if not (data/'基台测试1-46'/'world_result.stl').is_file():
        pytest.skip('optional local acceptance meshes unavailable')
    expected=[[-.157,.980,-.119],[.059,.969,-.239],[.165,.980,-.112]]
    for index,fdi in enumerate(('基台测试1-46','基台测试2-36','基台测试3-46')):
        folder=data/fdi
        candidates=implant_candidates(load_mesh(folder/'world_result.stl'),load_mesh(folder/'upperjaw_upper.stl'))
        assert len(candidates)==1
        component,result=candidates[0]
        assert result['confidence']=='high'
        assert np.dot(result['axis'],expected[index])>.94
        assert len(component.faces)>15000
    reference=Path(r'D:\code\牙冠自动生成\数据\1629-王煜-STL')
    crown_files=(reference/'03_牙冠解剖参考'/'Crown Unn19.stl',
                 reference/'03_牙冠解剖参考'/'UNN19.stl',
                 reference/'01_主要病例模型'/'1629-王煜 1.stl')
    if not all(path.is_file() for path in crown_files):
        pytest.skip('optional local crown controls unavailable')
    scan=load_mesh(data/'基台测试1-46'/'world_result.stl')
    jaw,component=sorted(scan.split(only_watertight=False),key=lambda mesh:-len(mesh.faces))[:2]
    opposing=load_mesh(data/'基台测试1-46'/'upperjaw_upper.stl')
    for path in crown_files:
        crown=load_mesh(path)
        crown.apply_translation(component.centroid-crown.centroid)
        controls=implant_candidates(trimesh.util.concatenate((jaw,crown)),opposing)
        assert not any(row['confidence']=='high' for _,row in controls),path.name
