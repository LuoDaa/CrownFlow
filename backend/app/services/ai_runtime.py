from pathlib import Path
import numpy as np
import trimesh

def _torch():
    try:
        import torch
        return torch
    except Exception as e:
        raise RuntimeError("AI 运行需要安装 backend/requirements-ai.txt") from e

def ai_runtime_available(checkpoint:Path|None):
    if checkpoint is None or not checkpoint.exists():
        return False
    try:
        _torch()
        return True
    except Exception:
        return False

def _sample_surface(mesh,n,seed=17):
    if len(mesh.faces)==0:
        raise ValueError("mesh 为空")
    rng=np.random.default_rng(seed)
    areas=np.asarray(mesh.area_faces,float)
    p=areas/areas.sum()
    face_idx=rng.choice(len(mesh.faces),size=n,replace=True,p=p)
    tri=np.asarray(mesh.triangles)[face_idx]
    r1=np.sqrt(rng.random(n)); r2=rng.random(n)
    pts=(1-r1)[:,None]*tri[:,0]+(r1*(1-r2))[:,None]*tri[:,1]+(r1*r2)[:,None]*tri[:,2]
    return pts

def predict_outer(checkpoint:Path,template_path:Path,upper_path:Path|None,lower_path:Path|None,
                  support_path:Path,output_path:Path,target_fdi:int):
    torch=_torch()
    model=torch.jit.load(str(checkpoint),map_location="cpu")
    model.eval()

    if not upper_path or not lower_path:
        raise ValueError("AI 推理需要已配准的上下颚 STL")
    support=trimesh.load(support_path,force="mesh",process=True)
    template=trimesh.load(template_path,force="mesh",process=True)
    if not isinstance(support,trimesh.Trimesh) or not isinstance(template,trimesh.Trimesh):
        raise ValueError("AI 输入 mesh 无效")

    ctx=[]
    ctx.append(_sample_surface(trimesh.load(upper_path,force="mesh",process=True),2048,11))
    ctx.append(_sample_surface(trimesh.load(lower_path,force="mesh",process=True),2048,13))
    ctx.append(_sample_surface(support,2048,15))
    context=np.concatenate(ctx,axis=0)

    center=np.asarray(support.vertices).mean(0)
    scale=float(np.linalg.norm(support.extents))
    if scale<1e-6: raise ValueError("Support 尺寸异常")

    context_n=(context-center)/scale
    # Canonical template is centered/scaled independently; the network learns conditioned placement.
    tv=np.asarray(template.vertices,float)
    tc=tv.mean(0); ts=float(np.linalg.norm(template.extents))
    template_n=(tv-tc)/max(ts,1e-6)

    with torch.no_grad():
        c=torch.tensor(context_n,dtype=torch.float32)[None]
        t=torch.tensor(template_n,dtype=torch.float32)[None]
        f=torch.tensor([int(target_fdi)],dtype=torch.long)
        pred=model(c,t,f)[0].cpu().numpy()

    world=pred*scale+center
    out=trimesh.Trimesh(vertices=world,faces=template.faces,process=False)
    out.remove_unreferenced_vertices()
    output_path.parent.mkdir(parents=True,exist_ok=True)
    out.export(output_path)
    return out
