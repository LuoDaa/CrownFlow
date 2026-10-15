"""
把真实历史病例整理为训练 NPZ。

输入目录约定（不包含任何样例数据）:
raw/
  case_x/
    upper.stl
    lower.stl
    support.stl
    crown_outer_gt.stl
    meta.json      {"fdi":46}

templates/
  FDI_46.stl ...

说明:
- target 是技师/医生确认的外表面 Crown Outer；最终空心冠不可直接作为外表面标签。
- 每个病例必须有明确 target_fdi。
- 本脚本只做点采样和 support-centered normalization，不伪造标签。
"""
from pathlib import Path
import argparse,json,numpy as np,trimesh

def sample(mesh,n,rng):
    pts,_=trimesh.sample.sample_surface(mesh,n)
    if len(pts)!=n: raise RuntimeError("surface sample failed")
    return pts

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--raw",required=True)
    ap.add_argument("--templates",required=True)
    ap.add_argument("--out",required=True)
    ap.add_argument("--context-points",type=int,default=6144)
    ap.add_argument("--template-points",type=int,default=2048)
    ap.add_argument("--target-points",type=int,default=2048)
    a=ap.parse_args()
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    rng=np.random.default_rng(7)

    made=0
    for case in sorted(Path(a.raw).iterdir()):
        if not case.is_dir(): continue
        need=["upper.stl","lower.stl","support.stl","crown_outer_gt.stl","meta.json"]
        if not all((case/x).exists() for x in need): continue
        meta=json.loads((case/"meta.json").read_text(encoding="utf-8"))
        fdi=int(meta["fdi"])
        tp=Path(a.templates)/f"FDI_{fdi}.stl"
        if not tp.exists(): continue

        upper=trimesh.load(case/"upper.stl",force="mesh",process=True)
        lower=trimesh.load(case/"lower.stl",force="mesh",process=True)
        support=trimesh.load(case/"support.stl",force="mesh",process=True)
        gt=trimesh.load(case/"crown_outer_gt.stl",force="mesh",process=True)
        template=trimesh.load(tp,force="mesh",process=True)

        n1=a.context_points//3;n2=a.context_points//3;n3=a.context_points-n1-n2
        ctx=np.concatenate([sample(upper,n1,rng),sample(lower,n2,rng),sample(support,n3,rng)])
        center=np.asarray(support.vertices).mean(0)
        scale=float(np.linalg.norm(support.extents))
        if scale<1e-6: continue

        context=(ctx-center)/scale
        target=(sample(gt,a.target_points,rng)-center)/scale

        tv=sample(template,a.template_points,rng)
        tc=np.asarray(template.vertices).mean(0);ts=float(np.linalg.norm(template.extents))
        template_n=(tv-tc)/max(ts,1e-6)

        np.savez_compressed(out/f"{case.name}.npz",
                           context=context.astype("float32"),
                           template=template_n.astype("float32"),
                           target=target.astype("float32"),
                           fdi=np.int64(fdi))
        made+=1
    if made == 0:
        raise RuntimeError("没有找到含 crown_outer_gt.stl 的完整病例；最终冠 STL 不能直接作为外形标签")
    print("built",made,"real training samples")

if __name__=="__main__": main()
