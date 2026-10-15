from pathlib import Path
import argparse,torch
from torch.utils.data import DataLoader,random_split
from tqdm import tqdm
from dataset import CrownNPZDataset
from model import CrownDeformer

def chamfer(a,b):
    d=torch.cdist(a,b)
    return d.min(2).values.mean()+d.min(1).values.mean()

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--data",required=True)
    ap.add_argument("--out",default="checkpoints/crown_deformer.pt")
    ap.add_argument("--epochs",type=int,default=100)
    ap.add_argument("--batch",type=int,default=4)
    ap.add_argument("--lr",type=float,default=2e-4)
    a=ap.parse_args()

    ds=CrownNPZDataset(a.data)
    if len(ds)<2: raise RuntimeError("训练至少需要 2 个真实病例；生产模型需要远多于此")
    nv=max(1,int(len(ds)*.1));nt=len(ds)-nv
    train,val=random_split(ds,[nt,nv],generator=torch.Generator().manual_seed(42))
    dl=DataLoader(train,batch_size=a.batch,shuffle=True,num_workers=0)
    vl=DataLoader(val,batch_size=a.batch,shuffle=False,num_workers=0)

    dev="cuda" if torch.cuda.is_available() else "cpu"
    model=CrownDeformer().to(dev)
    opt=torch.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=1e-4)

    best=1e9
    out=Path(a.out);out.parent.mkdir(parents=True,exist_ok=True)
    for ep in range(a.epochs):
        model.train(); tr=0
        for x in tqdm(dl,desc=f"train {ep+1}/{a.epochs}"):
            c=x["context"].to(dev);t=x["template"].to(dev);y=x["target"].to(dev);f=x["fdi"].to(dev)
            pred=model(c,t,f)
            loss=chamfer(pred,y)+0.02*((pred-t)**2).mean()
            opt.zero_grad();loss.backward();opt.step();tr+=loss.item()
        model.eval();va=0
        with torch.no_grad():
            for x in vl:
                pred=model(x["context"].to(dev),x["template"].to(dev),x["fdi"].to(dev))
                va+=chamfer(pred,x["target"].to(dev)).item()
        va/=max(1,len(vl))
        print("epoch",ep+1,"train",tr/max(1,len(dl)),"val",va)
        if va<best:
            best=va;torch.save({"state_dict":model.state_dict(),"val":va},out)
    print("best checkpoint:",out,"val",best)

if __name__=="__main__":main()
