import argparse,torch
from model import CrownDeformer

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--checkpoint",required=True)
    ap.add_argument("--out",required=True)
    a=ap.parse_args()
    ck=torch.load(a.checkpoint,map_location="cpu")
    m=CrownDeformer();m.load_state_dict(ck["state_dict"]);m.eval()
    scripted=torch.jit.script(m)
    scripted.save(a.out)
    print("saved TorchScript:",a.out)

if __name__=="__main__":main()
