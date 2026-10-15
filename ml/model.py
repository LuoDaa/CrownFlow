import torch
from torch import nn

FDI_VALUES=[18,17,16,15,14,13,12,11,21,22,23,24,25,26,27,28,38,37,36,35,34,33,32,31,41,42,43,44,45,46,47,48]

class PointEncoder(nn.Module):
    def __init__(self,out_dim=256):
        super().__init__()
        self.net=nn.Sequential(
            nn.Linear(3,64),nn.ReLU(),
            nn.Linear(64,128),nn.ReLU(),
            nn.Linear(128,out_dim),nn.ReLU()
        )
    def forward(self,x):
        return self.net(x).max(dim=1).values

class CrownDeformer(nn.Module):
    """
    context: [B,N,3] normalized patient+support points
    template: [B,M,3] normalized FDI template vertices/points
    fdi: [B] actual FDI number, e.g. 46
    output: [B,M,3] predicted normalized crown points using template topology
    """
    def __init__(self,latent=256,embed=32):
        super().__init__()
        self.context=PointEncoder(latent)
        self.fdi_embed=nn.Embedding(64,embed)
        self.decoder=nn.Sequential(
            nn.Linear(3+latent+embed,256),nn.ReLU(),
            nn.Linear(256,256),nn.ReLU(),
            nn.Linear(256,128),nn.ReLU(),
            nn.Linear(128,3)
        )
    def forward(self,context,template,fdi):
        g=self.context(context)
        e=self.fdi_embed(fdi.clamp(0,63))
        cond=torch.cat([g,e],dim=-1)
        cond=cond[:,None,:].expand(-1,template.shape[1],-1)
        delta=self.decoder(torch.cat([template,cond],dim=-1))
        return template+delta
