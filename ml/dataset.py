from pathlib import Path
import json, numpy as np
import torch
from torch.utils.data import Dataset

class CrownNPZDataset(Dataset):
    def __init__(self,root):
        self.files=sorted(Path(root).glob("*.npz"))
        if not self.files:
            raise RuntimeError("数据目录中没有 .npz")
    def __len__(self): return len(self.files)
    def __getitem__(self,i):
        z=np.load(self.files[i])
        return {
            "context":torch.tensor(z["context"],dtype=torch.float32),
            "template":torch.tensor(z["template"],dtype=torch.float32),
            "target":torch.tensor(z["target"],dtype=torch.float32),
            "fdi":torch.tensor(int(z["fdi"]),dtype=torch.long)
        }
