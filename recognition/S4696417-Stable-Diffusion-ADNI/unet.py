import torch
from torch import nn
import torch.nn.functional as F

class UNet(nn.Module):
    def __init__(self, in_channels=1, hidden_dims=[32, 64, 128, 256], time_emb_dim=256):
        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        self.ups.append(
            
        )