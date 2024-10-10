import math
import numpy.lib.arraypad as pad
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.container import Sequential


## Using RaoyonGming GFNet github repo as starting point for model architecture code
## https://github.com/raoyongming/GFNet/blob/master/gfnet.py
class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, dropout=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fnc1 = nn.Linear(in_features, hidden_features)
        self.activation = nn.GELU
        self.fnc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fnc1(x)
        x = self.activation(x)
        x = self.drop(x)
        x = self.fnc2(x)
        x = self.drop(x)
        return x
