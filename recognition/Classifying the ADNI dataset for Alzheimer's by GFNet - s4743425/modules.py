""" This file contains the source code of the components of this model.
Each component is implemented as a class or in a function.

The general structure of this vision Transformer was made in assistance by following 
these sources:

Shengjie, Z., Xiang, C., Bohan, R., & Haibo, Y. (2022, August 29). 
3D Global Fourier Network for Alzheimer’s Disease Diagnosis using Structural MRI. MICCAI 2022
 - Accepted Papers and Reviews. https://conferences.miccai.org/2022/papers/002-Paper1233.html

‌
"""
import torch
import torch.nn as nn
import torch.fft
from functools import partial
from collections import OrderedDict

# MLP Block similar to the example
class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.5):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.act(x)  # Additional activation layer
        x = self.drop(x)
        return x
    
# Global Filter
class Global_Filter(nn.Module):
    def __init__(self, h=9, w=10, d=5, dim=1000):
        super().__init__()
        # Learnable complex weight parameter for FFT
        self.complex_weight = nn.Parameter(torch.randn(
            h, w, d//2+1, dim, 2, dtype=torch.float32) * 0.02)
        self.dim = dim
        self.h = h
        self.w = w
        self.d = d

    def forward(self, x):
        B, N, C = x.shape
        x = x.to(torch.float32)
        x = x.view(B, self.h, self.w, self.d, self.dim)
        # Forward FFT
        x = torch.fft.rfftn(x, dim=(1, 2, 3), norm='ortho')
        weight = torch.view_as_complex(self.complex_weight)
        x = x * weight
        # Inverse FFT
        x = torch.fft.irfftn(x, s=(self.h, self.w, self.d), dim=(1, 2, 3), norm='ortho')
        x = x.reshape(B, N, C)
        return x