import torch
import torch.nn as nn
from functools import partial
from collections import OrderedDict
import math

"""
FeedForward Network
"""
class Mlp(nn.Module):
    """
    credit: https://www.youtube.com/watch?v=ovB0ddFtzzA&ab_channel=mildlyoverfitted
    """

    def __init__(self, in_features, hidden_features, out_features, drop=0.):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x