# -*- coding: utf-8 -*-
"""
U-Net for the HipMRI 2D task

Created on Sat Oct 5 14:31:17 2024

@author: al
"""

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


class UNetBasicBlock(nn.Module):
    """
    Basic building block for the U-Net
    """
    def __init__(self):
        pass
    
    def forward(self, x):
        pass

class UNet(nn.Module):
    """
    U-Net class. Basically going to consist of BasicBlocks (going off the 
    diagram from the U-Net paper).
    """
    def __init__(self):
        pass
    
    def forward(self, x):
        pass