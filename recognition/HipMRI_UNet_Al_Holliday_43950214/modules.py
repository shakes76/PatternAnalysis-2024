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
    
    Two things I could try:
        1. Structure like Ronneberger et. al. [1], that is 2 3x3 Conv2ds w/ ReLUs followed
        by a 2x2 Max Pool
        
        2. Structure like Sakboonyara and Taeprasartsit [2], which is a 3x3 Conv2d with ReLU, then
        a spatial dropout 25%, then batch norm, then another 3x3 Conv2d followed by a 2x2 Max Pool.
    
    References:
    [1] O. Ronneberger, P. Fischer, and T. Brox, “U-Net: Convolutional Networks for Biomedical
    Image Segmentation,” in Medical Image Computing and Computer-Assisted Intervention – MICCAI 2015,
    N. Navab, J. Hornegger, W. M. Wells, and A. F. Frangi, Eds., Cham: Springer International Publishing,
    2015, pp. 234–241. doi: 10.1007/978-3-319-24574-4_28.
 
    [2] B. Sakboonyara and P. Taeprasartsit, “U-Net and Mean-Shift Histogram for Efficient Liver
    Segmentation from CT Images,” in 2019 11th International Conference on Knowledge and Smart Technology
    (KST), Jan. 2019, pp. 51–56. doi: 10.1109/KST.2019.8687816.

    """
    def __init__(self, inDim, dim, kernSz):
        super().__init__()
        self.conv1 = nn.Conv2d(inDim, dim, kernel_size = kernSz)
        self.act1 = nn.ReLU()
        self.conv2 = nn.Conv2d(dim, dim, kernel_size = kernSz)
        self.act2 = nn.ReLU()
    
    def forward(self, x):
        c1 = self.act1(self.conv1(x))
        return self.act2(self.conv2(c1))

class UNet(nn.Module):
    """
    U-Net class. Basically going to consist of BasicBlocks (going off the 
    diagram from the U-Net paper).
    """
    def __init__(self):
        super().__init__()
        # encoder
        self.blk1 = UNetBasicBlock(1, 64, 3)
        self.blk2 = UNetBasicBlock(64, 128, 3)
        self.blk3 = UNetBasicBlock(128, 256, 3)
        self.blk4 = UNetBasicBlock(256, 512, 3)
        # latent
        self.latent = UNetBasicBlock(512, 1024, 3)
        # decoder
    
    def forward(self, x):
        # just the encoder pass first
        enc1 = self.blk1(x)
        maxPool1 = F.max_pool2d(enc1, 2)
        enc2 = self.blk2(maxPool1)
        maxPool2 = F.max_pool2d(enc2, 2)
        enc3 = self.blk3(maxPool2)
        maxPool3 = F.max_pool2d(enc3, 2)
        enc4 = self.blk4(maxPool3)
        maxPool4 = F.max_pool2d(enc4, 2)
        return maxPool4
        #return self.latent(maxPool4)
        # skip == blk1 + blk of some other level