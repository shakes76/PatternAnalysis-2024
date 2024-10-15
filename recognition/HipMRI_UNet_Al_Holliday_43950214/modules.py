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
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF


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
        self.conv1 = nn.Conv2d(inDim, dim, kernel_size = kernSz, padding = 1)
        self.act1 = nn.ReLU()
        self.conv2 = nn.Conv2d(dim, dim, kernel_size = kernSz, padding = 1)
        self.act2 = nn.ReLU()
    
    def forward(self, x):
        c1 = self.act1(self.conv1(x))
        return self.act2(self.conv2(c1))

class UNet(nn.Module):
    """
    U-Net class. Basically going to consist of BasicBlocks (going off the 
    diagram from the U-Net paper).
    """
    def __init__(self, inDim, outDim, segDim = 4):
        super().__init__()
        # encoder
        self.blk1 = UNetBasicBlock(inDim, outDim, 3)
        self.blk2 = UNetBasicBlock(outDim, outDim * 2, 3)
        self.blk3 = UNetBasicBlock(outDim * 2, outDim * 4, 3)
        
        #self.blk4 = UNetBasicBlock(256, 512, 3)
        
        # latent
        #self.latent = UNetBasicBlock(512, 1024, 3)
        
        self.latent = UNetBasicBlock(outDim * 4, outDim * 8, 3)
        
        # decoder
        # A note to Ronneberger et. al.: Why didn't you explicitly say that the up convolutions also
        # had a stride of 2 in your paper? I was scratching my head over why my upconvs were not
        # doing much upsizing.
        
        #self.up1 = nn.ConvTranspose2d(1024, 512, 2, stride = 2) 
        #self.dec1 = UNetBasicBlock(1024, 512, 3)
        
        self.up2 = nn.ConvTranspose2d(outDim * 8, outDim * 4, 2, stride = 2)
        self.dec2 = UNetBasicBlock(outDim * 8 , outDim * 4, 3)
        self.up3 = nn.ConvTranspose2d(outDim * 4, outDim * 2, 2, stride = 2)
        self.dec3 = UNetBasicBlock(outDim * 4, outDim * 2, 3)
        self.up4 = nn.ConvTranspose2d(outDim * 2, outDim, 2, stride = 2)
        self.dec4 = UNetBasicBlock(outDim * 2, outDim, 3)
        self.out = nn.Conv2d(outDim, segDim, kernel_size = 1)
    
    def forward(self, x):
        enc1 = self.blk1(x) # size (64, 252, 252)
        print("enc1: ", enc1.shape)
        maxPool1 = F.max_pool2d(enc1, 2) # size (64, 126, 126)
        print("maxPool1: ", maxPool1.shape)
        enc2 = self.blk2(maxPool1) # size (128, 122, 122)
        print("enc2: ", enc2.shape)
        maxPool2 = F.max_pool2d(enc2, 2) # size (128, 61, 61)
        print("maxPool2: ", maxPool2.shape)
        enc3 = self.blk3(maxPool2) # size (256, 57, 57)
        print("enc3: ", enc3.shape)
        maxPool3 = F.max_pool2d(enc3, 2) # size (256, 28, 28)
        print("maxPool3: ", maxPool3.shape)
        # bypassing the lowest layer in the 'U' before the latent space, that's why it's commented out
        #enc4 = self.blk4(maxPool3) # size (512, 24, 24)
        #maxPool4 = F.max_pool2d(enc4, 2) # size (512, 12, 12)
        #lat = self.latent(maxPool4) # size (1024, 8, 8)
        lat = self.latent(maxPool3)
        print("lat: ", lat.shape)
        #up1 = self.up1(lat) # size (512, 16, 16)
        #dec1 = self.dec1(torch.concat(
        #    (TF.center_crop(enc4, output_size=up1.size(1)), up1), 0)) # size (512, 12, 12)
        #up2 = self.up2(dec1) # size (256, 24, 24)
        up2 = self.up2(lat)
        print("up2: ", up2.shape)
        # to handle batches properly
        if len(up2.size()) == 4:
            cropSz = up2.size(2)
            chanDim = 1
        else:
            cropSz = up2.size(1)
            chanDim = 0
        dec2 = self.dec2(torch.concat(
            (TF.center_crop(enc3, output_size=cropSz), up2), chanDim)) # size (256, 20, 20)
        print("dec2: ", dec2.shape)
        up3 = self.up3(dec2) # size (128, 40, 40)
        print("up3: ", up3.shape)
        if len(up3.size()) == 4:
            cropSz = up3.size(2)
            chanDim = 1
        else:
            cropSz = up3.size(1)
            chanDim = 0
        dec3 = self.dec3(torch.concat(
            (TF.center_crop(enc2, output_size=cropSz), up3), chanDim)) # size (128, 36, 36)
        print("dec3: ", dec3.shape)
        up4 = self.up4(dec3) # size (64, 72, 72)
        print("up4: ", up4.shape)
        if len(up4.size()) == 4:
            cropSz = up4.size(2)
            chanDim = 1
        else:
            cropSz = up4.size(1)
            chanDim = 0
        dec4 = self.dec4(torch.concat(
            (TF.center_crop(enc1, output_size=cropSz), up4), chanDim)) # size (64, 68, 68)
        print("dec4: ", dec4.shape)
        return F.softmax(self.out(dec4), dim = chanDim) # size (2, 68, 68)
        # skip == blk1 + blk of some other level