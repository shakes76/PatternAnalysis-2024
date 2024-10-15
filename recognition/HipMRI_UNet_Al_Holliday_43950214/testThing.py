# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 09:35:54 2024

@author: al
"""

import numpy as np
import torch
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

import modules
import dataset



dev = torch.device('cuda')

hipmri2d = dataset.HipMRI2d("C:\\Users\\al\\HipMRI", transform = transforms.Resize((256, 256)), applyTrans = True)
hipMriLoader = DataLoader(hipmri2d, batch_size=64, shuffle=False)
trainImg, segments = next(iter(hipMriLoader))
firstImg = trainImg[0]
plt.imshow(firstImg.squeeze())
firstSeg = segments[0]
plt.figure()
plt.imshow(firstSeg.squeeze())
firstImg = firstImg.to(dev)

chan = 1
outDim = 64

#basicBlk = modules.UNetBasicBlock(chan, outDim, 3)
#basicBlk = basicBlk.to(dev) # WHY DO I HAVE TO DO IT THIS WAY? WHY CAN'T 'to()' BE IN-PLACE?

#out = basicBlk(firstImg)

net = modules.UNet(chan, outDim)
net = net.to(dev)
out = net(firstImg)