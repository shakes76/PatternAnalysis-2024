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
from dice import dice_coeff



dev = torch.device('cuda')

hipmri2d = dataset.HipMRI2d("H:\\HipMRI", imgSet = "test", transform = transforms.Resize((256, 256)), applyTrans = True)
hipMriLoader = DataLoader(hipmri2d, batch_size=8, shuffle=False)
trainImg, segments = next(iter(hipMriLoader))

print(np.max(segments.cpu().numpy()))

firstImg = trainImg[0]
plt.imshow(firstImg.squeeze())
firstSeg = segments[0]
plt.figure()
plt.imshow(firstSeg.squeeze())
firstImg = firstImg.to(dev)
firstSeg = firstSeg.to(dev)

chan = 1
outDim = 64
segDim = 6

trainImg = trainImg.to(dev)
#segments = segments.squeeze()
segments = segments.to(dev)

#basicBlk = modules.UNetBasicBlock(chan, outDim, 3)
#basicBlk = basicBlk.to(dev) # WHY DO I HAVE TO DO IT THIS WAY? WHY CAN'T 'to()' BE IN-PLACE?

#out = basicBlk(firstImg)

net = modules.UNet(chan, outDim, segDim = segDim)
net.load_state_dict(torch.load("H:\\python_work\\hipMri_unet_results\\weights_epoch_31.pth", weights_only = True))
net = net.to(dev)
net.eval()
out = net(trainImg)



# print a purrity picture
out1 = torch.permute(out, (0,2,3,1))
out2 = torch.argmax(out1, dim=-1)
out3 = out2[:,None, :, :].to(dev)
#plt.figure()
#plt.imshow(out2.cpu().numpy())

dc = dice_coeff(out, segments, dev, lbls = segDim)
print(dc.cpu().item())