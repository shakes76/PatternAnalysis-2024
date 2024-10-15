# -*- coding: utf-8 -*-
"""
Training script. Initially I'm just adapting Shakes' training code from the CNN example for
MNIST: https://colab.research.google.com/drive/1K2kiAJSCa6IiahKxfAIv4SQ4BFq7YDYO?usp=sharing

@author: al
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF

import dataset
import modules

chan = 1
outDim = 64

epochs = 8 # something small for now
dev = torch.device("cuda")
trans = transforms.Resize((256, 256))
# for Woomy
#hipmri2dtrain = dataset.HipMRI2d("C:\\Users\\al\\HipMRI", imgSet = "train", transform = trans, applyTrans = True)
#hipmri2dtest = dataset.HipMRI2d("C:\\Users\\al\\HipMRI", imgSet = "test", transform = trans, applyTrans = True)
# for the lab computers
hipmri2dtrain = dataset.HipMRI2d("H:\\HipMRI", imgSet = "train", transform = trans, applyTrans = True)
hipmri2dtest = dataset.HipMRI2d("H:\\HipMRI", imgSet = "test", transform = trans, applyTrans = True)
# for Rangpur
#hipmri2dtrain = dataset.HipMRI2d("/home/groups/comp3710/HipMRI_Study_open/keras_slices_data", imgSet = "train", transform = trans, applyTrans = True)
#hipmri2dtest = dataset.HipMRI2d("/home/groups/comp3710/HipMRI_Study_open/keras_slices_data", imgSet = "test", transform = trans, applyTrans = True)
trainLoader = DataLoader(hipmri2dtrain, batch_size=8, shuffle = False)

net = modules.UNet(chan, outDim)
net = net.to(dev)

lossFunc = nn.CrossEntropyLoss()
optm = torch.optim.SGD(net.parameters())

net.train()
for epoch in range(epochs):
    for i, (img, seg) in enumerate(trainLoader):
        #print(i)
        img = img.to(dev)
        seg = seg.to(dev)
        out = net(img)
        # This is no longer necessary!
        #seg = TF.center_crop(seg, output_size = out.size(2))
        loss = lossFunc(out, seg)
        optm.zero_grad()
        loss.backward()
        optm.step()
        if (i+1) % 100 == 0:
            print ("Epoch [{}/{}]"
                    .format(epoch+1, epochs))
            print("loss: ", loss.item())

print("Done")
# save the weights
torch.save(net.state_dict(), "./weights.pth")

# test Cross Entropy Loss
#imgBatch, segBatch = next(iter(trainLoader))
#img = imgBatch[0].to(dev)
#seg = segBatch[0].to(dev)

#out = net(img)
# one option is to upsize the output to the size of the original.
# this probably won't work well as it's just blowing up a smaller image
# and won't line up exactly with the target segment map.
#out = trans(out)
# Another option is to crop the target. This will probably work better
#seg = TF.center_crop(seg, output_size = out.size(1))
#loss = lossFunc(out[0], seg[0]) # probably will not work
    
