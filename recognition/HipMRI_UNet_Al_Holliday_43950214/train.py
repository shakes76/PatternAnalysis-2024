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
import matplotlib.pyplot as plt

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

net = modules.UNet(chan, outDim, segDim = 6)
net = net.to(dev)

lossFunc = nn.CrossEntropyLoss()
optm = torch.optim.SGD(net.parameters())

net.train()
for epoch in range(epochs):
    for i, (img, seg) in enumerate(trainLoader):
        #print(i)
        #plt.imshow(seg[0].squeeze().numpy())
        img = img.to(dev)
        #seg = nn.functional.one_hot(seg.long())
        seg = seg.squeeze().long()
        seg = seg.to(dev)
        
        out = net(img)
        #print(out.shape)
        # This is no longer necessary!
        #seg = TF.center_crop(seg, output_size = out.size(2))
        loss = lossFunc(out, seg)
        optm.zero_grad()
        loss.backward()
        optm.step()
        if (i+1) % 100 == 0:
            print ("Epoch [{}/{}], Loss: {:.5f}"
                    .format(epoch+1, epochs, loss.item()))
            #plt.imshow(out[0].cpu().detach().squeeze().numpy())

print("Done")
# save the weights
torch.save(net.state_dict(), "./weights.pth")

# one option is to upsize the output to the size of the original.
# this probably won't work well as it's just blowing up a smaller image
# and won't line up exactly with the target segment map.
#out = trans(out)
# Another option is to crop the target. This will probably work better
#seg = TF.center_crop(seg, output_size = out.size(1))
#loss = lossFunc(out[0], seg[0]) # probably will not work
    
