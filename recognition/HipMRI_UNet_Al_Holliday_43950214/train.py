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

import dataset
import modules

epochs = 8 # something small for now
dev = torch.device("cuda")
trans = transforms.Resize((256, 256))
# for the lab computers
hipmri2dtrain = dataset.HipMRI2d("H:\\HipMRI", imgSet = "train", transform = trans, applyTrans = True)
hipmri2dtest = dataset.HipMRI2d("H:\\HipMRI", imgSet = "test", transform = trans, applyTrans = True)
# for rangpur
#hipmri2dtrain = dataset.HipMRI2d("/home/groups/comp3710/HipMRI_Study_open/keras_slices_data", imgSet = "train", transform = trans, applyTrans = True)
#hipmri2dtest = dataset.HipMRI2d("/home/groups/comp3710/HipMRI_Study_open/keras_slices_data", imgSet = "test", transform = trans, applyTrans = True)
trainLoader = DataLoader(hipmri2dtrain, batch_size=128, shuffle = False)

net = modules.UNet()
net = net.to(dev)

lossFunc = nn.CrossEntropyLoss()
optm = torch.optim.SGD(net.parameters())

net.train()
for epoch in range(epochs):
    for i, (img, zero) in enumerate(trainLoader):
        img = img.to(dev)