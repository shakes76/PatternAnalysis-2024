# -*- coding: utf-8 -*-
"""
Training script

@author: al
"""
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

import dataset
import modules

trans = transforms.Resize((256, 256))
# for the lab computers
hipmri2dtrain = dataset.HipMRI2d("H:\\HipMRI", imgSet = "train", transform = trans, applyTrans = True)
# for rangpur
#hipmri2dtrain = dataset.HipMRI2d("/home/groups/comp3710/HipMRI_Study_open/keras_slices_data", imgSet = "train", transform = trans, applyTrans = True)

net = modules.UNet()