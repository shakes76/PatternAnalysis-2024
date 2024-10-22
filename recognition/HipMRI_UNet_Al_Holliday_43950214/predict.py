# -*- coding: utf-8 -*-
"""
Prediction code. Basically performing the validation

@author: al
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

import dataset
from dice import dice_coeff

def predict(net, dev):
    """
    Performs the validation. Basically serving as an example of how to use the trained model
    
    Parameters:
        net: the model (MUST be trained first)
        dev: the pytorch device to put the validation set on
    """
    trans = transforms.Resize((256, 256))
    # example HipMRI dataset root folders:
    # For Woomy (my laptop)
    #hipMriRoot = "C:\\Users\\al\\HipMRI"
    # For the lab computers
    hipMriRoot = "H:\\HipMRI"
    # For Rangpur
    #hipMriRoot = "/home/groups/comp3710/HipMRI_Study_open/keras_slices_data"
    hipmri2dval = dataset.HipMRI2d(hipMriRoot, imgSet = "validate", transform = trans, applyTrans = True)
    valLoader = DataLoader(hipmri2dval, batch_size = 8, shuffle = False)
    
    # perform validation
    print("Validating")
    diceLosses = []
    with torch.no_grad():
        for img, seg in valLoader:
            img = img.to(dev)
            #seg = seg.squeeze()
            seg = nn.functional.one_hot(seg.squeeze(), num_classes = 6)
            seg = torch.permute(seg, (0, 3, 1, 2))
            seg = seg.to(dev)
            seg = seg.to(dev)
            out = net(img)
            out = torch.permute(out, (0, 2, 3, 1)) # put the chan dim last
            out = torch.argmax(out, dim = -1)
            out = out[:, None, :, :] # reshape back to (batch, chan, h, w)
            diceSimilarity = dice_coeff(out, seg, dev, 6)
            print("current dice: {:.5f}".format(diceSimilarity.cpu().item()))
    print("Done!")
    avgDice = sum(diceLosses) / len(diceLosses)
    print("average dice from validation: {:.5f}".format(avgDice))
