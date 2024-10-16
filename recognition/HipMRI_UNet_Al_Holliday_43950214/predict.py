# -*- coding: utf-8 -*-
"""
Prediction code. Basically performing the validation

@author: al
"""

import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

import dataset

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
    hipMriRoot = "C:\\Users\\al\\HipMRI"
    # For the lab computers
    #hipMriRoot = "H:\\HipMRI"
    # For Rangpur
    #hipMriRoot = "/home/groups/comp3710/HipMRI_Study_open/keras_slices_data"
    hipmri2dval = dataset.HipMRI2d(hipMriRoot, imgSet = "validate", transform = trans, applyTrans = True)
    valLoader = DataLoader(hipmri2dval, batch_size = 8, shuffle = False)
    
    # TODO: perform validation!