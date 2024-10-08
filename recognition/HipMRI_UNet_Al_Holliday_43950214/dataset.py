# -*- coding: utf-8 -*-
"""
HipMRI data loader

@author: al
"""

import numpy as np
import nibabel as nib
import torch
import os

class HipMRI2d(torch.utils.data.Dataset):
    def __init__(self, root, imgSet = "seg_train", transform = False):
        self.root = root
        self.setPath = ""
        self.setFile = ""
        if imgSet == "seg_test":
            self.setPath = "keras_slices_seg_test"
            self.setFile = "hipmri_slices_seg_test.txt"
        elif imgSet == "seg_train":
            self.setPath = "keras_slices_seg_train"
            self.setFile = "hipmri_slices_seg_train_128.txt"
        elif imgSet == "seg_validate":
            self.setPath = "keras_slices_seg_validate"
            self.setFile = "hipmri_slices_seg_validate.txt"
        elif imgSet == "test":
            self.setPath = "keras_slices_test"
            self.setFile = "hipmri_slices_test.txt"
        elif imgSet == "train":
            self.setPath = "keras_slices_train"
            self.setFile = "hipmri_slices_train.txt"
        elif imgSet == "validate":
            self.setPath = "keras_slices_validate"
            self.setFile = "hipmri_slices_validate.txt"
        else:
            raise ValueError("imgSet must be one of 'seg_test', 'seg_train', 'seg_validate', 'test', 'train', 'validate'")
        self.pics = []
        with open(self.setFile) as f:
            for pic in f:
                self.pics.append(pic.strip())
        # TODO: change what trans represents. Something like:
        # trans = torchvision.transforms.Compose([transforms.ToTensor(), transforms.Resize()])
        self.trans = None
        self.applyTrans = transform
    
    def __len__(self):
        return len(self.pics)
    
    def __getitem__(self, i):
        """
        do the stuff in load_data_2D here
        """
        imgPath = os.path.join(self.root, self.setPath, self.pics[i])
        niftiImg = nib.load(imgPath)
        # get_fdata() returns an ndarray
        img = niftiImg.get_fdata(caching = "unchanged")
        if len(img.shape) == 3:
            img = img[: ,: ,0] # sometimes extra dims , remove
        if self.applyTrans:
            # TODO: look up what this Nibabel affine object is and how to use it
            # with PyTorch
            self.trans = niftiImg.affine
        # turn it back into 3D because PyTorch demands a channel dimension (and it demands that it be the first dim)
        img = img[np.newaxis,:,:]
        
        img = img.astype(np.float32)
        imgTensor = torch.tensor(img)
        return imgTensor, 0