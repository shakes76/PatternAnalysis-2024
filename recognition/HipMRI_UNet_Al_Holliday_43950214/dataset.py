# -*- coding: utf-8 -*-
"""
HipMRI data loader

@author: al
"""

import numpy as np
import nibabel as nib
import torch
import os
import torchvision.transforms as transforms

class HipMRI2d(torch.utils.data.Dataset):
    def __init__(self, root, imgSet = "train", transform = None, applyTrans = False):
        self.root = root
        self.setPath = ""
        self.segPath = ""
        self.setFile = ""
        self.segFile = ""
        if imgSet == "test":
            self.setPath = "keras_slices_test"
            self.setFile = "hipmri_slices_test.txt"
            self.segPath = "keras_slices_seg_test"
            self.segFile = "hipmri_slices_seg_test.txt"
        elif imgSet == "train":
            self.setPath = "keras_slices_train"
            self.setFile = "hipmri_slices_train.txt"
            self.segPath = "keras_slices_seg_train"
            self.segFile = "hipmri_slices_seg_train.txt"
        elif imgSet == "validate":
            self.setPath = "keras_slices_validate"
            self.setFile = "hipmri_slices_validate.txt"
            self.segPath = "keras_slices_seg_validate"
            self.segFile = "hipmri_slices_seg_validate.txt"
        else:
            raise ValueError("imgSet must be one of 'test', 'train', 'validate'")
        # note: len(pics) should equal len(segs)
        self.pics = []
        self.segs = []
        with open(self.setFile) as f:
            for pic in f:
                self.pics.append(pic.strip())
        with open(self.segFile) as g:
            for seg in g:
                self.segs.append(seg.strip())
        self.trans = transform
        self.applyTrans = applyTrans
    
    def __len__(self):
        return len(self.pics)
    
    def __getitem__(self, i):
        """
        do the stuff in load_data_2D here
        """
        imgPath = os.path.join(self.root, self.setPath, self.pics[i])
        segPath = os.path.join(self.root, self.segPath, self.segs[i])
        niftiImg = nib.load(imgPath)
        niftiSeg = nib.load(segPath)
        # get_fdata() returns an ndarray
        img = niftiImg.get_fdata(caching = "unchanged")
        seg = niftiSeg.get_fdata(caching = "unchanged")
        if len(img.shape) == 3:
            img = img[: ,: ,0] # sometimes extra dims , remove
        if len(seg.shape) == 3:
            seg = seg[: ,: ,0]
        # turn it back into 3D because PyTorch demands a channel dimension (and it demands that it be the first dim)
        img = img[np.newaxis,:,:]
        seg = seg[np.newaxis,:,:]
        img = img.astype(np.float32)
        seg = seg.astype(np.int64)
        imgTensor = torch.tensor(img)
        segTensor = torch.tensor(seg)
        if self.applyTrans:
            imgTensor = self.trans(imgTensor)
            segTensor = self.trans(segTensor)
        return imgTensor, segTensor
