# -*- coding: utf-8 -*-
"""
example code for loading NiFtI images
"""
import numpy as np
import nibabel as nib
from tqdm import tqdm
import torch
import os

def to_channels(arr: np.ndarray , dtype = np.uint8) -> np.ndarray:
    channels = np.unique(arr)
    res = np.zeros(arr.shape + (len(channels), ), dtype = dtype)
    for c in channels:
        c = int(c)
        res[..., c:c + 1][arr == c] = 1
    return res


# load medical image functions
def load_data_2D(imageNames , normImage = False , categorical = False , dtype = np.float32 , getAffines = False , early_stop = False):
    '''
    Load medical image data from names , cases list provided into a list for each .
    This function pre - allocates 4D arrays for conv2d to avoid excessive memory 
    usage .
    normImage : bool ( normalise the image 0.0 -1.0)
    early_stop : Stop loading pre - maturely , leaves arrays mostly empty , for quick 
    loading and testing scripts .
    '''
    affines = []
    
    # get fixed size
    num = len(imageNames)
    first_case = nib.load(imageNames [0]).get_fdata(caching = 'unchanged')
    if len(first_case.shape) == 3:
        first_case = first_case[: ,: ,0] # sometimes extra dims , remove
    if categorical :
        first_case = to_channels(first_case, dtype = dtype)
        rows, cols, channels = first_case.shape
        images = np.zeros((num, rows, cols, channels), dtype = dtype)
    else:
        rows, cols = first_case.shape
        images = np.zeros((num, rows, cols), dtype = dtype)
    
    #for i, inName in enumerate(tqdm(imageNames)):
    for i, inName in enumerate(imageNames):
        print(inName)
        niftiImage = nib.load(inName)
        print(niftiImage.shape)
        inImage = niftiImage.get_fdata(caching = 'unchanged') # read disk only
        affine = niftiImage.affine
        if len(inImage.shape) == 3:
            inImage = inImage[:, :, 0] # sometimes extra dims in HipMRI_study data
        inImage = inImage.astype(dtype)
        if normImage:
            #~ inImage = inImage / np.linalg.norm(inImage)
            #~ inImage = 255. * inImage / inImage.max ()
            inImage = (inImage - inImage.mean()) / inImage.std()
        if categorical:
            inImage = to_channels(inImage, dtype = dtype)
            images[i, :, :, :] = inImage
        else:
            images[i, :, :] = inImage
        affines.append(affine)
        if i > 20 and early_stop:
            break
    if getAffines:
        return images, affines
    else:
        return images

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
            self.setFile = "hipmri_slices_seg_train.txt"
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
        imgTensor = torch.tensor(img)
        return imgTensor, 0