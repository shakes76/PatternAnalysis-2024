import os
import os.path
import torch
import nibabel as nib
import numpy as np
from tqdm import tqdm
import torch


"""
REFERENCES (INCOMPLETE):
 Chandra, S. (2024).
Contains function to load Nifti files as well.
"""

def to_channels(arr: np.ndarray , dtype =np.uint8) -> np.ndarray :
    channels = np.unique(arr)
    res = np. zeros (arr.shape + (len(channels),), dtype=dtype)
    for c in channels :
        c = int(c)
        res [... , c:c +1][arr == c] = 1

    return res

def load_data_2D(imageNames, normImage=False, categorical=False, dtype=np.float32,
    getAffines=False , early_stop=False):
    '''
    Load medical image data from names , cases list provided into a list for each .

    This function pre - allocates 4D arrays for conv2d to avoid excessive memory &
    usage .

    normImage : bool ( normalise the image 0.0 -1.0)
    early_stop : Stop loading pre - maturely , leaves arrays mostly empty , for quick &
    loading and testing scripts .
    '''
    affines = []

    max_rows, max_cols = 0,0
    for inName in imageNames:
        niftiImage = nib.load(inName)
        inImage = niftiImage.get_fdata(caching='unchanged')
        if len(inImage.shape) == 3:
         inImage = inImage[: ,: ,0] # sometimes extra dims , remove
        # print(f"{inImage.shape[0]}, {inImage.shape[1]}")
        max_rows = max(max_rows, inImage.shape[0])
        max_cols = max(max_cols, inImage.shape[1])

    #get fixed size
    num = len(imageNames)
    first_case = nib.load(imageNames[0]).get_fdata(caching='unchanged')
    if len(first_case.shape) == 3:
        first_case = first_case[: ,: ,0] # sometimes extra dims , remove
    if categorical:
        first_case = to_channels(first_case,dtype = dtype)
        rows, cols, channels = first_case.shape
        images = np.zeros((num ,max_rows ,max_cols ,channels),dtype = dtype)
    else:
        # rows, cols = first_case.shape
        images = np.zeros((num, max_rows, max_cols), dtype = dtype)

    for i, inName in enumerate(tqdm(imageNames)):
        niftiImage = nib.load(inName)
        inImage = niftiImage.get_fdata(caching ='unchanged') # read disk only
        affine = niftiImage.affine
        if len(inImage.shape) == 3:
            inImage = inImage[: ,: ,0] # sometimes extra dims in HipMRI_study data
            inImage = inImage.astype(dtype)
        if normImage :
            #~ inImage = inImage / np. linalg . norm ( inImage )
            #~ inImage = 255. * inImage / inImage .max ()
            inImage = (inImage - inImage.mean()) / inImage.std()
        if categorical:
            inImage = utils.to_channels(inImage,dtype=dtype)
            images[i ,:inImage.shape[0] ,:inImage.shape[1] ,:] = inImage
        else :
            images[i ,:inImage.shape[0] ,:inImage.shape[1]] = inImage

        affines.append(affine)
        if i > 20 and early_stop:
            break

    images = torch.tensor(images)
    if getAffines:
        return images,affines
    else:
        return images



def SaveCheckpoint():
    print()
def LoadCheckpoint():
    print()
def Initiliase():
    print()