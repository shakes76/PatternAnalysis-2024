"""
Helper file with various functions to load 2D Nifti files.

Authors:
    to_channels: Shekhar "Shakes" Chandra, modified by Joseph Reid
    load_data_2D: Shekhar "Shakes" Chandra, modified by Joseph Reid
    load_data_2D_from_directory: Joseph Reid

Functions:
    to_channels: One hot encode segment data
    load_data_2D: Load 2D Nifti image files from iterable
    load_data_2D_from_directory: Create and pass in above iterable

Dependencies: 
    numpy 
    nibabel 
    tqdm 
    scikit-image
"""

import os
import numpy as np
import nibabel as nib
from tqdm import tqdm
from skimage.transform import resize

# Apply one hot encoding to segments
IMG_LABELS = 6 # To transform segments to correct size (6, 256, 128) instead of (256, 128)

def to_channels(arr: np.ndarray, dtype = np.uint8) -> np.ndarray:
    """ One hot encode segment data."""
    # channels = np.unique(arr)
    channels = IMG_LABELS
    # res = np.zeros(arr.shape + (len(channels),), dtype = dtype)
    res = np.zeros(arr.shape + (channels,), dtype = dtype)
    for c in range(channels):
        c = int(c)
        res[..., c:c+1][arr == c] = 1
    return res

# load medical image functions
def load_data_2D (imageNames, normImage=False, categorical=False, dtype = np.float32, 
                    getAffines = False, early_stop = False):
    '''
    Load medical image data from names, cases list provided into a list for each.
    
    This function pre-allocates 4D arrays for conv2d to avoid excessive memory usage.
    
    normImage: bool (normalise the image 0.0 - 1.0)
    early_stop: Stop loading pre-maturely, leaves arrays mostly empty, for quick loading and testing scripts.
    '''
    affines = []

    # get fixed size
    num = len(imageNames)
    first_case = nib.load(imageNames[0]).get_fdata(caching = 'unchanged')
    if len(first_case.shape) == 3:
        first_case = first_case[:, :, 0] # sometimes extra dims , remove
    if categorical: # To make segments/masks have the correct size
        first_case = to_channels(first_case, dtype = dtype)
        rows, cols, channels = first_case.shape
        images = np.zeros((num, rows, cols, channels), dtype = dtype)
    else:
        rows, cols = first_case.shape
        images = np.zeros(( num, rows, cols), dtype = dtype)

    for i, inName in enumerate(tqdm(imageNames)):
        niftiImage = nib.load(inName)
        inImage = niftiImage.get_fdata(caching = 'unchanged') # read disk only
        affine = niftiImage.affine
        # print(f"Index: {i}, Shape: {inImage.shape}")
        if len (inImage.shape) == 3:
            inImage = inImage[:, :, 0] # sometimes extra dims in HipMRI_study data
        inImage = resize(inImage, (rows, cols), order=1, preserve_range=True)
        inImage = inImage.astype(dtype)
        if normImage:
            # ~ inImage = inImage / np . linalg . norm ( inImage )
            # ~ inImage = 255. * inImage / inImage . max ()
            inImage = (inImage - inImage.mean()) / inImage.std()
        if categorical:
            inImage = to_channels(inImage, dtype = dtype)
            images[i, : , : , :] = inImage
        else:
            images[i, : , :] = inImage

        affines.append(affine)
        if i > 40 and early_stop:
            break
    
    if getAffines:
        return images, affines
    else :
        return images
    
def load_data_2D_from_directory(image_folder_path: str, normImage=False, categorical=False, dtype = np.float32, getAffines = False, early_stop = False) -> np.ndarray:
    """ Returns np array of all Nifti images in the specified image folder."""
    image_names = []
    for file in os.listdir(image_folder_path):
        image_names.append(os.path.join(image_folder_path, file))
    return load_data_2D(image_names, normImage, categorical, dtype, getAffines, early_stop)