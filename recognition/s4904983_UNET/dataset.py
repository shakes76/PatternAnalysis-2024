""" 
File: dataset.py
Author: Øystein Kvandal
Description: Contains the functions for loading the medical image data from the dataset for the UNET model.
"""

import numpy as np
import nibabel as nib
from tqdm import tqdm
import torchio as tio

# Local root path for the dataset
root_path = 'C:/Users/oykva/OneDrive - NTNU/Semester 7/PatRec/Project/HipMRI_study_keras_slices_data/'


def to_channels(arr: np.ndarray, dtype = np.uint8) -> np.ndarray:
    channels = np.unique(arr)
    res = np.zeros(arr. shape + (len(channels),), dtype=dtype)
    for c in channels:
        c = int(c)
        res[..., c: c +1][ arr == c ] = 1

    return res


# load medical image functions
def load_data_2D(imageNames, normImage = False, categorical = False, dtype = np.float32, getAffines = False, early_stop = False):
    '''
    Load medical image data from names, cases list provided into a list for each.
    This function pre - allocates 4D arrays for conv2d to avoid excessive memory &
    usage.

    normImage: bool (normalise the image 0.0 -1.0)
    early_stop: Stop loading pre - maturely, leaves arrays mostly empty, for quick &
    loading and testing scripts.
    '''
    affines = []

    # get fixed size
    num = len(imageNames)
    first_case = nib.load(imageNames[0]).get_fdata(caching = 'unchanged')
    if len(first_case.shape) == 3:
        first_case = first_case [:,:,0] # sometimes extra dims, remove
    if categorical:
        first_case = to_channels(first_case, dtype = dtype)
        rows, cols, channels = first_case.shape
        images = np.zeros((num, rows, cols, channels), dtype = dtype)
    else:
        rows, cols = first_case.shape
        images = np.zeros((num, rows, cols), dtype = dtype)

    for i, inName in enumerate(tqdm(imageNames)):
        niftiImage = nib.load(inName)
        inImage = niftiImage.get_fdata(caching = 'unchanged') # read disk only
        affine = niftiImage.affine
        if len(inImage.shape) == 3:
            inImage = inImage[:,:,0] # sometimes extra dims in HipMRI_study data
            inImage = inImage.astype(dtype)
        if normImage:
            #~ inImage = inImage / np.linalg.norm(inImage)
            #~ inImage = 255. * inImage / inImage.max()
            inImage = (inImage - inImage.mean()) / inImage.std()
        if categorical:
            inImage = to_channels(inImage, dtype = dtype)
            # Crop images to ensure they are all (256, 128) - some are originally (256, 144)
            # Raise exception if tha image is not (256, 128) or (256, 144)
            if inImage.shape != (inImage.shape[0], 256, 128) and inImage.shape != (inImage.shape[0], 256, 144):
                raise ValueError('Image format not (256, 128) or (256, 144). Shape: {}'.format(inImage.shape))
            inImage = inImage[:,:, (inImage.shape[2] - 128) // 2:(inImage.shape[2] + 128) // 2]
            images[i,:,:,:] = inImage
        else:
            # Crop images to ensure they are all (256, 128) - some are originally (256, 144)
            # Raise exception if tha image is not (256, 128) or (256, 144)
            if inImage.shape != (256, 128) and inImage.shape != (256, 144):
                raise ValueError('Image format not (256, 128) or (256, 144). Shape: {}'.format(inImage.shape))
            inImage = inImage[:, (inImage.shape[1] - 128) // 2:(inImage.shape[1] + 128) // 2]
            images[i,:,:] = inImage

        affines.append(affine)
        if i > 20 and early_stop:
            break

    if getAffines:
        return images, affines
    else:
        return images



### Unit test
# import os
# set_path = 'keras_slices_seg_' + 'train' + '/'
# imageNames = os.listdir(root_path + set_path)


# images = load_data_2D([root_path + set_path + i for i in imageNames])
# from matplotlib import pyplot as plt

# for i, image in enumerate(images[::1000]):
#     plt.figure(i)
#     plt.imshow(image, cmap = 'gray')
#     plt.show()