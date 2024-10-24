import os
import numpy as np
import nibabel as nib
import tensorflow as tf
import skimage.transform
from tqdm import tqdm
from matplotlib import pyplot as plt

def to_channels(arr: np.ndarray, dtype=np.uint8) -> np.ndarray:
    # channels = np.unique(arr)
    channels = np.arange(6)
    res = np.zeros(arr.shape + (len(channels), ), dtype=dtype)
    for c in channels:
        c = int(c)
        res[..., c : c + 1][arr == c] = 1
    return res

# load medical image functions
def load_data_2D(imageNames, normImage=False, categorical=False, dtype=np.float32, getAffines=False, early_stop=False):
    '''
    Load medical image data from names, cases list provided into a list for each.

    This function pre - allocates 4D arrays for conv2d to avoid excessive memory & usage.

    normImage : bool (normalise the image 0.0-1.0)
    early_stop : Stop loading pre - maturely, leaves arrays mostly empty, for quick & loading and testing scripts.
    '''

    affines = []

    # get fixed size
    num = len(imageNames)
    first_case = nib.load(imageNames[0]).get_fdata(caching='unchanged')
    if len(first_case.shape) == 3:
        first_case = first_case[:, :, 0] # sometimes extra dims, remove
    if categorical:
        first_case = to_channels(first_case, dtype=dtype)
        rows, cols, channels = first_case.shape
        images = np.zeros((num, rows, cols, channels), dtype=dtype)
    else:
        rows, cols = first_case.shape
        images = np.zeros((num, rows, cols), dtype=dtype)

    for i, inName in enumerate(tqdm(imageNames)):
        niftiImage = nib.load(inName)
        inImage = niftiImage.get_fdata(caching='unchanged') # read disk only
        affine = niftiImage.affine
        if len(inImage.shape) == 3:
            inImage = inImage[:, :, 0] # sometimes extra dims in HimMRI_study data
        if inImage.shape != (rows, cols):
            inImage = skimage.transform.resize(inImage, (rows, cols), 
                                              order=1, preserve_range=True)
        inImage = inImage.astype(dtype)
        if normImage:
            #~ inImage = inImage / np.linalg.norm(inImage)
            #~ inImage = 255. * inImage / inImage.max()
            inImage = (inImage - inImage.mean()) / inImage.std()
        if categorical:
            inImage = np.round(inImage)
            inImage = to_channels(inImage, dtype=dtype)
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

def load_data_tf(image_dir, seg_dir, dtype=np.float32):
    image_list = os.listdir(image_dir)
    image_path = []
    seg_path = []
    for image in image_list:
        image_path.append(os.path.join(image_dir, image))
        seg_path.append(os.path.join(seg_dir, image.replace("case", "seg")))
    images = load_data_2D(image_path)
    segs = load_data_2D(seg_path, categorical=True)
    return images, segs