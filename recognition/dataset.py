import numpy as np
import nibabel as nib
from tqdm import tqdm
import skimage.transform as skTrans
from pathlib import Path
import matplotlib.pyplot as plt  



def to_channels(arr: np.ndarray, dtype=np.uint8) -> np.ndarray:
    channels = np.unique(arr)
    res = np.zeros(arr.shape + (len(channels),), dtype=dtype)
    for c in channels:
        c = int(c)
        res[..., c:c + 1][arr == c] = 1
    return res

# load medical image functions
def load_data_2D(imageNames, normImage=False, categorical=False, dtype=np.float32, getAffines=False, early_stop=False):
    '''
    Load medical image data from names, cases list provided into a list for each.

    This function pre-allocates 4D arrays for conv2d to avoid excessive memory usage.

    normImage: bool (normalise the image 0.0 -1.0)
    early_stop: Stop loading prematurely, leaves arrays mostly empty, for quick loading and testing scripts.
    '''
    affines = []

    # get fixed size
    num = len(imageNames)
    first_case = nib.load(imageNames[0]).get_fdata(caching='unchanged')
    if len(first_case.shape) == 3:
        first_case = first_case[:, :, 0]  # sometimes extra dims, remove
    if categorical:
        first_case = to_channels(first_case, dtype=dtype)
        rows, cols, channels = first_case.shape
        images = np.zeros((num, rows, cols, channels), dtype=dtype)
    else:
        rows, cols = first_case.shape
        images = np.zeros((num, rows, cols), dtype=dtype)

    for i, inName in enumerate(tqdm(imageNames)):
        niftiImage = nib.load(inName)
        inImage = niftiImage.get_fdata(caching='unchanged')  # read disk only        

        affine = niftiImage.affine
        if len(inImage.shape) == 3:
            inImage = inImage[:, :, 0]  # sometimes extra dims in HipMRI_study data

        # Converts the image to a 256,128 image 
        inImage = skTrans.resize(inImage, (256, 128), order=1, preserve_range=True) 

        inImage = inImage.astype(dtype)
        if normImage:
            # ~ inImage = inImage / np.linalg.norm(inImage)
            # ~ inImage = 255. * inImage / inImage.max()
            inImage = (inImage - inImage.mean()) / inImage.std()

        if categorical:
            inImage = utils.to_channels(inImage, dtype=dtype)
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



baseDir = '/home/deb/Documents/3710DATA/HipMRI_study_keras_slices_data/'
testDir = '/home/deb/Documents/3710DATA/HipMRI_study_keras_slices_data/keras_slices_test/'
trainDir = '/home/deb/Documents/3710DATA/HipMRI_study_keras_slices_data/keras_slices_train/'
validateDir = '/home/deb/Documents/3710DATA/HipMRI_study_keras_slices_data/keras_slices_validate/'

testSegDir = '/home/deb/Documents/3710DATA/HipMRI_study_keras_slices_data/keras_slices_seg_test/'
trainSegDir = '/home/deb/Documents/3710DATA/HipMRI_study_keras_slices_data/keras_slices_seg_train/'
validateSegDir = '/home/deb/Documents/3710DATA/HipMRI_study_keras_slices_data/keras_slices_seg_validate/'

# Uncomment for rangpur and comment for local
# baseDir = /home/groups/comp3710/HipMRI_Study_open/keras_slices_data/
# testDir = '/home/groups/comp3710/HipMRI_Study_open/keras_slices_data/keras_slices_seg_test/'
# trainDir = '/home/groups/comp3710/HipMRI_Study_open/keras_slices_data/keras_slices_seg_train/'
# validateDir = '/home/groups/comp3710/HipMRI_Study_open/keras_slices_data/keras_slices_seg_validate/'
# 
# testSegDir = '/home/groups/comp3710/HipMRI_Study_open/keras_slices_data/keras_slices_seg_test/'
# trainSegDir = '/home/groups/comp3710/HipMRI_Study_open/keras_slices_data/keras_slices_seg_train/'
# validateSegDir = '/home/groups/comp3710/HipMRI_Study_open/keras_slices_data/keras_slices_seg_validate/'


# Load the scans 
testListNii = list(Path(testDir).glob('*.nii'))
testImages = load_data_2D(testListNii, normImage=True, categorical=False)

trainListNii = list(Path(trainDir).glob('*.nii'))
trainImages = load_data_2D(trainListNii, normImage=True, categorical=False)

validateListNii = list(Path(validateDir).glob('*.nii'))
validateImages = load_data_2D(validateListNii, normImage=True, categorical=False)

#Load the segmented scans 
testSegListNii = list(Path(testSegDir).glob('*.nii'))
testSegImages = load_data_2D(testSegListNii, normImage=True, categorical=False)

trainSegListNii = list(Path(trainSegDir).glob('*.nii'))
trainSegImages = load_data_2D(trainSegListNii, normImage=True, categorical=False)

validateSegListNii = list(Path(validateSegDir).glob('*.nii'))
validateSegImages = load_data_2D(validateSegListNii, normImage=True, categorical=False)

