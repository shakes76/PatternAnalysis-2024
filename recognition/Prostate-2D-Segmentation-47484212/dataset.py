import numpy as np
import nibabel as nib
from tqdm import tqdm
from utils import to_channels
import os
from random import shuffle

# Rangpur Path: /home/groups/comp3710/HipMRI_Study_open
def load_data_2D(imageNames, normImage=False, categorical=False, dtype=np.float32, getAffines=False, early_stop=False):
    '''
    Load medical image data from names, cases list provided into a list for each.

    This function pre-allocates 4D arrays for conv2d to avoid excessive memory usage.

    normImage: bool (normalize the image 0.0-1.0)
    early_stop: Stop loading prematurely, leaves arrays mostly empty, for quick loading and testing scripts.
    '''
    affines = []
    # get fixed count
    num = len(imageNames)
    first_case = nib.load(imageNames[0]).get_fdata(caching='unchanged')
    if len(first_case.shape) == 3:
        first_case = first_case[:, :, 0]  # sometimes extra dims, remove
    if categorical:
        first_case = to_channels(first_case, dtype=dtype)
        #rows, cols, channels = first_case.shape
        rows, cols, channels = 256, 128, 5 # first case does not include all classes so manually setting
        images = np.zeros((num, rows, cols, channels), dtype=dtype)
    else:
        #rows, cols = first_case.shape
        rows, cols = 256, 128
        images = np.zeros((num, rows, cols), dtype=dtype)

    for i, inName in enumerate(tqdm(imageNames)):
        niftiImage = nib.load(inName)
        inImage = niftiImage.get_fdata(caching='unchanged')  # read from disk only
        affine = niftiImage.affine
        if len(inImage.shape) == 3:
            inImage = inImage[:, :, 0]  # sometimes extra dims in HipMRI_study data
        inImage = inImage.astype(dtype)
        # crop image to expected resolution
        inImage = inImage[:256, :128]
        if normImage:
            # Normalize image
            inImage = (inImage - inImage.mean()) / inImage.std()
        if categorical:
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
    

def get_all_paths(dir):
    '''
    Retrieves all file paths of NIfTI images stored in subdirectories within a root directory.
    
    Params:
    str dir: The main directory containing subdirectories for each image.

    Returns a list of str: Paths to each NIfTI file in the root directory.
    '''
    paths = []
    
    # walk directory and get paths of files
    for subdir, _, files in os.walk(dir):
        for file in files:
            paths.append(os.path.join(subdir, file))
    return paths

def batch_paths(samplePaths, segPaths, batchSize):
    '''
    batch samples and segmentations randomly but mirrored for both 

    Params:
    List[string] samplePaths: list of paths to images being batched (same length as segPaths)
    List[string] segPaths: list of paths to segmentations being batched (same length as samplePaths)
    int batchSize: the size of batches to generate 

    Returns a list containing lists of size batchSize of paths for both samplePaths and segPaths
    '''
    count = len(samplePaths) # the amount of samples/segs
    indicies = [i for i in range(count)]
    shuffle(indicies)
    # shuffle samples and segs identically
    shuffledSamples = [samplePaths[i] for i in indicies]
    shuffledSegs = [segPaths[i] for i in indicies]
    # batch the shuffled data
    batchedSamples = [shuffledSamples[i: min(i + batchSize, count - 1)] for i in range(0, count, batchSize)]
    batchedSegs = [shuffledSegs[i: min(i + batchSize, count - 1)] for i in range(0, count, batchSize)]
    return batchedSamples, batchedSegs
