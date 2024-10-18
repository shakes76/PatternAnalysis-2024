"""
dataset.py

File to handle loading, and preprocessing of the dataset
"""
import os
from typing import Iterable

import nibabel as nib
import numpy as np
from const import DATASET_PATH
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from tqdm import tqdm

im = None
utils = None

def to_channels(arr: np.ndarray, dtype = np.uint8) -> np.ndarray:
    channels = np.unique(arr)
    res = np.zeros(arr.shape +(len(channels),), dtype = dtype)
    for c in channels:
        c = int(c)
        res [..., c:c + 1][ arr == c ] = 1

    return res

# load medical image functions
def load_data_2d(imageNames, normImage = False, categorical = False, dtype = np.float32, getAffines = False, early_stop = False):
    '''
    Load medical image data from names, cases list provided into a list for each.

    This function pre-allocates 4D arrays for conv2d to avoid excessive memory usage.

    normImage: bool (normalise the image 0.0 - 1.0)
    early_stop: Stop loading pre-maturely, leaves arrays mostly empty, for quick loading and testing scripts.
    '''
    affines = []

    # get fixed size
    num = len( imageNames)
    first_case = nib.load(imageNames[0]).get_fdata(caching = 'unchanged')
    if len( first_case.shape) == 3:
        first_case = first_case [:,:,0] # sometimes extra dims, remove
    if categorical:
        first_case = to_channels(first_case, dtype = dtype)
        rows, cols, channels = first_case.shape
        images = np.zeros(( num, rows, cols, channels), dtype = dtype)
    else:
        rows, cols = first_case.shape
        images = np.zeros(( num, rows, cols), dtype = dtype)
    
    for i, inName in enumerate(tqdm(imageNames)):
        niftiImage = nib.load(inName)
        inImage = niftiImage.get_fdata(caching = 'unchanged') # read disk only
        affine = niftiImage.affine
        if len( inImage.shape) == 3:
            inImage = inImage [:,:,0] # sometimes extra dims in HipMRI_study data
            inImage = inImage.astype(dtype)
        if normImage:
            #~ inImage = inImage / np. linalg.norm(inImage)
            #~ inImage = 255. * inImage / inImage.max()
            inImage =(inImage - inImage.mean()) / inImage.std()
        if categorical:
            inImage = utils.to_channels(inImage, dtype = dtype)
            images [i,:,:,:] = inImage
        else:
            images [i,:,:] = inImage

        affines.append(affine)
        if i > 20 and early_stop:
            break

    if getAffines:
        return images, affines
    else:
        return images

def load_data_3d(imageNames, normImage = False, categorical = False, dtype = np.float32, getAffines = False, orient = False, early_stop = False):
    '''
    Load medical image data from names, cases list provided into a list for each.

    This function pre-allocates 5D arrays for conv3d to avoid excessive memory usage.

    normImage: bool (normalise the image 0.0 -1.0)
    orient: Apply orientation and resample image ? Good for images with large slice thickness or anisotropic resolution
    dtype: Type of the data.If dtype =np.uint8, it is assumed that the data is labels
    early_stop: Stop loading pre - maturely ? Leaves arrays mostly empty, for quick loading and testing scripts.
    '''
    affines = []
    #~ interp = 'continuous'
    interp = 'linear'
    if dtype == np.uint8: # assume labels
        interp = 'nearest'

    # get fixed size
    num = len(imageNames)
    niftiImage = nib.load(imageNames[0])
    if orient:
        niftiImage = im.applyOrientation(niftiImage, interpolation = interp, scale = 1)
        #~ testResultName = "oriented.nii.gz"
        #~ niftiImage.to_filename(testResultName)
    first_case = niftiImage.get_fdata(caching = 'unchanged')
    if len( first_case.shape) == 4:
        first_case = first_case [:, :, :, 0] # sometimes extra dims, remove
    if categorical:
        first_case = to_channels(first_case, dtype = dtype)
        rows, cols, depth, channels = first_case.shape
        images = np.zeros(( num, rows, cols, depth, channels), dtype = dtype)
    else:
        rows, cols, depth = first_case.shape
        images = np.zeros(( num, rows, cols, depth), dtype = dtype)

    for i, inName in enumerate(tqdm(imageNames)):
        niftiImage = nib.load(inName)
        if orient:
            niftiImage = im.applyOrientation(niftiImage, interpolation = interp, scale =1)
        inImage = niftiImage.get_fdata(caching = 'unchanged') # read disk only
        affine = niftiImage.affine
        if len( inImage.shape) == 4:
            inImage = inImage[:, :, :, 0] # sometimes extra dims in HipMRI_study data
        inImage = inImage[:, :, :depth] # clip slices
        inImage = inImage.astype(dtype)
        if normImage:
            #~ inImage = inImage / np. linalg.norm(inImage)
            #~ inImage = 255. * inImage / inImage.max()
            inImage = (inImage - inImage.mean()) / inImage.std()
        if categorical:
            inImage = utils.to_channels(inImage, dtype = dtype)
            #~ images [i, :, :, :, :] = inImage
            images [i, : inImage.shape [0], : inImage.shape [1], : inImage.shape [2], : inImage.shape [3]] = inImage # with pad
        else:
            #~ images [i, :, :, :] = inImage
            images [i, : inImage.shape [0], : inImage.shape [1], : inImage.shape [2]] = inImage # with pad

        affines.append(affine)
        if i > 20 and early_stop:
            break

    if getAffines:
        return images, affines
    else:
        return images

class MriData3D(Dataset):
    """
    Mri Dataset class to Load in segments of the Hip Mri dataset. Is also able to load other simmilar
    datasets. 
    """
    def __init__(self, data_path:str = DATASET_PATH, target_data:list[str]|None=None, transform=None, label_transform=None, target_depth = 128) -> None:
        self.main_data_path = data_path + "/semantic_MRs/"
        self.data_label_path = data_path + "/semantic_labels_only/"
        self.transform = transform
        self.label_transform = label_transform
        self.data_files = os.listdir(self.main_data_path)
        self.label_files = os.listdir(self.data_label_path)
        self.target_depth = target_depth
        if target_data:
            self.data_files = [name for name in self.data_files if name[:10] in target_data]
            self.label_files = [name for name in self.label_files if name[:10] in target_data]

    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, index) -> tuple[np.ndarray, np.ndarray]:
        image = load_data_3d([os.path.join(self.main_data_path, self.data_files[index])])
        label = load_data_3d([os.path.join(self.data_label_path, self.label_files[index])])
        image = image[:, :, :, :self.target_depth]
        label = label[:, :, :, :self.target_depth]
        if self.transform:
            image = self.transform(image)
        if self.label_transform:
            label = self.label_transform(label)
        return image, label

def mri_split(data_path:str = DATASET_PATH,proportions:Iterable[float] = [0.7,0.2,0.1]) -> list[list[str]]:
    """Creates a split on sample names from the MRI's for training a neural network

    Args:
        data_path (str, optional): The path for the mri dataset. Defaults to DATASET_PATH.
        proportions (Iterable[float], optional): The set of proportions. Defaults to [0.7,0.2,0.1].

    Returns:
        list[list[str]]: List of the resulting splits (randomized)
    """
    if any(x < 0 for x in proportions):
        return []
    if sum(proportions) != 1:
        return []
    targets = [x[:10] for x in os.listdir(data_path + "/semantic_MRs/")]
    # targets = [x + 1 for x in range(10)]
    if len(targets) == 0:
        return []
    result = []
    while len(proportions) > 1:
        to_append, to_split = train_test_split(targets, train_size=proportions[0])
        targets = to_split
        result.append(to_append)
        proportions = proportions[1:]
        new_sum = sum(proportions)
        if new_sum != 1 and new_sum:
            proportions = [x/new_sum for x in proportions]
    result.append(to_split)
    return result
