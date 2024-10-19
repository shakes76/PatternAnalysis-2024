import os
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import nibabel as nib
from tqdm import tqdm


#image = load_data_3D('semantic_MRs_anon', normImage=False, categorical=False, dtype=np.float32, getAffines=False, orient=False, early_stop=False)
#label = load_data_3D('semantic_labels_anon', normImage=False, categorical=False, dtype=np.uint8, getAffines=False, orient=False, early_stop=False)

#image = torch.tensor(image, dtype=torch.float32)
#label = torch.tensor(label, dtype=torch.long)

def to_channels(arr: np.ndarray, dtype=np.uint8) -> np.ndarray:
    channels = np.unique(arr)
    res = np.zeros(arr.shape + (len(channels),), dtype=dtype)
    for c in channels:
        c = int(c)
        res[..., c:c+1][arr == c] = 1
    return res

# Load medical image functions
def load_data_2D(imageNames, normImage=False, categorical=False, dtype=np.float32, getAffines=False, early_stop=False):
    '''
    Load medical image data from names, cases list provided into a list for each.

    This function pre-allocates 4D arrays for conv2d to avoid excessive memory usage.

    Parameters:
        normImage : bool (normalize the image between 0.0 - 1.0)
        early_stop : Stop loading prematurely, leaves arrays mostly empty, for quick loading and testing scripts.
    '''
    affines = []

    # Get fixed size
    num = len(imageNames)
    first_case = nib.load(imageNames[0]).get_fdata(caching='unchanged')
    if len(first_case.shape) == 3:
        first_case = first_case[:, :, 0]  # Sometimes extra dims, remove
    
    if categorical:
        first_case = to_channels(first_case, dtype=dtype)
        rows, cols, channels = first_case.shape
        images = np.zeros((num, rows, cols, channels), dtype=dtype)
    else:
        rows, cols = first_case.shape
        images = np.zeros((num, rows, cols), dtype=dtype)

    for i, inName in enumerate(tqdm(imageNames)):
        niftiImage = nib.load(inName)
        inImage = niftiImage.get_fdata(caching='unchanged')  # Read disk only
        affine = niftiImage.affine
        if len(inImage.shape) == 3:
            inImage = inImage[:, :, 0]  # Sometimes extra dims in HipMRI_study data
        inImage = inImage.astype(dtype)
        
        if normImage:
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


def load_data_3D(imageNames, normImage=False, categorical=False, dtype=np.float32, getAffines=False, orient=False, early_stop=False):
    '''
    Load medical image data from names, cases list provided into a list for each.

    This function pre-allocates 5D arrays for conv3d to avoid excessive memory usage.

    Parameters:
        normImage : bool (normalize the image between 0.0 - 1.0)
        orient : Apply orientation and resample image, good for images with large slice thickness or anisotropic resolution
        dtype : Type of the data. If dtype = np.uint8, it is assumed that the data is labels
        early_stop : Stop loading prematurely, leaves arrays mostly empty, for quick loading and testing scripts.
    '''
    affines = []
    interp = 'linear'
    if dtype == np.uint8:  # Assume labels
        interp = 'nearest'

    # Get fixed size
    num = len(imageNames)
    niftiImage = nib.load(imageNames[0])
    if orient:
        niftiImage = im.applyOrientation(niftiImage, interpolation=interp, scale=1)

    first_case = niftiImage.get_fdata(caching='unchanged')
    if len(first_case.shape) == 4:
        first_case = first_case[:, :, :, 0]  # Sometimes extra dims, remove

    if categorical:
        first_case = to_channels(first_case, dtype=dtype)
        rows, cols, depth, channels = first_case.shape
        images = np.zeros((num, rows, cols, depth, channels), dtype=dtype)
    else:
        rows, cols, depth = first_case.shape
        images = np.zeros((num, rows, cols, depth), dtype=dtype)

    for i, inName in enumerate(tqdm(imageNames)):
        niftiImage = nib.load(inName)
        if orient:
            niftiImage = im.applyOrientation(niftiImage, interpolation=interp, scale=1)
        inImage = niftiImage.get_fdata(caching='unchanged')  # Read disk only
        affine = niftiImage.affine

        if len(inImage.shape) == 4:
            inImage = inImage[:, :, :, 0]  # Sometimes extra dims in HipMRI_study data
        inImage = inImage[:, :, :depth]  # Clip slices
        inImage = inImage.astype(dtype)

        if normImage:
            inImage = (inImage - inImage.mean()) / inImage.std()

        if categorical:
            inImage = to_channels(inImage, dtype=dtype)
            images[i, :inImage.shape[0], :inImage.shape[1], :inImage.shape[2], :inImage.shape[3]] = inImage  # With pad
        else:
            images[i, :inImage.shape[0], :inImage.shape[1], :inImage.shape[2]] = inImage  # With pad

        affines.append(affine)
        if i > 20 and early_stop:
            break

    if getAffines:
        return images, affines
    else:
        return images

class MRI3DDataset(Dataset):
    def __init__(self, image_folder, label_folder, normImage=False):
        self.image_folder = image_folder
        self.label_folder = label_folder
        self.image_filenames = sorted([os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.endswith('.nii.gz')])
        self.label_filenames = sorted([os.path.join(label_folder, f) for f in os.listdir(label_folder) if f.endswith('.nii.gz')])
        self.normImage = normImage

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        image = load_data_3D([self.image_filenames[idx]], normImage=self.normImage)
        label = load_data_3D([self.label_filenames[idx]], dtype=np.uint8)

        image = torch.tensor(image, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.long)

        return image, label



