import os
import numpy as np
import nibabel as nib
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from skimage.transform import resize  
def to_channels(arr: np.ndarray, dtype=np.uint8) -> np.ndarray:
    """
    Convert a 2D array to one-hot encoding across channels.
    
    Args:
        arr (np.ndarray): 2D array with categorical values.
        dtype: Data type of the output array.
        
    Returns:
        np.ndarray: One-hot encoded array with shape (rows, cols, num_channels).
    """
    channels = np.unique(arr)
    res = np.zeros(arr.shape + (len(channels),), dtype=dtype)
    for c in channels:
        c = int(c)
        res[..., c:c+1][arr == c] = 1
    return res


import cv2
def load_data_2D(imageNames, normImage=False, categorical=False, dtype=np.float32,
                 getAffines=False, early_stop=False, target_size=(252, 252)):
    '''
    Load medical image data from names into a list.

    Args:
        imageNames (list): List of image file paths.
        normImage (bool): Whether to normalize images.
        categorical (bool): Whether to convert images to categorical (one-hot) encoding.
        dtype: Data type of the output array.
        getAffines (bool): Whether to return affine matrices.
        early_stop (bool): Stop loading after a certain number of images.
        target_size (tuple): Desired output image size (height, width).

    Returns:
        np.ndarray: Loaded images.
        list: Affine matrices (if getAffines=True).
    '''
    affines = []
    num = len(imageNames)
    first_case = nib.load(imageNames[0]).get_fdata(caching='unchanged')
    
    if len(first_case.shape) == 3:
        first_case = first_case[:, :, 0]  # Remove extra dimensions if present

    # Prepare output array
    if categorical:
        first_case = to_channels(first_case, dtype=dtype)
        rows, cols, channels = first_case.shape
        images = np.zeros((num, *target_size, channels), dtype=dtype)
    else:
        rows, cols = first_case.shape
        images = np.zeros((num, *target_size), dtype=dtype)

    for i, inName in enumerate(tqdm(imageNames, desc="Loading Images")):
        niftiImage = nib.load(inName)
        inImage = niftiImage.get_fdata(caching='unchanged').astype(dtype)
        affine = niftiImage.affine

        if len(inImage.shape) == 3:
            inImage = inImage[:, :, 0]  # Remove extra dimensions

        # Resize image to target size
        inImage = cv2.resize(inImage, target_size[::-1])  # Resize to (width, height)

        if normImage:
            inImage = (inImage - inImage.mean()) / inImage.std()

        if categorical:
            inImage = to_channels(inImage, dtype=dtype)
            images[i, :, :, :] = inImage
        else:
            images[i, :, :] = inImage

        affines.append(affine)

        if early_stop and i >= 20:
            break

    if getAffines:
        return images, affines
    else:
        return images

def get_image_paths(folder, extensions=('.nii', '.nii.gz')):
    """
    Returns a list of image file paths from a specified folder.
    
    Parameters:
    - folder: str, Path to the folder containing the images.
    - extensions: tuple, File extensions to look for (default: NIfTI formats).
    
    Returns:
    - image_paths: list of full image file paths.
    """
    image_paths = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(extensions)]
    return image_paths

# Example usage:

train_paths = get_image_paths('HipMRI_study_keras_slices_data\keras_slices_train')
test_paths = get_image_paths('HipMRI_study_keras_slices_data\keras_slices_test')
val_paths = get_image_paths('HipMRI_study_keras_slices_data\keras_slices_validate')
 
#image_folder = "train"  # Path to the folder containing training images
train_images = load_data_2D(train_paths, normImage=True, categorical=False)
train_images = load_data_2D(test_paths, normImage=True, categorical=False)
train_images = load_data_2D(val_paths, normImage=True, categorical=False)