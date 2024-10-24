import numpy as np
import nibabel as nib
from tqdm import tqdm
import cv2
import os  # Ensure you have this to work with file paths
import torch
from torch.utils.data import Dataset  # Add this import for Dataset

def resize_image(image, target_shape):
    """Resize image to the target shape using OpenCV."""
    return cv2.resize(image, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_LINEAR)

def to_channels(arr: np.ndarray, dtype=np.uint8) -> np.ndarray:
    channels = np.unique(arr)
    res = np.zeros(arr.shape + (len(channels),), dtype=dtype)
    for c in channels:
        c = int(c)
        res[..., c:c + 1][arr == c] = 1
    return res

def load_data_2D(imageNames, normImage=False, categorical=False, dtype=np.float32, getAffines=False, early_stop=False):
    """
    Load medical image data from names, cases list provided into a list for each.

    Parameters:
    - imageNames: List of image file names
    - normImage: bool (normalize the image 0.0 - 1.0)
    - categorical: bool (indicates if the data is categorical)
    - dtype: Desired data type (default: np.float32)
    - getAffines: bool (return affine matrices along with images)
    - early_stop: bool (stop loading prematurely for testing purposes)

    Returns:
    - images: Loaded image data as a numpy array
    - affines: List of affine matrices (if getAffines is True)
    """
    affines = []
    num = len(imageNames)
    first_case = nib.load(imageNames[0]).get_fdata(caching='unchanged')

    if len(first_case.shape) == 3:
        first_case = first_case[:, :, 0]  # Remove extra dims if necessary
    if categorical:
        first_case = to_channels(first_case, dtype=dtype)
        rows, cols, channels = first_case.shape
        images = np.zeros((num, rows, cols, channels), dtype=dtype)
    else:
        rows, cols = first_case.shape
        images = np.zeros((num, rows, cols), dtype=dtype)

    for i, inName in enumerate(tqdm(imageNames)):
        niftiImage = nib.load(inName)
        inImage = niftiImage.get_fdata(caching='unchanged')
        affine = niftiImage.affine

        if len(inImage.shape) == 3:
            inImage = inImage[:, :, 0]  # Remove extra dims if necessary
        inImage = inImage.astype(dtype)

        # Resize the image if necessary
        if inImage.shape != (rows, cols):
            inImage = resize_image(inImage, (rows, cols))

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

