"""
Helper file with various functions to load 2D Nifti files.

Authors:
    to_channels: Shekhar "Shakes" Chandra, modified by Joseph Reid
    load_data_2D: Shekhar "Shakes" Chandra, modified by Joseph Reid
    load_data_2D_from_directory: Joseph Reid

Functions:
    to_channels: One hot encode segment data
    load_data_2D: Load 2D Nifti image files from a list
    load_data_2D_from_directory: Create and pass in the above list

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

IMG_LABELS = 6 # To apply one hot encoding to segments 
IMG_SIZE = 256 # Convert all images from (256, 128) or (256, 144) to (256, 256)


def to_channels(arr: np.ndarray, dtype = np.uint8) -> np.ndarray:
    """
    One hot encode segment data.

    Converts size (H, W) to (H, W, C) where:
    H is height, W is width, and C is the number of labels.
    
    Parameters:
        arr (np.ndarray): Array to be one hot encoded
        dtype (np.dtype): Type and size of the data in the array

    Returns:
        np.ndarray: Array that has been one hot encoded
    """
    channels = IMG_LABELS
    result = np.zeros(arr.shape + (channels,), dtype = dtype)
    for c in range(channels):
        result[..., c:c+1][arr == c] = 1
    return result


def load_data_2D(
        image_names: list[str], 
        norm_image: bool, 
        one_hot: bool, 
        resized: bool,
        resizing_masks: bool,
        dtype: np.dtype, 
        early_stop: bool
        ) -> np.ndarray:
    """
    Load 2D Nifti files, with optional resizing, normalising, one hot.
    
    Parameters:
        image_names (list[str]): List of Nifti image file paths
        norm_image (bool): Boolean flag to normalise images (mean=0, std=1)
        one_hot (bool): Boolean flag to one hot encode images
        resized (bool): Boolean flag to resize images to (IMG_SIZE, IMG_SIZE)
        resizing_masks (bool): Boolean flag to resize masks with order=0
        dtype (np.dtype): Type and size of the data to be returned as an array
        early_stop (bool): Boolean flag to prematurely stop loading files

    Returns:
        np.ndarray: Array containing the data from the Nifti files
    """
    # Get the number of images in the list
    num = len(image_names)
    # Get desired image shape
    if resized:
        rows = IMG_SIZE
        cols = IMG_SIZE
    else:
        rows = 256
        cols = 128
    channels = IMG_LABELS

    if one_hot: # Save images as (N, H, W, C)
        images = np.zeros((num, rows, cols, channels), dtype = dtype)
    else: # Save images as (N, H, W)
        images = np.zeros((num, rows, cols), dtype = dtype)

    for idx, image_name in enumerate(tqdm(image_names)):
        nifti_image = nib.load(image_name)
        image = nifti_image.get_fdata(caching = 'unchanged') # Read disk only
        if len(image.shape) == 3: # Remove extra dimensions, if any
            image = image[:, :, 0]
        if resizing_masks: # Make order=0 to keep 6 unique entries
            image = resize(image, (rows, cols), order=0, preserve_range=True)
        else:
            image = resize(image, (rows, cols), order=1, preserve_range=True)
        image = image.astype(dtype)
        if norm_image: # Normalise the image
            image = (image - image.mean()) / image.std()
        if one_hot: # One hot encode the image (should be a mask)
            image = to_channels(image, dtype = dtype)
            images[idx, : , : , :] = image
        else:
            images[idx, : , :] = image

        if idx > 40 and early_stop:
            break
    
    return images
    

def load_data_2D_from_directory(
        image_folder_path: str, 
        norm_image = True, 
        one_hot = False, 
        resized = True,
        resizing_masks = False,
        dtype: np.dtype = np.float32, 
        early_stop = False
        ) -> np.ndarray:
    """
    Returns np array of all Nifti images in the specified image folder.
    
    Creates a list of all image path names in a folder, then feeds 
    that into load_data_2D to return a np arrary of those images.
    
    Parameters:
        image_folder_path (str): Path to folder with Nifti images
        norm_image (bool): Boolean flag to normalise images (mean=0, std=1)
        one_hot (bool): Boolean flag to one hot encode images
        resized (bool): Boolean flag to resize images to (IMG_SIZE, IMG_SIZE)
        resizing_masks (bool): Boolean flag to resize masks with order=0
        dtype (np.dtype): Type and size of the data to be returned as an array
        early_stop (bool): Boolean flag to prematurely stop loading files
    
    Returns:
        np.ndarray: Array containing the data from the Nifti files
    """
    image_names = []
    for file in sorted(os.listdir(image_folder_path)):
        image_names.append(os.path.join(image_folder_path, file))
    return load_data_2D(image_names, norm_image, one_hot, resized, resizing_masks, dtype, early_stop)