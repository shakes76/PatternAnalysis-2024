import numpy as np
import nibabel as nib
from tqdm import tqdm
import skimage.transform as skTrans


def to_channels(arr: np.ndarray, dtype=np.uint8) -> np.ndarray:
    """
    Convert a categorical image array to a one-hot encoded format.

    Args:
        arr (np.ndarray): Input array of shape (H, W) containing categorical values.
        dtype (type, optional): Desired data type for the output array.

    Returns:
        np.ndarray: One-hot encoded array of shape (H, W, C), where C is number of unique categories.
    """
    channels = np.unique(arr)
    # One-Hot Code encoding category for image data
    res = np.zeros(arr.shape + (len(channels),), dtype=dtype)
    for c in channels:
        c = int(c)
        res[..., c:c+1][arr == c] = 1
    return res


def load_data_2D(imageNames, normImage=False, categorical=False, dtype=np.float32, 
                getAffines=False, early_stop=False, target_shape=(256, 144)):
    """
    Load and preprocess 2D medical image data from a list of file names.

    This function pre-allocates arrays for efficient memory usage and handles 
    normalization, categorical encoding, and image resizing.

    Args:
        imageNames (list): List of file paths to the image files.
        normImage (bool, optional): If True, normalize the image data to the range [0.0, 1.0].
        categorical (bool, optional): If True, convert images to one-hot encoded format.
        dtype (type, optional): Desired data type for the output array.
        getAffines (bool, optional): If True, return the affine matrices along with images.
        early_stop (bool, optional): If True, stop loading after the first 20 images for quick testing.
        target_shape (tuple, optional): Desired output shape for images after resizing.

    Returns:
        np.ndarray: Preprocessed images of shape (num_images, H, W) or (num_images, H, W, C) if categorical.
        list (optional): Affine matrices corresponding to each image, returned if getAffines is True.
    """
    affines = []

    # Get dataset size
    num = len(imageNames)
    
    # Load the first image to determine the initial shape
    first_case = nib.load(imageNames[0]).get_fdata(caching='unchanged')
    if len(first_case.shape) == 3:
        first_case = first_case[:, :, 0]  # Remove unwanted dims if exists

    # Resize first image to get dimensions for pre-allocated array
    resized_first_case = skTrans.resize(first_case, target_shape, order=1, preserve_range=True)

    if categorical: # Initialize numpy array with images and respective categories
        rows, cols = resized_first_case.shape
        images = np.zeros((num, rows, cols, len(np.unique(resized_first_case))), dtype=dtype)
    else:
        rows, cols = resized_first_case.shape
        images = np.zeros((num, rows, cols), dtype=dtype)

    for i, inName in enumerate(tqdm(imageNames)): # Progress bar
        niftiImage = nib.load(inName)
        inImage = niftiImage.get_fdata(caching='unchanged')
        affine = niftiImage.affine
        if len(inImage.shape) == 3:
            inImage = inImage[:, :, 0]

        # Resize the image to target shape
        inImage = skTrans.resize(inImage, target_shape, order=1, preserve_range=True)

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
