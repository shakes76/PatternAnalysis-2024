import numpy as np
import nibabel as nib
from tqdm import tqdm
import skimage.transform as skTrans

def to_channels(arr: np.ndarray, dtype=np.uint8) -> np.ndarray:
    channels = np.unique(arr)
    res = np.zeros(arr.shape + (len(channels),), dtype=dtype)
    for c in channels:
        c = int(c)
        res[..., c:c+1][arr == c] = 1
    return res

def load_data_2D(imageNames, normImage=False, categorical=False, dtype=np.float32, 
                getAffines=False, early_stop=False, target_shape=(256, 144)):
    '''
    Load medical image data from names, cases list provided into a list for each.

    This function pre-allocates 4D arrays for conv2d to avoid excessive memory usage.

    normImage : bool (normalize the image 0.0-1.0)
    early_stop : Stop loading prematurely, leaves arrays mostly empty for quick loading and testing scripts.
    '''
    affines = []

    # Get dataset size
    num = len(imageNames)
    
    # Load the first image to determine the initial shape
    first_case = nib.load(imageNames[0]).get_fdata(caching='unchanged')
    if len(first_case.shape) == 3:
        first_case = first_case[:, :, 0]  # sometimes extra dims, remove

    # Resize the first image to get dimensions for pre-allocated array
    resized_first_case = skTrans.resize(first_case, target_shape, order=1, preserve_range=True)

    if categorical:
        rows, cols = resized_first_case.shape
        images = np.zeros((num, rows, cols, len(np.unique(resized_first_case))), dtype=dtype)
    else:
        rows, cols = resized_first_case.shape
        images = np.zeros((num, rows, cols), dtype=dtype)

    for i, inName in enumerate(tqdm(imageNames)):
        niftiImage = nib.load(inName)
        inImage = niftiImage.get_fdata(caching='unchanged')  # Read from disk only
        affine = niftiImage.affine
        if len(inImage.shape) == 3:
            inImage = inImage[:, :, 0]  # Sometimes extra dims

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
