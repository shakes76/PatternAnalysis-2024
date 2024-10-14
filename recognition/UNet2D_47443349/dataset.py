import numpy as np
import nibabel as nib
from tqdm import tqdm
import os
import matplotlib.pyplot as plt

N_CLASSES = 6

def to_channels(arr: np.ndarray, dtype=np.uint8) -> np.ndarray:
    channels = np.unique(arr)
    res = np.zeros(arr.shape + (N_CLASSES,), dtype=dtype)
    for c in channels:
        c = int(c)
        res[..., c:c+1][arr == c] = 1
    return res

def load_data_2D(imageNames, normImage=False, categorical=False, dtype=np.float32, getAffines=False, early_stop=False):
    '''
    Load medical image data from names, cases list provided into a list for each.

    This function pre-allocates 4D arrays for conv2d to avoid excessive memory usage.

    normImage: bool (normalise the image 0.0 to 1.0)
    early_stop: Stop loading pre-maturely, leaves arrays mostly empty, for quick loading and testing scripts.
    '''
    affines = []

    # Get fixed size
    num = len(imageNames)
    first_case = nib.load(imageNames[0]).get_fdata(caching='unchanged')

    if len(first_case.shape) == 3:
        first_case = first_case[:,:,0] # Sometimes extra dims -> remove
    
    if categorical:
        first_case = to_channels(first_case, dtype=dtype)
        rows, cols, channels = first_case.shape
        images = np.zeros((num, rows, cols, channels), dtype=dtype)
    else:
        rows, cols = first_case.shape
        images = np.zeros((num, rows, cols), dtype=dtype)

    for i, inName in enumerate(tqdm(imageNames)):
        niftiImage = nib.load(inName)
        inImage = niftiImage.get_fdata(caching='unchanged') # Read disk only
        affine = niftiImage.affine
        if len(inImage.shape) == 3:
            inImage = inImage[:,:,0] # Sometimes extra dims in HipMRI study data
        inImage = inImage.astype(dtype)
        if normImage:
            inImage = (inImage - inImage.mean()) / inImage.std()
        if categorical:
            inImage = to_channels(inImage, dtype=dtype)
            images[i,:,:,:] = inImage
        else:
            images[i,:,:] = inImage

        affines.append(affine)
        if i > 20 and early_stop:
            break

    if getAffines:
        return images, affines
    else:
        return images

def get_names(data_path):
    names = [os.path.join(data_path, img) for img in os.listdir(data_path) if img.endswith(('.nii', '.nii.gz'))]
    return names

data_path = 'C:/Users/21pit/OneDrive/Desktop/COMP3710/Project/PatternAnalysis-2024/recognition/UNet2D_47443349'

# Test
imageNames = get_names(data_path + '/keras_slices_test')
images = load_data_2D(imageNames, early_stop=True, normImage=False)
plt.imshow(images[2])
plt.show()

maskNames = get_names(data_path + '/keras_slices_seg_test')
masks = load_data_2D(maskNames, early_stop=True, normImage=False, categorical=True)
plt.imshow(masks[2][:,:,1])
plt.show()

