from tqdm import tqdm
import nibabel as nib
import numpy as np
import cv2
import torch
import os
import numpy as np
import time
import matplotlib as plt

def to_channels(arr: np.ndarray, dtype=np.uint8) -> np.ndarray:
    channels = np.unique(arr)
    res = np.zeros(arr.shape + (len(channels),), dtype=dtype)
    for c in channels:
        c = int(c)
        res[..., c:c+1][arr == c] = 1

    return res

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        try:
            nn.init.xavier_uniform_(m.weight.data)
            m.bias.data.fill_(0)
        except AttributeError:
            print("Skipping initialization of ", classname)
    
def load_data_2D(imageNames, normImage=False, categorical=False, dtype=np.float32, getAffines=False, early_stop=False):
    '''
    Load medical image data from names, cases list provided into a list for each.

    This function pre-allocates 4D arrays for conv2d to avoid excessive memory usage.

    normImage : bool (normalize the image 0.0 - 1.0)
    early_stop : Stop loading pre-maturely, leaves arrays mostly empty, for quick loading and testing scripts.
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
        inImage = inImage.astype(dtype)
        if normImage:
            inImage = (inImage - inImage.mean()) / inImage.std()
        if categorical:
            inImage = to_channels(inImage, dtype=dtype)
            images[i, :, :, :] = inImage
        else:
            inImage = cv2.resize(inImage, (128, 256), interpolation=cv2.INTER_LINEAR)
            images[i, :, :] = inImage

        affines.append(affine)
        if i > 20 and early_stop:
            break

    if getAffines:
        return images, affines
    else:
        return images

def calc_ssim(x, y):
    # Ensure that the tensors are detached and moved to the CPU
    x = x.cpu().detach().numpy()
    y = y.cpu().detach().numpy()   
    # Get batch size
    size = x.shape[0]
    # Initialize a list to store SSIM values for each image in the batch
    ssim_values = []
    # Calculate SSIM for each image in the batch
    for i in range(size):
        # Assuming the images are (batch, channel, height, width), and channel=1 for grayscale
        ssim_val = ssim(x[i, 0], y[i, 0], data_range=1)  
        ssim_values.append(ssim_val)   
    # Calculate mean SSIM for the batch
    mean_ssim = np.mean(ssim_values)
    return mean_ssim
    
def clear_folder(folder_path):
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)

def folder_check()
    if not os.path.exists('models'):
        os.makedirs('models')
    else:
        clear_folder('models')
        
    if not os.path.exists('epoch_reconstructions'):
        os.makedirs('epoch_reconstructions')
    else:
        clear_folder('epoch_reconstructions')