import os
import numpy as np
import nibabel as nib
from torch.utils.data import Dataset
from tqdm import tqdm

def load_data_2D(imageNames, normImage=False, categorical=False, dtype=np.float32):
    images = []
    for inName in tqdm(imageNames):
        niftiImage = nib.load(inName)
        inImage = niftiImage.get_fdata(caching='unchanged')
        if len(inImage.shape) == 3:
            inImage = inImage[:,:,0] 
        inImage = inImage.astype(dtype)
        if normImage:
            inImage = (inImage - inImage.mean()) / inImage.std()
        images.append(inImage)
    return np.array(images)