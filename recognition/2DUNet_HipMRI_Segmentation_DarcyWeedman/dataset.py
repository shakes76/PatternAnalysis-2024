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

class HipMRIDataset(Dataset):
    def __init__(self, data_dir, seg_dir, transform=None):
        self.data_dir = data_dir
        self.seg_dir = seg_dir
        self.transform = transform
        self.image_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.nii.gz')])
        
    def __len__(self):
        return len(self.image_files)