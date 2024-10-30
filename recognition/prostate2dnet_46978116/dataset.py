
import numpy as np
import nibabel as nib
import glob
import os
import torch
from tqdm import tqdm
from torch.utils.data import Dataset
from skimage.transform import resize
import torchvision.transforms as transforms


def to_channels(arr: np.ndarray, dtype=np.uint8) -> np.ndarray:
    channels = np.unique(arr)
    res = np.zeros(arr.shape + (len(channels),), dtype=dtype)
    for c in channels:
        c = int(c)
        res[..., c:c+1][arr == c] = 1

    return res

def load_data_2D(imageNames, normImage=False, categorical=False, dtype=np.float32, getAffines=False, early_stop=False):
    images = []
    affines = []

    for inName in imageNames:
        niftiImage = nib.load(inName)
        inImage = niftiImage.get_fdata(caching='unchanged')
        affine = niftiImage.affine
        if len(inImage.shape) == 3:
            inImage = inImage[:, :, 0]
        inImage = inImage.astype(dtype)
        if normImage:
            inImage = (inImage - inImage.mean()) / inImage.std()
        if categorical:
            inImage = to_channels(inImage, dtype=dtype)
        images.append(inImage)
        affines.append(affine)

    images = np.stack(images)
    if getAffines:
        return images, affines
    else:
        return images

class ProstateDataset(Dataset):
    def __init__(self, image_path, mask_path, norm_image=False, transform=None, target_size=(128, 64)):
        self.transform = transform
        self.image_path = image_path
        self.mask_path = mask_path
        self.images = os.listdir(self.image_path)
        self.masks = os.listdir(self.mask_path)
        self.target_size = target_size
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_pth = os.path.join(self.image_path,self.images[idx])
        mask_pth= os.path.join(self.mask_path,self.images[idx].replace('case', 'seg'))
        image = load_data_2D([img_pth], normImage=True)
        mask = load_data_2D([mask_pth])

        if self.transform is not None:
            augm = self.transform(image=image,mask=mask)
        

        return image, mask
