
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
        return torch.tensor(images, dtype=torch.float32), affines
    else:
        return torch.tensor(images, dtype=torch.float32)

class ProstateDataset(Dataset):
    def __init__(self, image_path, mask_path, norm_image=False, transform=None, target_size=(128, 64)):
        self.transform = transform
        self.image_paths = sorted([os.path.join(image_path, img) for img in os.listdir(image_path) if img.endswith(('.nii', '.nii.gz'))])
        self.mask_paths = sorted([os.path.join(mask_path, img) for img in os.listdir(mask_path) if img.endswith(('.nii', '.nii.gz'))])
        self.norm_image = norm_image
        self.target_size = target_size

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = load_data_2D([self.image_paths[idx]], normImage=True)
        mask = load_data_2D([self.mask_paths[idx]])

        # Ensure image has shape (C, H, W)
        if image.dim() == 2:
            image = image.unsqueeze(0)
        elif image.dim() == 3 and image.shape[0] != 1:
            image = image.permute(2, 0, 1)

        # Ensure mask is of type torch.LongTensor
        mask = mask.long()


        
        image = transforms.Resize((128, 64))(image)
        mask = transforms.Resize((128, 64))(mask)

        return image, mask
