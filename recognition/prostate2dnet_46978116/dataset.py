# dataset.py
import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset
from skimage.transform import resize


def to_channels(arr: np.ndarray, dtype=np.uint8) -> np.ndarray:
    channels = np.unique(arr)
    res = np.zeros(arr.shape + (len(channels),), dtype=dtype)
    for c in channels:
        c = int(c)
        res[..., c:c+1][arr == c] = 1
    return res

def load_data_2D(image_names, norm_image=False, categorical=False, dtype=np.float32, get_affines=False, early_stop=False, target_size=(256, 256)):
    '''
    Load medical image data from names, cases list provided into a list for each.

    This function resizes images to a fixed size to ensure consistency.

    norm_image: bool (normalize the image)
    early_stop: Stop loading prematurely, leaves arrays mostly empty, for quick loading and testing scripts.
    '''
    affines = []
    images = []
    
    for i, inName in enumerate(image_names):
        niftiImage = nib.load(inName)
        inImage = niftiImage.get_fdata(caching='unchanged')
        affine = niftiImage.affine
        if len(inImage.shape) == 3:
            inImage = inImage[:, :, 0]  # Remove extra dimension if present
        inImage = inImage.astype(dtype)
        if norm_image:
            inImage = (inImage - inImage.mean()) / inImage.std()
        if categorical:
            inImage = to_channels(inImage, dtype=dtype)
        else:
            # Resize the image to the target size
            inImage = resize(inImage, target_size, anti_aliasing=True, preserve_range=True)
        images.append(inImage)
        affines.append(affine)
        if i > 20 and early_stop:
            break

    images = np.array(images, dtype=dtype)

    if get_affines:
        return images, affines
    else:
        return images

class ProstateDataset(Dataset):
    def __init__(self, image_paths, mask_paths, norm_image=False, transform=None):
        self.transform = transform

        # Load images and masks using load_data_2D
        self.images = load_data_2D(image_paths, norm_image=norm_image, categorical=False, dtype=np.float32)
        self.masks = load_data_2D(mask_paths, norm_image=False, categorical=False, dtype=np.uint8)

        # Expand dimensions if needed (e.g., add channel dimension)
        if len(self.images.shape) == 3:
            # Shape is (N, H, W), we need (N, 1, H, W)
            self.images = self.images[:, np.newaxis, :, :]
        elif len(self.images.shape) == 4:
            pass  # Shape is already (N, H, W, channels)
        else:
            raise ValueError(f"Unexpected image shape: {self.images.shape}")

        if len(self.masks.shape) == 3:
            # Shape is (N, H, W), we need (N, 1, H, W)
            self.masks = self.masks[:, np.newaxis, :, :]
        elif len(self.masks.shape) == 4:
            pass  # Shape is already (N, H, W, channels)
        else:
            raise ValueError(f"Unexpected mask shape: {self.masks.shape}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        mask = self.masks[idx]

        # Convert to tensors
        image = torch.from_numpy(image).float()
        mask = torch.from_numpy(mask).float()

        # Apply transformations if any
        if self.transform:
            image, mask = self.transform(image, mask)

        return image, mask