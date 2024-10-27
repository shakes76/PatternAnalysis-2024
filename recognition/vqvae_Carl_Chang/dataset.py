import os
import numpy as np
import nibabel as nib
import torch
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

def to_channels(arr: np.ndarray, dtype=np.uint8) -> np.ndarray:
    """Convert array to multi-channel format based on unique labels."""
    channels = np.unique(arr)
    res = np.zeros(arr.shape + (len(channels),), dtype=dtype)
    for c in channels:
        c = int(c)
        res[..., c:c+1][arr == c] = 1
    return res

def pad_to_fixed_size(img, target_size=(256, 144)):
    c, h, w = img.shape
    target_h, target_w = target_size

    # Calculate padding values
    pad_h = max(0, target_h - h)
    pad_w = max(0, target_w - w)
    
    # Calculate padding on left/right evenly
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left
    
    # Apply padding to the width, padding both sides equally
    padding = (pad_left, pad_right, 0, 0)  # (left, right, top, bottom)
    return F.pad(img, padding, mode='constant', value=0)  # Pad with zeros

class HipMRI2DDataset(Dataset):
    def __init__(self, data_dir, norm_image=False, categorical=False, dtype=np.float32, get_affines=False, early_stop=False):
        self.data_dir = data_dir
        self.norm_image = norm_image
        self.categorical = categorical
        self.dtype = dtype
        self.get_affines = get_affines
        self.early_stop = early_stop

        # Load all .nii.gz file paths
        self.image_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.nii.gz')]

        # Apply early stop limit if needed
        if self.early_stop:
            self.image_files = self.image_files[:20]  # Only keep the first 20 files for testing


    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        img, affine = self.load_nifti_image(img_path)

        # Convert to tensor
        img = torch.tensor(img, dtype=torch.float32).unsqueeze(0)

        # Pad to a fixed size (256x144)
        img = pad_to_fixed_size(img, target_size=(256, 144))

        # If affines are needed, return them as well
        if self.get_affines:
            return img, affine
        else:
            return img

    def load_nifti_image(self, file_path):
        """Load and preprocess a single Nifti image."""
        nifti_image = nib.load(file_path)
        in_image = nifti_image.get_fdata(caching='unchanged')
        affine = nifti_image.affine

        # Remove extra dimensions if necessary
        if len(in_image.shape) == 3:
            in_image = in_image[:, :, 0]

        # Convert to the specified data type
        in_image = in_image.astype(self.dtype)

        # Normalize the image
        if self.norm_image:
            in_image = (in_image - in_image.mean()) / in_image.std()

        # Convert to categorical if specified
        if self.categorical:
            in_image = to_channels(in_image, dtype=self.dtype)

        return in_image, affine

def get_data_loader(data_dir, batch_size=256, shuffle=True, norm_image=False, categorical=False, get_affines=False, early_stop=False):
    """Create a DataLoader for the dataset."""
    dataset = HipMRI2DDataset(data_dir, norm_image=norm_image, categorical=categorical, get_affines=get_affines, early_stop=early_stop)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
