import os
import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

def load_nifti_images_from_folder(folder_path):
    image_names = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.nii.gz'):
                image_names.append(os.path.join(root, file))
    return image_names

class NiftiDataset(Dataset):
    def __init__(self, image_names, norm_image=False, categorical=False):
        self.image_names = image_names
        self.norm_image = norm_image
        self.categorical = categorical
        self.images = self.load_data_2D()

    def load_data_2D(self):
        images = []
        for in_name in tqdm(self.image_names):
            nifti_image = nib.load(in_name)
            in_image = nifti_image.get_fdata(caching='unchanged')

            if len(in_image.shape) == 3:
                in_image = in_image[:, :, 0]

            in_image = in_image.astype(np.int32)

            if self.norm_image:
                in_image = (in_image - in_image.mean()) / in_image.std()

            if self.categorical:
                in_image = self.to_channels(in_image)
                images.append(in_image)
            else:
                images.append(in_image[np.newaxis, ...])

        return np.array(images)

    def to_channels(self, arr):
        channels = np.unique(arr)
        channels = channels[channels >= 0]
        res = np.zeros(arr.shape + (len(channels),), dtype=np.uint8)

        for c in channels:
            if isinstance(c, (int, np.integer)):
                res[..., c] = (arr == c).astype(np.uint8)

        return res

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        return torch.tensor(image, dtype=torch.float32)

