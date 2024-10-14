import os
import nibabel as nib
import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class HipMRIDataset(Dataset):
    def __init__(self, data_dir, transform, num_samples=None):
        self.data_dir = data_dir
        
        if num_samples:
            self.file_names = [f for f in os.listdir(data_dir) if f.endswith('.nii.gz')][:num_samples]
        else:
            self.file_names = [f for f in os.listdir(data_dir) if f.endswith('.nii.gz')]

        self.transform = transform

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        file_path = os.path.join(self.data_dir, self.file_names[idx])
        nii_image = nib.load(file_path)
        image_data = nii_image.get_fdata()
        
        image_data = (image_data - np.min(image_data)) / (np.max(image_data) - np.min(image_data)) * 255  # Normalize to 0-255
        image_data = image_data.astype(np.uint8)
        image_data = Image.fromarray(image_data)

        image_data = self.transform(image_data)

        return image_data
        