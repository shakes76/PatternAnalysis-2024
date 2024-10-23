import os
import nibabel as nib
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import Compose


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
    

def get_dataloader(data_dir, batch_size, transform, num_samples, shuffle):
    dataset = HipMRIDataset(data_dir, transform, num_samples)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader


def get_transforms(transform_config: list) -> Compose:
    transform_list = []
    for transform in transform_config:
        transform_name = transform.get('name')
        transform_params = transform.get('params', {})
        transform_fn = getattr(transforms, transform_name)
        transform_list.append(transform_fn(**transform_params))
    return Compose(transform_list)