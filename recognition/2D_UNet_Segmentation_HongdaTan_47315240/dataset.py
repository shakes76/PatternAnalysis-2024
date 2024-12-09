import os
import torch
from torch.utils.data import Dataset

class ProstateCancerDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_filenames = sorted([f for f in os.listdir(image_dir) if f.endswith('.pt')])
        self.mask_filenames = sorted([f for f in os.listdir(mask_dir) if f.endswith('.pt')])
        self.transform = transform

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        # Load preprocessed .pt files
        image_path = os.path.join(self.image_dir, self.image_filenames[idx])
        mask_path = os.path.join(self.mask_dir, self.mask_filenames[idx])

        image = torch.load(image_path, weights_only=True)
        mask = torch.load(mask_path, weights_only=True)

        # Ensure mask is binary
        mask = (mask > 0).float()  # Convert to binary (0, 1)

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        return image, mask

