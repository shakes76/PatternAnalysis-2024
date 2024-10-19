# Contains the dataloader for loading and preprocessing data

import os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image


class BrainDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Custom Dataset class to load brain images from a directory.
    
        Args:
            root_dir (str): Path to the directory (either 'train' or 'test') containing AD and NC subfolders.
            transform (callable, optional): Transform to apply to images.
        """
        self.image_paths = []
        self.labels = []
        self.transform = transform

        # Load all image paths and labels
        for label, subfolder in enumerate(['AD', 'NC']):
            folder_path = os.path.join(root_dir, subfolder)
            for filename in os.listdir(folder_path):
                if filename.endswith('.jpg') or filename.endswith('.jpeg') or filename.endswith('.png'):
                    self.image_paths.append(os.path.join(folder_path, filename))
                    self.labels.append(label)  # 0 for AD, 1 for NC

    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        # Load image
        image = Image.open(img_path).convert('RGB')

        # Apply any transformations
        if self.transform:
            image = self.transform(image)

        return image, label







