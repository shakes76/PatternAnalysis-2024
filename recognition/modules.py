"""
Contains the source code for the components of GFNet classifying the Alzheimer’s disease (normal and AD) of the ADNI brain data
Each component is implementated as a class or a function.
"""

import torch
import torch.nn as nn
from torchvision.models import vit_b_16, ViT_B_16_Weights
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

class ADNIDataset(torch.utils.data.Dataset):
    """
    PyTorch dataset class for the ADNI brain imaging data.
    """
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []

        # Load data and labels
        for label in ['normal', 'ad']:
            label_dir = os.path.join(data_dir, label)
            for filename in os.listdir(label_dir):
                self.image_paths.append(os.path.join(label_dir, filename))
                self.labels.append(0 if label == 'normal' else 1)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx])
        if self.transform:
            image = self.transform(image)
        return image, self.labels[idx]

class ViTClassifier(nn.Module):
    """
    Vision Transformer model for Alzheimer's classification.
    """
    def __init__(self, num_classes=2):
        super(ViTClassifier, self).__init__()
        self.vit = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
        self.vit.heads.head = nn.Linear(self.vit.heads.head.in_features, num_classes)

    def forward(self, x):
        return self.vit(x)

