"""
Contains the source code for the components of GFNet classifying the Alzheimer’s disease (normal and AD) of the ADNI brain data
Each component is implementated as a class or a function.
"""

import torch
import torch.nn as nn
from torchvision.models import vit_b_16, ViT_B_16_Weights
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
import torchvision
from torchvision import transforms


class ADNIDataset(Dataset):
    """
    Modified Dataset class to handle grayscale images
    """
    def __init__(self, root_path, train=True, transform=None):
        self.path = Path(root_path, 'train' if train else 'test')
        self.transform = transform

        # Initialize lists for each class
        self.files = []
        self.labels = []

        # Define class mapping
        self.class_to_idx = {
            'CN': 0,   # Cognitively Normal
            'MCI': 1,  # Mild Cognitive Impairment
            'AD': 2,   # Alzheimer's Disease
            'SMC': 3   # Subjective Memory Concern
        }

        # Load files for each class
        for class_name in self.class_to_idx.keys():
            class_path = Path(self.path, class_name)
            if class_path.exists():
                class_files = list(class_path.glob('*.jpg')) + list(class_path.glob('*.png'))
                self.files.extend(class_files)
                self.labels.extend([self.class_to_idx[class_name]] * len(class_files))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_path = self.files[idx]
        label = self.labels[idx]

        # Load as grayscale directly
        image = Image.open(img_path).convert('L')  # 'L' mode for grayscale

        if self.transform:
            image = self.transform(image)

        return image, label

class ViTClassifier(nn.Module):
    """
    Modified Vision Transformer for grayscale images
    """
    def __init__(self, num_classes=4):
        super(ViTClassifier, self).__init__()

        # Load the pre-trained ViT model
        self.vit = torchvision.models.vit_b_16(pretrained=True)

        # Modify the first layer to accept grayscale input
        # Create new patch embedding layer with 1 input channel instead of 3
        new_patch_embed = nn.Conv2d(
            in_channels=1,  # Changed from 3 to 1 for grayscale
            out_channels=768,
            kernel_size=16,
            stride=16
        )

        # Initialize the weights of the new layer
        # Average the weights across the RGB channels
        with torch.no_grad():
            new_patch_embed.weight = nn.Parameter(
                self.vit.conv_proj.weight.sum(dim=1, keepdim=True) / 3.0
            )
            new_patch_embed.bias = nn.Parameter(self.vit.conv_proj.bias)

        # Replace the patch embedding layer
        self.vit.conv_proj = new_patch_embed

        # Modify the classifier head for our number of classes
        num_features = self.vit.heads.head.in_features
        self.vit.heads.head = nn.Linear(num_features, num_classes)

    def forward(self, x):
        return self.vit(x)