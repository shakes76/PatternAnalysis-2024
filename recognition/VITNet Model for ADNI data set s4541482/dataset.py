# dataset.py

import os
import torch
from torch.utils.data import Dataset
from PIL import Image, ImageOps
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np

class ADNIDataset(Dataset):
    """
    Custom Dataset for loading MRI slices for Alzheimer's disease classification.
    """

    def __init__(self, data_dir, patient_ids=None, transform=None, mode='train'):
        """
        Args:
            data_dir (str): Path to the data directory containing class folders.
            patient_ids (list): List of patient IDs to include in this dataset. If None, include all patients.
            transform (callable, optional): Transform to be applied on a sample.
            mode (str): Mode of the dataset ('train', 'val', 'test') to control data augmentation.
        """
        self.data_dir = data_dir
        self.transform = transform
        self.mode = mode
        self.samples = []

        # Define Albumentations transforms
        if self.mode == 'train':
            self.augmentation_transforms = A.Compose([
                A.Resize(224, 224),  # Ensure images are resized to 224x224
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.ElasticTransform(alpha=1, sigma=50, p=0.2),  # Reduced p to prevent over-distortion
                A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.3),  # Reduced rotate_limit
                A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.3),
                A.RandomGamma(gamma_limit=(80, 120), p=0.3),
                A.GaussNoise(var_limit=(10.0, 30.0), p=0.3),
                A.GridDistortion(p=0.2),  # Reduced p
                # Additional Movement-Based Augmentations
                A.Affine(
                    scale=(0.95, 1.05),              # Slight scaling
                    translate_percent=(0.02, 0.02),  # Slight translation
                    rotate=(-5, 5),                   # Small rotations
                    shear=(-5, 5),                    # Small shears
                    p=0.3
                ),
                A.Normalize(mean=[0.5], std=[0.5]),
                ToTensorV2(),
            ])
        else:
            self.augmentation_transforms = A.Compose([
                A.Resize(224, 224),  # Ensure images are resized to 224x224
                A.Normalize(mean=[0.5], std=[0.5]),
                ToTensorV2(),
            ])

        # Loop through each class directory ('NC' and 'AD')
        for label, class_name in enumerate(['NC', 'AD']):
            class_dir = os.path.join(data_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
            # Iterate over all files in the class directory
            for filename in os.listdir(class_dir):
                if '_' in filename:
                    patient_id, _ = filename.split('_', 1)
                    if patient_ids is None or patient_id in patient_ids:
                        filepath = os.path.join(class_dir, filename)
                        if os.path.isfile(filepath):
                            self.samples.append((filepath, label))

    def __len__(self):
        # Returns the total number of slices
        return len(self.samples)

    def __getitem__(self, idx):
        # Get the file path and label for the given index
        image_path, label = self.samples[idx]

        # Load the image
        try:
            image = Image.open(image_path).convert('L')
        except IOError:
            print(f"Cannot open image {image_path}")
            # Return a zero tensor and label 0
            image = Image.new('L', (224, 224))
            label = 0

        # Resize the image to desired dimensions (224x224)
        image = image.resize((224, 224))  # Resize to match model input

        # Convert PIL image to numpy array
        image = np.array(image)

        # Apply transforms
        if self.mode == 'train':
            augmented = self.augmentation_transforms(image=image)
            image = augmented['image']
        else:
            transformed = self.augmentation_transforms(image=image)
            image = transformed['image']

        # Ensure image has 3 channels for ViT
        image = image.repeat(3, 1, 1)

        return image, label, image_path
