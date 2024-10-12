import torch
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from pathlib import Path
import os
import torchvision.transforms as transforms

# Define paths to the ADNI dataset
ADNI_ROOT_PATH = Path('/home', 'groups', 'comp3710', 'ADNI', 'AD_NC')

# Define the image transformations for training and testing
TRAIN_TRANSFORM = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

TEST_TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class ADNIDataset(Dataset):
    def __init__(self, root_dir, train=True, transform=None):
        """
        Custom Dataset for the ADNI data.
        Args:
            root_dir (str or Path): Root directory path to the ADNI dataset.
            train (bool): Flag to specify whether to load training or testing data.
            transform: Optional transform to be applied on the images.
        """
        self.root_dir = Path(root_dir, 'train' if train else 'test')
        self.transform = transform
        self.image_paths = []
        self.labels = []

        # Load Alzheimer's Disease (AD) images
        ad_path = self.root_dir / 'AD'
        ad_files = os.listdir(ad_path)
        self.image_paths.extend([ad_path / file for file in ad_files])
        self.labels.extend([1] * len(ad_files))  # Label 1 for AD class

        # Load Cognitive Normal (NC) images
        nc_path = self.root_dir / 'NC'
        nc_files = os.listdir(nc_path)
        self.image_paths.extend([nc_path / file for file in nc_files])
        self.labels.extend([0] * len(nc_files))  # Label 0 for NC class

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]

        # Open the image using PIL and apply any specified transformations
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        return image, label

def get_adni_dataloader(batch_size, train=True, val_split=0.2):
    """
    Function to create data loaders for the ADNI dataset.
    Args:
        batch_size (int): Batch size for the data loader.
        train (bool): If True, returns both train and validation loaders. If False, returns test loader.
        val_split (float): The proportion of data to use for validation (only applicable if train=True).
    Returns:
        DataLoader(s) for training, validation, or test sets.
    """
    if train:
        # Create training dataset and split into training and validation sets
        full_dataset = ADNIDataset(root_dir=ADNI_ROOT_PATH, train=True, transform=TRAIN_TRANSFORM)
        train_size = int((1 - val_split) * len(full_dataset))
        val_size = len(full_dataset) - train_size

        train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])

        # Create data loaders for training and validation sets
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        return train_loader, val_loader
    else:
        # Create test dataset and data loader
        test_dataset = ADNIDataset(root_dir=ADNI_ROOT_PATH, train=False, transform=TEST_TRANSFORM)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        return test_loader
