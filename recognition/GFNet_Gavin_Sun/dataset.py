import torch
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from pathlib import Path
import os
import torchvision.transforms as transforms

# Paths to the ADNI dataset
ADNI_ROOT_PATH = Path('/home', 'groups', 'comp3710', 'ADNI', 'AD_NC')


# Transformations for training set and preprocessing testing set
TRAIN_TRANSFORM = transforms.Compose([
    transforms.RandomRotation(degrees=10),
    transforms.RandomResizedCrop(size=224),
    transforms.ColorJitter(brightness=(0.8, 1.2)), 
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.0062], std=[0.0083])
])

TEST_TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.0062], std=[0.0083])
])


class ADNIDataset(Dataset):
    """
    Custom PyTorch Dataset for loading images from the ADNI dataset.
    """

    def __init__(self, root_dir, train=True, transform=None):
        """
        Initializes the ADNIDataset.

        Args:
            root_dir (str or Path): Path to the root directory of the dataset.
            train (bool): If True, load training data. If False, load test data.
            transform (callable, optional): Optional transform.
        """

        self.root_dir = Path(root_dir, 'train' if train else 'test')
        self.transform = transform
        self.image_paths = []
        self.labels = []

        # Load AD (Alzheimer's Disease) class images and labels
        ad_path = self.root_dir / 'AD'
        ad_files = os.listdir(ad_path)
        self.image_paths.extend([ad_path / file for file in ad_files])
        self.labels.extend([1] * len(ad_files))  
      
        # Load NC (Normal Control) class images and labels
        nc_path = self.root_dir / 'NC'
        nc_files = os.listdir(nc_path)
        self.image_paths.extend([nc_path / file for file in nc_files])
        self.labels.extend([0] * len(nc_files))  

    def __len__(self):
        """
        Returns the total number of samples in the dataset.

        Returns:
            int: Number of samples in the dataset.
        """

        return len(self.image_paths)

    def __getitem__(self, idx):
        """
        Retrieves an image and its corresponding label by index.

        Args:
            idx (int): Index of the image to retrieve.

        Returns:
            tuple: transformed image and its corresponding label
        """

        image_path = self.image_paths[idx]
        label = self.labels[idx]

        # Open image in grayscale
        image = Image.open(image_path).convert('L')
        if self.transform:
            image = self.transform(image)

        return image, label

def get_adni_dataloader(batch_size, train=True, val_split=0.2):
    """
    Creates DataLoader objects for the ADNI dataset.

    Args:
        batch_size (int): Number of samples per batch to load.
        train (bool): If True, creates data loaders for both training and validation datasets.
                      If False, creates a data loader for the test dataset.
        val_split (float): Fraction of the training data to be used for validation (applicable if train=True).

    Returns:
        If train=True:
            tuple: (train_loader, val_loader) where train_loader is for training data and val_loader is for validation data.
        If train=False:
            DataLoader: test_loader for the test dataset.
    """
    
    if train:
        # Create full training dataset and split into training and validation sets
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
