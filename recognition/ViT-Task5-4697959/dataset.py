# dataset.py
import os
import random
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import torch
from sklearn.utils import class_weight
import numpy as np

class ADNIDataset(Dataset):
    def __init__(self, data_dir, subset, transform=None):
        """
        Args:
            data_dir (string): Directory with all the images.
            subset (string): 'train' or 'test' to specify which subset of the data to use.
            transform (callable, optional): Optional transform to be applied on an image.
        """
        self.transform = transform
        self.root_dir = data_dir
        self.subset = subset

        # List all file paths with their corresponding label
        self.data_paths = []
        for label_class in ['AD', 'NC']:
            class_dir = os.path.join(data_dir, self.subset, label_class)
            for filename in os.listdir(class_dir):
                if os.path.isfile(os.path.join(class_dir, filename)):
                    self.data_paths.append((os.path.join(class_dir, filename), label_class))

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, idx):
        img_path, label_class = self.data_paths[idx]
        image = Image.open(img_path).convert('RGB')  # Convert image to RGB

        # Encode label: 1 for Alzheimer's (AD), 0 for normal control (NC)
        label = 1 if label_class == 'AD' else 0

        if self.transform:
            image = self.transform(image)

        return image, label

def get_data_loaders(data_dir, batch_size=32, img_size=224, val_split=0.2, num_workers=4):
    """
    Creates training, validation, and test data loaders.

    Args:
        data_dir (str): Path to the ADNI dataset directory.
        batch_size (int): Batch size for data loaders.
        img_size (int): Size to resize the images.
        val_split (float): Fraction of training data to use for validation.
        num_workers (int): Number of subprocesses for data loading.

    Returns:
        dict: Dictionary containing 'train', 'val', and 'test' DataLoaders.
        list: List of class names.
        torch.Tensor: Class weights for handling imbalance.
    """

    # Define image transformations
    train_transforms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet means
                             std=[0.229, 0.224, 0.225])   # ImageNet stds
    ])

    test_val_transforms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    ## Initialize the dataset for train and test
    train_dataset = ADNIDataset(data_dir=data_dir, subset='train', transform=train_transforms)
    test_dataset = ADNIDataset(data_dir=data_dir, subset='test', transform=test_val_transforms)

    # Patient-level splitting for train/validation sets
    unique_patient_ids = list(set('_'.join(path.split('/')[-1].split('_')[:-1]) for path, _ in train_dataset.data_paths))
    validation_patient_ids = set(random.sample(unique_patient_ids, int(val_split * len(unique_patient_ids))))

    # Split train_dataset based on patient IDs
    train_data_paths = [(path, label) for path, label in train_dataset.data_paths if '_'.join(path.split('/')[-1].split('_')[:-1]) not in validation_patient_ids]
    val_data_paths = [(path, label) for path, label in train_dataset.data_paths if '_'.join(path.split('/')[-1].split('_')[:-1]) in validation_patient_ids]

    # Update the data_paths for each dataset split
    train_dataset.data_paths = train_data_paths
    val_dataset = ADNIDataset(data_dir=data_dir, subset='train', transform=train_transforms)
    val_dataset.data_paths = val_data_paths

    # Create DataLoader instances
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # Compute class weights for handling class imbalance
    labels = [label for _, label in train_dataset.data_paths]
    class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(labels), y=labels)
    class_weights = torch.tensor(class_weights, dtype=torch.float)

    # Return data loaders and metadata
    return {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader
    }, ['NC', 'AD'], class_weights