import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class ADNIDataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None):
        self.root_dir = os.path.join(root_dir, split)
        self.transform = transform
        self.ad_files = sorted(os.listdir(os.path.join(self.root_dir, 'AD')))
        self.nc_files = sorted(os.listdir(os.path.join(self.root_dir, 'NC')))

        self.images = [(file, 1) for file in self.ad_files] + [(file, 0) for file in self.nc_files]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name, label = self.images[idx]
        img_path = os.path.join(self.root_dir, 'AD' if label == 1 else 'NC', img_name)
        image = Image.open(img_path).convert('L') 

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.float32)

def get_dataloaders(data_dir, batch_size=64, validation_split=0.15):
    """
    Creates DataLoader objects for training, validation, and testing datasets.
    
    Args:
        data_dir (str or Path): The root directory of the dataset.
        batch_size (int): The number of samples per batch.
        validation_split (float): Proportion of training data to be used for validation.
        
    Returns:
        tuple: (train_loader, val_loader, test_loader) DataLoader objects for each split.
    """
    
    # Define the custom transformations for training and testing
    train_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(30),
        transforms.ColorJitter(brightness=0.3, contrast=0.4, saturation=0.3, hue=0.2),
        transforms.RandomVerticalFlip(),
        transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    
    test_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    # Load training and testing datasets with their respective transforms
    train_dataset = ADNIDataset(data_dir, split='train', transform=train_transforms)
    test_dataset = ADNIDataset(data_dir, split='test', transform=test_transforms)

    # Split the training data into training and validation sets
    val_size = int(validation_split * len(train_dataset))
    train_size = len(train_dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])

    # Create DataLoader objects for each dataset
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader

