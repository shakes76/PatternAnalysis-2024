import torch
from torch.utils.data import random_split, DataLoader
from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder

def get_dataloaders(base_data_dir, batch_size=16, val_split=0.15):
    """
    This function loads the datasets from the data directory, splitting the train folder 
    into training and validation sets. The directory structure is as follows:
    - base_data_dir/train/AD
    - base_data_dir/train/NC
    - base_data_dir/test/AD
    - base_data_dir/test/NC
    
    :param base_data_dir: Directory containing train and test directories with AD and NC folders
    :param batch_size: Batch size for data loaders
    :param val_split: Percentage of train data used for validation
    :return: train_loader, val_loader, test_loader
    """
    train_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    test_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_data = ImageFolder(root=f"{base_data_dir}/train", transform=train_transforms)
    
    val_size = int(len(train_data) * val_split)
    train_size = len(train_data) - val_size
    train_data, val_data = random_split(train_data, [train_size, val_size])
    
    test_data = ImageFolder(root=f"{base_data_dir}/test", transform=test_transforms)
    
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader
