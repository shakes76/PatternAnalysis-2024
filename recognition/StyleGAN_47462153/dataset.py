import os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
from PIL import Image

def get_dataloader(image_size=64, batch_size=16, data_root='/home/groups/comp3710/ADNI/AD_NC/train', shuffle=True, num_workers=4):
    """
    Returns a DataLoader and the dataset for training.

    Args:
        image_size (int): Desired image size after transformations.
        batch_size (int): Number of samples per batch.
        data_root (str): Root directory of the dataset.
        shuffle (bool): Whether to shuffle the data.
        num_workers (int): Number of subprocesses for data loading.

    Returns:
        DataLoader: PyTorch DataLoader for the dataset.
        Dataset: The underlying dataset.
    """
    # Series of transformations: resize, center crop, convert to grayscale, tensor conversion, and normalization
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])
    
    dataset = datasets.ImageFolder(root=data_root, transform=transform)
    
    # DataLoader with the dataset, batch size, shuffling, and multiprocessing workers for efficient data loading
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True  # Enable pin_memory for faster data transfer to GPU
    )
    
    return dataloader, dataset
