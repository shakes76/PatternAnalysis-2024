import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_dataloaders(data_dir, batch_size=128):
    """
    Creates and returns DataLoader objects for training and testing datasets.

    Parameters:
    - data_dir (str): The directory containing 'train' and 'test' subdirectories.
    - batch_size (int): Number of samples per batch. Default is 128.

    Returns:
    - dataloaders (dict): DataLoader objects for training and testing datasets.
    - dataset_sizes (dict): Number of images in training and testing datasets.
    """
    
    # Define data transformations for training and testing datasets
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],  # Normalize image channels
                                 std=[0.229, 0.224, 0.225]),
        ]),
        'test': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ]),
    }

    # Load images from 'train' and 'test' directories with specified transformations
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
                      for x in ['train', 'test']}
    
    # Create DataLoader objects for training and testing datasets
    dataloaders = {
        'train': DataLoader(image_datasets['train'], batch_size=batch_size, shuffle=True, 
                            num_workers=4, pin_memory=True),
        'test': DataLoader(image_datasets['test'], batch_size=batch_size, shuffle=False, 
                           num_workers=4, pin_memory=True)
    }
    
    # Calculate the number of images in training and testing datasets
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'test']}
    
    return dataloaders, dataset_sizes