"""
This module provides utility functions for handling image datasets, including loading datasets,
splitting them into training, validation, and test sets, and calculating dataset statistics
such as mean and standard deviation. These statistics are used for normalization to improve
the training process for deep learning models.

Key Functions:
1. `get_mean_std`: Computes the mean and standard deviation of a dataset.
2. `get_dataloaders`: Loads the dataset, applies transformations, and returns DataLoaders for 
   the training, validation, and test datasets.
3. `main`: Example function to calculate the mean and standard deviation for the training set.

Note: The mean and standard deviation should be calculated using only the training dataset
to avoid data leakage. These values should then be used to normalize the training, validation, and test datasets during model training.
"""
import torch
import torchvision
from tqdm import tqdm
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

def get_mean_std(loader: DataLoader):
    """
    Computes the mean and standard deviation of the dataset for normalization.

    Args:
        loader (torch.utils.data.DataLoader): DataLoader for the dataset to calculate mean and std.

    Returns:
        mean (torch.Tensor): Tensor containing the mean value for each channel (R, G, B).
        std (torch.Tensor): Tensor containing the standard deviation for each channel (R, G, B).
    
    Note:
        This function assumes the input images have 3 channels (R, G, B). If using grayscale images, you should modify the mean and std calculation for a single channel.
    """
    mean = torch.zeros(3)
    squared_mean = torch.zeros(3)
    N = 0 # Number of batches
    for images, _ in tqdm(loader, desc="Computing mean and std"): 
        # Images are [32, 3, 224, 224]
        
        num_batches, num_channels, height, width = images.shape
        N += num_batches

        mean += images.sum(dim=(0,2,3)) # Mean is size [3], i.e it's the mean sum over each channel [R, G, B]...
        squared_mean += (images ** 2).sum(dim=(0,2,3)) # Accumulate squared mean
    mean /= N * height * width # Divide the summed mean by the number of pixels we've ever seen (i.e mean is on a per pixel basis)

    squared_mean /= N * height * width # Same with squared mean
    
    # Get std
    std = torch.sqrt((squared_mean - mean ** 2)) # Std per pixel
    
    return mean, std


def get_dataloaders(batch_size: int = 32, image_size: int = 224, train_fraction = 0.9, num_workers = 2, path: str = "./ADNI/AD_NC", use_transforms = True):
    """
    Loads the training, validation, and test datasets, applies optional transformations,
    and returns DataLoaders for each dataset.

    Args:
        batch_size (int, optional): Batch size for DataLoaders. Default is 32.
        image_size (int, optional): Image size to resize the images. Default is 224.
        train_fraction (float, optional): Fraction of the data used for training. Default is 0.9.
        num_workers (int, optional): Number of workers for data loading. Default is 2.
        path (str, optional): Path to the dataset directory. Default is './ADNI/AD_NC'.
        use_transforms (bool, optional): Whether to apply data augmentation and normalization 
            transforms. Default is True.

    Returns:
        tuple: A tuple containing:
            - train_dataloader (torch.utils.data.DataLoader): DataLoader for the training dataset.
            - val_dataloader (torch.utils.data.DataLoader): DataLoader for the validation dataset.
            - test_dataloader (torch.utils.data.DataLoader): DataLoader for the test dataset.
    
    Note:
        The mean and standard deviation for normalization are pre-calculated based on the training
        dataset. These values are used consistently across training, validation, and test datasets
        to avoid data leakage.
    """


    train_dir = f"{path}/train"
    test_dir = f"{path}/test"

    # Found from using get_mean_std in the past on training dataset.
    mean = torch.Tensor([0.1156, 0.1156, 0.1156]) 
    std =  torch.Tensor([0.2229, 0.2229, 0.2229])

    ### Optional parameter, where you can turn advanced transforms like normalisation off (NOT RECOMMENDED!!). This was only to see the difference in performance between two models.
    if use_transforms:
        print("Doing super aggressive transforms.  This should only be in v6 and not v5!! If it's in v5, that means that the sbatch ran this too.... ah...")
        data_transforms = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.CenterCrop(image_size // 1.2),  # Center crop
            transforms.RandomHorizontalFlip(),  # Horizontal flip for augmentation
            transforms.RandomRotation(degrees=10),  # Slight random rotation
            transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),  # Randomly crop and resize
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.1156, 0.1156, 0.1156], std=[0.2229, 0.2229, 0.2229])  # Assuming these are your normalization values
        ])
    else:
        print("Simple basic transform (only normalisation) being used!!")
        data_transforms = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.1156, 0.1156, 0.1156], std=[0.2229, 0.2229, 0.2229])  # Assuming these are your normalization values
        ])


    # Load the full train dataset
    train_data = datasets.ImageFolder(root=train_dir, transform=data_transforms)
    # Load the test dataset (no splitting needed for test)
    test_data = datasets.ImageFolder(root=test_dir, transform=data_transforms)

    # Split the train_data into training and validation datasets
    train_size = int(train_fraction * len(train_data))  
    val_size = len(train_data) - train_size  
    gen1 = torch.Generator().manual_seed(42) # Generator for reproducible results 
    train_dataset, val_dataset = random_split(train_data, [train_size, val_size], generator=gen1)


    # Create DataLoaders for the training, validation, and test datasets
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last= True) # Drops 8 images here out of 606 * 32 + 8 images...

    # Val/test dataloaders
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last= True)  # No need to shuffle validation set
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last= True)

    # Check dataset sizes
    print(f"Training set size: {len(train_dataset)}")
    print(f"Validation set size: {len(val_dataset)}")
    print(f"Test set size: {len(test_data)}")

    # Shuffle not needed for test because we aren't training on it. Shuffle is mainly used because, imagine if you had all your 0 classifications in the beginning and the dataloader ate this. You're feeding the NN hundreds of entire "0s" that it's going to backprop in this direction, instead of an 'unbiased sample'. This can affect how well it learns. Shuffling helps better convergence, and generalisation.

    return train_loader, val_loader, test_loader

