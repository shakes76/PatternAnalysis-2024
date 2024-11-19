"""
File: dataset.py
Author: Baibhav Mund (ID: 48548700)

Description:
    Defines `get_loader()` to load and process image datasets for model training
    using PyTorch’s DataLoader. Includes data augmentation, resizing, normalization, 
    and batch loading for efficient GAN training. Trains on classes such as AD/CN separately.

Usage:
    - Specify dataset path and call `get_loader()` with desired `log_resolution` and `batch_size`.
    - Use the returned DataLoader for model training.

Parameters:
    - dataset (str): Path to the dataset directory
    - log_resolution (int): Image resolution as a power of 2
    - batch_size (int): Images per batch
    - class_type(str): Either AD or NC depending on the training.

Functions:
    - get_loader(): Returns a DataLoader with transformations and batching.

Output:
    - DataLoader: Iterable for the dataset with processed images.
"""


# Import necessary libraries
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

# Define a function to create a data loader
def get_loader(dataset, log_resolution, batch_size, class_type):
    # Define a series of data transformations to be applied to the images
    transform = transforms.Compose(
        [
            # Resize images to a specified resolution (2^log_resolution x 2^log_resolution)
            transforms.Resize((2 ** log_resolution, 2 ** log_resolution)),
            
            # Convert the images to PyTorch tensors
            transforms.ToTensor(),
            
            # Apply random horizontal flips to augment the data (50% probability)
            transforms.RandomHorizontalFlip(p=0.5),
            
            # Normalize the pixel values of the images to have a mean and standard deviation of 0.5
            transforms.Normalize(
                [0.5, 0.5, 0.5],  # Mean for each channel
                [0.5, 0.5, 0.5],  # Standard deviation for each channel
            ),
        ]
    )
    
    # Create an ImageFolder dataset object that loads images from the specified directory
    dataset = datasets.ImageFolder(root=dataset, transform=transform)
    
    if class_type not in dataset.class_to_idx:
        raise ValueError(f"Invalid class_type '{class_type}'. Expected 'AD' or 'NC'.")
    class_idx = dataset.class_to_idx[class_type]

    # Filter indices for the specified class
    class_indices = [i for i, (_, label) in enumerate(dataset) if label == class_idx]

    # Create a subset dataset for the specified class
    class_dataset = Subset(dataset, class_indices)

    # Create a data loader that batches and shuffles the data
    loader = DataLoader(
        class_dataset,
        batch_size=batch_size,  # Number of samples per batch
        shuffle=True,          # Shuffle the data for randomness
    )
    
    return loader