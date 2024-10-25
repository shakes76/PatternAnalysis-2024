"""
=======================================================================
File Name: dataset.py
Author: Baibhav Mund
Student ID: 48548700
Description:
    This file defines a function for loading and processing image datasets 
    for training models, such as GANs, using PyTorch's DataLoader. It includes
    the following key functionalities:
    
    - Data Augmentation: The images are augmented with random horizontal flips.
    - Image Preprocessing: Images are resized and normalized.
    - Batch Loading: The images are loaded into batches for efficient processing 
      during training.

    The function `get_loader()` is designed to load images from directories, apply 
    specified transformations (e.g., resizing, normalization), and return a DataLoader 
    object that can be used during model training.

Usage:
    1. Specify the path to the dataset directory containing the images.
    2. Call `get_loader()` with the appropriate `log_resolution` (e.g., 6 for 64x64 images) 
       and `batch_size` for training.
    3. Use the returned DataLoader to iterate through the dataset during model training.
    
Parameters:
    - dataset: str
        The path to the root directory of the dataset (must contain subdirectories for each class).
    - log_resolution: int
        The resolution exponent (e.g., log_resolution=6 corresponds to 2^6=64x64 images).
    - batch_size: int
        The number of images to load per batch.

Functions:
    - get_loader(): Returns a DataLoader for the specified dataset, with specified
      image transformations and batching.

Output:
    - DataLoader object: Provides an iterable over the dataset with augmented and preprocessed images.

=======================================================================
"""

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Define a function to create a data loader
def get_loader(dataset, log_resolution, batch_size):
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
    
    # Create a data loader that batches and shuffles the data
    loader = DataLoader(
        dataset,
        batch_size=batch_size,  # Number of samples per batch
        shuffle=True,          # Shuffle the data for randomness
    )
    
    return loader