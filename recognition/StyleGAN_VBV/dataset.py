import os 
import glob  
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
import matplotlib.pylab as plt
from math import log2

# Constants for dataset and model configuration
AD_PATH = '/home/Student/s4742656/PatternAnalysis-2024/recognition/StyleGAN_VBV/AD_NC/AD'
NC_PATH = '/home/Student/s4742656/PatternAnalysis-2024/recognition/StyleGAN_VBV/AD_NC/NC'
DATASET_PATH = AD_PATH  # Path to the dataset
START_TRAIN_IMG_SIZE = 4  # Starting image size for training
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'  # Use GPU if available, otherwise use CPU
LR = 1e-3  # Learning rate
BATCH_SIZES = [256, 256, 128, 64, 32, 16]  # Batch sizes for different image sizes
CHANNELS_IMG = 3  # Number of image channels (RGB)
Z_Dim = 512  # Latent space dimension
W_DIM = 512  # Style space dimension
IN_CHANNELS = 512  # Input channels for the generator
LAMBDA_GP = 10  # Weight for the gradient penalty
PROGRESSIVE_EPOCHS = [30] * len(BATCH_SIZES)  # Number of epochs for each batch size

def get_loader(image_size):
    """
    Create a DataLoader for the dataset with the specified image size.

    Parameters:
        image_size (int): The target size for resizing images.

    Returns:
        loader (DataLoader): The DataLoader for the dataset.
        dataset (Dataset): The dataset object.
    """
    # Define image transformations
    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),  # Resize images to the specified size
            transforms.ToTensor(),  # Convert images to PyTorch tensors
            transforms.RandomHorizontalFlip(p=0.5),  # Randomly flip images horizontally
            transforms.Normalize(  # Normalize images to have mean 0 and std 1
                [0.5 for _ in range(CHANNELS_IMG)],  # Mean for each channel
                [0.5 for _ in range(CHANNELS_IMG)],  # Std for each channel
            )
        ]
    )
    
    # Select batch size based on the image size
    batch_size = BATCH_SIZES[int(log2(image_size / 4))]
    
    # Load the dataset from the specified directory
    dataset = datasets.ImageFolder(root=DATASET_PATH, transform=transform)
    
    # Create a DataLoader for the dataset
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True  # Shuffle data for training
    )
    return loader, dataset

def check_loader():
    """
    Visualize some samples from the DataLoader to verify loading functionality.
    """
    # Get a DataLoader for images of size 128
    loader, _ = get_loader(128)
    
    # Retrieve a batch of images
    cloth, _ = next(iter(loader))
    
    # Create a 3x3 subplot to display images
    _, ax = plt.subplots(3, 3, figsize=(8, 8))
    plt.suptitle('Some real samples')  # Title for the plot
    ind = 0  # Initialize index for image selection
    
    # Loop through the grid of subplots
    for k in range(3):
        for kk in range(3):
            # Display each image after permuting the dimensions
            ax[k][kk].imshow((cloth[ind].permute(1, 2, 0) + 1) / 2)  # Rescale to [0, 1] for visualization
            ind += 1  # Increment index for the next image
