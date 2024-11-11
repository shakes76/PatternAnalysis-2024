"""
dataset.py

Data loader for loading and preprocessing the ADNI dataset.

Author: Chiao-Yu Wang (Student No. 48007506)
"""
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from constants import IMAGE_SIZE, TRAIN_DATA_PATH, TEST_DATA_PATH, BATCH_SIZE

# Set manual seed for reproducibility
torch.manual_seed(1)

def load_data():
    """
    Loads the ADNI dataset and creates DataLoader objects for training, validation, and testing.

    This function applies different data augmentation and preprocessing transformations to the 
    training and testing datasets. It also splits the test dataset into a validation set for 
    model evaluation and hyperparameter tuning.

    Returns:
        tuple: A tuple containing three DataLoader objects:
            - train_loader (DataLoader): DataLoader for the training dataset.
            - validation_loader (DataLoader): DataLoader for the validation dataset.
            - test_loader (DataLoader): DataLoader for the testing dataset.
    """
    # Define data transformations for training
    transform_train = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),                        # Resize images
        transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.8, 1.0)),         # Random crop
        transforms.RandomHorizontalFlip(p=0.5),                             # Flip images horizontally
        transforms.RandomRotation(2),                                       # Rotate images randomly
        transforms.Grayscale(num_output_channels=1),                        # Convert to grayscale
        transforms.ToTensor(),                                              # Convert image to tensor
        transforms.Normalize(mean=[0.1156], std=[0.2385])                   # Normalise values
    ])

    # Define data transformations for testing (no data augmentation)
    transform_test = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),                        # Resize images
        transforms.Grayscale(num_output_channels=1),                        # Convert to grayscale
        transforms.ToTensor(),                                              # Convert image to tensor
        transforms.Normalize(mean=[0.1156], std=[0.2385])                   # Normalise values
    ])

    # Load and transform the training and test datasets
    train_data = datasets.ImageFolder(TRAIN_DATA_PATH, transform=transform_train)
    test_data = datasets.ImageFolder(TEST_DATA_PATH, transform=transform_test)

    # Split test_data into validation and test sets
    val_size = len(test_data) // 2
    test_size = len(test_data) - val_size
    validation_data, test_data = random_split(test_data, [val_size, test_size])

    # Create DataLoaders for training, validation, and test datasets
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    validation_loader = DataLoader(validation_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # Return the DataLoaders for training, validation, and testing
    return train_loader, validation_loader, test_loader
