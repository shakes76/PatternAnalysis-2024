"""
dataset.py

Data loader for loading and preprocessing the ADNI dataset.

Author: Chiao-Yu Wang (Student No. 48007506)
"""
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

TRAIN_DATA_PATH = "AD_NC/train"
TEST_DATA_PATH = "AD_NC/test"
IMAGE_SIZE = 128
BATCH_SIZE = 32

torch.manual_seed(1)

def load_data():
    """
    Loads the ADNI dataset into PyTorch DataLoader objects
    """
    # Define transformations for training and testing
    transform_train = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),      # Resize images to the given size
        transforms.RandomHorizontalFlip(p=0.5),           # Flip images horizontally
        transforms.RandomRotation(2),                     # Rotate images randomly
        transforms.ToTensor(),                            # Convert image to tensor
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Normalise values
    ])

    transform_test = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),      # Resize images to the given size
        transforms.ToTensor(),                            # Convert image to tensor
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Normalise values
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

    return train_loader, validation_loader, test_loader
