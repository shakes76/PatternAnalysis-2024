"""
File: predict.py
Description: Script to load a pre-trained Siamese Network model and classify a dataset of images.
             Calculates the accuracy of the model on a specified number of samples from the dataset.
             
Functions:
    load_model: Loads a trained Siamese Network from a file.
    classify_images: Classifies a batch of images and calculates accuracy on the dataset.
    
Main Process:
    1. Loads image data and metadata from Kaggle.
    2. Initializes and transforms the dataset.
    3. Loads the pre-trained model and evaluates its accuracy on the dataset.
"""

import torch
import os
import pandas as pd
import kagglehub
from torchvision import transforms
from torch.utils.data import DataLoader
from modules import CNN, SiameseNetwork
from dataset import ISICDataset

# Set device to GPU if available, otherwise use CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_model(model_path):
    """
    Load a pre-trained Siamese model and return its CNN component.

    Args:
        model_path (str): Path to the model file (.pth) to load.

    Returns:
        torch.nn.Module: CNN component of the Siamese network, moved to the specified device.
    """
    siamese_model = SiameseNetwork().to(device)  # Load Siamese model
    siamese_model.load_state_dict(torch.load(model_path, map_location=device))  # Load weights
    return siamese_model.cnn  # Return CNN component for classification

def classify_images(model, dataset, transform=None, num_samples=100):
    """
    Classify a set of images and compute the accuracy on the dataset.

    Args:
        model (torch.nn.Module): CNN model used for classification.
        dataset (torch.utils.data.Dataset): Dataset containing images and labels.
        transform (torchvision.transforms.Compose, optional): Transformations applied to images.
        num_samples (int, optional): Number of images to classify (default is 100).
    
    Prints:
        Accuracy of the model on the provided dataset.
    """
    dataloader = DataLoader(dataset, batch_size=16, shuffle=False)  # Load dataset in batches
    correct_predictions = 0
    total_images = 0

    with torch.no_grad():  # Disable gradient computation for efficiency
        for images, labels in dataloader:
            if total_images >= num_samples:
                break  # Stop if the required sample count is reached

            images, labels = images.to(device), labels.to(device)  # Move data to device
            outputs = model(images)  # Get model predictions
            _, predicted = torch.max(outputs, 1)  # Get predicted classes

            correct_predictions += (predicted == labels).sum().item()  # Count correct predictions
            total_images += labels.size(0)  # Update total image count

    accuracy = correct_predictions / total_images  # Calculate accuracy
    print(f'Accuracy of the model on the test dataset: {accuracy * 100:.2f}%')

if __name__ == "__main__":
    # Download dataset and load paths for images and metadata from Kaggle
    dataset_path = kagglehub.dataset_download("nischaydnk/isic-2020-jpg-256x256-resized")
    dataset_image_path = os.path.join(dataset_path, "train-image/image")
    meta_data_path = os.path.join(dataset_path, "train-metadata.csv")
    model_save_path = os.path.join(os.getcwd(), 'model.pth')  # Path to saved model

    # Define transformations for images
    transform = transforms.Compose([
        transforms.Resize((256, 256)),  # Resize images to 256x256 pixels
        transforms.ToTensor(),          # Convert images to tensor format
    ])

    # Define data augmentation transformations
    augment_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),          # Randomly flip images horizontally
        transforms.RandomVerticalFlip(),            # Randomly flip images vertically
        transforms.RandomRotation(30),              # Randomly rotate images up to 30 degrees
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),  # Random color jitter
    ])

    # Load the ISIC dataset with transformations and augmentations
    isic_dataset = ISICDataset(
        dataset_path=dataset_image_path,
        metadata_path=meta_data_path,
        transform=transform,
        augment_transform=augment_transform,
        num_augmentations=5  # Number of augmentations applied to each image
    )

    # Load and evaluate the pre-trained model
    model = load_model(model_save_path)  # Load the CNN model for classification
    model.eval()  # Set model to evaluation mode
    classify_images(model, isic_dataset, transform=transform)  # Classify images and print accuracy
