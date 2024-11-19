"""
File: train.py
Description: Trains a Siamese Network on pairs of images from the ISIC dataset to learn image similarity.
            Uses a contrastive loss to enforce similarity between pairs of images labeled as similar and dissimilarity for pairs labeled as different.

Functions:
    train_siamese_network: Trains the Siamese Network on the provided dataset.
    
Main Process:
    1. Loads the ISIC dataset from Kaggle.
    2. Initializes data transformations and the ISIC dataset class.
    3. Trains the Siamese Network and saves the model state to a file.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import kagglehub
import os
import time
from torchvision import transforms
from torch.utils.data import DataLoader
from modules import CNN, SiameseNetwork, ContrastiveLoss
from dataset import ISICDataset, SiameseDataset

# Use GPU if available, otherwise fall back to CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train_siamese_network(dataset, transform=None, epochs=10):
    """
    Trains a Siamese Network for image similarity on the provided dataset.
    
    Args:
        dataset (Dataset): The dataset of images to be used for training.
        transform (torchvision.transforms.Compose, optional): Transformations to apply to each image.
        epochs (int, optional): Number of epochs for training.
        
    Saves:
        Model weights to 'model.pth' after training.
    """
    # Create a DataLoader for the SiameseDataset with pairs of images
    train_dataloader = DataLoader(SiameseDataset(dataset, transform=transform), batch_size=16, shuffle=True)
    net = SiameseNetwork().to(device)  # Initialize the Siamese Network and move to device
    criterion = ContrastiveLoss()  # Use contrastive loss for similarity learning
    optimizer = optim.Adam(net.parameters(), lr=0.001)  # Use Adam optimizer

    print(f'Number of iterations per epoch: {len(train_dataloader)}')

    # Iterate over epochs
    for epoch in range(epochs):
        print(f'Starting epoch {epoch + 1}/{epochs}')
        
        for i, (img1, img2, labels) in enumerate(train_dataloader):
            # Move images and labels to device
            img1, img2, labels = img1.to(device), img2.to(device), labels.to(device)
            
            optimizer.zero_grad()  # Zero the parameter gradients
            
            # Forward pass through the network to get similarity outputs
            similarity = net(img1, img2)

            # Compute loss based on similarity and labels
            loss = criterion(*similarity, labels.float())

            # Backpropagation and optimization step
            loss.backward()
            optimizer.step()

        # Log loss after each epoch
        print(f'Epoch {epoch + 1}/{epochs} Loss: {loss.item()}')

    # Save the trained model's state
    torch.save(net.state_dict(), 'model.pth')
    print(f'Model saved to {os.path.abspath("model.pth")}')

if __name__ == "__main__":
    # Download and set paths for the ISIC dataset
    dataset_path = kagglehub.dataset_download("nischaydnk/isic-2020-jpg-256x256-resized")
    dataset_image_path = os.path.join(dataset_path, "train-image/image")
    meta_data_path = os.path.join(dataset_path, "train-metadata.csv")

    # Define transformations to resize and normalize images
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    # Initialize the ISIC dataset with image paths and metadata
    isic_dataset = ISICDataset(dataset_path=dataset_image_path, metadata_path=meta_data_path)

    # Track and print the time taken to complete training
    start_time = time.time()
    train_siamese_network(dataset=isic_dataset, transform=transform, epochs=25)
    elapsed_time = time.time() - start_time
    print(f'Training completed in {elapsed_time // 60:.0f}m {elapsed_time % 60:.0f}s')
