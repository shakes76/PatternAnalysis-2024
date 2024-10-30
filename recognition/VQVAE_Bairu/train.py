#containing the source code for training, validating, testing and saving your model. 
# The modelshould be imported from “modules.py” and the data loader should be imported from “dataset.py”. 
# Make sure to plot the losses and metrics during training

import os
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
from tqdm import tqdm
from modules import VQVAE  # Import your VQVAE model
from dataset import MedicalImageDataset, get_dataloaders  # Import your dataset classes

# Set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
num_epochs = 100
batch_size = 16
learning_rate = 1e-4
num_embeddings = 512  # Number of embeddings in vector quantizer
embedding_dim = 64  # Dimensionality of embedding
commitment_cost = 0.25  # Commitment cost
num_res_layers = 2  # Number of residual layers
hidden_channels = 64  # Hidden dimension of the VQVAE

# Data directories
/home/groups/comp3710/HipMRI_Study_open/keras_slices_data
#train_dir = "/Users/bairuan/Documents/uqsem8/comp3710/report/cloned/PatternAnalysis-2024/recognition/VQVAE_Bairu/HipMRI_study_keras_slices_data/keras_slices_train"
#val_dir = "/Users/bairuan/Documents/uqsem8/comp3710/report/cloned/PatternAnalysis-2024/recognition/VQVAE_Bairu/HipMRI_study_keras_slices_data/keras_slices_validate"
#test_dir = "/Users/bairuan/Documents/uqsem8/comp3710/report/cloned/PatternAnalysis-2024/recognition/VQVAE_Bairu/HipMRI_study_keras_slices_data/keras_slices_test"

train_dir = "/home/groups/comp3710/HipMRI_Study_open/keras_slices_data/keras_slices_train"
val_dir = "/home/groups/comp3710/HipMRI_Study_open/keras_slices_data/keras_slices_validate"
test_dir = "/home/groups/comp3710/HipMRI_Study_open/keras_slices_data/keras_slices_test"


def loss_fn(reconstructed, original, quantization_loss):
    """Calculate total loss."""
    recon_loss = F.mse_loss(reconstructed, original)
    total_loss = recon_loss + quantization_loss
    return total_loss

if __name__ == '__main__':
    # Create data loaders
    train_loader, val_loader, test_loader = get_dataloaders(train_dir, val_dir, test_dir, batch_size=batch_size)

    # Initialize the model
    model = VQVAE(1, hidden_channels, num_embeddings, embedding_dim, num_res_layers).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(num_epochs):
        model.train()  # Set the model to training mode
        train_loss = 0
        for batch in tqdm(train_loader):
            optimizer.zero_grad()
            batch = batch.to(device)

            # Forward pass
            reconstructed, quantization_loss = model(batch)

            # Calculate loss
            loss = loss_fn(reconstructed, batch, quantization_loss)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Average Training Loss: {avg_train_loss:.4f}")

        # Validation loop
        model.eval()  # Set the model to evaluation mode
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                reconstructed, quantization_loss = model(batch)
                loss = loss_fn(reconstructed, batch, quantization_loss)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Average Validation Loss: {avg_val_loss:.4f}")

        # Save the model at the end of training
        if (epoch + 1) % 10 == 0:  # Save every 10 epochs
            torch.save(model.state_dict(), f'vqvae_model_epoch_{epoch + 1}.pth')
            print(f"Model saved at epoch {epoch + 1}.")

    # Final model save
    torch.save(model.state_dict(), 'vqvae_final_model.pth')
    print("Final model saved.")
