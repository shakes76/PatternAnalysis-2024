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
train_dir = "/Users/bairuan/Documents/uqsem8/comp3710/report/cloned/PatternAnalysis-2024/recognition/VQVAE_Bairu/keras_slices_train"
val_dir = "/Users/bairuan/Documents/uqsem8/comp3710/report/cloned/PatternAnalysis-2024/recognition/VQVAE_Bairu/keras_slices_validate"
test_dir = "/Users/bairuan/Documents/uqsem8/comp3710/report/cloned/PatternAnalysis-2024/recognition/VQVAE_Bairu/keras_slices_test"

# Create data loaders
train_loader, val_loader, test_loader = get_dataloaders(train_dir, val_dir, test_dir, batch_size=batch_size)

# Initialize the model
model = VQVAE(1, hidden_channels, num_embeddings, embedding_dim, num_res_layers).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

def loss_fn(reconstructed, original, quantization_loss):
    """Calculate total loss."""
    recon_loss = F.mse_loss(reconstructed, original)
    total_loss = recon_loss + quantization_loss
    return total_loss

# Training loop
for epoch in range(num_epochs):
    model.train()  # Set the model to training mode
    

# Final model save
torch.save(model.state_dict(), 'vqvae_final_model.pth')
print("Final model saved.")
