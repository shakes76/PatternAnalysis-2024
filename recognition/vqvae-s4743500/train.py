# train.py

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms # type: ignore
import matplotlib.pyplot as plt
from modules import VQVAE # type: ignore # Import your VQVAE model
from dataset import ProstateMRIDataset  # Import the custom dataset

# Hyperparameters
image_size = 256  # Image size for resizing
batch_size = 32  # Adjust this based on available memory
num_epochs = 35  # Number of training epochs
learning_rate = 0.0001  # Learning rate for optimizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define data transformations
transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.Grayscale(num_output_channels=1),  # Ensuring grayscale images
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # Normalize pixel values to [-1, 1]
])

# Dataset and DataLoader
dataroot = '/home/groups/comp3710/HipMRI_Study_open/keras_slices_data/keras_slices_train'  # Path to MRI training data
dataset = ProstateMRIDataset(img_dir=dataroot, transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=1)

# Initialize the VQ-VAE model
model = VQVAE(
    in_channels=1,  # Grayscale images
    num_hiddens=128, # number of feature maps/channels
    num_downsampling_layers=3,  # Adjustable for image size
    num_residual_layers=2,
    num_residual_hiddens=32,
    embedding_dim=64, # Set to 64, but try 128 if images are not clear enough
    num_embeddings=256,
    decay=0.99,
    epsilon=1e-5
).to(device)

# Optimizer and loss function
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.MSELoss()

# Loss logging
eval_every = 100
model_save_path = './saved_models'
os.makedirs(model_save_path, exist_ok=True)

# Training loop
def train():
    print("Starting training loop")
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for batch_idx, images in enumerate(dataloader):
            images = images.to(device)  # Move images to the appropriate device

            # Forward pass through the VQ-VAE model
            outputs = model(images)
            x_recon = outputs['x_recon']
            commitment_loss = outputs['commitment_loss']

            # Compute reconstruction loss (mean squared error)
            recon_loss = criterion(x_recon, images)

            # Total loss is the sum of reconstruction loss and commitment loss
            loss = recon_loss + commitment_loss

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            # Print log info
            if batch_idx % eval_every == 0:
                print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{batch_idx}/{len(dataloader)}], '
                      f'Loss: {loss.item():.4f}, Recon Loss: {recon_loss.item():.4f}, '
                      f'Commitment Loss: {commitment_loss.item():.4f}')

        # Save the model checkpoint after every epoch
        torch.save(model.state_dict(), os.path.join(model_save_path, f'vqvae_epoch_{epoch + 1}.pth'))

# TO DO LIST: ADD FUNCTION HERE TO VISUALISE RECONSTRUCTED IMAGES