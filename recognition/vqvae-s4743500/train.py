# train.py

import os
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms # type: ignore
import matplotlib.pyplot as plt
from modules import VQVAE # type: ignore # Import your VQVAE model
from dataset import ProstateMRIDataset  # Import the custom dataset
from skimage.metrics import structural_similarity as ssim

# Hyperparameters
image_size = 256  # Image size for resizing
batch_size = 32  # Adjust this based on available memory
num_epochs = 15  # Number of training epochs
learning_rate = 0.001  # PLAY AROUND WITH 0.001 or 0.0001 Learning rate for optimizer
beta = 0.25  # EXPERIMENT WITH 0.1, 0.2, OR 0.5 IF IMAGE IS NOT CLEAR ENOUGH
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define data transformations
transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.Grayscale(num_output_channels=1),  # Ensuring grayscale images
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # Normalize pixel values to [-1, 1]
])

# Dataset and DataLoader
train_data_path = '/home/groups/comp3710/HipMRI_Study_open/keras_slices_data/keras_slices_train' # Path to MRI training data
test_data_path = '/home/groups/comp3710/HipMRI_Study_open/keras_slices_data/keras_slices_test' # Path to MRI test data
val_data_path = '/home/groups/comp3710/HipMRI_Study_open/keras_slices_data/keras_slices_validate' # Path to MRI validation data

# Loads datasets using the custom data loader
train_dataset = ProstateMRIDataset(img_dir=train_data_path, transform=transform)
test_dataset = ProstateMRIDataset(img_dir=test_data_path, transform=transform)
val_dataset = ProstateMRIDataset(img_dir=val_data_path, transform=transform)

# Creates DataLoaders for each dataset
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=1)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=1)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=1)

# Initialize the VQ-VAE model
model = VQVAE(
    in_channels=1,  # Grayscale images
    num_hiddens=128, # number of feature maps/channels
    num_downsampling_layers=3,  # Adjustable for image size
    num_residual_layers=2,
    num_residual_hiddens=32,
    embedding_dim=128, # Set to 64, but try 128 if images are not clear enough
    num_embeddings=512,
    beta=0.25, # I ADDED THIS PARAMETER
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

# Lists to store SSIM scores and losses for plotting later
train_ssim_scores, val_ssim_scores = [], []
train_losses, val_losses = [], []
batch_losses = []

# Training loop
def train():
    # Print number of images in each dataset
    print(f'Number of training images: {len(train_dataset)}')
    print(f'Number of validation images: {len(val_dataset)}')
    print(f'Number of testing images: {len(test_dataset)}')
    print("Starting training loop")
    best_val_loss = float('inf') # Tracks the best validation loss for model saving 
    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0
        ssim_train_scores = []  # Track SSIM for training images
        for batch_idx, images in enumerate(train_loader):
            images = images.to(device)  # Move images to the appropriate device

            # Forward pass through the VQ-VAE model
            outputs = model(images)
            x_recon = outputs['x_recon']
            commitment_loss = outputs['commitment_loss']

            # Compute reconstruction loss (mean squared error)
            recon_loss = criterion(x_recon, images)

            # Total loss is the sum of reconstruction loss and commitment loss
            loss = recon_loss + beta * commitment_loss

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # ADDED THIS: Track batch-level losses for the plot
            batch_losses.append(loss.item())

            total_train_loss += loss.item()

            # Calculate SSIM between original and reconstructed images (in the same batch)
            batch_ssim = calculate_ssim(images, x_recon)
            ssim_train_scores.append(batch_ssim)

            # Print log info
            if batch_idx % eval_every == 0:
                print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{batch_idx}/{len(train_loader)}], '
                      f'Loss: {loss.item():.4f}, Recon Loss: {recon_loss.item():.4f}, '
                      f'Commitment Loss: {commitment_loss.item():.4f}')
        
        avg_train_loss = total_train_loss / len(train_loader)
        avg_ssim_train = np.mean(ssim_train_scores)

        print(f'Epoch [{epoch + 1}], Train Loss: {avg_train_loss:.4f}, Train SSIM: {avg_ssim_train:.4f}')

        # Stores the train SSIM and loss values in the list
        train_ssim_scores.append(avg_ssim_train)
        train_losses.append(avg_train_loss)

        # Visualize and save reconstructed training images
        visualize_reconstruction(images, x_recon, epoch, phase='train')

        # Switch to Validation Phase
        model.eval() # Switches the model to evaluation mode
        total_val_loss = 0
        ssim_val_scores = []  # Track SSIM for validation images
        with torch.no_grad():
            for val_images in val_loader:
                val_images = val_images.to(device)
                outputs = model(val_images)
                x_recon = outputs['x_recon']
                commitment_loss = outputs['commitment_loss']

                # Validation reconstruction loss
                val_recon_loss = criterion(x_recon, val_images)

                val_loss = val_recon_loss + beta * commitment_loss

                total_val_loss += val_loss.item()

                # Calculate SSIM between original and reconstructed validation images
                batch_ssim = calculate_ssim(val_images, x_recon)
                ssim_val_scores.append(batch_ssim)
        
        avg_val_loss = total_val_loss / len(val_loader)
        avg_ssim_val = np.mean(ssim_val_scores)

        print(f'Epoch [{epoch + 1}], Val Loss: {avg_val_loss:.4f}, Val SSIM: {avg_ssim_val:.4f}')

        # Stores the validation SSIM and loss values in the list 
        val_ssim_scores.append(avg_ssim_val)
        val_losses.append(avg_val_loss)

        # Visualize and save reconstructed validation images
        visualize_reconstruction(val_images, x_recon, epoch, phase='val')

        # Saves the best model based on validation loss
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), os.path.join(model_save_path, 'best_vqvae_model.pth'))

    # After training, plot and save the SSIM scores and losses
    plot_ssim_scores(train_ssim_scores, val_ssim_scores)
    plot_losses(train_losses, val_losses)
    plot_batch_losses(batch_losses)


# Plot SSIM scores
def plot_ssim_scores(train_ssim_scores, val_ssim_scores):
    plt.figure()
    plt.plot(train_ssim_scores, label='Train SSIM')
    plt.plot(val_ssim_scores, label='Val SSIM')
    plt.xlabel('Epoch')
    plt.ylabel('SSIM')
    plt.title('SSIM Scores per Epoch')
    # Add a red dotted line at y=0.6 to represent the benchmark
    plt.axhline(y=0.6, color='red', linestyle='--', label='Benchmark SSIM 0.6')
    plt.legend()
    plt.savefig('ssim_scores.png')
    plt.close()

# Plot losses
def plot_losses(train_losses, val_losses):
    plt.figure()
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Losses per Epoch')
    plt.legend()
    plt.savefig('losses.png')
    plt.close()

# Plot batch losses 
def plot_batch_losses(train_losses):
    plt.figure()
    plt.plot(train_losses, label='Train Loss')
    plt.xlabel('Batch no.')
    plt.ylabel('Reconstruction loss')
    plt.title('Training reconstruction Losses')
    plt.ylim(0, 0.2) # I ADDED THIS TO REDUCE THE Y-VALUE RANGE
    plt.legend()
    plt.savefig('batch_losses.png')
    plt.close()

def calculate_ssim(original_images, reconstructed_images):
    original_images = original_images.cpu().data
    reconstructed_images = reconstructed_images.cpu().data

    ssim_scores = []
    for i in range(original_images.size(0)):
        original_np = original_images[i][0].numpy()
        reconstructed_np = reconstructed_images[i][0].numpy()
        ssim_score = ssim(original_np, reconstructed_np, data_range=reconstructed_np.max() - reconstructed_np.min())
        ssim_scores.append(ssim_score)

    return np.mean(ssim_scores)

# Function to visualize and save reconstructed images
def visualize_reconstruction(original_images, reconstructed_images, epoch, phase='train'):
    original_images = original_images.cpu().data
    reconstructed_images = reconstructed_images.cpu().data

    # Denormalize images from [-1, 1] to [0, 1]
    original_images = (original_images + 1) / 2
    reconstructed_images = (reconstructed_images + 1) / 2

    # Create the directory for saving images if it doesn't exist
    recon_save_dir = f'./reconstructions/{phase}'  # Separate folders for train/val
    os.makedirs(recon_save_dir, exist_ok=True)

    # Plot original and reconstructed images side by side
    num_images = min(8, original_images.size(0))  # Show at most 8 images
    fig, axes = plt.subplots(2, num_images, figsize=(num_images * 2, 4))

    for i in range(num_images):
        # Original images
        axes[0, i].imshow(original_images[i][0], cmap='gray')  # Assuming grayscale
        axes[0, i].axis('off')
        # Reconstructed images
        axes[1, i].imshow(reconstructed_images[i][0], cmap='gray')  # Assuming grayscale
        axes[1, i].axis('off')

    # Save the plot with the phase (train/val) in the filename
    plt.savefig(f'{recon_save_dir}/epoch_{epoch + 1}_phase_{phase}.png')
    plt.close()

if __name__ == "__main__":
    train()

