import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import os
from modules import VQVAE
from dataset import get_dataloader
from skimage.metrics import structural_similarity as ssim
import numpy as np

# Calculate SSIM score between two images
def calculate_ssim(img1, img2):
    """
    Calculate the Structural Similarity Index Measure (SSIM) between two images.

    Args:
        img1 (torch.Tensor): The first batch of images as a tensor of shape (B, C, H, W).
        img2 (torch.Tensor): The second batch of images as a tensor of shape (B, C, H, W).

    Returns:
        float: The average SSIM score across the batch.
    """
    # Convert tensors to numpy arrays with channels last
    img1 = img1.detach().permute(0, 2, 3, 1).cpu().numpy()
    img2 = img2.detach().permute(0, 2, 3, 1).cpu().numpy()
    # Calculate SSIM for each image in the batch
    ssim_vals = [
        ssim(img1[i], img2[i], data_range=1.0, channel_axis=-1, win_size=11)
        for i in range(img1.shape[0])
    ]
    # Return average SSIM across batch
    return np.mean(ssim_vals)

# Train function for the VQ-VAE model
def train_vqvae(num_epochs=38, batch_size=32, lr=1e-4, device='cpu'):
    """
    Train a Vector Quantized Variational Autoencoder (VQ-VAE) model.

    Parameters:
        num_epochs (int): The number of training epochs.
        batch_size (int): The size of each batch for training and validation.
        lr (float): The learning rate for the optimizer.
        device (str): The device to run the model on ('cpu' or 'cuda').

    Returns:
        None. The function trains the VQ-VAE model and may save training metrics or model weights.
    """
    # Model hyperparameters
    input_dim = 1
    hidden_dim = 128
    res_channels = 32
    nb_res_layers = 2
    nb_levels = 3
    embed_dim = 128  
    nb_entries = 1024  
    scaling_rates = [8, 4, 2]

    # Initialize VQ-VAE model with specified parameters
    vqvae = VQVAE(in_channels=input_dim, hidden_channels=hidden_dim, res_channels=res_channels, 
                  nb_res_layers=nb_res_layers, nb_levels=nb_levels, embed_dim=embed_dim, 
                  nb_entries=nb_entries, scaling_rates=scaling_rates)
    vqvae.to(device)  # Move model to specified device

    # Define optimizer and loss function
    optimizer = optim.Adam(vqvae.parameters(), lr=lr)
    mse_loss_fn = nn.MSELoss()  # Mean Squared Error for reconstruction

    # Define directories for training and validation images
    current_dir = os.path.dirname(__file__)
    train_image_dir = os.path.join(current_dir, "keras_slices", "keras_slices", "keras_slices_train")
    val_image_dir = os.path.join(current_dir, "keras_slices", "keras_slices", "keras_slices_seg_validate")

    # Load data for training and validation
    train_loader = get_dataloader(train_image_dir, batch_size=batch_size)
    val_loader = get_dataloader(val_image_dir, batch_size=batch_size)

    # Lists to store loss and SSIM metrics for plotting
    train_loss_list = []
    val_loss_list = []
    train_ssim_list = []
    val_ssim_list = []

    # Training loop
    for epoch in range(num_epochs):
        vqvae.train()  # Set model to training mode
        epoch_loss = 0
        epoch_ssim = 0

        for (batch, _) in train_loader:
            batch = batch.to(device)  # Move batch to device
            optimizer.zero_grad()  # Zero gradients for each batch

            # Forward pass
            reconstructed, diffs = vqvae(batch)
            
            # Compute combined loss: MSE reconstruction loss + quantization differences
            mse_loss = mse_loss_fn(reconstructed, batch)
            total_loss = mse_loss + sum(diffs)

            # Backward pass and optimization
            total_loss.backward()
            optimizer.step()
            epoch_loss += total_loss.item()
            
            # Calculate SSIM for reconstructed images
            batch_ssim = calculate_ssim(batch, reconstructed)
            epoch_ssim += batch_ssim

        # Average losses and SSIM for the epoch
        avg_train_loss = epoch_loss / len(train_loader)
        avg_train_ssim = epoch_ssim / len(train_loader)
        train_loss_list.append(avg_train_loss)
        train_ssim_list.append(avg_train_ssim)

        # Validation phase
        vqvae.eval()  # Set model to evaluation mode
        epoch_val_loss = 0
        epoch_val_ssim = 0
        
        with torch.no_grad():  # No gradients needed for validation
            for (val_batch, _) in val_loader:
                val_batch = val_batch.to(device)  # Move validation batch to device
                reconstructed_val, diffs_val = vqvae(val_batch)
                
                # Compute validation loss
                val_mse_loss = mse_loss_fn(reconstructed_val, val_batch)
                val_loss = val_mse_loss + sum(diffs_val)

                epoch_val_loss += val_loss.item()
                # Calculate SSIM for validation images
                val_batch_ssim = calculate_ssim(val_batch, reconstructed_val)
                epoch_val_ssim += val_batch_ssim

        # Average validation loss and SSIM for the epoch
        avg_val_loss = epoch_val_loss / len(val_loader)
        avg_val_ssim = epoch_val_ssim / len(val_loader)
        val_loss_list.append(avg_val_loss)
        val_ssim_list.append(avg_val_ssim)

        # Print metrics for the current epoch
        print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}")
        print(f"Epoch {epoch + 1}/{num_epochs}, Train SSIM: {avg_train_ssim:.4f}, Validation SSIM: {avg_val_ssim:.4f}")

    # Save model weights
    torch.save(vqvae.state_dict(), 'vqvae_model.pth')
    print("Model weights saved as 'vqvae_model.pth'.")

    # Plot training and validation metrics
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, num_epochs + 1), train_loss_list, label='Train Loss')
    plt.plot(range(1, num_epochs + 1), val_loss_list, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(range(1, num_epochs + 1), train_ssim_list, label='Train SSIM')
    plt.plot(range(1, num_epochs + 1), val_ssim_list, label='Validation SSIM')
    plt.xlabel('Epoch')
    plt.ylabel('SSIM')
    plt.title('Training and Validation SSIM')
    plt.legend()

    plt.savefig("training_validation_metrics.png")  # Save metrics plot

# Main execution block
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # Use GPU if available
    train_vqvae(device=device)  # Start training
