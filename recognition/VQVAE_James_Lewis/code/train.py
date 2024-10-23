import torch
import os
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from modules import Encoder, Decoder, VectorQuantizer, VQVAE
from dataset import load_data_2D, DataLoader
import torchmetrics.image
import torch.optim as optim

import torchmetrics


def train_vqvae(model, train_images, val_images, test_images, num_epochs, learning_rate, device, batch_size,
                lambda_ssim=0.1, max_grad_norm=1.0):
    """
    Train the VQ-VAE model.

    @param model: VQVAE, the VQ-VAE model to train
    @param train_images: np.ndarray, the training images
    @param val_images: np.ndarray, the validation images
    @param test_images: np.ndarray, the test images
    @param num_epochs: int, the number of epochs to train
    @param learning_rate: float, the learning rate for the optimizer
    @param device: torch.device, the device to train on (CPU or GPU)
    @param batch_size: int, the size of each batch for training
    @param lambda_ssim: float, the weight for SSIM loss
    @param max_grad_norm: float, the maximum gradient norm for gradient clipping
    """

    model.to(device)

    # Define the optimizer and learning rate scheduler
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    # Initialize SSIM metric
    ssim_metric = torchmetrics.StructuralSimilarityIndexMeasure(data_range=1.0).to(device)

    # Create DataLoaders
    train_loader = DataLoader(train_images, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_images, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_images, batch_size=batch_size, shuffle=False)

    # Training loop
    for epoch in range(num_epochs):
        model.train()  # Set the model to training mode
        total_loss = 0  # Track total loss for the epoch
        total_ssim = 0  # Track total SSIM for the epoch

        num_batches = len(train_images) // batch_size
        for batch_idx in range(num_batches):
            # Get the batch of images
            start_idx = batch_idx * batch_size
            end_idx = start_idx + batch_size
            data = torch.tensor(train_images[start_idx:end_idx]).float()  # Convert to a tensor
            data = data.permute(0, 2, 1)  # Move to [batch_size, height, width], if needed

            # If you have grayscale images and want to expand to RGB:
            data = data.unsqueeze(1).repeat(1, 3, 1, 1)
            data = data.to(device)  # Convert to float and move to device



            optimizer.zero_grad()  # Zero the gradients

            # Forward pass through the model
            reconstructed_data, commitment_loss = model(data)

            # Calculate reconstruction loss (Mean Squared Error)
            reconstruction_loss = F.mse_loss(reconstructed_data, data)

            # Compute SSIM between the reconstructed images and original images
            ssim_score = ssim_metric(reconstructed_data, data)
            ssim_loss = 1 - ssim_score  # Convert SSIM score to a loss (1 - SSIM)

            # Total loss: MSE + VQ commitment loss + weighted SSIM loss
            total_loss_value = reconstruction_loss + commitment_loss + lambda_ssim * ssim_loss
            total_loss_value.backward()  # Backpropagate the loss

            # Gradient Clipping to stabilize training
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

            # Update the weights
            optimizer.step()

            # Accumulate total SSIM and total loss for reporting
            total_loss += total_loss_value.item()
            total_ssim += ssim_score.item()

            if batch_idx % 100 == 0:  # Print every 100 batches
                print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{batch_idx}/{len(train_loader)}], '
                      f'Loss: {total_loss_value.item():.4f}, '
                      f'Reconstruction Loss: {reconstruction_loss.item():.4f}, '
                      f'Commitment Loss: {commitment_loss.item():.4f}, '
                      f'SSIM: {ssim_score.item():.4f}')

        # Step the learning rate scheduler
        scheduler.step()

        # Average loss and SSIM for the epoch
        avg_loss = total_loss / len(train_loader)
        avg_ssim = total_ssim / len(train_loader)
        print(f'Epoch [{epoch + 1}/{num_epochs}], Average Loss: {avg_loss:.4f}, Average SSIM: {avg_ssim:.4f}')

        # Validate the model on the validation set
        validate(model, val_loader, device, ssim_metric)

def validate(model, val_loader, device, ssim_metric):
    """
    Validate the model on validation data.

    @param model: VQVAE, the model being validated
    @param val_loader: DataLoader, validation data
    @param device: torch.device, the device for computations
    @param ssim_metric: torchmetrics.Metric, SSIM metric instance
    """
    model.eval()
    # Set the model to evaluation mode
    total_val_ssim = 0  # Track total SSIM for validation
    with torch.no_grad():
        for data in val_loader:

            data = data[0].permute(0, 2, 1).unsqueeze(1).repeat(1, 3, 1, 1)  # Prepare data
            data = data.to(device)
            reconstructed_data, _ = model(data)
            ssim_score = ssim_metric(reconstructed_data, data)
            total_val_ssim += ssim_score.item()

    avg_val_ssim = total_val_ssim / len(val_loader)
    print(f"Validation SSIM: {avg_val_ssim:.4f}")

if __name__ == "__main__":
    # Hyperparameters
    input_dim = 3
    dim = 256
    n_res_block = 2
    n_res_channel = 64
    stride = 2
    n_embed = 512
    embedding_dims = 64
    commitment_cost = 0.25
    num_epochs = 50
    learning_rate = 1e-3
    batch_size = 32

    train_image_directory = '/Users/jameslewis/PatternAnalysis-2024/recognition/VQVAE_James_Lewis/data/HipMRI_study_keras_slices_data/keras_slices_train'
    val_image_directory = '/Users/jameslewis/PatternAnalysis-2024/recognition/VQVAE_James_Lewis/data/HipMRI_study_keras_slices_data/keras_slices_validate'
    test_image_directory = '/Users/jameslewis/PatternAnalysis-2024/recognition/VQVAE_James_Lewis/data/HipMRI_study_keras_slices_data/keras_slices_test'

    train_names =[os.path.join(train_image_directory, img) for img in os.listdir(train_image_directory) if img.endswith(('.nii', '.nii.gz'))]

    val_names= [os.path.join(val_image_directory, img) for img in os.listdir(val_image_directory) if img.endswith(('.nii', '.nii.gz'))]
    test_names= [os.path.join(test_image_directory, img) for img in os.listdir(test_image_directory) if img.endswith(('.nii', '.nii.gz'))]

    train_images = load_data_2D(train_names)
    val_images = load_data_2D(val_names)
    test_images = load_data_2D(test_names)



    # Create the VQVAE model
    model = VQVAE(input_dim, dim, n_res_block, n_res_channel, stride, n_embed, commitment_cost, embedding_dims)

    # Specify the device (CPU or GPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Train the model
    train_vqvae(model, train_images, val_images, test_images, num_epochs, learning_rate, device, batch_size)
