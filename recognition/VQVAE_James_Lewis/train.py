import torch
import os
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from modules import Encoder, Decoder, VectorQuantizer, VQVAE
from dataset import load_data_2D
import torchmetrics.image
import torch.optim as optim

import torchmetrics

def train_vqvae(model, images, num_epochs, learning_rate, device, batch_size):
    """
    Train the VQ-VAE model.

    @param model: VQVAE, the VQ-VAE model to train
    @param images: np.ndarray, the training images
    @param num_epochs: int, the number of epochs to train
    @param learning_rate: float, the learning rate for the optimizer
    @param device: torch.device, the device to train on (CPU or GPU)
    @param batch_size: int, the size of each batch for training
    """

    # Move model to the specified device
    model.to(device)

    # Define the optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Initialize SSIM metric
    ssim_metric = torchmetrics.StructuralSimilarityIndexMeasure(data_range=1.0).to(device)

    # Training loop
    for epoch in range(num_epochs):
        model.train()  # Set the model to training mode
        total_loss = 0  # Track total loss for the epoch
        total_ssim = 0  # Track total SSIM for the epoch

        # Create batches of images
        num_batches = len(images) // batch_size
        for batch_idx in range(num_batches):
            # Get the batch of images
            start_idx = batch_idx * batch_size
            end_idx = start_idx + batch_size
            data = torch.tensor(images[start_idx:end_idx]).float()  # Convert to a tensor
            data = data.permute(0, 2, 1)  # Move to [batch_size, height, width], if needed

            # If you have grayscale images and want to expand to RGB:
            data = data.unsqueeze(1).repeat(1, 3, 1, 1)
            data = data.to(device)  # Move data to the specified device

            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass through the model
            reconstructed_data, commitment_loss = model(data)

            # Calculate reconstruction loss (Mean Squared Error)
            reconstruction_loss = F.mse_loss(reconstructed_data, data)

            # Compute SSIM between the reconstructed images and original images
            ssim_score = ssim_metric(reconstructed_data, data)

            # Total loss is the sum of reconstruction loss and commitment loss
            total_loss = reconstruction_loss + commitment_loss
            total_loss.backward()  # Backpropagate the loss

            # Update the weights
            optimizer.step()

            # Accumulate total SSIM
            total_ssim += ssim_score.item()

            if batch_idx % 100 == 0:  # Print every 100 batches
                print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{batch_idx}/{num_batches}], '
                      f'Loss: {total_loss.item():.4f}, '
                      f'Reconstruction Loss: {reconstruction_loss.item():.4f}, '
                      f'Commitment Loss: {commitment_loss.item():.4f}, '
                      f'SSIM: {ssim_score.item():.4f}')

        avg_ssim = total_ssim / num_batches
        print(f'Epoch [{epoch + 1}/{num_epochs}], Average Loss: {total_loss.item() / num_batches:.4f}, Average SSIM: {avg_ssim:.4f}')




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
    num_epochs = 25
    learning_rate = 1e-3
    batch_size = 32

    # Load the data
    image_directory = '/Users/jameslewis/PatternAnalysis-2024/recognition/VQVAE_James_Lewis/data/HipMRI_study_keras_slices_data/keras_slices_seg_train'

    imageNames = [os.path.join(image_directory, img) for img in os.listdir(image_directory) if
                  img.endswith(('.nii', '.nii.gz'))]

    if not imageNames:
        print("No .nii or .nii.gz files found in the specified directory.")
    else:
        # Proceed with loading the images
        images, affines = load_data_2D(
            imageNames=imageNames,
            normImage=True,
            categorical=False,
            dtype=np.float32,
            getAffines=True,
            early_stop=False
        )
    # Initialize your dataset and dataloader
    # dataset = YourCustomDataset(...)
    # dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Create the VQVAE model
    model = VQVAE(input_dim, dim, n_res_block, n_res_channel, stride, n_embed, commitment_cost, embedding_dims)

    # Specify the device (CPU or GPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Train the model
    train_vqvae(model, images, num_epochs, learning_rate, device, batch_size)
