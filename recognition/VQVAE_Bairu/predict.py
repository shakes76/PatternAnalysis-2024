"""
Usage of the trained VQ-VAE model for image reconstruction.

This script loads the model, processes test images, calculates SSIM scores, 
and visualizes original vs. reconstructed images.

Author: Bairu An, s4702833.
"""


import os
import torch
from torchvision import transforms
from modules import VQVAE  # Import your VQVAE model
from dataset import MedicalImageDataset, get_dataloaders  # Import your dataset classes
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from pytorch_msssim import ssim  # Import SSIM metric for image quality assessment

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Directories for the dataset and model
test_dir = "/Users/bairuan/Documents/uqsem8/comp3710/report/cloned/PatternAnalysis-2024/recognition/VQVAE_Bairu/HipMRI_study_keras_slices_data/keras_slices_test"
model_path = "vqvae_final_model.pth"  # Path to the trained model
save_dir = "reconstructed_images"  # Directory to save reconstructed images

# Hyperparameters
batch_size = 1  # Set batch size for inference

# Pre-processing transformation
input_transf = transforms.Compose([
    transforms.Resize((256, 128)),  # Resize images to required dimensions
    transforms.Normalize((0.5,), (0.5,))  # Normalize images
])

def main():
    # Load the trained VQVAE model
    model = VQVAE(1, 64, 512, 64, 2).to(device)  # Initialize model with parameters
    model.load_state_dict(torch.load(model_path, map_location=device))  # Load model weights
    model.eval()  # Set model to evaluation mode

    # Create directory to save images if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    # Load test data
    test_loader = get_dataloaders(test_dir, test_dir, test_dir, batch_size=batch_size)[2]  # Get test data loader

    # Initialize lists for SSIM scores and images
    ssim_scores = []  # List to store SSIM scores
    original_images = []  # List to store original images
    reconstructed_images = []  # List to store reconstructed images

    # Inference and Visualization
    with torch.no_grad():  # Disable gradient calculation for inference
        for batch_idx, batch in tqdm(enumerate(test_loader), total=len(test_loader)):
            batch = batch.to(device)  # Move batch to device
            
            # Forward pass through the model
            reconstructed_data, quantization_loss = model(batch)

            # Denormalize images to original scale
            original_image = (batch.cpu().numpy().squeeze() * 0.5 + 0.5)  # Original image
            reconstructed_image = (reconstructed_data.cpu().numpy().squeeze() * 0.5 + 0.5)  # Reconstructed image

            # Calculate SSIM score for the current image
            ssim_score = ssim(batch, reconstructed_data, data_range=1.0).item()
            ssim_scores.append(ssim_score)  # Append SSIM score to the list
            original_images.append(original_image)  # Store original image
            reconstructed_images.append(reconstructed_image)  # Store reconstructed image

            # Save individual images for comparison
            plt.figure(figsize=(10, 5))
            plt.subplot(1, 2, 1)
            plt.imshow(original_image, cmap='gray')  # Display original image
            plt.title(f'Original Image {batch_idx + 1}')  # Title for original image
            plt.axis('off')  # Hide axes

            plt.subplot(1, 2, 2)
            plt.imshow(reconstructed_image, cmap='gray')  # Display reconstructed image
            plt.title(f'Reconstructed Image {batch_idx + 1}\nSSIM: {ssim_score:.4f}')  # Title for reconstructed image with SSIM
            plt.axis('off')  # Hide axes

            # Save the comparison image
            plt.savefig(os.path.join(save_dir, f'image_{batch_idx + 1}.png'))
            plt.close()  # Close the figure to avoid display in interactive environments

    # Calculate and print average SSIM for the test set
    average_ssim = np.mean(ssim_scores)
    print(f"Average SSIM on test set: {average_ssim:.4f}")

    # Identify the best and worst 4 images based on SSIM scores
    best_indices = np.argsort(ssim_scores)[-4:]  # Indices of the top 4 SSIM scores
    worst_indices = np.argsort(ssim_scores)[:4]  # Indices of the bottom 4 SSIM scores

    # Create a figure for the best reconstructed images
    plt.figure(figsize=(20, 10))
    for i, idx in enumerate(best_indices):
        plt.subplot(2, 4, i + 1)  # Original images
        plt.imshow(original_images[idx], cmap='gray')
        plt.title(f'Best Original {idx}\nSSIM: {ssim_scores[idx]:.4f}')  # Title with index and SSIM
        plt.axis('off')  # Hide axes

        plt.subplot(2, 4, i + 5)  # Reconstructed images
        plt.imshow(reconstructed_images[idx], cmap='gray')
        plt.title(f'Best Reconstructed {idx}\nSSIM: {ssim_scores[idx]:.4f}')  # Title with index and SSIM
        plt.axis('off')  # Hide axes

    plt.savefig(os.path.join(save_dir, 'best_reconstructed_images.png'))  # Save best images
    plt.close()  # Close the figure

    # Create a figure for the worst reconstructed images
    plt.figure(figsize=(20, 10))
    for i, idx in enumerate(worst_indices):
        plt.subplot(2, 4, i + 1)  # Original images
        plt.imshow(original_images[idx], cmap='gray')
        plt.title(f'Worst Original {idx}\nSSIM: {ssim_scores[idx]:.4f}')  # Title with index and SSIM
        plt.axis('off')  # Hide axes

        plt.subplot(2, 4, i + 5)  # Reconstructed images
        plt.imshow(reconstructed_images[idx], cmap='gray')
        plt.title(f'Worst Reconstructed {idx}\nSSIM: {ssim_scores[idx]:.4f}')  # Title with index and SSIM
        plt.axis('off')  # Hide axes

    plt.savefig(os.path.join(save_dir, 'worst_reconstructed_images.png'))  # Save worst images
    plt.close()  # Close the figure

if __name__ == "__main__":
    main()  # Execute the main function
