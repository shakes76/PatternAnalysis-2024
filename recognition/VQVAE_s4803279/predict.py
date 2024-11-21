"""
Author: Farhaan Rashid

Student Number: s4803279

This file shows example usage of the trained model.

The trained VQVAE model here can generate new images.
"""
import numpy as np
import torch
import os
from torchvision.utils import save_image
from modules import VQVAE2
from dataset import create_nifti_data_loaders
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm


def load_model(model_path, in_channels, hidden_dims, num_embeddings, embedding_dims, commitment_cost, device):
    """
    Loads a pre-trained VQVAE2 model from a checkpoint file.

    Args:
        model_path (str): Path to the saved model checkpoint.
        in_channels (int): Number of input channels (e.g., 1 for grayscale images).
        hidden_dims (list): List of hidden dimensions for the model layers.
        num_embeddings (list): List of codebook sizes for each level of VQVAE2.
        embedding_dims (list): List of embedding dimensions for each level of VQVAE2.
        commitment_cost (float): Weight for the commitment loss in VQVAE2.
        device (torch.device): Device to run the model on (CPU or GPU).

    Returns:
        torch.nn.Module: The loaded VQVAE2 model.
    """
    model = VQVAE2(in_channels, hidden_dims, num_embeddings, embedding_dims, commitment_cost)
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()
    return model


def calculate_ssim(original, reconstructed):
    """
    Calculate SSIM between original and reconstructed images.
    
    Args:
        original: Original image (torch tensor or numpy array).
        reconstructed: Reconstructed image (torch tensor or numpy array).
    
    Returns:
        SSIM score (float).
    """
    original = original.squeeze().cpu().numpy()
    reconstructed = reconstructed.squeeze().cpu().numpy()

    # Ensure the arrays are in shape [H, W] for SSIM calculation (remove channel dimension if necessary)
    if original.ndim == 3:
        original = original[0]
    if reconstructed.ndim == 3:
        reconstructed = reconstructed[0]

    # Calculate SSIM
    ssim_score = ssim(original, reconstructed, data_range = original.max() - original.min())
    return ssim_score


def test_vqvae(data_loader, model, device, output_dir, num_samples = 5):
    """
    Tests the VQVAE2 model on test data, generates reconstructed images, and saves them.

    Args:
        data_loader (torch.utils.data.DataLoader): DataLoader providing the test data.
        model (torch.nn.Module): The trained VQVAE2 model to test.
        device (torch.device): Device to run the model on (CPU or GPU).
        output_dir (str): Directory to save the reconstructed and original images.
        num_samples (int): Number of batches of images to generate and save. Default is 10.
    """
    model.eval()
    ssim_scores = []

    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Testing loop
    with torch.no_grad():  # Disable gradient calculations for testing
        for i, images in enumerate(tqdm(data_loader)):
            # Move the input images to the GPU if available
            images = images.to(device)

            # Encode and decode using the VQVAE model
            _, reconstructed_images = model(images)

            # Save a batch of reconstructed images
            save_image(reconstructed_images, os.path.join(output_dir, f'reconstruction_{i}.png'))

            # Save original images for comparison
            save_image(images, os.path.join(output_dir, f'original_{i}.png'))

            # Calculate SSIM for each image in the batch
            for orig_img, recon_img in zip(images, reconstructed_images):
                ssim_score = calculate_ssim(orig_img, recon_img)
                ssim_scores.append(ssim_score)

            # Stop after generating 'num_samples' batches
            if i >= num_samples - 1:
                break

    print(f"Generated {num_samples} samples and saved to {output_dir}.")

    # Print and return the average SSIM score
    avg_ssim = np.mean(ssim_scores)
    print(f"Average SSIM: {avg_ssim:.4f}")
    return avg_ssim


def main_test(
        model_path,
        test_data_dir,
        output_dir,
        batch_size = 16,
        num_samples = 5,
        hidden_dims = [64, 128],
        num_embeddings = [256, 256],
        embedding_dims = [32, 64],
        commitment_cost = 0.25,
        num_workers = 4):
    """
    Main function to test a pre-trained VQVAE2 model on test data and save the results.

    Args:
        model_path (str): Path to the pre-trained VQVAE2 model checkpoint.
        test_data_dir (str): Directory containing the test dataset.
        output_dir (str): Directory to save the test results (reconstructed images).
        batch_size (int): Batch size for data loading. Default is 16.
        num_samples (int): Number of batches of samples to generate and save. Default is 10.
        hidden_dims (list): List of hidden dimensions for the model layers.
        num_embeddings (list): List of codebook sizes for each level of VQVAE2.
        embedding_dims (list): List of embedding dimensions for each level of VQVAE2.
        commitment_cost (float): Weight for the commitment loss in VQVAE2.
        num_workers (int): Number of worker threads for data loading. Default is 4.
    """
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device being used for testing:", device, "\n")

    # Data loader for test data
    test_loader = create_nifti_data_loaders(test_data_dir, batch_size, num_workers)

    # Load the trained VQVAE model
    in_channels = 1
    model = load_model(model_path, in_channels, hidden_dims, num_embeddings, embedding_dims, commitment_cost, device)

    # Test the VQVAE model
    print("Generating images using the VQVAE model")
    test_vqvae(test_loader, model, device, output_dir, num_samples)
