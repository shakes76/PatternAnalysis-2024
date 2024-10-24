"""
This modules is for visulization utilities.
"""

# Importing libraries
import torch
from torchvision.utils import save_image
from skimage.metrics import structural_similarity as compute_ssim
import numpy as np
import matplotlib.pyplot as plt
from train_VQVAE import *


# Load the saved model
def load_model(model, model_path):
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

# Function to generate reconstructed images from VQ-VAE model
def generate_images(model, dataloader):
    with torch.no_grad():
        for batch in dataloader:
            original_imgs = batch.to(device)  # Take one batch of images
            loss, recon_imgs, perplexity, _ = model(original_imgs)
            return original_imgs, recon_imgs

# Function to calculate SSIM scores for a batch of images
def calculate_ssim_batch(original_imgs, recon_imgs):
    ssim_scores = []
    for i in range(original_imgs.size(0)):  # Loop through the batch
        original_img = original_imgs[i].cpu().squeeze().numpy()  # Convert to NumPy [H, W]
        recon_img = recon_imgs[i].cpu().squeeze().numpy()  # Convert to NumPy [H, W]
        
        # Calculate SSIM score
        ssim_score = compute_ssim(original_img, recon_img, data_range=2.0, channel_axis=None)
        ssim_scores.append(ssim_score)
    
    return ssim_scores

# Function to save and display a batch of images
def save_and_display_images(original_imgs, recon_imgs):
    batch_size = original_imgs.size(0)
    
    for i in range(batch_size):
        # Save the images
        save_image(original_imgs[i], f'Output/original_img_{i}.png')
        save_image(recon_imgs[i], f'Output/reconstructed_img_{i}.png')

        # Display the images
        original_np = original_imgs[i].cpu().numpy().squeeze()  # [H, W]
        recon_np = recon_imgs[i].cpu().numpy().squeeze()  # [H, W]

        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        axs[0].imshow(original_np, cmap='gray')
        axs[0].set_title(f'Original Image {i}')
        axs[0].axis('off')

        axs[1].imshow(recon_np, cmap='gray')
        axs[1].set_title(f'Reconstructed Image {i}')
        axs[1].axis('off')

        # plt.show()

    plt.close()

# Main code to load model, generate images, and calculate SSIM
if __name__ == '__main__':
    model_path = './Model/Vqvae.pth'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize the model
    model = Model(HIDDEN_DIM, RESIDUAL_HIDDEN_DIM, NUM_RESIDUAL_LAYER, NUM_EMBEDDINGS, EMBEDDING_DIM, COMMITMENT_COST).to(device)
    
    # Load the saved model
    model = load_model(model, model_path)
    
    # Load your test dataloader (make sure you define your dataloader)
    dataloader = get_dataloader("HipMRI_study_keras_slices_data", batch_size = BATCH_SIZE)
    
    # Generate the reconstructed images for the batch
    original_imgs, recon_imgs = generate_images(model, dataloader)

    # Calculate SSIM scores for the entire batch
    ssim_scores = calculate_ssim_batch(original_imgs, recon_imgs)

    for i, ssim_score in enumerate(ssim_scores):
        print(f'SSIM Score for image {i}: {ssim_score}')

    # Save and display the batch of original and reconstructed images
    save_and_display_images(original_imgs, recon_imgs)
