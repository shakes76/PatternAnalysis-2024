import torch
import os
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torchvision.utils import make_grid, save_image # type: ignore
from modules import VQVAE  # Import trained VQ-VAE model
from dataset import ProstateMRIDataset  # Import the custom dataset
from skimage.metrics import structural_similarity as ssim
from torchvision import transforms # type: ignore
import numpy as np

# Load the trained model
def load_model(model, model_path):
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu')))
    model.eval()
    return model

# Function to calculate SSIM score between original and reconstructed images
def calculate_ssim(original_images, reconstructed_images):
    original_images = original_images.cpu().data
    reconstructed_images = reconstructed_images.cpu().data

    ssim_scores = []
    for i in range(original_images.size(0)):
        original_np = original_images[i][0].numpy()
        reconstructed_np = reconstructed_images[i][0].numpy()
        ssim_score = ssim(original_np, reconstructed_np, data_range=reconstructed_np.max() - reconstructed_np.min())
        ssim_scores.append(ssim_score)

    return ssim_scores

# Function to plot SSIM scores for each image
def plot_ssim_scores(ssim_scores):
    plt.figure()
    plt.scatter(range(len(ssim_scores)), ssim_scores, label="SSIM")
    plt.xlabel("Test image")
    plt.ylabel("SSIM")
    plt.title("SSIM scores for test images")
    plt.legend()
    plt.savefig("ssim_scores_test_set.png")
    plt.show()

# Function to visualize and save the best image (highest SSIM)
def visualise_best_image(original_image, reconstructed_image, save_dir='./results', img_name="best_image.png"):
    os.makedirs(save_dir, exist_ok=True)

    original_image = original_image.cpu().data
    reconstructed_image = reconstructed_image.cpu().data

    # Denormalize the images from [-1, 1] to [0, 1]
    original_image = (original_image + 1) / 2
    reconstructed_image = (reconstructed_image + 1) / 2

    # Plot original and reconstructed images side by side
    fig, axes = plt.subplots(1, 2, figsize=(6, 4))

    # Original image
    axes[0].imshow(original_image[0], cmap='gray')
    axes[0].set_title("Original Image")
    axes[0].axis('off')

    # Reconstructed image
    axes[1].imshow(reconstructed_image[0], cmap='gray')
    axes[1].set_title("Reconstructed Best Image")
    axes[1].axis('off')

    # Save the plot
    plt.savefig(os.path.join(save_dir, img_name))
    plt.close()

# Function to visualize and save the worst image (lowest SSIM)
def visualise_worst_image(original_image, reconstructed_image, save_dir='./results', img_name="worst_image.png"):
    os.makedirs(save_dir, exist_ok=True)

    original_image = original_image.cpu().data
    reconstructed_image = reconstructed_image.cpu().data

    # Denormalize the images from [-1, 1] to [0, 1]
    original_image = (original_image + 1) / 2
    reconstructed_image = (reconstructed_image + 1) / 2

    # Plot original and reconstructed images side by side
    fig, axes = plt.subplots(1, 2, figsize=(6, 4))

    # Original image
    axes[0].imshow(original_image[0], cmap='gray')
    axes[0].set_title("Original Image")
    axes[0].axis('off')

    # Reconstructed image
    axes[1].imshow(reconstructed_image[0], cmap='gray')
    axes[1].set_title("Reconstructed Worst Image")
    axes[1].axis('off')

    # Save the plot
    plt.savefig(os.path.join(save_dir, img_name))
    plt.close()

def main():
    # Load test dataset
    image_size = 256
    batch_size = 32
    test_data_path = '/home/groups/comp3710/HipMRI_Study_open/keras_slices_data/keras_slices_test'

    # Define data transformations (same as training)
    transform = transforms.Compose([
        transforms.Resize((296, 144)),
        transforms.Grayscale(num_output_channels=1),  # Grayscale images
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # Normalize to [-1, 1]
    ])

    # Load test dataset and DataLoader
    test_dataset = ProstateMRIDataset(img_dir=test_data_path, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=1)

    # Load the trained VQ-VAE model
    model = VQVAE(
        in_channels=1,  # Grayscale images
        num_hiddens=128,  # Number of feature maps/channels
        num_downsampling_layers=3,
        num_residual_layers=2,
        num_residual_hiddens=32,
        embedding_dim=128,
        num_embeddings=512,
        beta=0.25,  # Beta value used in training
        decay=0.99,
        epsilon=1e-5
    ).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

    # Load the saved model 
    model_save_path = './saved_models/best_vqvae_model.pth'
    model = load_model(model, model_save_path)

    # Initialize lists to track SSIM scores and image indices
    all_ssim_scores = []
    best_ssim = -1
    worst_ssim = 1
    best_original_image = None
    best_reconstructed_image = None
    worst_original_image = None
    worst_reconstructed_image = None

    # Loop through the test dataset
    for test_images in test_loader:
        test_images = test_images.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

        # Pass the test images through the model to get the reconstructed images
        with torch.no_grad():
            outputs = model(test_images)
            reconstructed_images = outputs['x_recon']

        # Calculate SSIM for this batch of images
        batch_ssim_scores = calculate_ssim(test_images, reconstructed_images)
        all_ssim_scores.extend(batch_ssim_scores)

        # Track the best and worst SSIM with their corresponding images
        batch_max_ssim = max(batch_ssim_scores)
        batch_min_ssim = min(batch_ssim_scores)

        if batch_max_ssim > best_ssim:
            best_ssim = batch_max_ssim
            best_idx = batch_ssim_scores.index(batch_max_ssim)
            best_original_image = test_images[best_idx]
            best_reconstructed_image = reconstructed_images[best_idx]

        if batch_min_ssim < worst_ssim:
            worst_ssim = batch_min_ssim
            worst_idx = batch_ssim_scores.index(batch_min_ssim)
            worst_original_image = test_images[worst_idx]
            worst_reconstructed_image = reconstructed_images[worst_idx]
        
    # Plot SSIM scores for each test image
    plot_ssim_scores(all_ssim_scores)

    # Print the best and lowest SSIM scores
    print(f"Highest SSIM score: {best_ssim:.4f}")
    print(f"Lowest SSIM score: {worst_ssim:.4f}")

    # Print the average SSIM score
    average_ssim = sum(all_ssim_scores) / len(all_ssim_scores)
    print(f"Average SSIM score: {average_ssim:.4f}")

    # Visualize the best original and reconstructed image
    visualise_best_image(best_original_image, best_reconstructed_image)

    # Visualize the best original and reconstructed image
    visualise_worst_image(worst_original_image, worst_reconstructed_image)

if __name__ == "__main__":
    main()

