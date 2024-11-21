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
import random

def load_model(model, model_path):
    """
    Loads the trained model from a specified directory and checkpoint

    Args:
        model (nn.Module): The model to load the weights into
        model_path (str): The specified path to the saved trained model

    Returns: 
        nn.Module: The saved model with the loaded weights 
    """
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu')))
    model.eval()  # Sets the model to evaluation mode
    return model

def calculate_ssim(original_images, reconstructed_images):
    """
    Calculates SSIM score between original and reconstructed images

    Args:
        original_images (torch.Tensor): Original images
        reconstructed_images (torch.Tensor): Reconstructed images

    Returns:
        list: A list of SSIM scores for each image in the batch
    """
    # Moves original and reconstructed images to CPU 
    original_images = original_images.cpu().data
    reconstructed_images = reconstructed_images.cpu().data

    ssim_scores = []
    for i in range(original_images.size(0)):
        # Converts tensor images to numpy arrays to calculate its SSIM score
        original_np = original_images[i][0].numpy()
        reconstructed_np = reconstructed_images[i][0].numpy()

        # Calculates SSIM for each image
        ssim_score = ssim(original_np, reconstructed_np, data_range=reconstructed_np.max() - reconstructed_np.min())
        ssim_scores.append(ssim_score)

    return ssim_scores

def plot_ssim_scores(ssim_scores):
    """
    Plots the SSIM scores for every single reconstructed image for the test dataset

    Args:
        ssim_scores (list): A list of SSIM scores for the reconstructed test images
    """
    plt.figure()

    # Creates a scatter plot to display the SSIM scores
    plt.scatter(range(len(ssim_scores)), ssim_scores, label="SSIM")  

    plt.xlabel("Test image")
    plt.ylabel("SSIM")
    plt.title("SSIM scores for test images")
    plt.legend()
    plt.savefig("ssim_scores_test_set.png")  # Saves the plot 
    plt.show()

def save_reconstructed_images(original_images, reconstructed_images, save_dir='./results', img_name="reconstructed_images.png"):
    """
    Saves a random sample of 8 original images on the top row and reconstructed images on the bottom row for visual comparisons

    Args:
        original_images (torch.Tensor): Random sample of original image
        reconstructed_images (torch.Tensor): The reconstructions of the original images
        save_dir (str): Directory where the image will be saved
        img_name (str): The name of the saved image
    """
    # Creates the directory if it doesn't exist 
    os.makedirs(save_dir, exist_ok=True)

    # Denormalizes the images from [-1, 1] to [0, 1] for visualisation purposes
    original_images = (original_images.cpu() + 1) / 2
    reconstructed_images = (reconstructed_images.cpu() + 1) / 2

    # Randomly selects 8 different images from the batch
    num_images = original_images.size(0)
    indices = random.sample(range(num_images), min(8, num_images))
    selected_original_images = original_images[indices]
    selected_reconstructed_images = reconstructed_images[indices]

    # Concatenates the selected original and reconstructed images along the batch dimension
    concatenated_images = torch.cat((selected_original_images, selected_reconstructed_images), 0)

    # Creates and saves the grid of original and reconstructed images 
    grid = make_grid(concatenated_images, nrow=len(indices))
    save_image(grid, os.path.join(save_dir, img_name))

def visualise_best_image(original_image, reconstructed_image, save_dir='./results', img_name="best_image.png"):
    """
    Saves the best reconstructed test image, along with its corresponding input image, based on which reconstruction 
    had the highest SSIM score

    Args: 
        original_image (torch.Tensor): The original image with the highest SSIM score from the reconstruction
        reconstructed_image (torch.Tensor): The reconstructed image with the highest SSIM score
        save_dir (str): Directory where the image will be saved
        img_name (str): The name of the saved image
    """
    # Creates the directory for the image to be saved in 
    os.makedirs(save_dir, exist_ok=True)

    # Moves images to CPU
    original_image = original_image.cpu().data
    reconstructed_image = reconstructed_image.cpu().data

    # Denormalizes the images from [-1, 1] to [0, 1] for visualisation purposes
    original_image = (original_image + 1) / 2
    reconstructed_image = (reconstructed_image + 1) / 2

    # Plots original and reconstructed images side by side
    fig, axes = plt.subplots(1, 2, figsize=(6, 4))

    # Original image is plotted on the left side
    axes[0].imshow(original_image[0], cmap='gray')
    axes[0].set_title("Original Image")
    axes[0].axis('off')

    # Reconstructed image is plotted on the right side
    axes[1].imshow(reconstructed_image[0], cmap='gray')
    axes[1].set_title("Reconstructed Best Image")
    axes[1].axis('off')

    plt.savefig(os.path.join(save_dir, img_name))
    plt.close()

def visualise_worst_image(original_image, reconstructed_image, save_dir='./results', img_name="worst_image.png"):
    """
    Saves the worst reconstructed test image, along with its corresponding input image, based on which reconstruction 
    had the lowest SSIM score

    Args: 
        original_image (torch.Tensor): The original image with the highest SSIM score from the reconstruction
        reconstructed_image (torch.Tensor): The reconstructed image with the highest SSIM score
        save_dir (str): Directory where the image will be saved
        img_name (str): The name of the saved image
    """
    # Creates the directory for the image to be saved in 
    os.makedirs(save_dir, exist_ok=True)

    # Moves images to CPU
    original_image = original_image.cpu().data
    reconstructed_image = reconstructed_image.cpu().data

    # Denormalizes the images from [-1, 1] to [0, 1] for visualisation purposes
    original_image = (original_image + 1) / 2
    reconstructed_image = (reconstructed_image + 1) / 2

    # Plots original and reconstructed images side by side
    fig, axes = plt.subplots(1, 2, figsize=(6, 4))

    # Original image is plotted on the left side 
    axes[0].imshow(original_image[0], cmap='gray')
    axes[0].set_title("Original Image")
    axes[0].axis('off')

    # Reconstructed image is plotted on the right side 
    axes[1].imshow(reconstructed_image[0], cmap='gray')
    axes[1].set_title("Reconstructed Worst Image")
    axes[1].axis('off')

    plt.savefig(os.path.join(save_dir, img_name))
    plt.close()

def main():
    """
    Main function that loads the test dataset and passes it through to the best trained VQ-VAE  
    model. It also calculates the SSIM scores and saves the best and worst reconstructed test images.
    """
    # Loads test dataset
    batch_size = 32
    test_data_path = '/home/groups/comp3710/HipMRI_Study_open/keras_slices_data/keras_slices_test'

    # Defines data transformations (same transformations used during training)
    transform = transforms.Compose([
        transforms.Resize((296, 144)),
        transforms.Grayscale(num_output_channels=1),  
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)) 
    ])

    # Loads the test dataset and DataLoader
    test_dataset = ProstateMRIDataset(img_dir=test_data_path, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=1)

    # Loads the trained VQ-VAE model
    model = VQVAE(
        in_channels=1,  
        num_hiddens=128,  
        num_downsampling_layers=3,
        num_residual_layers=2,
        num_residual_hiddens=32,
        embedding_dim=128,
        num_embeddings=512,
        beta=0.25,  
        decay=0.99,
        epsilon=1e-5
    ).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

    # Loads the saved model at the specific directory 
    model_save_path = './saved_models/best_vqvae_model.pth'
    model = load_model(model, model_save_path)

    # Initializes lists to track SSIM scores and image with the best and worst SSIM scores
    all_ssim_scores = []
    best_ssim = -1
    worst_ssim = 1
    best_original_image = None
    best_reconstructed_image = None
    worst_original_image = None
    worst_reconstructed_image = None

    # Loops through the test dataset
    for test_images in test_loader:
        test_images = test_images.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

        # Passes the test images through the saved model to get the reconstructed images
        with torch.no_grad():
            outputs = model(test_images)
            reconstructed_images = outputs['x_recon']

        # Calculates SSIM for this batch of images
        batch_ssim_scores = calculate_ssim(test_images, reconstructed_images)
        all_ssim_scores.extend(batch_ssim_scores)

        # Tracks the best and worst SSIM with their corresponding reconstructions 
        batch_max_ssim = max(batch_ssim_scores)
        batch_min_ssim = min(batch_ssim_scores)

        # Continually checks if the best SSIM score has been seen
        if batch_max_ssim > best_ssim:
            # Updates the best SSIM accordingly
            best_ssim = batch_max_ssim
            best_idx = batch_ssim_scores.index(batch_max_ssim) # Retrieves the index of the best SSIM score 
            best_original_image = test_images[best_idx]
            best_reconstructed_image = reconstructed_images[best_idx]

        # Continually checks if the worst SSIM score has been seen
        if batch_min_ssim < worst_ssim:
            # Updates the worst SSIM accordingly 
            worst_ssim = batch_min_ssim
            worst_idx = batch_ssim_scores.index(batch_min_ssim)
            worst_original_image = test_images[worst_idx]
            worst_reconstructed_image = reconstructed_images[worst_idx] # Retrieves the index of the worst SSIM score

        # Saves a random sample of 8 original and reconstructed images for visualization
        save_reconstructed_images(test_images, reconstructed_images, save_dir='./results', img_name=f'reconstructed_images.png')

    # Plots the SSIM scores for every reconstructed test image
    plot_ssim_scores(all_ssim_scores)

    # Prints the best and lowest SSIM scores
    print(f"Highest SSIM score: {best_ssim:.4f}")
    print(f"Lowest SSIM score: {worst_ssim:.4f}")

    # Prints the average SSIM score
    average_ssim = sum(all_ssim_scores) / len(all_ssim_scores)
    print(f"Average SSIM score: {average_ssim:.4f}")

    # Visualises the best and worst original and reconstructed image 
    visualise_best_image(best_original_image, best_reconstructed_image)
    visualise_worst_image(worst_original_image, worst_reconstructed_image)

if __name__ == "__main__":
    main()
