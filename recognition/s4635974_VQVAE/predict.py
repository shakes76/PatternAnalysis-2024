"""
COMP3710 VQ-VAE report
Author: David Collie
Date: October, 2024

VQ-VAE Model Evaluation and Visualization

This file contains the implementation for evaluating the VQ-VAE (Vector Quantized Variational Autoencoder) 
model on a test dataset.
It loads the pre-trained model, computes the Structural Similarity Index Measure (SSIM) for the reconstructed 
images, and visualizes the results.

The key components of this implementation are:
- **Model Loading**: Loads the VQ-VAE model from a specified path and configures it for evaluation.
- **Test Data Preparation**: Utilizes the `HipMRILoader` class to prepare the test dataset for evaluation.
- **SSIM Calculation**: Computes the SSIM between real and reconstructed images to assess the model's performance.
- **Image Visualization**: Displays real and decoded images side by side for visual comparison.
- **Loss and SSIM Graphs**: Loads previously saved training metrics and generates plots for training and validation 
reconstruction loss, VQ loss, and SSIM over epochs.
"""

import torch
from torchmetrics.functional.image \
    import structural_similarity_index_measure as ssim
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle

from dataset import HipMRILoader
import modules


# Model path for loading model
model_path = 'saved_model/VQ-VAE.pth'

# Hyperparmeters for loading model
batch_size = 16            # Batch size for the dataloader
num_hiddens = 128          # Number of hidden units in the VQ-VAE model
num_residual_hiddens = 32  # Number of residual hidden units
num_channels = 1           # Number of input channels (grayscale image)
num_embeddings = 512       # Number of embeddings in the VQ-VAE model
dim_embedding = 64         # Dimensionality of each embedding
beta = 0.25                # Beta parameter for commitment loss in VQ-VAE

# Directory for saving test images
test_save_dir = f'test_images'
os.makedirs(test_save_dir, exist_ok=True)

def predict(
        model_path=model_path, 
        test_save_dir=test_save_dir 
        ):
    """
    Loads the saved VQ-VAE model, evaluates it on the test set, calculates 
    SSIM, and visualizes real vs. decoded images.

    Args:
        model_path (str): Path to the pre-trained model file.
        test_save_dir (str): Directory where test images will be saved.

    Returns:
        None
    """
    
    # Configure Pytorch
    seed = 42
    torch.manual_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device: ", device)
    print()
    
    # Create a new instance of the model
    model = modules.VQVAE(
        num_channels=num_channels,
        num_hiddens=num_hiddens,
        num_residual_hiddens=num_hiddens,
        num_embeddings=num_embeddings,
        dim_embedding=dim_embedding,
        beta=beta).to(device)

    # Load the saved state_dict into the model
    model.load_state_dict(torch.load(model_path))
    print('Using saved model\n')

    # model.to(device)
    model.eval()
    
    # Directories for datasets
    train_dir = '/home/groups/comp3710/HipMRI_Study_open/keras_slices_data/keras_slices_train'
    test_dir = '/home/groups/comp3710/HipMRI_Study_open/keras_slices_data/keras_slices_test'
    validate_dir = '/home/groups/comp3710/HipMRI_Study_open/keras_slices_data/keras_slices_validate'

    # Get test loader
    test_loader = HipMRILoader(
        train_dir, validate_dir, test_dir,
        batch_size=batch_size, transform=None
        ).get_test_loader()
    
    # Save average SSIM per batch
    SSIM  = []

    # Test loop
    with torch.no_grad():

        batch_SSIM = [] # Store SSIM per batch

        # Loop through the test dataset
        for i, training_images in enumerate(test_loader):

            # Pass images through the model to get reconstructed images
            training_images = training_images.to(device)
            _, test_output_images = model(training_images)

            # Reshape images for SSIM calculation
            real_image = training_images.view(-1, 1, 256, 128).detach()
            decoded_image = test_output_images.view(-1, 1, 256, 128).detach()
            
            # Calculate SSIM and store it
            similarity = ssim(decoded_image, real_image).item()
            batch_SSIM.append(similarity)
        
        # Calculate average SSIM per batch
        batch_ssim = np.mean(batch_SSIM)
        SSIM.append(batch_ssim)

    average_SSIM = np.mean(SSIM)
    print("Average SSIM on test set: ", average_SSIM)

    # Visualize 4 images through the model and save in test_save_dir
    # Create a figure to plot the images
    fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(8, 16))

    # Loop through random indices to visualize
    for idx, real_image in enumerate(test_loader):  # Get batch from DataLoader
        if idx >= 4:  # We only need 4 images
            break

        # Ensure the real_image is on the device
        real_image = real_image.to(device)

        # Pass it through the model to get the decoded image
        _, decoded_image = model(real_image)

        # Detach and move to CPU for visualization (select only the first in the batch)
        real_image = real_image[0].cpu().numpy().squeeze()  # Assuming it's a single-channel image
        decoded_image = decoded_image[0].cpu().detach().numpy().squeeze()

        # Plot real image on the left
        axes[idx, 0].imshow(real_image, cmap='gray')
        axes[idx, 0].set_title('Real Image')
        axes[idx, 0].axis('off')

        # Plot decoded image on the right
        axes[idx, 1].imshow(decoded_image, cmap='gray')
        axes[idx, 1].set_title('Decoded Image')
        axes[idx, 1].axis('off')

    # Save the figure
    plt.tight_layout()
    plt.savefig(os.path.join(test_save_dir, 'real_vs_decoded.png'))

    # Print graphs from trained model
    # Save directory for graphs
    save_dir = "graphs"
    os.makedirs(save_dir, exist_ok=True)

    open_dir = 'data_viz/VQ-VAE.pkl'

    # Load the saved data using pickle
    with open(open_dir, 'rb') as f:
        data = pickle.load(f)

    # Access the lists
    training_reconstruction_loss = data["training_reconstruction_loss"]
    training_vq_loss = data["training_vq_loss"]
    validation_reconstruction_loss = data["validation_reconstruction_loss"]
    validation_vq_loss = data["validation_vq_loss"]
    ssim = data["ssim"]

    epochs = 150

    if epochs > len(training_reconstruction_loss) + 1:
        epochs = len(training_reconstruction_loss) + 1

    # Plot training and validation reconstruction loss
    epochs = range(1, epochs)
    plt.figure(figsize=(12, 6))
    plt.plot(epochs, training_reconstruction_loss, label="Training Reconstruction Loss")
    plt.plot(epochs, validation_reconstruction_loss, label="Validation Reconstruction Loss")
    plt.xlabel('Epoch')
    plt.ylabel('Training Reconstruction Loss')
    plt.title('Training Reconstruction Loss over Epochs')
    plt.legend()
    plt.grid(True)

    # Save the figure to save_dir
    save_path = os.path.join(save_dir, 'training_reconstruction_loss.png')
    plt.savefig(save_path)
    plt.close()  # Close the plot to prevent it from displaying


    # Plot Training and Validation VQ Loss
    plt.figure(figsize=(12, 6))
    plt.plot(epochs, training_vq_loss, label="Training VQ Loss")
    plt.plot(epochs, validation_vq_loss, label="Validation VQ Loss")
    plt.ylim(0, 50000) 
    plt.xlabel('Epoch')
    plt.ylabel('VQ Loss')
    plt.title('VQ Loss over Epochs')
    plt.legend()
    plt.grid(True)

    # Save the VQ Loss figure
    save_path_vq = os.path.join(save_dir, 'vq_loss.png')
    plt.savefig(save_path_vq)
    plt.close()  # Close the plot to prevent displaying


    # Plot SSIM over epochs
    plt.figure(figsize=(12, 6))
    plt.plot(epochs, ssim, label="SSIM")
    plt.xlabel('Epoch')
    plt.ylabel('SSIM')
    plt.title('SSIM over Epochs')
    plt.legend()
    plt.grid(True)

    # Save the SSIM figure
    save_path_ssim = os.path.join(save_dir, 'ssim.png')
    plt.savefig(save_path_ssim)
    plt.close()  # Close the plot to prevent displaying
    
    print("End")

if (__name__ == "__main__"):
    predict(model_path=model_path,
            test_save_dir=test_save_dir)




    



            


