import os
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms # type: ignore
import matplotlib.pyplot as plt
from modules import VQVAE # type: ignore # Import the VQVAE model
from dataset import ProstateMRIDataset  # Import the custom dataset
from torchvision.utils import make_grid, save_image # type: ignore
from skimage.metrics import structural_similarity as ssim

# Training hyperparameters
batch_size = 32  # Determines how many images are processed at once before the model updates its parameters
num_epochs = 20  # Number of training epochs
learning_rate = 0.001  # Learning rate for the optimizer
beta = 0.25 # The commitment loss coefficient

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Defines the data transformations
transform = transforms.Compose([
    transforms.Resize((296, 144)), 
    transforms.Grayscale(num_output_channels=1),  # Ensuring grayscale images
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # Normalizes the pixel values to [-1, 1]
])

# Dataset and DataLoader
train_data_path = '/home/groups/comp3710/HipMRI_Study_open/keras_slices_data/keras_slices_train' # Path to MRI training data
test_data_path = '/home/groups/comp3710/HipMRI_Study_open/keras_slices_data/keras_slices_test' # Path to MRI test data
val_data_path = '/home/groups/comp3710/HipMRI_Study_open/keras_slices_data/keras_slices_validate' # Path to MRI validation data

# Loads datasets using the custom data loader from dataset.py
train_dataset = ProstateMRIDataset(img_dir=train_data_path, transform=transform)
test_dataset = ProstateMRIDataset(img_dir=test_data_path, transform=transform)
val_dataset = ProstateMRIDataset(img_dir=val_data_path, transform=transform)

# Creates DataLoaders for each dataset
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=1)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=1)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=1)

# Initialises the VQ-VAE model which was defined in modules.py
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
).to(device)

# Initialise the optimizer (Adam Optimizer) and loss function (MSE loss)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.MSELoss()

# Loss logging
eval_every = 100
model_save_path = './saved_models'
os.makedirs(model_save_path, exist_ok=True)

# Lists to store SSIM scores and losses for plotting graphs
train_ssim_scores, val_ssim_scores = [], []
train_losses, val_losses = [], []
batch_losses = []

# Training loop
def train():
    """
    This is the main function that trains the VQ-VAE model over a certain number of epochs
    """
    # Prints the number of images in each dataset
    print(f'Number of training images: {len(train_dataset)}')
    print(f'Number of validation images: {len(val_dataset)}')
    print(f'Number of testing images: {len(test_dataset)}')

    # Calls the save original image function to save a batch of original images before training
    save_original_images(train_loader)

    print("Starting training loop")

    # Tracks the best validation loss for model saving 
    best_val_loss = float('inf') 
    
    # Iterates over the set number of epochs 
    for epoch in range(num_epochs):
        model.train()  # Sets the model to training mode
        total_train_loss = 0
        ssim_train_scores = []  # Tracks the SSIM scores for the training images
        for batch_idx, images in enumerate(train_loader):
            images = images.to(device)  # Moves images to GPU for training

            # Forward pass through the VQ-VAE model
            outputs = model(images)
            x_recon = outputs['x_recon']
            commitment_loss = outputs['commitment_loss']
            codebook_embeddings = outputs['codebook_embeddings']

            # Calculates the reconstruction loss which is the MSE between the original and reconstructed images
            recon_loss = criterion(x_recon, images)

            # Total loss is the sum of reconstruction loss and commitment loss
            loss = recon_loss + beta * commitment_loss

            # Performs backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Tracks the batch-level losses for plotting
            batch_losses.append(loss.item())

            # Accumulates the training loss for the current epoch 
            total_train_loss += loss.item()

            # Calculates SSIM between original and reconstructed images in the same batch
            batch_ssim = calculate_ssim(images, x_recon)
            ssim_train_scores.append(batch_ssim)

            # Prints log information at specific intervals (every 100th)
            if batch_idx % eval_every == 0:
                print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{batch_idx}/{len(train_loader)}], '
                      f'Loss: {loss.item():.4f}, Recon Loss: {recon_loss.item():.4f}, '
                      f'Commitment Loss: {commitment_loss.item():.4f}')
        
        # Calculates the average training loss and SSIM for the epoch
        avg_train_loss = total_train_loss / len(train_loader)
        avg_ssim_train = np.mean(ssim_train_scores)

        print(f'Epoch [{epoch + 1}], Train Loss: {avg_train_loss:.4f}, Train SSIM: {avg_ssim_train:.4f}')

        # Stores the train SSIM and loss values in the initialised list
        train_ssim_scores.append(avg_ssim_train)
        train_losses.append(avg_train_loss)

        # Calls the function that saves the reconstructed training images
        visualise_reconstruction(images, x_recon, codebook_embeddings, epoch, phase='train')

        model.eval() # Switches the model to evaluation mode
        total_val_loss = 0
        ssim_val_scores = []  # Track SSIM for validation images
        with torch.no_grad():
            for val_images in val_loader:
                val_images = val_images.to(device)

                # Forward pass through the model using the validation images
                outputs = model(val_images)
                x_recon = outputs['x_recon']
                commitment_loss = outputs['commitment_loss']
                codebook_embeddings = outputs['codebook_embeddings']

                # Validation reconstruction loss
                val_recon_loss = criterion(x_recon, val_images)

                # Sums up the reconstruction and commitment losses to get the total validation loss
                val_loss = val_recon_loss + beta * commitment_loss
                total_val_loss += val_loss.item()

                # Calculate SSIM between original and reconstructed validation images
                batch_ssim = calculate_ssim(val_images, x_recon)
                ssim_val_scores.append(batch_ssim)
        
        # Calculates the average validation loss and SSIM for the epoch
        avg_val_loss = total_val_loss / len(val_loader)
        avg_ssim_val = np.mean(ssim_val_scores)

        print(f'Epoch [{epoch + 1}], Val Loss: {avg_val_loss:.4f}, Val SSIM: {avg_ssim_val:.4f}')

        # Stores the validation SSIM and loss values in the list 
        val_ssim_scores.append(avg_ssim_val)
        val_losses.append(avg_val_loss)

        # Calls the function that saves the reconstructed validation images
        visualise_reconstruction(val_images, x_recon, codebook_embeddings, epoch, phase='val')

        # Saves the best model based on the best validation loss so far
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), os.path.join(model_save_path, 'best_vqvae_model.pth'))
            print(f"Model saved at epoch {epoch + 1} with validation loss: {best_val_loss:.4f}")

    # Plots and saves the SSIM scores and losses after training is finished
    plot_ssim_scores(train_ssim_scores, val_ssim_scores)
    plot_losses(train_losses, val_losses)
    plot_batch_losses(batch_losses)


def plot_ssim_scores(train_ssim_scores, val_ssim_scores):
    """
    Plots the SSIM scores for every epoch and saves the plot as 'ssim_scores.png'

    Args:
        train_ssim_scores (list): List of SSIM scores for the training set
        val_ssim_scores (list): List of SSIM scores for the validation set
    """
    plt.figure()
    plt.plot(train_ssim_scores, label='Train SSIM')
    plt.plot(val_ssim_scores, label='Val SSIM')
    plt.xlabel('Epoch')
    plt.ylabel('SSIM')
    plt.title('SSIM Scores per Epoch')

    # Added a red dotted line at y=0.6 to represent the benchmark
    plt.axhline(y=0.6, color='red', linestyle='--', label='Benchmark SSIM 0.6')

    plt.legend()
    plt.savefig('ssim_scores.png')
    plt.close()

def plot_losses(train_losses, val_losses):
    """
    Plots the loss values for every epoch and saves the plot as 'losses.png'

    Args:
        train_losses (list): List of training passes per epoch
        val_losses (list): List of validation losses per epoch
    """
    plt.figure()
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Losses per Epoch')
    plt.legend()
    plt.savefig('losses.png')
    plt.close()
 
def plot_batch_losses(train_losses):
    """
    Plots the reconstruction losses for each batch during training and saves the plot as 'batch_losses.png'

    Args:
        train_losses (list): List of batch-level training losses
    """
    plt.figure()
    plt.plot(train_losses, label='Train Loss')
    plt.xlabel('Batch no.')
    plt.ylabel('Reconstruction loss')
    plt.title('Training reconstruction Losses')
    plt.ylim(0, 0.2) # Reduced the y-axis range to zoom in on values 0 to 0.2
    plt.legend()
    plt.savefig('batch_losses.png')
    plt.close()

def calculate_ssim(original_images, reconstructed_images):
    """
    Calculates the SSIM between the original and reconstructed images 

    Args: 
        original_images (torch.Tensor): Batch of original images
        reconstructed_images (torch.Tensor): Batch of reconstructed images
    
    Returns:
        float: The mean SSIM score for that specific batch
    """

    # Moves both original and reconstructed images to CPU 
    original_images = original_images.cpu().data
    reconstructed_images = reconstructed_images.cpu().data

    ssim_scores = []  # Lists to store SSIM scores for each image

    # Iterates through each image in the batch 
    for i in range(original_images.size(0)):
        # Converts the first channel of the original and reconstructed images to NumPy arrays
        original_np = original_images[i][0].numpy()
        reconstructed_np = reconstructed_images[i][0].numpy()

        # Calculates the SSIM score between the original and reconstruced images
        ssim_score = ssim(original_np, reconstructed_np, data_range=reconstructed_np.max() - reconstructed_np.min())
        ssim_scores.append(ssim_score)

    return np.mean(ssim_scores)

def visualise_reconstruction(original_images, reconstructed_images, codebook_embeddings, epoch, phase='train'):
    """
    Visualises and saves the reconstructed images, along with the original and codebook embedding images

    Args:
        original_images (torch.Tensor): Original images
        reconstructed (torch.Tensor): Reconstructed images by the VQ-VAE model
        codebook_embeddings (torch.Tensor): Embedding vectors representing the codebook used for quantization
        epoch (int): The current epoch to be used for naming the saved image
        phase (str): Indicates whether the images are from the train or validation phase
    """
    # Moves images and embeddings to CPU 
    original_images = original_images.cpu().data
    reconstructed_images = reconstructed_images.cpu().data
    codebook_embeddings = codebook_embeddings.cpu().data  

    # Denormalizes images from [-1, 1] to [0, 1] for visualisation purposes
    original_images = (original_images + 1) / 2
    reconstructed_images = (reconstructed_images + 1) / 2

    # Creates the directory for saving images if it doesn't exist
    recon_save_dir = f'./reconstructions/{phase}'  
    os.makedirs(recon_save_dir, exist_ok=True)

    # Plots original image and reconstructed images 
    num_images = min(8, original_images.size(0))  
    fig, axes = plt.subplots(2, num_images, figsize=(num_images * 2, 4))

    for i in range(num_images):
        # Original images are plotted on the top row
        axes[0, i].imshow(original_images[i][0], cmap='gray')  
        axes[0, i].axis('off')

        # Reconstructed images are plotted on the bottom row
        axes[1, i].imshow(reconstructed_images[i][0], cmap='gray')  
        axes[1, i].axis('off')

        # Calls the visualise_comparison funcion to save the input, embedding, and reconstructed images together
        visualise_comparison(
            original_images[i].unsqueeze(0),  
            codebook_embeddings[i].unsqueeze(0), 
            reconstructed_images[i].unsqueeze(0), 
            epoch,
            save_dir=f'./comparisons/{phase}', 
            img_name=f'comparison_epoch_{epoch}_image_{i}.png'  
        )

    # Saves the plot with the phase (train/val) in the filename
    plt.savefig(f'{recon_save_dir}/epoch_{epoch + 1}_phase_{phase}.png')
    plt.close()

def save_original_images(loader, save_dir='./original_images', img_name="original_images.png"):
    """
    Saves a batch of original images in a grid like format and saves it as 'original_images.png'

    Args:
        loader (DataLoader): DataLoader for the original images 
        save_dir (str): Directory to save the original images
        img_name (str): The name of the saved image
    """
    # Fetches a batch of images from the data loader
    original_imgs = next(iter(loader))
    
    # Denormalizes the images to the range [0, 1] for visualisation purposes
    original_imgs = denormalize(original_imgs)
    
    # Uses the make_grid function to create the grid of original images 
    grid = make_grid(original_imgs, nrow=8)
    
    # Creates the directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Saves the grid of images in the specified directory 
    save_image(grid, os.path.join(save_dir, img_name))

    print("Saved original images")

def denormalize(tensor):
    """
    Denomarlises an image tensor from the range [-1, 1] to [0, 1] for visualisation
    """
    return (tensor * 0.5) + 0.5  

def visualise_comparison(input_image, codebook_embedding, reconstructed_image, epoch, save_dir='./comparisons', img_name="comparison.png"):
    """
    Save a comparison of the input image, codebook embedding, and reconstructed image as 'comparison.png'

    Args:
        input_image (torch.Tensor): The original input image from the dataset
        codebook_embedding (torch.Tensor): The quantized embedding vector
        reconstructed_image (torch.Tensor): The reconstructed image by the VQ-VAE model
        epoch (int): The current epoch number
        save_dir (str): The name of the directory where the image will be saved
        img_name (str): The name of the saved image 
    """
    # Creates the directory for saving comparisons if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    # Denormalizes the images from [-1, 1] to [0, 1] for visualisation purposes
    input_image = denormalize(input_image)
    reconstructed_image = denormalize(reconstructed_image)

    # Reduce the codebook embedding to a 2D representation by averaging across channels 
    codebook_embedding_vis = codebook_embedding.mean(dim=1).detach().cpu()  

    # Creates a plot with 3 columns to easily compare the input image, codebook embedding, and reconstructed image
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    # Input Image
    axes[0].imshow(input_image.cpu().squeeze(0).permute(1, 2, 0), cmap='gray')
    axes[0].set_title("Input Image")
    axes[0].axis('off')

    # Codebook Embedding
    axes[1].imshow(codebook_embedding_vis[0].cpu(), cmap='nipy_spectral')  # Different color map for embeddings
    axes[1].set_title("Codebook Embedding")
    axes[1].axis('off')

    # Reconstructed Image
    axes[2].imshow(reconstructed_image.cpu().squeeze(0).permute(1, 2, 0), cmap='gray')
    axes[2].set_title("Reconstructed Image")
    axes[2].axis('off')

    # Saves the comparison plot
    comparison_path = os.path.join(save_dir, f'epoch_{epoch}_comparison.png')
    plt.savefig(comparison_path)
    plt.close()

if __name__ == "__main__":
    train()

