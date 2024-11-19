import argparse
import logging
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import shutil
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import get_dataloader, get_transforms
from modules import VQVAE
from utils import calculate_ssim, read_yaml_file, combine_images
    

def calculate_batch_ssim(batch: torch.Tensor, reconstructed_batch: torch.Tensor) -> float:
    """Calculate the Structural Similarity Index (SSIM) for a batch of images.
    
    Args:
        batch (torch.Tensor): the batch of images.
        
    Returns:
        float: the average SSIM for the batch of images.
    """
    batch_ssim = 0.0
    for i in range(batch.size(0)):
        original_image = batch[i, 0].cpu().detach().numpy()
        reconstructed_image = reconstructed_batch[i, 0].cpu().detach().numpy()
        batch_ssim += calculate_ssim(original_image, reconstructed_image)
    return batch_ssim / batch.size(0)


def train_one_epoch(
    model: VQVAE, 
    train_loader: DataLoader, 
    criterion: nn.Module, 
    optimizer: torch.optim.Optimizer, 
    device: torch.device,
    epoch: int,
    num_epochs: int
) -> tuple:
    """Train the model for one epoch.
    
    Args:
        model (VQVAE): the model to train.
        train_loader (DataLoader): the DataLoader for the training data.
        criterion (nn.Module): the loss function.
        optimizer (torch.optim.Optimizer): the optimizer.
        device (torch.device): the device to train on.
        epoch (int): the current epoch.
        num_epochs (int): the total number of epochs.
        
    Returns:
        tuple: the average commitment loss, average reconstruction loss, average total loss, average SSIM, original image, and reconstructed image.
    """
    # Keep track of losses and SSIM
    train_total_commitment_loss = 0.0
    train_total_recon_loss = 0.0
    train_total_loss = 0.0
    train_total_ssim = 0.0
    for train_batch in tqdm(train_loader, desc=f"Training Epoch {epoch}/{num_epochs}"):
        train_batch = train_batch.to(device).float()
        optimizer.zero_grad()
        
        # Get model output
        train_reconstructed, train_commitment_loss = model(train_batch)
        
        # Calculate loss
        train_recon_loss = criterion(train_reconstructed, train_batch)
        train_loss = train_recon_loss + train_commitment_loss
        
        # Backpropagation
        train_loss.backward()
        optimizer.step()
        
        # Update losses and SSIM
        train_total_commitment_loss += train_commitment_loss.item()
        train_total_recon_loss += train_recon_loss.item()
        train_total_loss += train_loss.item()
        train_total_ssim += calculate_batch_ssim(train_batch, train_reconstructed)
    
    # Calculate average losses and SSIM
    avg_train_commitment_loss = train_total_commitment_loss / len(train_loader)
    avg_train_recon_loss = train_total_recon_loss / len(train_loader)
    avg_train_loss = train_total_loss / len(train_loader)
    avg_train_ssim = train_total_ssim / len(train_loader)
        
    # Get original and reconstructed images of the first image
    train_original_image = train_batch[0, 0].cpu().detach().numpy()
    train_reconstructed_image = train_reconstructed[0, 0].cpu().detach().numpy()
    
    return (
        avg_train_commitment_loss,
        avg_train_recon_loss,
        avg_train_loss,
        avg_train_ssim,
        train_original_image,
        train_reconstructed_image
    )


def validate_one_epoch(
    model: VQVAE,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
    num_epochs: int
) -> tuple:
    """Validate the model for one epoch.
    
    Args:
        model (VQVAE): the model to validate.
        val_loader (DataLoader): the DataLoader for the validation data.
        criterion (nn.Module): the loss function.
        device (torch.device): the device to validate on.
        epoch (int): the current epoch.
        num_epochs (int): the total number of epochs.
        
    Returns:
        tuple: the average commitment loss, average reconstruction loss, average total loss, average SSIM, original image, and reconstructed image.
    """
    # Keep track of losses and SSIM
    val_total_commitment_loss = 0.0
    val_total_recon_loss = 0.0
    val_total_loss = 0.0
    val_total_ssim = 0.0
    with torch.no_grad():
        for val_batch in tqdm(val_loader, desc=f"Validation Epoch {epoch}/{num_epochs}"):
            val_batch = val_batch.to(device).float()
            
            # Get model output
            val_reconstructed, val_commitment_loss = model(val_batch)
            
            # Calculate loss
            val_recon_loss = criterion(val_reconstructed, val_batch)
            val_loss = val_recon_loss + val_commitment_loss
            
            # Update losses and SSIM
            val_total_commitment_loss += val_commitment_loss.item()
            val_total_recon_loss += val_recon_loss.item()
            val_total_loss += val_loss.item()
            val_total_ssim += calculate_batch_ssim(val_batch, val_reconstructed)
            
    # Calculate average losses and SSIM
    avg_val_commitment_loss = val_total_commitment_loss / len(val_loader)
    avg_val_recon_loss = val_total_recon_loss / len(val_loader)
    avg_val_loss = val_total_loss / len(val_loader)
    avg_val_ssim = val_total_ssim / len(val_loader)
            
    # Get original and reconstructed images of the first image
    val_original_image = val_batch[0, 0].cpu().detach().numpy()
    val_reconstructed_image = val_reconstructed[0, 0].cpu().detach().numpy()
    
    return (
        avg_val_commitment_loss,
        avg_val_recon_loss,
        avg_val_loss,
        avg_val_ssim,
        val_original_image,
        val_reconstructed_image
    )
    
    
def save_epoch_image(
    train_original_image: np.ndarray,
    train_reconstructed_image: np.ndarray,
    val_original_image: np.ndarray,
    val_reconstructed_image: np.ndarray,
    epoch: int,
    image_log_dir: str
) -> None:
    """Save the original and reconstructed images for the train and validation sets.
    
    Args:
        train_original_image (np.ndarray): the original image from the training set.
        train_reconstructed_image (np.ndarray): the reconstructed image from the training set.
        val_original_image (np.ndarray): the original image from the validation set.
        val_reconstructed_image (np.ndarray): the reconstructed image from the validation set.
        epoch (int): the current epoch.
        image_log_dir (str): the directory to save the images.
    """
    with torch.no_grad():
        # Combine original and reconstructed images
        train_image = combine_images(train_original_image, train_reconstructed_image)
        val_image = combine_images(val_original_image, val_reconstructed_image)
        
        plt.figure(figsize=(8, 4))

        plt.subplot(1, 2, 1)
        plt.imshow(train_image, cmap='gray')
        plt.title("Train Original and Reconstructed Image")
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(val_image, cmap='gray')
        plt.title("Validation Original and Reconstructed Image")
        plt.axis('off')

        plt.tight_layout()
        plt.savefig(os.path.join(image_log_dir, f'epoch_{epoch}.png'))
        plt.close()
    

def train(config: dict) -> None:
    """Train the VQVAE model.
    
    Args:
        config (dict): configuration dictionary.
    """
    # Load configuration
    model_parameters = config['model_parameters']
    
    log_dir = os.path.join(config['logs_root'], config['log_dir_name']) if config['log_dir_name'] else \
        os.path.join(config['logs_root'], f"{time.strftime('%Y%m%d_%H%M%S')}")
    image_log_dir = os.path.join(log_dir, 'images')
    log_frequency = config['log_frequency']
    image_frequency = config['image_frequency']
    
    batch_size = config['batch_size']
    learning_rate = config['learning_rate']
    num_epochs = config['num_epochs']
    weight_decay = config['weight_decay']
    
    train_dataset_dir = config['train_dataset_dir']
    val_dataset_dir = config['val_dataset_dir']
    test_dataset_dir = config['test_dataset_dir']
    train_num_samples = config['train_num_samples']
    val_num_samples = config['val_num_samples']
    test_num_samples = config['test_num_samples']
    
    train_transforms = config['train_transforms']
    val_test_transforms = config['val_test_transforms']
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Set up logging
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(image_log_dir, exist_ok=True)
    logging.basicConfig(filename=os.path.join(log_dir, 'training.log'), level=logging.INFO, 
                        format='%(asctime)s - %(levelname)s - %(message)s')
    shutil.copy(config_path, log_dir)
    
    # Load model
    model = VQVAE(**model_parameters).to(device)
    
    # Optimizer and Loss
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = nn.MSELoss()
    
    # Data Loaders
    train_transform = get_transforms(train_transforms)
    val_test_transform = get_transforms(val_test_transforms)
    
    train_loader = get_dataloader(train_dataset_dir, batch_size, train_transform, train_num_samples, shuffle=True)
    val_loader = get_dataloader(val_dataset_dir, batch_size, val_test_transform, val_num_samples, shuffle=False)
    test_loader = get_dataloader(test_dataset_dir, 1, val_test_transform, test_num_samples, shuffle=False)
    
    # Training Loop
    train_commitment_losses = []
    train_recon_losses = []
    train_total_losses = []
    train_ssim_scores = []
    val_commitment_losses = []
    val_recon_losses = []
    val_total_losses = []
    val_ssim_scores = []
    
    best_val_loss = float('inf')
    
    start_time = time.time()
    for epoch in range(1, num_epochs + 1):
        # Train
        model.train()
        train_commitment_loss, train_recon_loss, train_loss, train_ssim, train_original_image, train_reconstructed_image = \
            train_one_epoch(model, train_loader, criterion, optimizer, device, epoch, num_epochs)
        
        # Validation
        model.eval()
        val_commitment_loss, val_recon_loss, val_loss, val_ssim, val_original_image, val_reconstructed_image = \
            validate_one_epoch(model, val_loader, criterion, device, epoch, num_epochs)
        
        # Save original and reconstructed images
        if epoch % image_frequency == 0:
            save_epoch_image(
                train_original_image, 
                train_reconstructed_image, 
                val_original_image, 
                val_reconstructed_image, 
                epoch, 
                image_log_dir
            )
        
        # Save metrics
        train_commitment_losses.append(train_commitment_loss)
        train_recon_losses.append(train_recon_loss)
        train_total_losses.append(train_loss)
        train_ssim_scores.append(train_ssim)
        val_commitment_losses.append(val_commitment_loss)
        val_recon_losses.append(val_recon_loss)
        val_total_losses.append(val_loss)
        val_ssim_scores.append(val_ssim)
        
        # Log
        if epoch % log_frequency == 0:
            logging.info(f"Epoch [{epoch}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Train SSIM: {train_ssim:.4f}, Val SSIM: {val_ssim:.4f}")
            
        # Save best model
        if val_loss < best_val_loss:
            torch.save(model.state_dict(), os.path.join(log_dir, 'best_model.pth'))
            best_val_loss = val_loss
    
    # Log training time
    end_time = time.time()
    logging.info(f"Training took {(end_time - start_time) / 60:.2f} minutes\n")
        
    # Plot metrics
    train_metrics = [
        train_commitment_losses, 
        train_recon_losses,
        train_total_losses, 
        train_ssim_scores
    ]
    val_metrics = [
        val_commitment_losses, 
        val_recon_losses,
        val_total_losses, 
        val_ssim_scores
    ]
    metric_names = [
        "Vector Quantization Loss", 
        "Reconstruction Loss",
        "Total Loss", 
        "Structural Similarity Index"
    ]
    
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    axs = axs.flatten()
    
    for i in range(4):
        axs[i].plot(train_metrics[i], label='Train')
        axs[i].plot(val_metrics[i], label='Validation')
        axs[i].set_title(metric_names[i])
        axs[i].set_xlabel('Epoch')
        axs[i].set_ylabel(metric_names[i])
        axs[i].legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(log_dir, 'train_metrics.png'))
    plt.close()
    
    # Save latest model
    torch.save(model.state_dict(), os.path.join(log_dir, 'latest_model.pth'))
    
    # Evaluation
    model.eval()
    test_losses = []
    test_ssim_values = []
    with torch.no_grad():
        for test_batch in tqdm(test_loader, desc="Evaluation"):
            test_batch = test_batch.to(device).float()
            
            # Get model output
            test_reconstructed, test_commitment_loss = model(test_batch)
            
            # Calculate loss
            test_recon_loss = criterion(test_reconstructed, test_batch)
            
            # Get original and reconstructed images
            test_original_image = test_batch[0, 0].cpu().detach().numpy()
            test_reconstructed_image = test_reconstructed[0, 0].cpu().detach().numpy()
            
            # Update losses and SSIM
            test_losses.append(test_recon_loss.item() + test_commitment_loss.item())
            test_ssim_values.append(calculate_ssim(test_original_image, test_reconstructed_image))
        
    # Log test metrics
    logging.info(f"Test Loss: {sum(test_losses) / len(test_losses):.4f}, Test SSIM: {sum(test_ssim_values) / len(test_ssim_values):.4f}")
    
    # Save distribution of scores
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs = axs.flatten()
    
    axs[0].hist(test_losses, bins=20, color='blue', alpha=0.7)
    axs[0].set_title("Loss Distribution")
    axs[0].set_xlabel("Loss")
    axs[0].set_ylabel("Frequency")
    
    axs[1].hist(test_ssim_values, bins=20, color='green', alpha=0.7)
    axs[1].set_title("SSIM Distribution")
    axs[1].set_xlabel("SSIM")
    axs[1].set_ylabel("Frequency")
    
    plt.tight_layout()
    plt.savefig(os.path.join(log_dir, f'test_metrics.png'))
    plt.close()
    
    # Save test images
    fig, axs = plt.subplots(2, 3, figsize=(12, 8))
    axs = axs.flatten()
    with torch.no_grad():
        for i, test_batch in enumerate(test_loader):
            if i == 6:
                break
            test_batch = test_batch.to(device).float()
            test_reconstructed, _ = model(test_batch)
            test_original_image = test_batch[0, 0].cpu().detach().numpy()
            test_reconstructed_image = test_reconstructed[0, 0].cpu().detach().numpy()
            
            test_combined_image = combine_images(test_original_image, test_reconstructed_image)

            axs[i].imshow(test_combined_image, cmap='gray')
            axs[i].axis('off')
        
    plt.tight_layout()
    plt.savefig(os.path.join(log_dir, f'test_images.png'))
    plt.close()
 
 
if __name__ == '__main__':
    # Load configuration
    parser = argparse.ArgumentParser(description='Train VQVAE model.')
    parser.add_argument('--config', type=str, required=True, 
                        help='Path to the configuration YAML file.')
    
    config_path = parser.parse_args().config
    config = read_yaml_file(config_path)
    
    train(config)
