"""
Utility files for the training and prediction scripts.

@author George Reid-Smith
"""
import logging
from pathlib import Path
import yaml
import numpy as np
import random

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torchvision import utils

def load_config(config_path):
    """Load the configuration yaml.

    Args:
        config_path (str): path to configuration yaml

    Returns:
        dict: dictionary of training and prediction values
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config

def setup_logging(log_dir: str, title: str):
    """Global logging setup for all modules.

    Args:
        log_dir (str): output logging directory
    """
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    print(log_dir / 'training.log')
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] %(levelname)s:%(name)s:%(message)s',
        handlers=[
            logging.FileHandler(log_dir / title),
            logging.StreamHandler()
        ]
    )

def set_seed(seed):
    """Seed randomness for result reproducibility.

    Args:
        seed (int): integer to seed randomness
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

def save_samples(images, outputs, sample_size, output_dir, title):
    """Generate and save image samples for a given model.

    Args:
        images (torch.Tensor): a batch of images
        sample_size (int): number of samples to be generated
        device (torch.device): device to generate images on
        model (nn.Module): the VQVAE model
        output_dir (str): directory to save images to
        title (str): the title of the sample set
    """
    sample_images = images[:sample_size].cpu()
    sample_outputs = outputs[:sample_size].cpu()
    
    combined = torch.cat([sample_images, sample_outputs], 0)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_dir = output_dir / f'{title}.png'

    utils.save_image(
        combined,
        output_dir,
        nrow=sample_size,
        normalize=True
    )

def save_model(epoch, ssim, model, optimizer, model_dir):
    """Save a model version.

    Args:
        epoch (int): current model training epoch
        ssim (float): current structured similarity of model
        model (nn.Module): VQVAE model
        optimizer (torch.optim): optimizer of the model
        model_dir (str): directory to save model
    """
    model_dir = Path(model_dir) / f'vqvae_model.pth'

    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'ssim': ssim
    }, model_dir)

def _plot_loss(iterations, training_losses, validation_losses, loss_dir):
    fig, ax1 = plt.subplots()

    # Plot losses
    ax1.plot(iterations, training_losses, label='Training Loss', color='blue')
    ax1.plot(iterations, validation_losses, label='Validation Loss', color='orange')

    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Iterations')
    ax1.set_ylabel('Reconstruction Loss')
    ax1.legend()

    ax1.grid(True)

    plt.savefig(loss_dir)

def _plot_ssim(iterations, training_ssims, validation_ssims, ssim_dir):
    fig, ax1 = plt.subplots()

    # Plot losses
    ax1.plot(iterations, training_ssims, label='Training SSIM', color='blue')
    ax1.plot(iterations, validation_ssims, label='Validation SSIM', color='orange')

    ax1.set_title('Training and Validation SSIM')
    ax1.set_xlabel('Iterations')
    ax1.set_ylabel('Structured Similarity Index')
    ax1.legend()

    ax1.grid(True)

    plt.savefig(ssim_dir)

def plot_results(iterations, training_losses, training_ssims, validation_losses, validation_ssims, visual_dir):
    """Plot SSIM and reconstruction loss over training proces.

    Args:
        iterations (int): iteration for each loss and ssim
        ssim_scores (float): ssim score at given iteration
        losses (float): recon loss at given iteration
        visual_dir (str): directory to save the plot
    """
    loss_dir = Path(visual_dir) / 'losses.png'
    ssim_dir = Path(visual_dir) / 'ssims.png'

    _plot_loss(iterations, training_losses, validation_losses, loss_dir)
    _plot_ssim(iterations, training_ssims, validation_ssims, ssim_dir)

    