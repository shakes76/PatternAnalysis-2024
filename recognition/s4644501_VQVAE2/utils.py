"""
Utility files for the training and prediction scripts.

@author George Reid-Smith
"""
import logging
from pathlib import Path
import yaml

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

def setup_logging(log_dir: str):
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
            logging.FileHandler(log_dir / 'training.log'),
            logging.StreamHandler()
        ]
    )

def save_samples(images, sample_size, device, model, output_dir, title):
    """Generate and save image samples for a given model.

    Args:
        images (torch.Tensor): a batch of images
        sample_size (int): number of samples to be generated
        device (torch.device): device to generate images on
        model (nn.Module): the VQVAE model
        output_dir (str): directory to save images to
        title (str): the title of the sample set
    """
    sample_images = images[:sample_size]

    with torch.no_grad():
        outputs, _ = model(sample_images.to(device))
    
    combined = torch.cat([sample_images.cpu(), outputs.cpu()], 0)
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

    logging.info('Model saved')

def plot_results(iterations, ssim_scores, losses, visual_dir):
    """Plot SSIM and reconstruction loss over training proces.

    Args:
        iterations (int): iteration of loss and ssim
        ssim_scores (float): ssim score at given iteration
        losses (float): recon loss at given iteration
        visual_dir (str): directory to save the plot
    """
    visual_dir = Path(visual_dir)

    fig, ax1 = plt.subplots()

    ax1.plot(iterations, ssim_scores, color='orange', label='SSIM')
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('SSIM', color='orange')
    ax1.tick_params(axis='y', labelcolor='orange')

    ax2 = ax1.twinx()
    ax2.plot(iterations, losses, color='b', label='Loss')
    ax2.set_ylabel('Loss', color='b')
    ax2.tick_params(axis='y', labelcolor='b')

    plt.title('SSIM and Loss over Model Training')
    ax1.grid(True)
    plt.savefig(visual_dir / 'ssim_loss_plot.png')