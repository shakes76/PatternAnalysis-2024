import logging
import os
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torchvision import utils
import yaml

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config

def setup_logging(log_dir):
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

def save_samples(epoch, iteration, model, images, sample_size, device, sample_dir):
    sample_images = images[:sample_size]

    with torch.no_grad():
        outputs, _ = model(sample_images.to(device))
    
    combined = torch.cat([sample_images.cpu(), outputs.cpu()], 0)
    sample_path = Path(sample_dir)
    sample_path.mkdir(parents=True, exist_ok=True)

    utils.save_image(
        combined,
        sample_path / f"epoch_{epoch+1:03d}_iter_{iteration+1:05d}.png",
        nrow=sample_size,
        normalize=True
    )

def plot_ssim(iterations, ssim_scores, visual_dir):
    visual_dir = Path(visual_dir)
    plt.figure()
    plt.plot(iterations, ssim_scores, marker='o')
    plt.xlabel('Iteration')
    plt.ylabel('SSIM')
    plt.title('SSIM over Training')
    plt.grid(True)
    plt.savefig(visual_dir / 'ssim_scores.png')  # You can customize the path and filename
    plt.show()  # Optional: comment this out if running on a headless server



