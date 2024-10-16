import argparse
import logging
import os
import shutil
import time

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from dataset import HipMRIDataset
from modules import VQVAE
from utils import calculate_ssim, read_yaml_file, get_transforms, combine_images


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test VQVAE model.')
    parser.add_argument('--config', type=str, required=True, 
                        help='Path to the configuration YAML file.')
    config_path = parser.parse_args().config
    config = read_yaml_file(config_path)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model_parameters = config['model_parameters']
    pretrained_path = config['pretrained_path']
    
    dataset_dir = config['test_dataset_dir']
    num_samples = config['num_samples']
    
    log_dir = os.path.join(config['logs_root'], config['log_dir_name']) if config['log_dir_name'] else \
        os.path.join(config['logs_root'], f"{time.strftime('%Y%m%d_%H%M%S')}")
    
    transforms = get_transforms(config['test_transforms'])
    
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(filename=os.path.join(log_dir, 'evaluation.log'), level=logging.INFO, 
                        format='%(asctime)s - %(levelname)s - %(message)s')
    shutil.copy(config_path, log_dir)
    
    dataset = HipMRIDataset(dataset_dir, transforms, num_samples=num_samples)
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    model = VQVAE(**model_parameters).to(device)
    
    model.load_state_dict(torch.load(pretrained_path, map_location=device))
    
    # Evaluation
    losses = []
    ssim_values = []
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluation"):
            batch = batch.to(device).float()
            reconstructed, vq_loss = model(batch)
            recon_loss = F.mse_loss(reconstructed, batch)
            
            original_image = batch[0, 0].cpu().detach().numpy()
            reconstructed_image = reconstructed[0, 0].cpu().detach().numpy()
            
            losses.append(recon_loss.item() + vq_loss.item())
            ssim_values.append(calculate_ssim(original_image, reconstructed_image))
        
    logging.info(f"Average Loss: {sum(losses) / len(losses)}, Average SSIM: {sum(ssim_values) / len(ssim_values)}")
    
    # Display distribution of scores
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs = axs.flatten()
    
    axs[0].hist(losses, bins=20, color='blue', alpha=0.7)
    axs[0].set_title("Loss Distribution")
    axs[0].set_xlabel("Loss")
    axs[0].set_ylabel("Frequency")
    
    axs[1].hist(ssim_values, bins=20, color='green', alpha=0.7)
    axs[1].set_title("SSIM Distribution")
    axs[1].set_xlabel("SSIM")
    axs[1].set_ylabel("Frequency")
    
    plt.tight_layout()
    plt.savefig(os.path.join(log_dir, f'evaluation_metrics.png'))
    plt.close()
    
    # Display some images
    fig, axs = plt.subplots(2, 3, figsize=(12, 8))
    axs = axs.flatten()
    with torch.no_grad():
        for i, batch in enumerate(data_loader):
            if i == 6:
                break
            batch = batch.to(device).float()
            reconstructed, _ = model(batch)
            original_image = batch[0, 0].cpu().detach().numpy()
            reconstructed_image = reconstructed[0, 0].cpu().detach().numpy()
            
            combined_image = combine_images(original_image, reconstructed_image)

            axs[i].imshow(combined_image, cmap='gray')
            axs[i].set_title(f"Image {i + 1}")
            axs[i].axis('off')
        
    plt.tight_layout()
    plt.savefig(os.path.join(log_dir, f'images.png'))
    plt.close()
    