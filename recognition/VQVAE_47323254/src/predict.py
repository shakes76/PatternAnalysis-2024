import argparse
import os

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchvision import transforms
from tqdm import tqdm

from dataset import get_dataloader, get_transforms
from modules import VQVAE
from utils import calculate_ssim, read_yaml_file, combine_images


if __name__ == '__main__':
    # Load configuration
    parser = argparse.ArgumentParser(description='Predict using VQVAE model.')
    parser.add_argument('--config', type=str, required=True, 
                        help='Path to the configuration YAML file.')
    config_path = parser.parse_args().config
    config = read_yaml_file(config_path)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model_parameters = config['model_parameters']
    pretrained_path = config['pretrained_path']
    dataset_dir = config['dataset_dir']
    save_dir = config['save_dir']
    
    os.makedirs(save_dir, exist_ok=True)
    
    transforms = get_transforms(config['transforms'])
    data_loader = get_dataloader(dataset_dir, 1, transforms, num_samples=None, shuffle=False)
    
    model = VQVAE(**model_parameters).to(device)
    model.load_state_dict(torch.load(pretrained_path, map_location=device))
    
    criterion = nn.MSELoss()
    
    # Prediction
    # Save the original and reconstructed images
    model.eval()
    image_count = 1
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Predicting"):
            batch = batch.to(device).float()
            reconstructed, commitment_loss = model(batch)
            recon_loss = criterion(reconstructed, batch)
            loss = recon_loss + commitment_loss
            
            original_image = batch[0, 0].cpu().detach().numpy()
            reconstructed_image = reconstructed[0, 0].cpu().detach().numpy()
            
            ssim = calculate_ssim(original_image, reconstructed_image)
            
            combined_image = combine_images(original_image, reconstructed_image)
            
            plt.imshow(combined_image, cmap='gray')
            plt.title(f"Loss: {loss.item():.4f}, SSIM: {ssim:.4f}")
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f'image_{image_count}.png'))
            plt.close()
            
            image_count += 1
