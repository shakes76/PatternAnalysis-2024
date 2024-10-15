import logging
import os

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from dataset import HipMRIDataset
from modules import VQVAE
from utils import calculate_ssim


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Path
    dir = "recognition/VQVAE_47323254"
    log_dir = f'{dir}/logs/v1'
    
    data_dir = f'{dir}/HipMRI_study_keras_slices_data'
    pretrained_path = f"{log_dir}/model.pth"

    # Configuration
    num_samples = None
    
    logging.basicConfig(filename=os.path.join(log_dir, 'evaluation.log'), level=logging.INFO, 
                        format='%(asctime)s - %(levelname)s - %(message)s')

    test_transform = transforms.Compose([
        transforms.CenterCrop((256, 128)),
        # transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    dataset = HipMRIDataset(data_dir + "/keras_slices_test", test_transform, num_samples=num_samples)
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    model = VQVAE(in_channels=1, hidden_channels=128, res_channels=32, nb_res_layers=2, 
                embed_dim=64, nb_entries=512, downscale_factor=4).to(device)
    
    model.load_state_dict(torch.load(pretrained_path, map_location=device))
    
    losses = []
    ssim_values = []
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
    plt.show()
    
    # Display some images
    n = 5
    for batch in data_loader:
        if n == 0:
            break
        n -= 1
        batch = batch.to(device).float()
        reconstructed, _ = model(batch)
        original_image = batch[0, 0].cpu().detach().numpy()
        reconstructed_image = reconstructed[0, 0].cpu().detach().numpy()
        
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        axs = axs.flatten()
        
        axs[0].imshow(original_image, cmap='gray')
        axs[0].set_title("Original Image")
        axs[0].axis('off')
        
        axs[1].imshow(reconstructed_image, cmap='gray')
        axs[1].set_title("Reconstructed Image")
        axs[1].axis('off')
        
        plt.tight_layout()
        plt.show()
    

