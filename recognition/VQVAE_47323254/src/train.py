import logging
import os
import time

import matplotlib.pyplot as plt
import shutil
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import HipMRIDataset
from modules import VQVAE
from utils import calculate_ssim, read_yaml_file, get_transforms
    
    
if __name__ == '__main__':
    # config_path = "recognition/VQVAE_47323254/configs/train.yaml"
    config_path = "/home/Student/s4732325/project2/configs/train.yaml"
    config = read_yaml_file(config_path)
    
    model_parameters = config['model_parameters']
    
    log_dir = os.path.join(config['logs_root'], config['log_dir_name']) if config['log_dir_name'] else \
        os.path.join(config['logs_root'], f"{time.strftime('%Y%m%d_%H%M%S')}")
    log_frequency = config['log_frequency']
    image_frequency = config['image_frequency']
    
    batch_size = config['batch_size']
    learning_rate = config['learning_rate']
    num_epochs = config['num_epochs']
    weight_decay = config['weight_decay']
    
    train_dataset_dir = config['train_dataset_dir']
    val_dataset_dir = config['val_dataset_dir']
    num_samples = config['num_samples']
    
    train_transforms = config['train_transforms']
    val_transforms = config['val_transforms']
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(filename=os.path.join(log_dir, 'training.log'), level=logging.INFO, 
                        format='%(asctime)s - %(levelname)s - %(message)s')
    shutil.copy(config_path, log_dir)

    train_transform = get_transforms(train_transforms)
    val_test_transform = get_transforms(val_transforms)

    model = VQVAE(**model_parameters).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # Dataset loader
    train_dataset = HipMRIDataset(train_dataset_dir, train_transform, num_samples=num_samples)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataset = HipMRIDataset(val_dataset_dir, val_test_transform, num_samples=num_samples)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Training Loop
    train_vq_losses = []
    train_recon_losses = []
    train_total_losses = []
    val_vq_losses = []
    val_recon_losses = []
    val_total_losses = []
    train_ssim_scores = []
    val_ssim_scores = []
    
    start_time = time.time()
    for epoch in range(1, num_epochs + 1):
        # Train
        model.train()
        train_total_vq_loss = 0.0
        train_total_recon_loss = 0.0
        train_total_loss = 0.0
        train_total_ssim = 0.0
        for train_batch in tqdm(train_loader, desc=f"Training Epoch {epoch}/{num_epochs}"):
            train_batch = train_batch.to(device).float()
            optimizer.zero_grad()
            train_reconstructed, train_vq_loss = model(train_batch)
            train_recon_loss = F.mse_loss(train_reconstructed, train_batch)
            train_loss = train_recon_loss + train_vq_loss
            train_loss.backward()
            optimizer.step()
            
            train_total_vq_loss += train_vq_loss.item()
            train_total_recon_loss += train_recon_loss.item()
            train_total_loss += train_loss.item()
            train_total_ssim += calculate_ssim(train_batch[0, 0].cpu().detach().numpy(), 
                                        train_reconstructed[0, 0].cpu().detach().numpy())
        
        # Validation
        model.eval()
        val_total_vq_loss = 0.0
        val_total_recon_loss = 0.0
        val_total_loss = 0.0
        val_total_ssim = 0.0
        with torch.no_grad():
            for val_batch in tqdm(val_loader, desc=f"Validation Epoch {epoch}/{num_epochs}"):
                val_batch = val_batch.to(device).float()
                val_reconstructed, val_vq_loss = model(val_batch)
                val_recon_loss = F.mse_loss(val_reconstructed, val_batch)
                val_loss = val_recon_loss + val_vq_loss
                
                val_total_vq_loss += val_vq_loss.item()
                val_total_recon_loss += val_recon_loss.item()
                val_total_loss += val_loss.item()
                val_total_ssim += calculate_ssim(val_batch[0, 0].cpu().detach().numpy(), 
                                        val_reconstructed[0, 0].cpu().detach().numpy())
        
        # Save original and reconstructed images every 10 epochs
        if epoch % image_frequency == 0:
            with torch.no_grad():
                train_original_image = train_batch[0, 0].cpu().detach().numpy()
                train_reconstructed_image = train_reconstructed[0, 0].cpu().detach().numpy()
                val_original_image = val_batch[0, 0].cpu().detach().numpy()
                val_reconstructed_image = val_reconstructed[0, 0].cpu().detach().numpy()
                
                plt.figure(figsize=(10, 10))

                plt.subplot(2, 2, 1)
                plt.imshow(train_original_image, cmap='gray')
                plt.title(f"Train Original Image")
                plt.axis('off')

                plt.subplot(2, 2, 2)
                plt.imshow(train_reconstructed_image, cmap='gray')
                plt.title(f"Train Reconstructed Image")
                plt.axis('off')
                
                plt.subplot(2, 2, 3)
                plt.imshow(val_original_image, cmap='gray')
                plt.title(f"Val Original Image")
                plt.axis('off')
                
                plt.subplot(2, 2, 4)
                plt.imshow(val_reconstructed_image, cmap='gray')
                plt.title(f"Val Reconstructed Image")
                plt.axis('off')

                plt.savefig(os.path.join(log_dir, f'epoch_{epoch}.png'))
                plt.close()
        
        train_loss = train_total_loss / len(train_loader)
        val_loss = val_total_loss / len(val_loader)
        train_vq_losses.append(train_total_vq_loss / len(train_loader))
        train_recon_losses.append(train_total_recon_loss / len(train_loader))
        train_total_losses.append(train_loss)
        train_ssim_scores.append(train_total_ssim / len(train_loader))
        val_vq_losses.append(val_total_vq_loss / len(val_loader))
        val_recon_losses.append(val_total_recon_loss / len(val_loader))
        val_total_losses.append(val_loss)
        val_ssim_scores.append(val_total_ssim / len(val_loader))
            
        if epoch % log_frequency == 0:
            logging.info(f"Epoch [{epoch}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Train SSIM: {train_total_ssim / len(train_loader):.4f}, Val SSIM: {val_total_ssim / len(val_loader):.4f}")
    
    end_time = time.time()
    logging.info(f"Training took {(end_time - start_time) / 60:.2f} minutes")
        
    train_metrices = [
        train_vq_losses, train_recon_losses,
        train_total_losses, train_ssim_scores
    ]
    val_metrices = [
        val_vq_losses, val_recon_losses,
        val_total_losses, val_ssim_scores
    ]
    metric_names = [
        "Vector Quantization Loss", "Reconstruction Loss",
        "Total Loss", "Structural Similarity Index"
    ]
    
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    axs = axs.flatten()
    
    for i in range(4):
        axs[i].plot(train_metrices[i], label='Train')
        axs[i].plot(val_metrices[i], label='Validation')
        axs[i].set_title(metric_names[i])
        axs[i].set_xlabel('Epoch')
        axs[i].set_ylabel(metric_names[i])
        axs[i].legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(log_dir, 'metrices_plot.png'))
    plt.close()
    
    # Save model
    torch.save(model.state_dict(), os.path.join(log_dir, 'model.pth'))
