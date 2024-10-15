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
    if device.type == 'cpu':
        dir = "recognition/VQVAE_47323254"
        log_dir = f'{dir}/logs/log_local'
    else:
        dir = "/home/Student/s4732325/project2"
        log_dir = f'{dir}/logs/v2'
    data_dir = f'{dir}/HipMRI_study_keras_slices_data'
    os.makedirs(log_dir, exist_ok=True)

    # Configuration
    num_epochs = 40
    batch_size = 64
    learning_rate = 1e-3
    num_samples = None

    # Set up logging
    logging.basicConfig(filename=os.path.join(log_dir, 'training.log'), level=logging.INFO, 
                        format='%(asctime)s - %(levelname)s - %(message)s')

    train_transform = transforms.Compose([
        transforms.CenterCrop((256, 128)),
        # transforms.Resize((256, 256)),
        transforms.RandomVerticalFlip(),
        transforms.RandomHorizontalFlip(),
        transforms.GaussianBlur(3, sigma=(0.1, 0.5)),
        transforms.RandomAffine(degrees=0, translate=(0.0, 0.0), scale=(0.9, 1.1)), 
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    val_test_transform = transforms.Compose([
        transforms.CenterCrop((256, 128)),
        # transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    model = VQVAE(in_channels=1, hidden_channels=128, res_channels=32, nb_res_layers=2, 
                embed_dim=64, nb_entries=512, downscale_factor=4).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Dataset loader
    train_dataset = HipMRIDataset(data_dir + "/keras_slices_train", train_transform, num_samples=num_samples)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataset = HipMRIDataset(data_dir + "/keras_slices_validate", val_test_transform, num_samples=num_samples)
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
        if epoch % 10 == 0:
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
            
        logging.info(f"Epoch [{epoch}/{num_epochs}], \
                        Train Loss: {train_loss:.4f}, \
                        Val Loss: {val_loss:.4f}, \
                        Train SSIM: {train_total_ssim / len(train_loader):.4f}, \
                        Val SSIM: {val_total_ssim / len(val_loader):.4f}")
        
        
    train_metrices = [
        train_vq_losses,
        train_recon_losses,
        train_total_losses,
        train_ssim_scores
    ]
    val_metrices = [
        val_vq_losses,
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
        axs[i].plot(train_metrices[i], label='Train')
        axs[i].plot(val_metrices[i], label='Validation')
        axs[i].set_title(metric_names[i])
        axs[i].legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(log_dir, 'metrices_plot.png'))
    plt.close()
    
    # Save model
    torch.save(model.state_dict(), os.path.join(log_dir, 'model.pth'))
