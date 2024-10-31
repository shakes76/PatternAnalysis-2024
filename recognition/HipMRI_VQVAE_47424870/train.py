import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import os
from modules import VQVAE
from dataset import get_dataloader
from skimage.metrics import structural_similarity as ssim
import numpy as np

def calculate_ssim(img1, img2):
    img1 = img1.detach().permute(0, 2, 3, 1).cpu().numpy()
    img2 = img2.detach().permute(0, 2, 3, 1).cpu().numpy()
    ssim_vals = [
        ssim(img1[i], img2[i], data_range=1.0, channel_axis=-1, win_size=11)
        for i in range(img1.shape[0])
    ]
    return np.mean(ssim_vals)

def train_vqvae(num_epochs=38, batch_size=32, lr=1e-4, device='cpu'):
    input_dim = 1
    hidden_dim = 128
    res_channels = 32
    nb_res_layers = 2
    nb_levels = 3
    embed_dim = 128  
    nb_entries = 1024  
    scaling_rates = [8, 4, 2]

    vqvae = VQVAE(in_channels=input_dim, hidden_channels=hidden_dim, res_channels=res_channels, 
                  nb_res_layers=nb_res_layers, nb_levels=nb_levels, embed_dim=embed_dim, 
                  nb_entries=nb_entries, scaling_rates=scaling_rates)
    vqvae.to(device)

    optimizer = optim.Adam(vqvae.parameters(), lr=lr)
    mse_loss_fn = nn.MSELoss()

    current_dir = os.path.dirname(__file__)
    train_image_dir = os.path.join(current_dir, "keras_slices", "keras_slices", "keras_slices_train")
    val_image_dir = os.path.join(current_dir, "keras_slices", "keras_slices", "keras_slices_seg_validate")

    train_loader = get_dataloader(train_image_dir, batch_size=batch_size)
    val_loader = get_dataloader(val_image_dir, batch_size=batch_size)

    train_loss_list = []
    val_loss_list = []
    train_ssim_list = []
    val_ssim_list = []

    for epoch in range(num_epochs):
        vqvae.train()
        epoch_loss = 0
        epoch_ssim = 0

        for (batch, _) in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()

            reconstructed, diffs = vqvae(batch)
            
            # Compute combined loss using MSE only
            mse_loss = mse_loss_fn(reconstructed, batch)
            total_loss = mse_loss + sum(diffs)

            total_loss.backward()
            optimizer.step()
            epoch_loss += total_loss.item()
            
            batch_ssim = calculate_ssim(batch, reconstructed)
            epoch_ssim += batch_ssim

        avg_train_loss = epoch_loss / len(train_loader)
        avg_train_ssim = epoch_ssim / len(train_loader)
        train_loss_list.append(avg_train_loss)
        train_ssim_list.append(avg_train_ssim)

        vqvae.eval()
        epoch_val_loss = 0
        epoch_val_ssim = 0
        
        with torch.no_grad():
            for (val_batch, _) in val_loader:
                val_batch = val_batch.to(device)
                reconstructed_val, diffs_val = vqvae(val_batch)
                
                val_mse_loss = mse_loss_fn(reconstructed_val, val_batch)
                val_loss = val_mse_loss + sum(diffs_val)

                epoch_val_loss += val_loss.item()
                val_batch_ssim = calculate_ssim(val_batch, reconstructed_val)
                epoch_val_ssim += val_batch_ssim

        avg_val_loss = epoch_val_loss / len(val_loader)
        avg_val_ssim = epoch_val_ssim / len(val_loader)
        val_loss_list.append(avg_val_loss)
        val_ssim_list.append(avg_val_ssim)

        print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}")
        print(f"Epoch {epoch + 1}/{num_epochs}, Train SSIM: {avg_train_ssim:.4f}, Validation SSIM: {avg_val_ssim:.4f}")

    torch.save(vqvae.state_dict(), 'vqvae_model.pth')
    print("Model weights saved as 'vqvae_model.pth'.")

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, num_epochs + 1), train_loss_list, label='Train Loss')
    plt.plot(range(1, num_epochs + 1), val_loss_list, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(range(1, num_epochs + 1), train_ssim_list, label='Train SSIM')
    plt.plot(range(1, num_epochs + 1), val_ssim_list, label='Validation SSIM')
    plt.xlabel('Epoch')
    plt.ylabel('SSIM')
    plt.title('Training and Validation SSIM')
    plt.legend()

    plt.savefig("training_validation_metrics.png")

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_vqvae(device=device)
