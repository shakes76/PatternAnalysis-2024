"""
REFERENCES:

(1) This code was developed with assistance from the Claude AI assistant,
    created by Anthropic, PBC. Claude provided guidance on implementing
    StyleGAN2 architecture and training procedures.

    Date of assistance: 8/10/2024
    Claude version: Claude-3.5 Sonnet
    For more information about Claude: https://www.anthropic.com

(2) GitHub Repository: stylegan2-ada-pytorch
    URL: https://github.com/NVlabs/stylegan2-ada-pytorch/tree/main
    Accessed on: 29/09/24 - 8/10/24
    
(3) Karras, T., Laine, S., Aittala, M., Hellsten, J., Lehtinen, J., & Aila, T. (2020). 
    Analyzing and improving the image quality of StyleGAN.
    arXiv. https://arxiv.org/abs/1912.04958

(4) Karras, T., Laine, S., & Aila, T. (2019).
    A Style-Based Generator Architecture for Generative Adversarial Networks.
    arXiv. https://arxiv.org/abs/1812.04948
"""


import torch
import torch.nn as nn
import torch.optim as optim
import torch.amp as amp
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from modules import StyleGAN2Generator, StyleGAN2Discriminator
from dataset import ADNIDataset
import os
import gc
import numpy as np
import matplotlib.pyplot as plt
from torch.nn.utils import clip_grad_norm_

# Hyperparams - mostly following StyleGAN2 paper, adjusted for smaller network
z_dim = 128 # Latent dims (z: input, w: intermediate)
w_dim = 128  
num_mapping_layers = 3
mapping_dropout = 0.0
label_dim = 2  # AD and NC
num_layers = 5
ngf = 64 # Num generator features
ndf = 64 # Num discriminator features
batch_size = 16
num_epochs = 100
lr = 0.0002
beta1 = 0.5
beta2 = 0.999  # Adam betas
r1_gamma = 10.0  # R1 regularisation weight
d_reg_interval = 16  # Discrim regularisation interval
max_grad_norm = 1.0  # Maximum norm for gradient clipping
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create results dir
for dir in ["results/AD", "results/NC", "results/UMAP", "checkpoints"]:
    os.makedirs(dir, exist_ok=True)

# Init dataset and loader
dataset = ADNIDataset(root_dir="/home/groups/comp3710/ADNI/AD_NC", split="train") # CHANGE DIR
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=1)

# Init generator and discriminator
generator = StyleGAN2Generator(z_dim, w_dim, num_mapping_layers, mapping_dropout, label_dim, num_layers, ngf).to(device)
discriminator = StyleGAN2Discriminator(image_size=(256, 240), num_channels=1, ndf=ndf, num_layers=num_layers).to(device)

# Init optimisers
g_optim = optim.Adam(generator.parameters(), lr=lr, betas=(beta1, beta2))
d_optim = optim.Adam(discriminator.parameters(), lr=lr, betas=(beta1, beta2))

# Loss function
criterion = nn.BCEWithLogitsLoss()

# Init GradScaler
scaler = amp.GradScaler()

# Helper funcs
def requires_grad(model, flag=True):
    """Enable/disable gradients for model params"""
    for p in model.parameters():
        p.requires_grad = flag

def d_r1_loss(real_pred, real_img):
    """R1 regularisation for discriminator"""
    grad_real, = torch.autograd.grad(outputs=real_pred.sum(), inputs=real_img, create_graph=True)
    return grad_real.pow(2).reshape(grad_real.shape[0], -1).sum(1).mean()

def save_images(generator, z, labels, epoch, batch=None):
    """Save generated images, categorise AD and NC"""
    with torch.no_grad(), amp.autocast(device_type='cuda'):
        fakes = generator(z, labels)
        for i, (img, lbl) in enumerate(zip(fakes, labels)):
            label_str = "AD" if lbl == 0 else "NC"
            filename = f"results/{label_str}/fake_e{epoch+1}_" + (f"b{batch}_" if batch else "") + f"s{i+1}.png"
            save_image(img.float(), filename, normalize=True)  # Ensure float32 for save_image

def plot_losses(d_losses, g_losses):
    plt.figure(figsize=(10, 5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(g_losses, label="Generator", alpha=0.5)
    plt.plot(d_losses, label="Discriminator", alpha=0.5)
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig("results/loss_plot.png")
    plt.close()

def clear_cache():
    torch.cuda.empty_cache()
    gc.collect()

# Training loop
total_batches = len(dataloader)
print_interval = 50
save_interval = 5 # Every 5 epochs save and gen progress images
fixed_z = torch.randn(16, z_dim).to(device)
fixed_labels = torch.cat([torch.zeros(8), torch.ones(8)], dim=0).long().to(device)
d_losses = []
g_losses = []

for epoch in range(num_epochs):
    clear_cache()
    for i, (real_images, labels) in enumerate(dataloader):
        real_images, labels = real_images.to(device), labels.to(device)
        
        ### Train Discriminator ###
        requires_grad(generator, False)
        requires_grad(discriminator, True)
        
        with amp.autocast(device_type='cuda'):
            z = torch.randn(real_images.size(0), z_dim).to(device)
            fake_images = generator(z, labels)
            fake_output = discriminator(fake_images.detach())
            real_output = discriminator(real_images)
            d_loss = criterion(fake_output, torch.zeros_like(fake_output)) + criterion(real_output, torch.ones_like(real_output))
            d_losses.append(d_loss.item())
            
        d_optim.zero_grad()
        scaler.scale(d_loss).backward()
        scaler.unscale_(d_optim)
        clip_grad_norm_(discriminator.parameters(), max_grad_norm)
        scaler.step(d_optim)
        scaler.update()
        
        # Discriminator R1 regularisation 
        if i % d_reg_interval == 0:
            real_images.requires_grad = True
            with amp.autocast(device_type='cuda'):
                real_pred = discriminator(real_images)
                r1_loss = d_r1_loss(real_pred, real_images)
            d_optim.zero_grad()
            scaler.scale(r1_gamma / 2 * r1_loss * d_reg_interval).backward()
            scaler.unscale_(d_optim)
            clip_grad_norm_(discriminator.parameters(), max_grad_norm)
            scaler.step(d_optim)
            scaler.update()
        
        ### Train Generator ###
        requires_grad(generator, True)
        requires_grad(discriminator, False)
        
        with amp.autocast(device_type='cuda'):
            z = torch.randn(real_images.size(0), z_dim).to(device)
            fake_images = generator(z, labels)
            fake_pred = discriminator(fake_images)
            g_loss = criterion(fake_pred, torch.ones_like(fake_pred))
            g_losses.append(g_loss.item())
        
        g_optim.zero_grad()
        scaler.scale(g_loss).backward()
        scaler.unscale_(g_optim)
        clip_grad_norm_(generator.parameters(), max_grad_norm)
        scaler.step(g_optim)
        scaler.update()

        # Print losses and check for NaN
        if i % print_interval == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}] Batch [{i+1}/{total_batches}] "
                  f"D_loss: {d_loss.item():.4f} G_loss: {g_loss.item():.4f}")
            
        if torch.isnan(d_loss) or torch.isnan(g_loss):
            print(f"NaN loss detected at Epoch {epoch+1}, Batch {i+1}. Break.")
            break
    
    print(f"Epoch [{epoch+1}/{num_epochs}] completed")
    
    # Save checkpoints and generate images every 5 epochs
    if (epoch + 1) % save_interval == 0:
        save_images(generator, fixed_z, fixed_labels, epoch)
        torch.save({
            'gen_state_dict': generator.state_dict(),
            'discrim_state_dict': discriminator.state_dict(),
        }, f"checkpoints/stylegan2_checkpoint_epoch_{epoch+1}.pth")

plot_losses(d_losses, g_losses)

print("Training complete!")