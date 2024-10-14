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

# Hyperparams - mostly following StyleGAN2 paper
z_dim = 256 # Latent dims (z: input, w: intermediate)
w_dim = 256  
num_mapping_layers = 4
mapping_dropout = 0.1
label_dim = 2  # AD and NC
num_layers = 7
ngf = 256 # Num generator features
ndf = 256 # Num disciminator features
batch_size = 4
num_epochs = 100
lr = 0.0001
beta1 = 0.0
beta2 = 0.99  # Adam betas
r1_gamma = 10.0  # R1 regularisation weight
pl_weight = 2.0  # Path length regularisation weight
pl_decay = 0.01  # Path length decay
d_reg_interval = 16  # Discrim regularisation interval
g_reg_interval = 4  # Generator regularisation interval
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create restuls dir
for dir in ["results/AD", "results/NC", "results/UMAP", "checkpoints"]:
    os.makedirs(dir, exist_ok=True)

# Init dataset and loader
dataset = ADNIDataset(root_dir="/home/groups/comp3710/ADNI/AD_NC", split="train") # CHANGE DIR
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

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

def g_path_regularise(fake_img, latents, mean_path_length, decay=0.01):
    """Path length regularisation for generator"""
    noise = torch.randn_like(fake_img) / np.sqrt(fake_img.shape[2] * fake_img.shape[3])
    grad, = torch.autograd.grad(outputs=(fake_img * noise).sum(), inputs=latents, create_graph=True)
    path_lengths = torch.sqrt(grad.pow(2).sum(1))
    path_mean = mean_path_length + decay * (path_lengths.mean() - mean_path_length)
    path_penalty = (path_lengths - path_mean).pow(2).mean()
    
    return path_penalty, path_mean.detach()

def save_images(generator, z, labels, epoch, batch=None):
    """Save generated images, categorise AD and NC"""
    with torch.no_grad(), amp.autocast(device_type='cuda'):
        fakes = generator(z, labels)
        for i, (img, lbl) in enumerate(zip(fakes, labels)):
            label_str = "AD" if lbl == 0 else "NC"
            filename = f"results/{label_str}/fake_e{epoch+1}_" + (f"b{batch}_" if batch else "") + f"s{i+1}.png"
            save_image(img.float(), filename, normalize=True)  # Ensure float32 for save_image

def plot_umap(generator, discriminator, dataloader, epoch):
    """Generate UMAP plot to visualise latent space"""
    generator.eval()
    discriminator.eval()
    real_feats, fake_feats, labels = [], [], []
    
    with torch.no_grad(), amp.autocast(device_type='cuda'):
        for real_imgs, lbls in dataloader:
            real_imgs, lbls = real_imgs.to(device), lbls.to(device)
            real_feats.append(discriminator(real_imgs, feature_output=True).float().cpu())
            labels.extend(lbls.cpu().numpy())
            z = torch.randn(real_imgs.size(0), z_dim).to(device)
            fake_imgs = generator(z, lbls)
            fake_feats.append(discriminator(fake_imgs, feature_output=True).float().cpu())
    
    real_feats = torch.cat(real_feats).numpy()
    fake_feats = torch.cat(fake_feats).numpy()
    labels = np.array(labels)
    combined_feats = np.vstack([real_feats, fake_feats])
    combined_labels = np.concatenate([labels, labels])
    is_real = np.concatenate([np.ones(len(real_feats)), np.zeros(len(fake_feats))])
    
    normalised_feats = StandardScaler().fit_transform(combined_feats)
    embeddings = UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42).fit_transform(normalised_feats)
    
    for plot_type, c in [("labels", combined_labels), ("real_vs_fake", is_real)]:
        plt.figure(figsize=(12, 10))
        plt.scatter(embeddings[:, 0], embeddings[:, 1], c=c, cmap='coolwarm' if plot_type == "labels" else 'viridis', alpha=0.7)
        plt.colorbar()
        plt.title(f"UMAP - {'AD vs NC' if plot_type == 'labels' else 'Real vs Fake'} (Epoch {epoch+1})")
        plt.savefig(f"results/UMAP/umap_{plot_type}_e{epoch+1}.png")
        plt.close()

    # Clear memory
    del real_feats, fake_feats, combined_feats, normalised_feats, embeddings
    clear_cache()

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
print_interval = 1
save_interval = 5 # Every 5 epochs save and gen progress images
fixed_z = {
    'AD': torch.randn(8, z_dim).to(device),
    'NC': torch.randn(8, z_dim).to(device)
}
fixed_labels = {
    'AD': torch.zeros(8, dtype=torch.long).to(device),
    'NC': torch.ones(8, dtype=torch.long).to(device)
}
mean_path_length = 0
d_losses = []
g_losses = []

for epoch in range(num_epochs):
    clear_cache()
    for i, (real_images, labels) in enumerate(dataloader):
        real_images, labels = real_images.to(device), labels.to(device)
        
        ### Train Discriminator ###
        requires_grad(generator, False)
        requires_grad(discriminator, True)
        
        with amp.autocast(device_type='cuda'): # Use mixed precision
            # Generate fake images
            z = torch.randn(real_images.size(0), z_dim).to(device)
            fake_images = generator(z, labels)
            fake_output = discriminator(fake_images.detach()) # Want these predictions close to 0 for Discrim
            real_output = discriminator(real_images) # Want these predictions clsoe to 1
            d_loss = criterion(fake_output, torch.zeros_like(fake_output)) + criterion(real_output, torch.ones_like(real_output))
            d_losses.append(d_loss.item())
            
        d_optim.zero_grad()
        scaler.scale(d_loss).backward()
        scaler.step(d_optim)
        scaler.update()
        
        # Discrimnator R1 regularisation 
        if i % d_reg_interval == 0:
            real_images.requires_grad = True
            with amp.autocast(device_type='cuda'):
                real_pred = discriminator(real_images)
                r1_loss = d_r1_loss(real_pred, real_images)
            d_optim.zero_grad()
            scaler.scale(r1_gamma / 2 * r1_loss * d_reg_interval).backward()
            scaler.step(d_optim)
            scaler.update()
        
        ### Train Generator ###
        requires_grad(generator, True)
        requires_grad(discriminator, False)
        
        with amp.autocast(device_type='cuda'):
            z = torch.randn(real_images.size(0), z_dim).to(device)
            fake_images = generator(z, labels)
            fake_pred = discriminator(fake_images) # Want this close to 1 for Gen
            # Generator loss
            g_loss = criterion(fake_pred, torch.ones_like(fake_pred))
            g_losses.append(g_loss.item())
        
        g_optim.zero_grad()
        scaler.scale(g_loss).backward()
        scaler.step(g_optim)
        scaler.update()
        
        # Generator path length regularisation
        if i % g_reg_interval == 0:
            with amp.autocast(device_type='cuda'):
                fake_images, latents = generator(z, labels, return_latents=True)
                path_loss, mean_path_length = g_path_regularise(fake_images, latents, mean_path_length, pl_decay)
            g_optim.zero_grad()
            scaler.scale(pl_weight * path_loss * g_reg_interval).backward()
            scaler.step(g_optim)
            scaler.update()

        # Print losses
        if i % print_interval == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}] Batch [{i+1}/{total_batches}] "
                  f"D_loss: {d_loss.item():.4f} G_loss: {g_loss.item():.4f}")

        # # Save progress images
        # if i % save_interval == 0:
        #     save_images(generator, fixed_z['AD'], fixed_labels['AD'], epoch, i)
        #     save_images(generator, fixed_z['NC'], fixed_labels['NC'], epoch, i)
    
    print(f"Epoch [{epoch+1}/{num_epochs}] completed")
    
    # Save checkpoints and generate images every 5 epochs
    if (epoch + 1) % save_interval == 0:
        save_images(generator, fixed_z['AD'], fixed_labels['AD'], epoch)
        save_images(generator, fixed_z['NC'], fixed_labels['NC'], epoch)
        torch.save({
            'gen_state_dict': generator.state_dict(),
            'discrim_state_dict': discriminator.state_dict(),
        }, f"checkpoints/stylegan2_checkpoint_epoch_{epoch+1}.pth")

    # Plot UMAP only after 50th epoch and at the end
    if epoch + 1 == 50 or epoch + 1 == num_epochs:
        print("Importing UMAP")
        from umap import UMAP
        import numpy as np
        import matplotlib.pyplot as plt
        from sklearn.preprocessing import StandardScaler
        print(f"Generating UMAP plot for epoch {epoch+1}")
        plot_umap(generator, discriminator, dataloader, epoch)


    
plot_losses(d_losses, g_losses)

print("Training complete!")