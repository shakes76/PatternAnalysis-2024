import torch, wandb
import numpy as np
from sklearn.manifold import TSNE
from torchvision import transforms
from dataset import get_dataloader
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Set up parameters
IMAGE_SIZE = 128
BATCH_SIZE = 32
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the trained VAE model
os.chdir('recognition/S4696417-Stable-Diffusion-ADNI')
vae_path = f'checkpoints/VAE/ADNI-vae_e80_b16_im{IMAGE_SIZE}.pt'
vae = torch.load(vae_path, map_location=device)
vae.eval()

# Set up data loading
image_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
])

ad_loader, _ = get_dataloader('data/train/AD', batch_size=BATCH_SIZE, transform=image_transform)
nc_loader, _ = get_dataloader('data/train/NC', batch_size=BATCH_SIZE, transform=image_transform)

def get_latent_representations(loader, label):
    latents = []
    labels = []
    with torch.no_grad():
        for images, _ in loader:
            images = images.to(device)
            mu, _ = vae.encode(images)
            # Flatten the latent representations
            mu_flat = mu.view(mu.size(0), -1).cpu().numpy()
            latents.append(mu_flat)
            labels.extend([label] * images.shape[0])
    return np.concatenate(latents), np.array(labels)

# Get latent representations for both classes
ad_latents, ad_labels = get_latent_representations(ad_loader, 'AD')
nc_latents, nc_labels = get_latent_representations(nc_loader, 'NC')

# Combine latents and labels
latents = np.concatenate([ad_latents, nc_latents])
labels = np.concatenate([ad_labels, nc_labels])

print(f"Total number of samples: {len(latents)}")
print(f"Shape of latents: {latents.shape}")
print(f"Unique labels: {np.unique(labels)}")

# Apply t-SNE
tsne = TSNE(n_components=2, random_state=42)
latents_2d = tsne.fit_transform(latents)

print(f"Shape of t-SNE output: {latents_2d.shape}")

# Create a custom plot
plt.figure(figsize=(12, 10))
ad_mask = labels == 'AD'
nc_mask = labels == 'NC'

plt.scatter(latents_2d[ad_mask, 0], latents_2d[ad_mask, 1], c='red', alpha=0.6, label='AD')
plt.scatter(latents_2d[nc_mask, 0], latents_2d[nc_mask, 1], c='blue', alpha=0.6, label='NC')

plt.title("t-SNE visualization of VAE latent space", fontsize=16)
plt.xlabel("t-SNE 1", fontsize=12)
plt.ylabel("t-SNE 2", fontsize=12)
plt.legend(fontsize=10)

# Add some padding to the plot
plt.tight_layout()

# Save the plot to a file
os.makedirs('visualizations', exist_ok=True)
plot_path = 'visualizations/tsne_plot.png'
plt.savefig(plot_path, dpi=300)
plt.close()
