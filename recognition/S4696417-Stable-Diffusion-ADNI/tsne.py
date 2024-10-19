"""
This file is used to generate the TSNE embeddings for the generated latent space
This will help to visualise the separation between the ADNI class samples.

Author: Liam O'Sullivan
"""

import os
import torch
import numpy as np
from sklearn.manifold import TSNE
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('Agg')

# Set up parameters
IMAGE_SIZE = 256
SAMPLE_SIZE = 10000
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the trained AD/NC diffusion models
print("Loading trained models...")
os.chdir('recognition/S4696417-Stable-Diffusion-ADNI')
ad_path = 'checkpoints/Diffusion/ADNI_AD_diffusion_e500_im256.pt'
nc_path = 'checkpoints/Diffusion/ADNI_NC_diffusion_e500_im256.pt'

ad_model = torch.load(ad_path).to(device)
nc_model = torch.load(nc_path).to(device)
ad_model.eval()
nc_model.eval()


def get_latent_representations(model, num_samples):
    """
    Sample new images from the diffusion model and get their latent representations

    Args:
        model: Trained diffusion model
        num_samples: Number of samples to generate
    """
    with torch.no_grad():

        # Generate samples using the diffusion model
        latent_samples = model.sample(num_samples, device=device, log=False, latents=True)
        return latent_samples.cpu().numpy().reshape(num_samples, -1)


# Get the latent representations for AD and NC samples
print("Generating latent representations for AD samples...")
ad_latents = get_latent_representations(ad_model, SAMPLE_SIZE)
print("Generating latent representations for NC samples...")
nc_latents = get_latent_representations(nc_model, SAMPLE_SIZE)

latents = np.concatenate([ad_latents, nc_latents])
labels = np.array(['AD'] * SAMPLE_SIZE + ['NC'] * SAMPLE_SIZE)

print(f"Total number of samples: {len(latents)}")
print(f"Shape of latents: {latents.shape}")

# Apply t-SNE
tsne = TSNE(n_components=2, random_state=0)
latents_2d = tsne.fit_transform(latents)

print(f"Shape of t-SNE output: {latents_2d.shape}")

# Create a custom plot
plt.figure(figsize=(12, 10))
ad_mask = labels == 'AD'
nc_mask = labels == 'NC'

plt.scatter(latents_2d[ad_mask, 0], latents_2d[ad_mask, 1], c='red', alpha=0.6, label='AD')
plt.scatter(latents_2d[nc_mask, 0], latents_2d[nc_mask, 1], c='blue', alpha=0.6, label='NC')

plt.title("t-SNE visualisation of VAE latent space", fontsize=16)
plt.xlabel("t-SNE 1", fontsize=12)
plt.ylabel("t-SNE 2", fontsize=12)
plt.legend(fontsize=10)

# Add some padding to the plot
plt.tight_layout()

# Save the plot to a file
os.makedirs('visualisations', exist_ok=True)
plot_path = 'visualisations/tsne_plot.png'
plt.savefig(plot_path, dpi=300)
plt.close()
