import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from modules import StyleGAN2Generator, StyleGAN2Discriminator
from dataset import ADNIDataset
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from umap import UMAP

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters - match to train.py
z_dim = 128
w_dim = 128
num_mapping_layers = 3
mapping_dropout = 0.0
label_dim = 2
num_layers = 5
ngf = 64
ndf = 64
batch_size = 32

# Init dataset and loader
dataset = ADNIDataset(root_dir="/home/groups/comp3710/ADNI/AD_NC", split="train")
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)

# Init generator and discriminator
generator = StyleGAN2Generator(z_dim, w_dim, num_mapping_layers, mapping_dropout, label_dim, num_layers, ngf).to(device)
discriminator = StyleGAN2Discriminator(image_size=(256, 240), num_channels=1, ndf=ndf, num_layers=num_layers).to(device)

# Load trained model
checkpoint = torch.load("checkpoints/stylegan2_checkpoint_epoch_100.pth")  # Adjust epoch number as needed
generator.load_state_dict(checkpoint['gen_state_dict'])
discriminator.load_state_dict(checkpoint['discrim_state_dict'])

generator.eval()
discriminator.eval()

def get_embeddings(model, dataloader, is_real=True):
    all_embeddings = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            if not is_real:
                z = torch.randn(images.size(0), z_dim).to(device)
                images = generator(z, labels.to(device))
            
            embeddings = discriminator(images, feature_output=True)
            all_embeddings.append(embeddings.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    return np.concatenate(all_embeddings), np.array(all_labels)

# Get embeddings for real and fake images
real_embeddings, real_labels = get_embeddings(discriminator, dataloader, is_real=True)
fake_embeddings, fake_labels = get_embeddings(discriminator, dataloader, is_real=False)

# Combine embeddings and labels
combined_embeddings = np.vstack([real_embeddings, fake_embeddings])
combined_labels = np.concatenate([real_labels, fake_labels])
is_real = np.concatenate([np.ones(len(real_embeddings)), np.zeros(len(fake_embeddings))])

# Normalise embeddings
scaler = StandardScaler()
normalized_embeddings = scaler.fit_transform(combined_embeddings)

# Perform UMAP dimensionality reduction
umap_model = UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
umap_embeddings = umap_model.fit_transform(normalized_embeddings)

# Plot UMAP embeddings
plt.figure(figsize=(12, 10))
scatter = plt.scatter(umap_embeddings[:, 0], umap_embeddings[:, 1], c=combined_labels, cmap='coolwarm', alpha=0.7)
plt.colorbar(scatter)
plt.title("UMAP Visualisation of StyleGAN2 Embeddings (AD vs NC)")
plt.savefig("results/umap_ad_nc.png")
plt.close()

plt.figure(figsize=(12, 10))
scatter = plt.scatter(umap_embeddings[:, 0], umap_embeddings[:, 1], c=is_real, cmap='viridis', alpha=0.7)
plt.colorbar(scatter)
plt.title("UMAP Visualisation of StyleGAN2 Embeddings (Real vs Fake)")
plt.savefig("results/umap_real_fake.png")
plt.close()

print("UMAP visualisation completed.")