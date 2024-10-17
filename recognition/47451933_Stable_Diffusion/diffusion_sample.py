import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from dataset import *

device = torch.device('cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))

data = Dataset()

noise_level = 0.5

# Function to display images
def display_images(images, num_images=5):
    images = images[:num_images].detach().cpu().numpy()  # Select a few images
    images = (images + 1) / 2  # Rescale images from [-1, 1] to [0, 1] for display

    fig, axs = plt.subplots(1, num_images, figsize=(15, 5))
    for i in range(num_images):
        img = np.transpose(images[i], (1, 2, 0))  # Convert from (C, H, W) to (H, W, C)
        axs[i].imshow(img)
        axs[i].axis('off')
    plt.show()

data_loader = data.get_train()

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np

# VAE Encoder for mapping input images to latent space
class VAEDecoder(nn.Module):
    def __init__(self, latent_dim=256):
        super(VAEDecoder, self).__init__()
        self.decoder = nn.Sequential(
            #nn.Linear(latent_dim, 512 * 4 * 4),  # Update for the new size
            #nn.ReLU(),
            #nn.Unflatten(1, (512, 4, 4)),  # Update for the new size
            #nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),  # 4x4 -> 8x8
            #nn.BatchNorm2d(256),
            #nn.ReLU(),
            nn.ConvTranspose2d(latent_dim, 128, kernel_size=4, stride=2, padding=1),  # 8x8 -> 16x16
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # 16x16 -> 32x32
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1),  # 32x32 -> 64x64
            nn.Tanh()
        )

    def forward(self, z):
        return self.decoder(z)

# VAE Decoder for reconstructing images from latent space
class VAEEncoder(nn.Module):
    def __init__(self, latent_dim=256):
        super(VAEEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1),  # 64x64 -> 32x32
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # 32x32 -> 16x16
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, latent_dim, kernel_size=4, stride=2, padding=1),  # 16x16 -> 8x8
            nn.BatchNorm2d(latent_dim),
            #nn.ReLU(),
            #nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),  # 8x8 -> 4x4
            #nn.BatchNorm2d(512),
            #nn.ReLU(),
            #nn.Flatten()
        )
        self.fc_mu = nn.Linear(512 * 4 * 4, latent_dim)  # Update for the new size
        self.fc_logvar = nn.Linear(512 * 4 * 4, latent_dim)  # Update for the new size

    def forward(self, x):
        h = self.encoder(x)
        #mu = self.fc_mu(h)
        #logvar = self.fc_logvar(h)
        return h#mu, logvar

# Reparameterization trick for VAE
def reparameterize(mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std

# UNet for denoising, now accepts label embeddings
class UNet(nn.Module):
    def __init__(self, latent_dim=256, num_classes=2):
        super(UNet, self).__init__()
        self.label_embedding = nn.Embedding(num_classes, latent_dim)

        # Encoder part (downsampling)
        self.enc1 = nn.Sequential(
            nn.Conv2d(latent_dim, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=2),  # Downsample
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.enc3 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1, stride=2),  # Downsample
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU()
        )

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(1024, 512, kernel_size=3, padding=1),
            nn.ReLU()
        )

        # Decoder part (upsampling)
        self.dec3 = nn.Sequential(
            nn.ConvTranspose2d(512, 512, kernel_size=4, stride=2, padding=1),  # Upsample
            nn.ReLU(),
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(256, 256, kernel_size=4, stride=2, padding=1),  # Upsample
            nn.ReLU(),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.dec1 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, latent_dim, kernel_size=3, padding=1)
        )

    def forward(self, x, labels):
        # Get the label embeddings
        label_emb = self.label_embedding(labels).unsqueeze(-1).unsqueeze(-1)
        # Concatenate the label embeddings with the input
        #x = torch.cat((x, label_emb.repeat(1, 1, x.size(2), x.size(3))), dim=1)
        
        # Pass through the encoder
        enc1_out = self.enc1(x)
        enc2_out = self.enc2(enc1_out)
        enc3_out = self.enc3(enc2_out)
        
        # Pass through the bottleneck
        bottleneck_out = self.bottleneck(enc3_out)
        
        # Pass through the decoder with skip connections
        dec3 = self.crop_and_add(self.dec3(bottleneck_out), enc2_out)
        dec2 = self.crop_and_add(self.dec2(dec3), enc1_out)
        dec1 = self.crop_and_add(self.dec1(dec2), x)

        return dec1

    def crop_and_add(self, upsampled, skip_connection):
        """
        Crops the upsampled tensor to match the size of the skip connection tensor.
        """
        _, _, h, w = skip_connection.size()
        upsampled = upsampled[:, :, :h, :w]
        return upsampled + skip_connection


# Function to add noise at each step
def add_noise(x, noise_level):
    noise = torch.randn_like(x)
    return x + noise * noise_level, noise

# Training setup
latent_dim = 100
num_classes = 2  # Update this according to your dataset
vae_encoder = VAEEncoder(latent_dim=latent_dim)
vae_decoder = VAEDecoder(latent_dim=latent_dim)
unet = UNet(latent_dim=latent_dim, num_classes=num_classes)

# Loss functions and optimizers
vae_criterion = nn.MSELoss()
diffusion_criterion = nn.MSELoss()
vae_optimizer = optim.Adam(list(vae_encoder.parameters()) + list(vae_decoder.parameters()), lr=1e-4)
unet_optimizer = optim.Adam(unet.parameters(), lr=0.001)

def weights_init(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
vae_encoder.apply(weights_init)
vae_decoder.apply(weights_init)
unet.apply(weights_init)

# Assume you have a DataLoader `data_loader` that provides (images, labels)
num_epochs = 4
for epoch in range(num_epochs):
    for images, labels in tqdm(data_loader):
        images = images.to('cuda' if torch.cuda.is_available() else 'cpu')
        labels = labels.to(images.device)
        vae_encoder = vae_encoder.to(images.device)
        vae_decoder = vae_decoder.to(images.device)
        
        # VAE Forward pass
        z = vae_encoder(images)

        #z = reparameterize(mu, logvar)
        reconstructed_images = vae_decoder(z)
        
        # VAE Loss
        recon_loss = vae_criterion(reconstructed_images, images)
        #kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        vae_loss = recon_loss #+ 0.05 * kl_div
        
        # Backpropagation for VAE
        vae_optimizer.zero_grad()
        vae_loss.backward()
        vae_optimizer.step()

    print(f"Epoch [{epoch+1}/{num_epochs}], VAE Loss: {vae_loss.item()}")
    if epoch % 20 == 0:
        with torch.no_grad():
            # Decode denoised latent to generate images
            generated_images = vae_decoder(z).clamp(-1, 1)
            display_images(generated_images, num_images=5)

torch.save(vae_encoder, "models/encoder.model")
torch.save(vae_decoder, "models/decoder.model")

vae_encoder = torch.load("models/encoder.model", weights_only=False)
vae_encoder.eval()

vae_decoder = torch.load("models/decoder.model", weights_only=False)
vae_decoder.eval()

for epoch in range(num_epochs):
    for images, labels in tqdm(data_loader):
        images = images.to('cuda' if torch.cuda.is_available() else 'cpu')
        labels = labels.to(images.device)
        vae_encoder.eval()
        vae_decoder.eval()
        vae_encoder = vae_encoder.to(images.device)
        vae_decoder = vae_decoder.to(images.device)
        unet = unet.to(images.device)
        
        # VAE Forward pass
        z = vae_encoder(images)
        #z = reparameterize(mu, logvar)
        
        # Detach latent variables to avoid retaining the graph
        z = z.detach()

        # Add noise to latent space for diffusion training
        noisy_latent, noise = add_noise(z, noise_level=noise_level)
        
        # Predict the noise using the UNet
        predicted_noise = unet(noisy_latent, labels)  # Pass labels to UNet
        
        # Diffusion Loss: Predict the added noise
        diffusion_loss = diffusion_criterion(predicted_noise, noise)
        
        # Backpropagation for diffusion
        unet_optimizer.zero_grad()
        diffusion_loss.backward()
        unet_optimizer.step()
        
        # Denoise the latent representation using the predicted noise
        denoised_latent = noisy_latent - predicted_noise

    # Print losses and display samples every 10 epochs
    print(f"Epoch [{epoch+1}/{num_epochs}], VAE Loss: {vae_loss.item()}, Diffusion Loss: {diffusion_loss.item()}")
    if epoch % 2 == 0:
        with torch.no_grad():
            # Decode denoised latent to generate images
            generated_images = vae_decoder(denoised_latent).clamp(-1, 1)
            display_images(generated_images, num_images=5)
            generated_images = vae_decoder(predicted_noise).clamp(-1, 1)
            display_images(generated_images, num_images=5)

print("Training completed!")

def generate_sample(label, model, vae_decoder, num_samples=1, noise_level=noise_level):
    model.eval()
    vae_decoder.eval()
    
    with torch.no_grad():
        # Create a random latent vector
        z = torch.randn(num_samples, latent_dim, 25, 25).to('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Add noise to the latent space
        noisy_latent, noise = add_noise(z, noise_level=noise_level)
        
        # Predict the noise using the UNet (conditioning on the provided label)
        label_tensor = torch.tensor([label] * num_samples).to(noisy_latent.device)
        predicted_noise = model(noisy_latent, label_tensor)
        
        # Denoise the latent representation using the predicted noise
        denoised_latent = noisy_latent - predicted_noise
        
        # Decode the denoised latent representation to generate images
        output_images = vae_decoder(denoised_latent).clamp(-1, 1)
        
    return output_images

# Example usage to generate images for class label 0
sample_images = generate_sample(0, unet, vae_decoder, num_samples=5)

# Function to display the generated images
def display_sample_images(images):
    images = (images + 1) / 2  # Normalize from [-1, 1] to [0, 1]
    images = images.cpu().permute(0, 2, 3, 1).numpy()
    
    plt.figure(figsize=(10, 5))
    for i, img in enumerate(images):
        plt.subplot(1, len(images), i + 1)
        plt.imshow(img)
        plt.axis('off')
    plt.show()

# Display the generated images
display_sample_images(sample_images)

