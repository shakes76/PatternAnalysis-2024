import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import math
from PIL import Image

from dataset import *

device = torch.device('cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))

data = Dataset()

def get_timestep_embedding(time_step, embedding_dim):
        scaled_time_step = torch.tensor(time_step, dtype=torch.float32).view(1, 1)  # Reshape to (1, 1)
        
        # Calculate the sinusoidal embeddings
        half_dim = embedding_dim // 2
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -(math.log(10000.0) / half_dim))
        emb = scaled_time_step * emb.unsqueeze(0)

        # Apply sin and cos functions
        pos_enc = torch.zeros(1, embedding_dim, device=scaled_time_step.device)
        pos_enc[0, 0::2] = torch.sin(emb)  # sin for even indices
        pos_enc[0, 1::2] = torch.cos(emb)  # cos for odd indices
        
        return pos_enc

class BetaScheduler:
    def __init__(self, num_timesteps=1000, beta_start=1e-4, beta_end=0.1):
        self.num_timesteps = num_timesteps
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps)

    def get_beta(self, t):
        return self.betas[t]

    def get_alphas(self):
        return 1 - self.betas

    def get_alphas_cumprod(self):
        return torch.cumprod(self.get_alphas(), dim=0)

# Create the noise scheduler
num_timesteps = 10
beta_scheduler = BetaScheduler(num_timesteps=num_timesteps)
alphas_cumprod = beta_scheduler.get_alphas_cumprod()

# Function to display images
def display_images(images, num_images=5):
    images = images[:num_images].detach().cpu().numpy()  # Select a few images
    images = (images + 1) / 2  # Rescale images from [-1, 1] to [0, 1] for display

    fig, axs = plt.subplots(1, num_images, figsize=(15, 5))
    for i in range(num_images):
        img = np.transpose(images[i], (1, 2, 0))  # Convert from (C, H, W) to (H, W, C)
        axs[i].imshow(img, cmap = 'gray')
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
            nn.ReLU(),
            #nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),  # 8x8 -> 4x4
            #nn.BatchNorm2d(512),
            #nn.ReLU(),
            #nn.Flatten()
        )
        self.conv_mu = nn.Conv2d(latent_dim, latent_dim, kernel_size=1)
        self.conv_logvar = nn.Conv2d(latent_dim, latent_dim, kernel_size=1)

    def forward(self, x):
        h = self.encoder(x)
        mu = self.conv_mu(h)  # Get the mean using a 1x1 convolution
        logvar = self.conv_logvar(h)  # Get the log variance using a 1x1 convolution
        return mu, logvar

# Reparameterization trick for VAE
def reparameterize(mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std

# UNet for denoising, now accepts label embeddings
class UNet(nn.Module):
    def __init__(self, latent_dim=256, num_classes=2):
        super(UNet, self).__init__()
        #self.label_embedding = nn.Embedding(num_classes, latent_dim)

        # Encoder part (downsampling)
        self.enc1 = nn.Sequential(
            nn.Conv2d(latent_dim, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=2),  # Downsample
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        self.enc3 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1, stride=2),  # Downsample
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
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
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(256, 256, kernel_size=4, stride=2, padding=1),  # Upsample
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.dec1 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, latent_dim, kernel_size=3, padding=1)
        )

        self.time_embedding_layer = torch.nn.Linear(32, latent_dim)

    def forward(self, x, labels, t):
        # Get the label embeddings
        #label_emb = self.label_embedding(labels).unsqueeze(-1).unsqueeze(-1)
        # Concatenate the label embeddings with the input
        #x = torch.cat((x, label_emb.repeat(1, 1, x.size(2), x.size(3))), dim=1)
        t_embedding = get_timestep_embedding(t, 32).to(device)
        t_embedding = self.time_embedding_layer(t_embedding)
        t_embedding = t_embedding[:, :, None, None]
        
        # Pass through the encoder
        enc1_out = self.enc1(x+ t_embedding)
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
    
    def crop(self, upsampled, skip_connection):
        """
        Crops the upsampled tensor to match the size of the skip connection tensor.
        """
        _, _, h, w = skip_connection.size()
        upsampled = upsampled[:, :, :h, :w]
        return upsampled


def add_noise(x, t):
    noise = torch.randn_like(x).to(device)
    alpha_t = alphas_cumprod[t].to(x.device)
    return x + torch.sqrt(1 - alpha_t) * noise, noise

# Reverse diffusion using the noise prediction
def reverse_diffusion(noisy_latent, predicted_noise, t):
    alpha_t = alphas_cumprod[t].view(-1, 1, 1, 1).to(noisy_latent.device)
    beta_t = beta_scheduler.get_beta(t).view(-1, 1, 1, 1).to(noisy_latent.device)
    alpha_prev = alphas_cumprod[t - 1].view(-1, 1, 1, 1).to(noisy_latent.device) if t > 0 else torch.tensor(1.0).view(-1, 1, 1, 1).to(noisy_latent.device)

    # Predicted noise should be subtracted
    mean = (noisy_latent - beta_t * predicted_noise / torch.sqrt(1 - alpha_t)) / torch.sqrt(alpha_t)

    if t > 0:
        variance = beta_t * (1 - alpha_prev) / (1 - alpha_t)
        z = torch.randn_like(noisy_latent)
        return mean + torch.sqrt(variance) * z
    else:
        return mean

# Training setup
latent_dim = 100
num_classes = 2  # Update this according to your dataset
vae_encoder = VAEEncoder(latent_dim=latent_dim).to(device)
vae_decoder = VAEDecoder(latent_dim=latent_dim).to(device)
unet = UNet(latent_dim=latent_dim, num_classes=num_classes).to(device)

# Loss functions and optimizers
vae_criterion = nn.MSELoss(reduction='sum')
diffusion_criterion = nn.MSELoss(reduction='sum')
vae_optimizer = optim.AdamW(list(vae_encoder.parameters()) + list(vae_decoder.parameters()), lr=1e-3)
vae_scheduler = torch.optim.lr_scheduler.LinearLR(vae_optimizer, start_factor=1.0, end_factor=0.1, total_iters=50)

unet_optimizer = optim.AdamW(unet.parameters(), lr=1e-5)
unet_scheduler = torch.optim.lr_scheduler.LinearLR(vae_optimizer, start_factor=1.0, end_factor=0.1, total_iters=30)

def weights_init(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
vae_encoder.apply(weights_init)
vae_decoder.apply(weights_init)
unet.apply(weights_init)

# Assume you have a DataLoader `data_loader` that provides (images, labels)
num_epochs = 50
'''for epoch in range(num_epochs):
    for images, labels in tqdm(data_loader):
        images = images.to(device)
        labels = labels.to(images.device)
        vae_encoder = vae_encoder.to(images.device)
        vae_decoder = vae_decoder.to(images.device)
        
        # VAE Forward pass
        mu, logvar = vae_encoder(images)

        z = reparameterize(mu, logvar)
        reconstructed_images = vae_decoder(z)
        
        # VAE Loss
        recon_loss = vae_criterion(reconstructed_images, images)
        kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        vae_loss = recon_loss + kl_div
        
        # Backpropagation for VAE
        vae_optimizer.zero_grad()
        vae_loss.backward()
        vae_optimizer.step()

    vae_scheduler.step()
    print(f"Epoch [{epoch+1}/{num_epochs}], VAE Loss: {vae_loss.item()}")
    if (epoch + 1) % 25 == 0:
        with torch.no_grad():
            # Decode denoised latent to generate images
            generated_images = vae_decoder(z).clamp(-1, 1)
            display_images(generated_images, num_images=5)

torch.save(vae_encoder, "models/encoder.model")
torch.save(vae_decoder, "models/decoder.model")'''

vae_encoder = torch.load("models/encoder.model", weights_only=False)
vae_encoder.eval()

vae_decoder = torch.load("models/decoder.model", weights_only=False)
vae_decoder.eval()

def generate_sample(label, model, vae_decoder, num_samples=1):
    model.eval()
    vae_decoder.eval()
    output_images = torch.tensor(())
    
    with torch.no_grad():
        x = torch.randn(num_samples, latent_dim, 25, 25).to(device) # Ensure dimensions match the expected input size
        for t in reversed(range(num_timesteps)):
            predicted_noise = model(x, label, t)
            x = reverse_diffusion(x, predicted_noise, t).clamp(-1, 1)
            output_image = vae_decoder(x)#.clamp(-1, 1)  # Clamp only the final images for proper display
            if t in [0,1,2,3,4,5,7,8,9]:
                output_images = torch.cat((output_images.to(device), output_image.to(device)), 0)

        
            
    
    return output_images


# Function to display the generated images
def display_sample_images(images):
    images = (images + 1) / 2  # Normalize from [-1, 1] to [0, 1]
    images = images.cpu().permute(0, 2, 3, 1).numpy()
    
    plt.figure(figsize=(10, 5))
    for i, img in enumerate(images):
        plt.subplot(1, len(images), i + 1)
        plt.imshow(img, cmap = 'gray')
        plt.axis('off')
    plt.show()


#'''

# Visualize noisy images at different timesteps
z = 0 * torch.randn(5, latent_dim, 25, 25).to(device)
for t in [0, 2, 5, 9]:
    noisy_latent, _ = add_noise(z, t)
    generated_images = vae_decoder(noisy_latent).clamp(-1, 1)
    display_images(generated_images, num_images=5)

for epoch in range(num_epochs):
    for images, labels in tqdm(data_loader):
        images = images.to(device)
        labels = labels.to(device)
        
        unet.train()
        vae_encoder.eval()
        vae_decoder.eval()
        
        mu, logvar = vae_encoder(images)
        z = reparameterize(mu, logvar).detach()
        
        # Select a random time step for noise addition
        t = torch.randint(0, num_timesteps, (1,)).item()
        
        # Add noise to latent space
        noisy_latent, noise = add_noise(z, t)
        
        # Predict the noise using the UNet
        predicted_noise = unet(noisy_latent, labels, t)
        
        # Compute diffusion loss
        diffusion_loss = diffusion_criterion(predicted_noise, noise)
        
        # Backpropagation for UNet
        unet_optimizer.zero_grad()
        diffusion_loss.backward()
        unet_optimizer.step()
        torch.nn.utils.clip_grad_norm_(unet.parameters(), max_norm=1.0)
        
        #sample_images = generate_sample(0, unet, vae_decoder, num_samples=5)
        #display_images(sample_images, num_images=5)

    unet_scheduler.step()
    print(f"Epoch [{epoch+1}/{num_epochs}], Diffusion Loss: {diffusion_loss.item()}")
    
    if (epoch + 1) % 10 == 0:
        with torch.no_grad():
            # Reverse the diffusion process
            print(t)
            denoised_latent = reverse_diffusion(noisy_latent, predicted_noise, t)
            generated_images = vae_decoder(denoised_latent).clamp(-1, 1)
            display_images(generated_images, num_images=5)
       
            sample_images = generate_sample(0, unet, vae_decoder, num_samples=1)
            display_sample_images(sample_images)

print("Training completed!")

torch.save(unet, "models/unet.model")
#'''
unet = torch.load("models/unet.model", weights_only=False)
unet.eval()


# Example usage to generate images for class label 0
sample_images = generate_sample(0, unet, vae_decoder, num_samples=5)

# Display the generated images
display_sample_images(sample_images)

