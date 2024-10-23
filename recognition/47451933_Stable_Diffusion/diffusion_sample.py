import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import math
import umap
import torchvision.utils as tvutils

reducer = umap.UMAP(min_dist=0, n_neighbors=35)

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

# VAE Encoder for mapping input images to latent space
class VAEDecoder(nn.Module):
    def __init__(self, latent_dim=256):
        super(VAEDecoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 64, kernel_size=4, stride=2, padding=1),   # Output: (64, 50, 50)
            #nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),    # Output: (32, 100, 100)
            #nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1),     # Output: (1, 200, 200)
            nn.Tanh()  # To normalize output between 0 and 1
        )

    def forward(self, z):
        x_recon = self.decoder(z)
        return x_recon

# VAE Decoder for reconstructing images from latent space
class VAEEncoder(nn.Module):
    def __init__(self, latent_dim=256):
        super(VAEEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1),  # Output: (32, 100, 100)
            #nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # Output: (64, 50, 50)
            #nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, latent_dim, kernel_size=4, stride=2, padding=1),  # Output: (128, 25, 25)
            #nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Flatten(),  # Output: (512 * 6 * 6)
        )
        self.fc_mu = nn.Linear(latent_dim * 25 * 25, latent_dim)
        self.fc_logvar = nn.Linear(latent_dim * 25 * 25, latent_dim)

        self.decoder_fc = nn.Linear(latent_dim, latent_dim * 25 * 25)

    def forward(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)

        z = self.reparameterize(mu, logvar)
        h = self.decoder_fc(z)
        h = h.view(-1, latent_dim, 25, 25)

        return h, mu, logvar
    
    def decode(self,x):
        h = self.decoder_fc(x)
        h = h.view(-1, latent_dim, 25, 25)

        return h

    # Reparameterization trick for VAE
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

class CrossAttention(nn.Module):
    def __init__(self, embed_dim, heads):
        super(CrossAttention, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads=heads, batch_first=True)

    def forward(self, x, context):
        # x: (batch_size, num_channels, height, width), context: label embedding (batch_size, num_classes)
        b, c, h, w = x.size()
        x_flat = x.view(b, c, h * w).permute(0, 2, 1)  # (batch_size, h*w, num_channels)
        context = context.unsqueeze(1).repeat(1, h * w, 1)  # (batch_size, 1, num_classes)
        
        attn_output, _ = self.attention(x_flat, context, context)  # (batch_size, h*w, num_channels)
        attn_output = attn_output.permute(0, 2, 1).view(b, c, h, w)  # Reshape back to (batch_size, num_channels, height, width)
        return attn_output

# UNet for denoising
class UNet(nn.Module):
    def __init__(self, latent_dim=256, num_classes=2):
        super(UNet, self).__init__()
        self.channel_size = 128

        self.label_embedding = nn.Embedding(num_classes, 512)
        self.cross_attention = CrossAttention(embed_dim=512, heads=4)

        # Encoder part (downsampling)
        self.enc1 = nn.Sequential(
            nn.Conv2d(latent_dim, self.channel_size, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.channel_size),
            nn.ReLU(),
            nn.Conv2d(self.channel_size, self.channel_size, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.channel_size),
            nn.ReLU()
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(self.channel_size, self.channel_size * 2, kernel_size=3, padding=1, stride=2),  # Downsample
            nn.BatchNorm2d(self.channel_size * 2),
            nn.ReLU(),
            nn.Conv2d(self.channel_size * 2, self.channel_size * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.channel_size * 2),
            nn.ReLU()
        )
        self.enc3 = nn.Sequential(
            nn.Conv2d(self.channel_size * 2, self.channel_size * 4, kernel_size=3, padding=1, stride=2),  # Downsample
            nn.BatchNorm2d(self.channel_size * 4),
            nn.ReLU(),
            nn.Conv2d(self.channel_size * 4, self.channel_size * 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.channel_size * 4),
            nn.ReLU()
        )

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(self.channel_size * 4, self.channel_size * 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(self.channel_size * 8, self.channel_size * 4, kernel_size=3, padding=1),
            nn.ReLU()
        )

        # Decoder part (upsampling)
        self.dec3 = nn.Sequential(
            nn.ConvTranspose2d(self.channel_size * 4, self.channel_size * 4, kernel_size=4, stride=2, padding=1),  # Upsample
            nn.BatchNorm2d(self.channel_size * 4),
            nn.ReLU(),
            nn.Conv2d(self.channel_size * 4, self.channel_size * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.channel_size * 2),
            nn.ReLU()
        )
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(self.channel_size * 2, self.channel_size * 2, kernel_size=4, stride=2, padding=1),  # Upsample
            nn.BatchNorm2d(self.channel_size * 2),
            nn.ReLU(),
            nn.Conv2d(self.channel_size * 2, self.channel_size, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.channel_size),
            nn.ReLU()
        )
        self.dec1 = nn.Sequential(
            nn.Conv2d(self.channel_size, self.channel_size, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.channel_size),
            nn.ReLU(),
            nn.Conv2d(self.channel_size, latent_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(latent_dim),
            nn.Sigmoid()  # Final activation to scale outputs to [0, 1]
        )

        self.time_embedding_layer = nn.Linear(32, 128)

    def forward(self, x, label, t):
        # Time embedding
        t_embedding = get_timestep_embedding(t, 32).to(x.device)  # Ensure tensor is on the same device
        t_embedding = self.time_embedding_layer(t_embedding)  # Map time embedding to latent_dim
        t_embedding = t_embedding.unsqueeze(2).unsqueeze(3).repeat(x.shape[0], 1, 25, 25) # Shape: (batch_size, latent_dim, 1, 1)

        print(t_embedding)

        label_embedding = label.view(x.shape[0], 1).expand(-1,512).float()

        # Pass through the encoder with added time embedding
        enc1_out = self.enc1(torch.concat((x,t_embedding), dim=2))  # Ensure shapes match for addition
        enc2_out = self.enc2(enc1_out)
        enc3_out = self.enc3(enc2_out)

        # Pass through the bottleneck
        bottleneck_out = self.bottleneck(enc3_out)

        bottleneck_out = self.cross_attention(bottleneck_out, label_embedding)

        # Pass through the decoder with skip connections
        dec3 = self.crop_and_add(self.dec3(bottleneck_out), enc2_out)
        dec2 = self.crop_and_add(self.dec2(dec3), enc1_out)
        dec1 = self.crop_and_add(self.dec1(dec2), x)  # Skip connection with original input

        return dec1, bottleneck_out

    def crop_and_add(self, upsampled, skip_connection):
        """Crops the upsampled tensor to match the size of the skip connection tensor."""
        upsampled_cropped = self.crop(upsampled, skip_connection)
        return upsampled_cropped + skip_connection
    
    def crop(self, upsampled, skip_connection):
        """Crops the upsampled tensor to match the size of the skip connection tensor."""
        _, _, h, w = skip_connection.size()
        upsampled = upsampled[:, :, :h, :w]
        return upsampled


def add_noise(x, t):
    noise = torch.randn_like(x).to(device)
    alpha_t = alphas_cumprod[t].to(x.device)
    return torch.sqrt(alpha_t) * x + torch.sqrt(1 - alpha_t) * noise, noise

# Reverse diffusion using the noise prediction
def reverse_diffusion(noisy_latent, predicted_noise, t):
    alpha_t = alphas_cumprod[t].to(noisy_latent.device)
    beta_t = beta_scheduler.get_beta(t).to(noisy_latent.device)
    alpha_prev = alphas_cumprod[t - 1].to(noisy_latent.device) if t > 0 else torch.tensor(1.0).to(noisy_latent.device)

    # Predicted noise should be subtracted
    mean = (noisy_latent - beta_t * predicted_noise / torch.sqrt(1 - alpha_t)) / torch.sqrt(alpha_t)

    if t > 0:
        variance = beta_t * (1 - alpha_prev) / (1 - alpha_t)
        z = torch.randn_like(noisy_latent)
        return mean + torch.sqrt(variance) * z
    else:
        return mean

# Training setup
latent_dim = 128
num_classes = 2  # Update this according to your dataset
vae_encoder = VAEEncoder(latent_dim=latent_dim).to(device)
vae_decoder = VAEDecoder(latent_dim=latent_dim).to(device)
unet = UNet(latent_dim=latent_dim, num_classes=num_classes).to(device)

# Loss functions and optimizers
vae_criterion = nn.MSELoss(reduction='sum')
diffusion_criterion = nn.MSELoss()
vae_optimizer = optim.AdamW(list(vae_encoder.parameters()) + list(vae_decoder.parameters()), lr=1e-3)
vae_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(vae_optimizer, 50)

unet_optimizer = optim.AdamW(unet.parameters(), lr=1e-3)
unet_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(unet_optimizer, 200)

def weights_init(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
vae_encoder.apply(weights_init)
vae_decoder.apply(weights_init)
unet.apply(weights_init)

vae_losses = []

# Assume you have a DataLoader `data_loader` that provides (images, labels)
num_epochs = 10
'''for epoch in range(num_epochs):
    vae_it_loss = []
    for images, labels in tqdm(data_loader):
        images = images.to(device)
        labels = labels.to(images.device)
        vae_encoder = vae_encoder.to(images.device)
        vae_decoder = vae_decoder.to(images.device)
        
        # VAE Forward pass
        z, mu, logvar = vae_encoder(images)

        reconstructed_images = vae_decoder(z)
        
        # VAE Loss
        recon_loss = vae_criterion(reconstructed_images, images)
        kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        vae_loss = 0.5*recon_loss + kl_div

        vae_it_loss.append(vae_loss.item())
        
        # Backpropagation for VAE
        vae_optimizer.zero_grad()
        vae_loss.backward()
        vae_optimizer.step()

    vae_scheduler.step()
    print(f"Epoch [{epoch+1}/{num_epochs}], VAE Loss: {vae_loss.item()}")
    vae_losses.append(np.mean(vae_it_loss))
    #if (epoch + 1) % 20 == 0:
    #    with torch.no_grad():
    #        # Decode denoised latent to generate images
    #        generated_images = vae_decoder(z).clamp(-1, 1)
    #        display_images(generated_images, num_images=5)
    #        sample = vae_encoder.decode(torch.randn(5, latent_dim).to(device))
    #        generated_images = vae_decoder(sample)
    #        display_images(generated_images, num_images=5)

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
    label = torch.tensor([label] * num_samples).to(device)
    
    with torch.no_grad():
        x = torch.randn(num_samples, latent_dim).to(device) # Ensure dimensions match the expected input size
        x = vae_encoder.decode(x)
        for t in reversed(range(num_timesteps)):
            #x, _ = add_noise(x,t)
            predicted_noise, _ = model(x, label, t)
            x = reverse_diffusion(x, predicted_noise, t)#.clamp(-1, 1)
            output_image = vae_decoder(x)#.clamp(-1, 1)  # Clamp only the final images for proper display
            if t in [0,1,2,3,4,5,6,7,9,10]:
                output_images = torch.cat((output_images.to(device), output_image.to(device)), 0)

    return output_images

def generate_sample_latent(label, model, vae_decoder, num_samples=1):
    model.eval()
    vae_decoder.eval()
    output_images = torch.tensor(())
    label = torch.tensor([label] * num_samples).to(device)
    
    with torch.no_grad():
        x = torch.randn(num_samples, latent_dim).to(device) # Ensure dimensions match the expected input size
        x = vae_encoder.decode(x)
        for t in reversed(range(num_timesteps)):
            #x, _ = add_noise(x,t)
            predicted_noise, y = model(x, label, t)
            output_image = y#.clamp(-1, 1)  # Clamp only the final images for proper display
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

diffusion_losses = []
for epoch in range(num_epochs):
    losses = []
    for images, labels in tqdm(data_loader):
        images = images.to(device)
        labels = labels.to(device)
        
        unet.train()
        vae_encoder.eval()
        vae_decoder.eval()
        
        z, mu, logvar = vae_encoder(images)

        z = z.view(-1, latent_dim, 25, 25)

        # Select a random time step for noise addition
        t = torch.randint(0, num_timesteps, (1,)).item()

        # Add noise to latent space
        noisy_latent, noise = add_noise(z, t)

        # Predict the noise using the UNet
        predicted_noise, _ = unet(noisy_latent, labels, t)

        # Compute diffusion loss
        diffusion_loss = diffusion_criterion(predicted_noise, noise)
        losses.append(diffusion_loss.item())
        
        # Backpropagation for UNet
        unet_optimizer.zero_grad()
        diffusion_loss.backward()
        unet_optimizer.step()

    unet_scheduler.step()
    print(f"Epoch [{epoch+1}/{num_epochs}], Diffusion Loss: {np.mean(losses)}")
    diffusion_losses.append(np.mean(losses))
    
    if (epoch+1) % 99 == 0:
        with torch.no_grad():
            # Reverse the diffusion process
            print(t)
            denoised_latent = reverse_diffusion(noisy_latent, predicted_noise, t)
            generated_images = vae_decoder(denoised_latent).clamp(-1, 1)
            display_images(generated_images, num_images=5)
       
            sample_images = generate_sample(0, unet, vae_decoder, num_samples=10)
            plt.imshow(np.transpose(tvutils.make_grid(sample_images.to(device)[:100], nrow=10, padding=2, normalize=True).cpu(),(1,2,0)))
            plt.show()
            sample_images = generate_sample(1, unet, vae_decoder, num_samples=10)
            plt.imshow(np.transpose(tvutils.make_grid(sample_images.to(device)[:100], nrow=10, padding=2, normalize=True).cpu(),(1,2,0)))
            plt.show()

print("Training completed!")

torch.save(unet, "models/unet.model")
#'''
unet = torch.load("models/unet.model", weights_only=False)
unet.eval()

plt.plot(vae_losses)
plt.title("VAE Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()

plt.plot(diffusion_losses)
plt.title("Diffusion Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()


n_images = 10

# Display the generated images
sample_images = generate_sample(0, unet, vae_decoder, num_samples=n_images)
display_sample_images(sample_images)

n_images = 100

# Example usage to generate images for class label 0
embedded0 = generate_sample_latent(0, unet, vae_decoder, num_samples=n_images).view(-1,512*13*7)
#h,mu,logvar = vae_encoder(sample_images)
#embedded0 = vae_encoder.reparameterize(mu,logvar)

print(embedded0.shape)

embedded1 = generate_sample_latent(1, unet, vae_decoder, num_samples=n_images).view(-1,512*13*7)
#h,mu,logvar = vae_encoder(sample_images)
#embedded1 = vae_encoder.reparameterize(mu,logvar)
print(embedded1.shape)

embedded = torch.cat((embedded0,embedded1),dim=0)

print(embedded.shape)

embedding = reducer.fit_transform(embedded.cpu().numpy())
plt.scatter(embedding[:, 0], embedding[:, 1], c=[0] * n_images *10 + [1] * n_images*10, cmap='Spectral', s=10)
plt.show()