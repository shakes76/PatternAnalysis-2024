'''
[desc]
contains eveything need to
train the stable diffusion model

@author Jamie Westerhout
@project Stable Diffusion
@date 2024
'''

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import math
import umap

#project imports
from utils import *
from modules import *
from dataset import *

##################
# What to train #
TRAIN_UNET = True
TRAIN_VAE = True
##################

#get best device
device = torch.device('cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))

#number of timesteps for diffusion
num_timesteps = 10
noise_scheduler = NoiseScheduler(num_timesteps=num_timesteps)
alphas_cumprod = noise_scheduler.get_alphas_cumprod()

#get data
data = Dataset()
data_loader = data.get_train()
data_loader_val = data.get_val()
data_loader_test = data.get_test()

# training params
latent_dim = 128
num_classes = 2 #AD and NC
image_size = data.image_size
epochs = 100
base_lr = 1e-3

# creating models and putting them on device
vae_encoder = VAEEncoder(latent_dim=latent_dim).to(device)
vae_decoder = VAEDecoder(latent_dim=latent_dim).to(device)
unet = UNet(latent_dim=latent_dim, num_classes=num_classes).to(device)

# loss criterions
vae_criterion = nn.MSELoss(reduction='sum')
diffusion_criterion = nn.MSELoss()

# vae optimizer and noise schedular
vae_optimizer = optim.AdamW(list(vae_encoder.parameters()) + list(vae_decoder.parameters()), lr=base_lr)
vae_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(vae_optimizer, epochs)

# unet optimizer and noise schedular
unet_optimizer = optim.AdamW(unet.parameters(), lr=base_lr)
unet_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(unet_optimizer, epochs)

def train_vae(encoder, decoder, intermediate_outputs = True, save_model = True):
    vae_losses = []
    vae_val_losses = []

    #epochs
    for epoch in range(epochs):
        vae_it_loss = []
        vae_val_loss = []

        #train
        for images, labels in tqdm(data_loader):
            images = images.to(device)
            labels = labels.to(images.device)
            vae_encoder = encoder.to(images.device)
            vae_decoder = decoder.to(images.device)
            
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
        
        #validate
        for images, labels in tqdm(data_loader_val):
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

            vae_val_loss.append(vae_loss.item())

        #outputs results
        vae_scheduler.step()
        print(f"Epoch [{epoch+1}/{epochs}], VAE Loss: {np.mean(vae_it_loss)}, VAE validation loss: {np.mean(vae_val_loss)}")
        vae_losses.append(np.mean(vae_it_loss))
        vae_val_losses.append(np.mean(vae_val_loss))

        if (epoch + 1) % 20 == 0 and intermediate_outputs:
            with torch.no_grad():
                # using output from last val it to give ideal outpout
                generated_images = vae_decoder(z)
                display_images(generated_images, num_images=4, title="Best Possible Output")

                # display new generated images
                sample = vae_encoder.decode(torch.randn(4, latent_dim).to(device))
                generated_images = vae_decoder(sample)
                display_images(generated_images, num_images=4, title="Newly Generated Outputs")

                #validation images
                display_images(images, num_images=4, title="Validation Images")
    
    if save_model:
        torch.save(vae_encoder, "models/encoder.model")
        torch.save(vae_decoder, "models/decoder.model")
    
    return vae_losses, vae_val_losses

def train_unet(unet, intermediate_outputs = True, save_model = True):
    diffusion_losses = []
    diffusion_losses_val = []

    unet.train()

    #epochs
    for epoch in range(epochs):
        losses = []
        losses_val = []

        #train
        for images, labels in tqdm(data_loader):

            images = images.to(device)
            labels = labels.to(device)
            
            z, _, _ = vae_encoder(images)
            z = z.view(-1, latent_dim, 25, 25)

            # select random timestep of which noise to add
            t = torch.randint(0, num_timesteps, (1,)).item()

            # add noise
            noisy_latent, noise = add_noise(z, t, noise_scheduler)

            # Predict the noise
            predicted_noise, _ = unet(noisy_latent, labels, t)

            # caculate the loss
            diffusion_loss = diffusion_criterion(predicted_noise, noise)
            losses.append(diffusion_loss.item())
            
            # Bwakwards step
            unet_optimizer.zero_grad()
            diffusion_loss.backward()
            unet_optimizer.step()

        #validate
        for images, labels in tqdm(data_loader_val):

            images = images.to(device)
            labels = labels.to(device)
            
            z, _, _ = vae_encoder(images)
            z = z.view(-1, latent_dim, 25, 25)

            # select random timestep of which noise to add
            t = torch.randint(0, num_timesteps, (1,)).item()

            # add noise
            noisy_latent, noise = add_noise(z, t, noise_scheduler)

            # Predict the noise
            predicted_noise, _ = unet(noisy_latent, labels, t)

            # Caculate the loss
            diffusion_loss = diffusion_criterion(predicted_noise, noise)
            losses_val.append(diffusion_loss.item())

        #print losses
        unet_scheduler.step()
        print(f"Epoch [{epoch+1}/{epochs}], Diffusion Loss: {np.mean(losses)}, Diffusion Loss Val: {np.mean(losses_val)}")

        #step lr schedular
        diffusion_losses.append(np.mean(losses))
        diffusion_losses_val.append(np.mean(losses_val))
        
        #show intermedit image generation
        if (epoch+1) % 20 == 0 and intermediate_outputs:
            with torch.no_grad():
                #display images from las denoised latent should give iteal model
                denoised_latent = reverse_diffusion(noisy_latent, predicted_noise, t, noise_scheduler)
                generated_images = vae_decoder(denoised_latent).clamp(-1, 1)
                display_images(generated_images, num_images=4, title="ideal images")
        
                #generate new images from random noise for 0 class
                sample_images = generate_sample(0, unet, vae_decoder, vae_encoder, latent_dim, num_timesteps, noise_scheduler, num_samples=10)
                plt.imshow(np.transpose(tvutils.make_grid(sample_images.to(device)[:100], nrow=10, padding=2, normalize=True).cpu(),(1,2,0)))
                plt.title("Generate Images Label: 0")
                plt.show()

                #generate new images from random noise for 1 class
                sample_images = generate_sample(1, unet, vae_decoder, vae_encoder, latent_dim, num_timesteps, noise_scheduler, num_samples=10)
                plt.imshow(np.transpose(tvutils.make_grid(sample_images.to(device)[:100], nrow=10, padding=2, normalize=True).cpu(),(1,2,0)))
                plt.title("Generate Images Label: 1")
                plt.show()
    
    if save_model:
        torch.save(unet, "models/unet.model")

    return diffusion_losses, diffusion_losses_val


if __name__ == '__main__':
    if TRAIN_VAE:
        train_vae(vae_encoder, vae_decoder, False, True)
        vae_encoder.eval()
    else:
        vae_encoder = torch.load("models/encoder.model", weights_only=False)
        vae_encoder.eval()

        vae_decoder = torch.load("models/decoder.model", weights_only=False)
        vae_decoder.eval()
    
    if TRAIN_UNET:
        train_unet(unet, False, True)
        vae_decoder.eval()
    else:
        unet = torch.load("models/unet.model", weights_only=False)
        unet.eval()