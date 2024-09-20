import torch, io, matplotlib, os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import wandb
from tqdm import tqdm
from torchvision.utils import make_grid
import torch.nn as nn

matplotlib.use('Agg')

def generate_images(model, device, epoch, num_images=10, save_dir='generated_images'):
    """
    Fuinction to generate and save images to Wandb for checking model outputs during training

    Args:
        model (torch.nn.Module): Model to be used for generating images
        device (str): Device to be used for generating images
        epoch (int): Epoch to be used for generating images
        num_images (int, optional): Number of images to be generated. Defaults to 10.

    Usage:
        generate_images(model, device, epoch, num_images=10)
    """
    print(f"Generating images for epoch {epoch}...")
    model.eval()
    with torch.no_grad():
        # Start from random noise
        latents = torch.randn(num_images, model.vae.latent_dim, 1, 1).to(device)
        
        # Gradual denoising
        for t in reversed(range(model.noise_scheduler.num_timesteps)):
            t_batch = torch.full((num_images,), t, device=device, dtype=torch.long)
            
            # Predict and remove noise
            predicted_noise = model.unet(latents, t_batch)
            latents = model.noise_scheduler.remove_noise(latents, t_batch, predicted_noise)
        
        # Decode the final latents
        generated_images = model.vae.decode(latents)
        
        # Denormalize and clamp to [0, 1]
        generated_images = (generated_images + 1) / 2.0
        generated_images = torch.clamp(generated_images, 0, 1)
    
    # Create a grid of images
    img_grid = make_grid(generated_images, nrow=int(np.sqrt(num_images)))
    
    # Convert to numpy and transpose for correct image format
    img_grid = img_grid.cpu().numpy().transpose((1, 2, 0))
    
    # Create a figure and display the image grid
    plt.figure(figsize=(10, 10))
    plt.imshow(img_grid)
    plt.axis('off')
    
    # Save the figure locally
    plt.savefig(f"{save_dir}/generated_images_epoch_{epoch}.png", bbox_inches='tight', pad_inches=0.1)
    plt.close()
    
    # Log to wandb
    wandb.log({
        f"generated_images_epoch_{epoch}": wandb.Image(img_grid),
        "epoch": epoch
    })
    
    # Also log individual images to wandb
    wandb_images = [wandb.Image(img) for img in generated_images.cpu()]
    wandb.log({f"individual_images_epoch_{epoch}": wandb_images})

    print(f"Images for epoch {epoch} have been generated and logged to wandb.")


def generate_samples(model, noise_scheduler, device, epoch, num_samples=5):
    model.eval()
    with torch.no_grad():
        # Start from random noise
        latents = torch.randn(num_samples, model.unet.in_channels, 32, 32).to(device)

        # Gradually denoise the latents
        for t in tqdm(reversed(range(noise_scheduler.num_timesteps)), desc="Sampling"):
            t_batch = torch.full((num_samples,), t, device=device, dtype=torch.long)
            
            # Predict noise
            noise_pred = model.unet(latents, t_batch)
            
            # Update latents
            latents = noise_scheduler.step(noise_pred, t, latents)

        # Decode the final latents to images
        samples = model.decode(latents)

    # Create a grid of images
    samples_grid = make_grid(samples, nrow=num_samples) 
    samples_grid_np = samples_grid.cpu().numpy().transpose((1, 2, 0))
    
    # Log to wandb
    wandb.log({
        f"generated_samples_epoch_{epoch}": wandb.Image(samples_grid_np),
        "epoch": epoch
    })

    print(f"Samples for epoch {epoch} have been generated and logged to wandb.")
    return samples

def calculate_gradient_norm(model):
    total_norm = 0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    return total_norm
