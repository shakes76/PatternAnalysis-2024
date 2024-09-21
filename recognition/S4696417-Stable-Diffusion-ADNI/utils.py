import torch, io, matplotlib, os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import wandb
from tqdm import tqdm
from torchvision.utils import make_grid
import torch.nn as nn
matplotlib.use('Agg')

def generate_samples(model, noise_scheduler, device, epoch, num_samples=5):
    model.eval()
    with torch.no_grad():
        # Start from random noise
        latents = torch.randn(num_samples, model.vae.latent_dim, 4, 4).to(device)

        # Gradually denoise the latents
        for t in tqdm(reversed(range(noise_scheduler.num_timesteps)), desc="Sampling"):
            t_batch = torch.full((num_samples,), t, device=device, dtype=torch.long)
            
            # Predict noise
            noise_pred = model.unet(latents, t_batch)
            
            # Update latents
            latents = noise_scheduler.step(noise_pred, t, latents)

            # Optionally, log intermediate samples to wandb
            if t % (noise_scheduler.num_timesteps // 10) == 0:  # Log every 10% of steps
                # Decode the latents to images at this step
                intermediate_images = model.vae.decode(latents)
                samples_grid = make_grid(intermediate_images, nrow=num_samples)
                samples_grid_np = samples_grid.cpu().numpy().transpose((1, 2, 0))
                
                # Log to wandb
                wandb.log({
                    f"intermediate_samples_epoch_{epoch}_step_{t}": wandb.Image(samples_grid_np),
                    "epoch": epoch
                })

        # Decode the final latents to images
        images = model.vae.decode(latents)

    # Create a grid of images
    samples_grid = make_grid(images, nrow=num_samples) 
    samples_grid_np = samples_grid.cpu().numpy().transpose((1, 2, 0))
    
    # Log to wandb
    wandb.log({
        f"generated_samples_epoch_{epoch}": wandb.Image(samples_grid_np),
        "epoch": epoch
    })

    print(f"Samples for epoch {epoch} have been generated and logged to wandb.")
    return images

def calculate_gradient_norm(model):
    total_norm = 0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    return total_norm
