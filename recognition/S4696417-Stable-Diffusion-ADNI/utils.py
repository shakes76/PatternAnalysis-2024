from torch.optim.lr_scheduler import LambdaLR 
import wandb
import matplotlib
import torch
from tqdm import tqdm
from torchvision.utils import make_grid
import torch.nn as nn
matplotlib.use('Agg')

@torch.inference_mode()
def generate_samples(model, noise_scheduler, device, epoch, num_samples=5):
    model.eval()
    with torch.no_grad():
        # Start from random noise
        latents = torch.randn(num_samples, model.vae.latent_dim, 4, 4).to(device)
        # Gradually denoise the latents
        for t in tqdm(reversed(range(noise_scheduler.num_timesteps)), desc="Sampling"):
            # Create a batch of the same timestep
            t_batch = torch.full((num_samples,), t, device=device, dtype=torch.long)
           
            # Predict noise
            noise_pred = model.unet(latents, t_batch)
           
            # Update latents
            latents = noise_scheduler.step(noise_pred, t_batch, latents)
            
            # Optionally, log intermediate samples to wandb
            if t % (noise_scheduler.num_timesteps // 10) == 0:  # Log every 10% of steps
                # Decode the latents to images at this step
                with torch.no_grad():
                    intermediate_images = model.vae.decode(latents)
                samples_grid = make_grid(intermediate_images, nrow=num_samples)
                samples_grid_np = samples_grid.cpu().numpy().transpose((1, 2, 0))
               
                # Log to wandb
                wandb.log({
                    f"intermediate_samples_epoch_{epoch}_step_{t}": wandb.Image(samples_grid_np),
                    "epoch": epoch
                })
        
        # Decode the final latents to images
        with torch.no_grad():
            images = model.vae.decode(latents)
    
    # Create a grid of final images
    samples_grid = make_grid(images, nrow=num_samples)
    samples_grid_np = samples_grid.cpu().numpy().transpose((1, 2, 0))
   
    # Log final samples to wandb
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

def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
        )
    return LambdaLR(optimizer, lr_lambda, last_epoch)

@torch.no_grad()
def visualize_denoising_process(model, noise_scheduler, num_inference_steps=10, num_images=5):
    # Start from random noise
    latents = torch.randn(num_images, 8, 16, 16).to('cuda')
    
    images = []
    for i, t in enumerate(reversed(range(0, noise_scheduler.num_timesteps, noise_scheduler.num_timesteps // num_inference_steps))):
        # Create a batch of the same timestep
        timesteps = torch.full((num_images,), t, device='cuda', dtype=torch.long)
        
        # Get model prediction (predicted noise)
        noise_pred = model.unet(latents, timesteps)
        
        # Update latents with predicted noise
        latents = noise_scheduler.step(model.unet, latents, timesteps)
        
        if i % (num_inference_steps // 10) == 0:  # Log every 10% of steps
            # Decode latents to image space
            with torch.no_grad():
                images.append(wandb.Image(model.vae.decode(latents).cpu(), caption=f"Step {i}"))

    #noise_scheduler.fast_sampling(model, (num_images,8,32,32), 'cuda', num_steps=50)
    
    wandb.log({"denoising_process": images})

