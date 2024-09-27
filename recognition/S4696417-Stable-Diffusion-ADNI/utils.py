from torchmetrics.functional.image import peak_signal_noise_ratio, structural_similarity_index_measure
from torch.optim.lr_scheduler import LambdaLR 
import wandb
import torch

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
        
        # Update latents with predicted noise
        latents = noise_scheduler.step(model.unet, latents, timesteps)
        
        if i % (num_inference_steps // 10) == 0:  # Log every 10% of steps
            # Decode latents to image space
            with torch.no_grad():
                images.append(wandb.Image(model.vae.decode(latents).cpu(), caption=f"Step {i}"))
    
    wandb.log({"denoising_process": images})

def calculate_metrics(model, noisy_latents, timesteps, images, loss, optimizer, mode):
    """
    Calculate the PSNR and SSIM metrics for the denoised images. 
    Will log to wandb based on train/val

    Args:
        model (torch.nn.Module): The diffusion model instance
        noisy_latents (torch.Tensor): The noisy latents to use for denoising
        timesteps (torch.Tensor): The timesteps to use for denoising
        images (torch.Tensor): The clean images to use for calculating metrics
        loss (torch.Tensor): The loss to use for calculating metrics
        optimizer (torch.optim.Optimizer): The optimizer to use for calculating metrics
        mode (string): Whether the model is in training, validation mode

    Returns:
        psnr (torch.Tensor): The PSNR metric
        ssim (torch.Tensor): The SSIM metric
    """
    assert mode in ['train', 'val'], "mode must be 'train' or 'val'"
    
    with torch.no_grad():
        denoised_latents = model.noise_scheduler.step(model.unet, noisy_latents, timesteps)
        denoised_images = model.vae.decode(denoised_latents)
        ssim = structural_similarity_index_measure(denoised_images, images)
        psnr = peak_signal_noise_ratio(denoised_images, images)

    if mode == 'train':
        wandb.log({
            'train_psnr': psnr.item(),
            'train_ssim': ssim.item(),
            'train_loss': loss.item(),
            'learning_rate': optimizer.param_groups[0]['lr'],
        })
    elif mode == 'val':
        wandb.log({
            'val_psnr': psnr.item(),
            'val_ssim': ssim.item(),
            'val_loss': loss.item(),
            'learning_rate': optimizer.param_groups[0]['lr'],
        })

    return psnr, ssim


