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

    #noise_scheduler.fast_sampling(model, (num_images,8,32,32), 'cuda', num_steps=50)
    
    wandb.log({"denoising_process": images})

