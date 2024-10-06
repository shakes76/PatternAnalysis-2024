import torch, wandb, os
from dataset import get_dataloader
from modules import StableDiffusion, UNet, NoiseScheduler_Fast_DDPM

@torch.no_grad()
def visualize_denoising_process(model, noise_scheduler, num_inference_steps=10, num_images=5):
    """
    Function to visualise the denoising process when generating samples from the model. 
    Will log the process to wandb

    Args:
        model: Stable diffusion model
        noise_scheduler: Noise scheduler
        num_inference_steps: Number of inference steps
        num_images: Number of images to generate
    """
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

def run_setup(
        method, 
        image_size,
        batch_size, 
        image_transform,
        vae_path,
        hidden_dims=[64, 128, 256, 512, 1024],
        time_emb_dim=256,
        noise_timesteps=100):
    
    """
    Setup function to initialise model and dataloaders

    Args:
        method: Local or Slurm
        image_size: Image size
        batch_size: Batch size
        image_transform: Image transform
        vae_path: Path to VAE model
        hidden_dims: Hidden dimensions
        time_emb_dim: Time embedding dimension
        noise_timesteps: Noise timesteps

    Returns:
        train_loader: Training dataloader
        val_loader: Validation dataloader
        model: Stable diffusion model
    """
    
    print("Loading data...")
    if method == 'Local':
        os.chdir('recognition/S4696417-Stable-Diffusion-ADNI')
        train_loader, val_loader = get_dataloader('data/train/AD', batch_size=batch_size, transform=image_transform)
    elif method == 'Slurm':
        train_loader, val_loader = get_dataloader('/home/groups/comp3710/ADNI/AD_NC/train/AD', batch_size=batch_size, transform=image_transform)

    # Settings
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using {device}')

    # Load pretrained VAE model
    print("Loading VAE model...")
    vae = torch.load(vae_path)
    vae.eval() 
    for param in vae.parameters():
        param.requires_grad = False 

    unet = UNet(in_channels=vae.latent_dim, hidden_dims=hidden_dims, time_emb_dim=time_emb_dim)
    noise_scheduler = NoiseScheduler_Fast_DDPM(num_timesteps=noise_timesteps).to(device)
    model = StableDiffusion(unet, vae, noise_scheduler, image_size=image_size, device=device).to(device)

    return train_loader, val_loader, model


def get_warmup_steps(train_loader, epochs):
    """
    Function to return parameters for learning rate scheduler with warmup stage

    Args:
        train_loader: Training dataloader
        epochs: Number of epochs

    Returns:
        warmup_steps: Number of warmup steps
        total_steps: Total number of steps
    """
    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * epochs
    warmup_steps = int(0.1 * total_steps)

    return warmup_steps, total_steps

def init_wandb(lr, epochs, optimizer, scheduler, criterion, scaler, IMAGE_SIZE, BATCH_SIZE):
    """
    Function to initialise wandb project logging

    Args:
        lr: Learning rate
        epochs: Number of epochs
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        criterion: Loss function
        scaler: Gradient scaler
        IMAGE_SIZE: Image size
        BATCH_SIZE: Batch size
    """
    wandb.init(
    project="Stable-Diffusion-ADNI", 
    entity="s1lentcs-uq",
    config={
        "learning rate": lr,
        "epochs": epochs,
        "optimizer": type(optimizer).__name__,
        "scheduler": type(scheduler).__name__,
        "loss": type(criterion).__name__,
        "scaler": type(scaler).__name__,
        "name": "SD-ADNI - VAE and Unet",
        "image size": IMAGE_SIZE,
        "batch size": BATCH_SIZE
    })
    print("Wandb initialized")
