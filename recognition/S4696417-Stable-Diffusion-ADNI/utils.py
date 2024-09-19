import torch, io, matplotlib, os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import wandb

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
    print("Generating images...")
    model.eval()
    with torch.no_grad():
        # Generate random noise
        noise = torch.randn(num_images, 1, 32, 32).to(device)
        
        # Generate timestamps (you may need to adjust this based on your model's requirements)
        timestamps = torch.randint(0, 1000, (num_images,), device=device).long()
        
        # Generate images
        generated_images = model(noise, timestamps)
        
        # Denormalize and convert to numpy for plotting
        generated_images = generated_images.cpu().numpy()
        generated_images = (generated_images + 1) / 2.0
        generated_images = np.clip(generated_images, 0, 1)  # Ensure values are in [0, 1]
        generated_images = np.squeeze(generated_images, axis=1) # for MNIST
        #generated_images = np.transpose(generated_images, (0, 2, 3, 1))
    
    # Make directory if it doesnt exist
    os.makedirs(save_dir, exist_ok=True)

    # Plot and save the generated images
    fig, axes = plt.subplots(1, num_images, figsize=(20, 2))
    wandb_images = []

    for i, img in enumerate(generated_images):
        img_path = os.path.join(save_dir, f'epoch_{epoch}_image_{i}.png')

        # For single image
        if num_images == 1:
            axes.imshow(img, cmap='gray')
            axes.axis('off')
        else:
            axes[i].imshow(img, cmap='gray')
            axes[i].axis('off')
            
        # plt.imsave(img_path, img)
        plt.imsave(img_path, img, cmap='gray') #MNIST
        
        #wandb_images.append(wandb.Image(img_path, caption=f"Epoch {epoch}, Image {i+1}"))

        # Plot for combined figure
        if num_images > 1:
            axes[i].imshow(img, cmap='gray')
            axes[i].axis('off')
        else:
            axes.imshow(img, cmap='gray')
            axes.axis('off')

    plt.tight_layout()

    # Save the combined figure
    combined_path = os.path.join(save_dir, f'epoch_{epoch}_combined.png')
    plt.savefig(combined_path)
    plt.close(fig)

    # Log to wandb
    wandb.log({
        f"generated_images_epoch_{epoch}": wandb.Image(combined_path, caption=f"Generated Images at Epoch {epoch}"),
        # f"individual_images_epoch_{epoch}": wandb_images
    })
    print(f"Images for epoch {epoch} have been logged to wandb.")

def calculate_gradient_norm(model):
    total_norm = 0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    return total_norm


def get_noise_schedule(num_timesteps=1000, beta_start=0.0001, beta_end=0.02):
    """
    Create a noise schedule for the diffusion process.
    """
    return torch.linspace(beta_start, beta_end, num_timesteps)

def add_noise(x, t, noise_schedule):
    """
    Add noise to the input images based on the timesteps.
    x: [B, C, H, W] tensor of images
    t: [B] tensor of timesteps
    noise_schedule: [num_timesteps] tensor of noise schedule
    """
    batch_size = x.shape[0]
    cumulative_alphas = torch.cumprod(1 - noise_schedule, dim=0)
    alphas_t = cumulative_alphas[t]
    alphas_t = alphas_t.view(-1, 1, 1, 1)  # reshape for broadcasting

    noise = torch.randn_like(x)
    noisy_images = torch.sqrt(alphas_t) * x + torch.sqrt(1 - alphas_t) * noise
    return noisy_images