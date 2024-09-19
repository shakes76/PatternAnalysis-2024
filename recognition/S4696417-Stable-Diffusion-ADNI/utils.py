import torch, io
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import wandb

def generate_images(model, device, epoch, num_images=10):
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
        noise = torch.randn(num_images, 3, 32, 32).to(device)
        
        # Generate timestamps (you may need to adjust this based on your model's requirements)
        timestamps = torch.randint(0, 1000, (num_images,), device=device).long()
        
        # Generate images
        generated_images = model(noise, timestamps)
        
        # Denormalize and convert to numpy for plotting
        generated_images = generated_images.cpu().numpy()
        generated_images = (generated_images + 1) / 2.0
        generated_images = np.clip(generated_images, 0, 1)  # Ensure values are in [0, 1]
        generated_images = np.transpose(generated_images, (0, 2, 3, 1))

    # Plot and save the generated images
    fig, axes = plt.subplots(1, num_images, figsize=(20, 2))
    wandb_images = []
    for i, img in enumerate(generated_images):
        axes[i].imshow(img)
        axes[i].axis('off')
        
        # Convert to PIL Image for wandb
        pil_img = Image.fromarray((img * 255).astype(np.uint8))
        wandb_images.append(wandb.Image(pil_img, caption=f"Epoch {epoch}, Image {i+1}"))

    plt.tight_layout()

    # Save the figure to a buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)

    # Log the whole figure and individual images to wandb
    wandb.log({
        f"generated_images_epoch_{epoch}": wandb.Image(buf, caption=f"Generated Images at Epoch {epoch}"),
        f"individual_images_epoch_{epoch}": wandb_images
    })

    plt.close(fig)
    print(f"Images for epoch {epoch} have been logged to wandb.")

def calculate_gradient_norm(model):
    total_norm = 0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    return total_norm