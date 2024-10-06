import torch
import wandb
import os

num_images = 5
IMAGE_SIZE = 256  # must match loaded model
method = 'Local'

if method == 'Local':
    os.chdir('recognition/S4696417-Stable-Diffusion-ADNI')

device = 'cuda' if torch.cuda.is_available() else 'cpu'
wandb.init(project="Stable-Diffusion-ADNI-Inference", name="Inference")

# Load Trained Diffusion Model
model = torch.load(f'checkpoints/Diffusion/ADNI_diffusion_e400_im{IMAGE_SIZE}.pt').to(device)


# Generate images
with torch.no_grad():
    sample_images = model.sample(num_images, device=device)
    model.fast_sample(num_images, device=device, steps=250)
