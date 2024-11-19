"""
This file is used to generate new sample images from the trained diffusion model.

Author: Liam O'Sullivan
"""

import torch
import wandb
import os

num_images = 64
IMAGE_SIZE = 256  # must match loaded model
method = 'Local'

if method == 'Local':
    os.chdir('recognition/S4696417-Stable-Diffusion-ADNI')

device = 'cuda' if torch.cuda.is_available() else 'cpu'
wandb.init(project="Stable-Diffusion-ADNI-Inference", name="Inference")

# Load Trained Diffusion Model (download either AD or NC checkpoint from below and place in checkpoints/Diffusion)
# AD Checkpoint: https://drive.google.com/file/d/1eVkB2aPTVc8dtPLJkcCfOaJs6yEOkYS9/view?usp=sharing
# NC Checkpoint: https://drive.google.com/file/d/1rxdQhUixX2N9tbVXLW_jyyg-jt7HbYWh/view?usp=sharing
model = torch.load(f'checkpoints/Diffusion/ADNI_AD_diffusion_e500_im{IMAGE_SIZE}.pt').to(device)


# Generate images
with torch.no_grad():
    sample_images = model.sample(num_images, device=device)
    model.fast_sample(num_images, device=device, steps=100)
