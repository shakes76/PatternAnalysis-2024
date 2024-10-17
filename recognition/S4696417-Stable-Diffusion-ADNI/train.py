"""
This file contains the primary training and validation loop for the diffusion model.
The image size and latent dimension must match the VAE model used for pretraining.
Wandb account will need to be setup and logged in before running the training.

Author: Liam O'Sullivan
"""

import os
import wandb
import torch
from tqdm import tqdm
import torch.nn as nn
from torch.amp import autocast
from torch.cuda.amp import GradScaler
from torchvision import transforms
from torchmetrics.functional.image import peak_signal_noise_ratio, \
    structural_similarity_index_measure

from modules import CosineAnnealingWarmupScheduler
from utils import run_setup, get_warmup_steps, init_wandb


# SETUP - Must Match Image Size of VAE
IMAGE_SIZE = 256
BATCH_SIZE = 16
method = 'Local'

image_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
])

train_loader, val_loader, model = run_setup(
    method=method,
    batch_size=BATCH_SIZE,
    image_size=IMAGE_SIZE,
    image_class='AD',
    image_transform=image_transform,
    hidden_dims=[64, 128, 256, 512, 1024],
    time_emb_dim=256,
    vae_path=f'checkpoints/VAE/ADNI-vae_e100_b8_im{IMAGE_SIZE}_l16.pt',
    noise_timesteps=100
)

criterion = nn.MSELoss()
scaler = GradScaler()

lr = 1e-4
epochs = 500
warmup_steps, total_steps = get_warmup_steps(train_loader, epochs)

optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
scheduler = CosineAnnealingWarmupScheduler(optimizer, warmup_steps, total_steps)

init_wandb(lr, epochs, optimizer, scheduler, criterion, scaler, IMAGE_SIZE, BATCH_SIZE)

# Training/Validation Loop
print("Training model...")
for epoch in range(epochs):
    model.train()
    train_loss, val_loss = 0, 0
    train_psnr, val_psnr = 0, 0
    train_ssim, val_ssim = 0, 0

    loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
    for i, batch in enumerate(loop):

        images, _ = batch  # retrieve clean image batch
        images = images.to(model.device)
        optimizer.zero_grad()

        # Encode images to latent space
        with torch.no_grad():
            mu, logvar = model.encode(images)
            latents = model.sample_latent(mu, logvar)

        # Sample noise and timesteps
        timesteps = torch.randint(
            0,
            model.noise_scheduler.num_timesteps,
            (latents.size(0),),
            device=model.device)

        noise = torch.randn_like(latents)

        # Add noise to latents
        noisy_latents = model.noise_scheduler.add_noise(latents, noise, timesteps)

        with torch.no_grad():
            denoised_images = model.vae.decode(
                model.noise_scheduler.step(model.unet, noisy_latents, timesteps))

            ssim = structural_similarity_index_measure(denoised_images, images)
            psnr = peak_signal_noise_ratio(denoised_images, images)

        # Train UNet
        with autocast('cuda'):
            predicted_noise = model.predict_noise(noisy_latents, timesteps)
            loss = criterion(predicted_noise, noise)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # Update metrics
        train_loss += loss.item()
        train_psnr += psnr.item()
        train_ssim += ssim.item()

        wandb.log({
            'train_psnr': psnr.item(),
            'train_ssim': ssim.item(),
            'train_loss': loss.item(),
            'learning_rate': optimizer.param_groups[0]['lr'],
        })

        # Update progress bar
        loop.set_postfix(loss=loss.item(), psnr=psnr.item(), ssim=ssim.item())

    # Compute average metrics
    avg_train_loss = train_loss / len(train_loader)
    avg_train_psnr = train_psnr / len(train_loader)
    avg_train_ssim = train_ssim / len(train_loader)

    # Validation
    model.eval()
    with torch.no_grad():
        for images, _ in tqdm(val_loader, desc="Validating"):
            images = images.to(model.device)

            with torch.no_grad():
                mu, logvar = model.encode(images)
                latents = model.sample_latent(mu, logvar)

            noise = torch.randn_like(latents)
            timesteps = torch.randint(
                0,
                model.noise_scheduler.num_timesteps,
                (images.size(0),),
                device=model.device)

            noisy_latents = model.noise_scheduler.add_noise(latents, noise, timesteps)
            with torch.no_grad():
                denoised_images = model.vae.decode(
                    model.noise_scheduler.step(model.unet, noisy_latents, timesteps))

                ssim = structural_similarity_index_measure(denoised_images, images)
                psnr = peak_signal_noise_ratio(denoised_images, images)

            with autocast('cuda'):
                predicted_noise = model.predict_noise(noisy_latents, timesteps)
                loss = criterion(predicted_noise, noise)

            val_loss += loss.item()
            val_psnr += psnr.item()
            val_ssim += ssim.item()

    # Compute average validation metrics
    avg_val_loss = val_loss / len(val_loader)
    avg_val_psnr = val_psnr / len(val_loader)
    avg_val_ssim = val_ssim / len(val_loader)

    # Log epoch-level metrics
    wandb.log({
        'epoch': epoch,
        'avg_train_loss': avg_train_loss,
        'avg_val_loss': avg_val_loss,
        'avg_train_psnr': avg_train_psnr,
        'avg_val_psnr': avg_val_psnr,
        'avg_train_ssim': avg_train_ssim,
        'avg_val_ssim': avg_val_ssim
    })

    print(f'Epoch: {epoch}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')
    print(f'Train PSNR: {avg_train_psnr:.4f}, Val PSNR: {avg_val_psnr:.4f}')
    print(f'Train SSIM: {avg_train_ssim:.4f}, Val SSIM: {avg_val_ssim:.4f}')

    # Generate and log sample images
    if (epoch) % 10 == 0:
        sample_images = model.sample(BATCH_SIZE, device=model.device)
        ssim = structural_similarity_index_measure(sample_images, images)
        psnr = peak_signal_noise_ratio(sample_images, images)
        wandb.log({
            'Generated SSIM': ssim,
            'Generated PSNR': psnr
        })

    if epoch == epochs:
        sample_images = model.sample(num_images=8, device=model.device)

print("Training complete")
path = os.path.join(os.getcwd(),
                    f'checkpoints/Diffusion/ADNI_diffusion_e{epoch+1}_im{IMAGE_SIZE}.pt')
torch.save(model, path)
wandb.finish()
