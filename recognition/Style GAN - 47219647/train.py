import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.cuda.amp import autocast, GradScaler
from dataset import data_set_creator
from diffusers import DDPMScheduler
from modules import vae, unet, text_encoder, tokenizer

data_loader, label_map = data_set_creator()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

unet.to(device)
vae.to(device)
text_encoder.to(device)

num_epochs = 10

criterion = nn.MSELoss()
optimizer = AdamW(unet.parameters(), lr=1e-4)
scaler = GradScaler()
noise_scheduler = DDPMScheduler(num_train_timesteps=1000)

for epoch in range(num_epochs):
    unet.train()
    for batch_images, batch_labels in data_loader:
        batch_images = batch_images.to(device)

        #Changes the lable to the desired text encoding
        text_labels = [label_map[label.item()] for label in batch_labels]

        text_inputs = tokenizer(batch_labels, padding="max_length", return_tensors="pt", truncation=True).input_ids.to(device)
        
        text_embeddings = text_encoder(text_inputs).last_hidden_state


        latents = vae.encode(batch_images).latent_dist.sample() * 0.18215  # Scale factor

        # Generate random noise
        noise = torch.randn_like(latents).to(device)

        # Sample random timesteps
        timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (batch_images.size(0),), device=device).long()

        # Add noise to the latent representations based on timesteps
        noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

        # Forward pass through the UNet (with mixed precision)
        with autocast():
            noise_pred = unet(noisy_latents, timesteps, text_embeddings).sample

            # Compute the loss (MSE between predicted noise and actual noise)
            loss = criterion(noise_pred, noise)

        # Backpropagation with mixed precision
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

torch.save(unet.state_dict(), "unet_trained.pth")
torch.save(vae.state_dict(), "vae_trained.pth")