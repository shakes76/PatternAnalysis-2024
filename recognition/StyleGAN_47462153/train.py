import os
import torch
from modules import Generator, Discriminator
from dataset import get_dataloader
from torch import optim
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import torch.nn.functional as F

def train():
    dataloader, dataset = get_dataloader(root_dir, batch_size)
    generator = Generator(z_dim=512, w_dim=512, in_channels=512, img_channels=1).to(device)
    discriminator = Discriminator(in_channels=512, img_channels=1).to(device)
    
    gen_optimizer = optim.Adam(generator.parameters(), lr=0.001, betas=(0.5, 0.999))
    disc_optimizer = optim.Adam(discriminator.parameters(), lr=0.0005, betas=(0.5, 0.999))
    
    scaler = GradScaler()
    
    print(f"Loaded {len(dataset)} images for training.")
    for epoch in range(num_epochs):
        gen_loss_accum = 0.0
        disc_loss_accum = 0.0
        loop = tqdm(dataloader, desc=f"Epoch [{epoch+1}/{num_epochs}]")
        for batch in loop:
            real_images, _ = batch
            real_images = real_images.to(device)
            batch_size_current = real_images.size(0)
            
            disc_optimizer.zero_grad()
            noise = torch.randn(batch_size_current, 512).to(device)
            fake_images = generator(noise).detach()
            with autocast():
                real_output = discriminator(real_images)
                fake_output = discriminator(fake_images)
                disc_loss_real = F.binary_cross_entropy_with_logits(real_output, torch.ones_like(real_output))
                disc_loss_fake = F.binary_cross_entropy_with_logits(fake_output, torch.zeros_like(fake_output))
                disc_loss = (disc_loss_real + disc_loss_fake) / 2
            scaler.scale(disc_loss).backward()
            scaler.step(disc_optimizer)
            scaler.update()
            disc_loss_accum += disc_loss.item()
            
            gen_optimizer.zero_grad()
            noise = torch.randn(batch_size_current, 512).to(device)
            fake_images = generator(noise)
            with autocast():
                fake_output = discriminator(fake_images)
                gen_loss = F.binary_cross_entropy_with_logits(fake_output, torch.ones_like(fake_output))
            scaler.scale(gen_loss).backward()
            scaler.step(gen_optimizer)
            scaler.update()
            gen_loss_accum += gen_loss.item()
            
            loop.set_postfix(gen_loss=gen_loss.item(), disc_loss=disc_loss.item())
        
        print(f"Epoch [{epoch+1}/{num_epochs}] Generator Loss: {gen_loss_accum:.4f}, Discriminator Loss: {disc_loss_accum:.4f}")

if __name__ == "__main__":
    root_dir = '/path/to/data'
    batch_size = 16
    num_epochs = 10
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train()
