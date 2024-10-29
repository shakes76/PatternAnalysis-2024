import os
import torch
from modules import Generator, Discriminator
from dataset import get_dataloader
from torch import optim
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import torch.nn.functional as F
import argparse
import matplotlib.pyplot as plt

def train(root_dir, batch_size, num_epochs, output_dir):
    dataloader, dataset = get_dataloader(root_dir, batch_size)
    generator = Generator(z_dim=512, w_dim=512, in_channels=512, img_channels=1).to(device)
    discriminator = Discriminator(in_channels=512, img_channels=1).to(device)
    
    gen_optimizer = optim.Adam(generator.parameters(), lr=0.001, betas=(0.5, 0.999))
    disc_optimizer = optim.Adam(discriminator.parameters(), lr=0.0005, betas=(0.5, 0.999))
    
    scaler = GradScaler()
    
    os.makedirs(output_dir, exist_ok=True)
    
    gen_losses = []
    disc_losses = []
    
    print(f"Loaded {len(dataset)} images for training.")
    for epoch in range(num_epochs):
        gen_loss_accum = 0.0
        disc_loss_accum = 0.0
        loop = tqdm(dataloader, desc=f"Epoch [{epoch+1}/{num_epochs}]")
        for batch in loop:
            real_images, _ = batch
            real_images = real_images.to(device)
            batch_size_current = real_images.size(0)
            
            # Train Discriminator
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
            
            # Train Generator
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
        
        avg_gen_loss = gen_loss_accum / len(dataloader)
        avg_disc_loss = disc_loss_accum / len(dataloader)
        gen_losses.append(avg_gen_loss)
        disc_losses.append(avg_disc_loss)
        print(f"Epoch [{epoch+1}/{num_epochs}] Generator Loss: {avg_gen_loss:.4f}, Discriminator Loss: {avg_disc_loss:.4f}")

    plt.figure(figsize=(10,5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(gen_losses, label="Generator")
    plt.plot(disc_losses, label="Discriminator")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(output_dir, "loss_plot.png"))
    plt.show()
    print(f"Training loss plot saved at {os.path.join(output_dir, 'loss_plot.png')}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train StyleGAN2 on ADNI Dataset')
    parser.add_argument('--data_root', type=str, default='/path/to/data', help='Path to training data')
    parser.add_argument('--num_epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--output_dir', type=str, default='./training_outputs', help='Directory to save training plots')
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    train(
        root_dir=args.data_root,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        output_dir=args.output_dir
    )