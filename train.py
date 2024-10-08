import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from stylegan2_network import StyleGAN2Generator, StyleGAN2Discriminator
from adni_dataset import ADNIDataset
import os
import numpy as np

# Hyperparameters
z_dim = 512 # Input latent dim
w_dim = 512 # intermediate latent dim
num_mapping_layers = 8
mapping_dropout = 0.1
label_dim = 2  # AD and NC
num_layers = 7
ngf = 256 # Num generator features
ndf = 256 # Num discriminator features
batch_size = 32
num_epochs = 100
lr = 0.0002
beta1 = 0.5
beta2 = 0.999
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create restuls dir
os.makedirs("results", exist_ok=True)
os.makedirs("checkpoints", exist_ok=True)

# Init dataset and loader
dataset = ADNIDataset(root_dir="path", split="train")
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

# Init generator and discriminator
generator = StyleGAN2Generator(z_dim, w_dim, num_mapping_layers, mapping_dropout, label_dim, num_layers, ngf).to(device)
discriminator = StyleGAN2Discriminator(image_size=(256, 240), num_channels=1, ndf=ndf, num_layers=num_layers).to(device)

# Init optimisers
g_optim = optim.Adam(generator.parameters(), lr=lr, betas=(beta1, beta2))
d_optim = optim.Adam(discriminator.parameters(), lr=lr, betas=(beta1, beta2))

# Loss function
criterion = nn.BCEWithLogitsLoss()

# Training loop
total_batches = len(dataloader)
print_interval = 100

for epoch in range(num_epochs):
    for i, (real_images, labels) in enumerate(dataloader):
        batch_size = real_images.size(0)
        real_images = real_images.to(device)
        labels = labels.to(device)

        # Train Discriminator
        d_optim.zero_grad()

        # Generate fake images
        z = torch.randn(batch_size, z_dim).to(device)
        fake_images = generator(z, labels)

        # Discriminator loss for real and fake
        real_output = discriminator(real_images)
        fake_output = discriminator(fake_images.detach())
        d_loss_real = criterion(real_output, torch.ones_like(real_output))
        d_loss_fake = criterion(fake_output, torch.zeros_like(fake_output))
        d_loss = d_loss_real + d_loss_fake

        d_loss.backward()
        d_optim.step()

        # Train Generator
        g_optim.zero_grad()

        # Regenerate fake images (since were detached before)
        fake_images = generator(z, labels)
        fake_output = discriminator(fake_images)

        # Generator loss
        g_loss = criterion(fake_output, torch.ones_like(fake_output))

        g_loss.backward()
        g_optim.step()

        # Print losses
        if i % print_interval == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}] Batch [{i+1}/{total_batches}] "
                  f"D_loss: {d_loss.item():.4f} G_loss: {g_loss.item():.4f}")

    # Save generated images
    with torch.no_grad():
        fixed_z = torch.randn(16, z_dim).to(device)
        fixed_labels = torch.randint(0, 2, (16,)).to(device)
        fake_images = generator(fixed_z, fixed_labels)
        save_image(fake_images, f"results/fake_images_epoch_{epoch+1}.png", normalize=True)

    # Save model
    torch.save({
        'generator_state_dict': generator.state_dict(),
        'discriminator_state_dict': discriminator.state_dict(),
        'g_optimizer_state_dict': g_optim.state_dict(),
        'd_optimizer_state_dict': d_optim.state_dict(),
    }, f"checkpoints/stylegan2_checkpoint_epoch_{epoch+1}.pth")

    print(f"Epoch [{epoch+1}/{num_epochs}] completed")

print("Training complete!")