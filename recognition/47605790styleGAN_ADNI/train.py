'''
This file contain the training loop for this project,
initialize every components required
and control the flow of data 
'''

import torch
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader
from modules import Generator, Discriminator, MappingNetwork, adversarial_loss
from dataset import load_data

# Training function
def train(generator, discriminator, mapping_network, train_loader, epochs, device, lr=0.0001, gen_updates_per_disc=4):
    """
    Trains the generator and discriminator over multiple epochs.
    
    Args:
        generator: The generator model.
        discriminator: The discriminator model.
        mapping_network: The mapping network for transforming latent space.
        train_loader: DataLoader for training data.
        epochs: Number of training epochs.
        device: The device (GPU) to train on.
        lr: Learning rate for the optimizers.
        gen_updates_per_disc: Number of times to update the generator for each discriminator update.
        
    Returns:
        None: Trains and prints progress during training.
    """
    # Optimizers
    optimizer_G = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
    
    # Schedulers to reduce learning rates over time
    scheduler_G = torch.optim.lr_scheduler.StepLR(optimizer_G, step_size=20, gamma=0.5)
    scheduler_D = torch.optim.lr_scheduler.StepLR(optimizer_D, step_size=20, gamma=0.5)
    
    # Fixed latent vector for evaluation
    fixed_z = torch.randn(8, 512, device=device)

    for epoch in range(epochs):
        for i, (real_imgs, _) in enumerate(train_loader):
            real_imgs = real_imgs.to(device)
            batch_size = real_imgs.size(0)
            
            # Adversarial ground truths
            valid = torch.ones(batch_size, 1, device=device)
            fake = torch.zeros(batch_size, 1, device=device)

            # Train Generator More Frequently 
            for _ in range(gen_updates_per_disc):
                optimizer_G.zero_grad()
                
                # Sample noise as generator input
                z = torch.randn(batch_size, 512, device=device)
                
                # Map latent vector to intermediate space
                w = mapping_network(z)

                # Generate images
                gen_imgs = generator(z, w)

                # Loss for the generator
                g_loss = adversarial_loss(discriminator(gen_imgs), valid)
                g_loss.backward()
                optimizer_G.step()

            # Train Discriminator
            optimizer_D.zero_grad()

            # Loss for real images
            real_loss = adversarial_loss(discriminator(real_imgs), valid)

            # Loss for fake images
            fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)

            # Total discriminator loss
            d_loss = (real_loss + fake_loss) / 2
            d_loss.backward()
            optimizer_D.step()

            # Print progress
            if i % 100 == 0:
                print(f"[Epoch {epoch}/{epochs}] [Batch {i}/{len(train_loader)}] "
                      f"[D loss: {d_loss.item()}] [G loss: {g_loss.item()}]")

        # Step the schedulers to reduce the learning rate after each epoch
        scheduler_G.step()
        scheduler_D.step()

        # Save images at the end of each epoch
        save_generated_images(generator, mapping_network, fixed_z, epoch, device)

def save_generated_images(generator, mapping_network, fixed_z, epoch, device):
    """
    Saves generated images at the end of each epoch.
    
    Args:
        generator: The generator model.
        mapping_network: The mapping network for style modulation.
        fixed_z: Fixed latent vector for image generation.
        epoch: The current epoch number.
        device: The device (CPU/GPU) to use.

    Returns:
        None: Saves generated images to disk.
    """
    generator.eval()
    with torch.no_grad():
        w = mapping_network(fixed_z)
        generated_imgs = generator(fixed_z, w).cpu()
        # Save images to a directory
        for idx, img in enumerate(generated_imgs):
            img = img.permute(1, 2, 0).numpy()  # Move channels to last dimension
            img = (img * 127.5 + 127.5).astype("uint8")  # Denormalize
            img_path = f"C:/Users/Admin/Downloads/images/epoch_{epoch}_img_{idx}.png"
            Image.fromarray(img).save(img_path)
    generator.train()

# Main entry point for training
if __name__ == "__main__":
    """
    Initializes components, loads data, and begins training the StyleGAN.
    """
    # Directories
    train_dir = r'C:/Users/Admin/Downloads/ADNI_AD_NC_2D/AD_NC/train'
    test_dir = r'C:/Users/Admin/Downloads/ADNI_AD_NC_2D/AD_NC/test'
    
    # Load data
    train_loader, _ = load_data(train_dir, test_dir, batch_size=32)

    # Initialize models
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator = Generator().to(device)
    discriminator = Discriminator().to(device)
    mapping_network = MappingNetwork().to(device)

    # Train the models
    train(generator, discriminator, mapping_network, train_loader, epochs=100, device=device, gen_updates_per_disc=4)
