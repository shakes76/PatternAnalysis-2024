import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm
import os
import logging
from dataset import ADNIDataset, get_transform
from modules import StyleGAN2

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Hyperparameters
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LEARNING_RATE_G = 0.0005
LEARNING_RATE_D = 0.0001
BATCH_SIZE = 50
NUM_EPOCHS = 100
LATENT_DIM = 512
IMAGE_SIZE = 256
CHECKPOINT_INTERVAL = 10
SAMPLE_INTERVAL = 500
LAMBDA_GP = 5  # Gradient penalty lambda
DATA_PATH = "/home/groups/comp3710/ADNI/AD_NC/train/"


def compute_gradient_penalty(discriminator, real_samples, fake_samples):
    """Compute gradient penalty for WGAN-GP"""
    # Generate random interpolation factors
    alpha = torch.rand(real_samples.size(0), 1, 1, 1).to(real_samples.device)

    # Create interpolated images
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)

    # Calculate discriminator output for interpolated images
    d_interpolates = discriminator(interpolates)

    # Calculate gradients of discriminator output with respect to inputs
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones_like(d_interpolates),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    # Calculate gradient penalty
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
    elif classname == 'Linear':
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0)
    elif classname == 'NoiseInjection':
        nn.init.constant_(m.weight.data, 0.0)
    elif classname == 'Generator':
        nn.init.normal_(m.input.data, 0.0, 0.02)


def save_sample_images(model, epoch, device, sample_dir="output/samples"):
    """Generate and save sample images"""
    os.makedirs(sample_dir, exist_ok=True)
    model.eval()
    with torch.no_grad():
        # Generate fixed number of samples
        fixed_noise = torch.randn(25, LATENT_DIM).to(device)
        fake_images = model.generate(fixed_noise)
        save_image(fake_images,
                   f"{sample_dir}/epoch_{epoch}.png",
                   nrow=5,
                   normalize=True)
    model.train()


def train():
    # Create output directories
    os.makedirs("output/images", exist_ok=True)
    os.makedirs("output/checkpoints", exist_ok=True)
    os.makedirs("output/samples", exist_ok=True)
    os.makedirs("output/final", exist_ok=True)

    # Load the dataset
    dataset = ADNIDataset(root_dir=DATA_PATH, transform=get_transform())
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

    # Initialize the StyleGAN2 model
    model = StyleGAN2(
        w_dim=LATENT_DIM,
        num_layers=7,
        channels=[512, 512, 512, 512, 256, 128, 64],
        img_size=256
    ).to(DEVICE)
    model.apply(weights_init)

    # Optimizers
    optimizer_G = optim.Adam(model.generator.parameters(), lr=LEARNING_RATE_G, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(model.discriminator.parameters(), lr=LEARNING_RATE_D, betas=(0.5, 0.999))

    # Learning rate schedulers
    scheduler_G = optim.lr_scheduler.ReduceLROnPlateau(optimizer_G, mode='min', factor=0.5, patience=5, verbose=True)
    scheduler_D = optim.lr_scheduler.ReduceLROnPlateau(optimizer_D, mode='min', factor=0.5, patience=5, verbose=True)

    # Loss function
    adversarial_loss = nn.BCEWithLogitsLoss()

    # Training loop
    total_steps = 0
    best_g_loss = float('inf')
    for epoch in range(NUM_EPOCHS):
        g_losses = []
        d_losses = []

        for i, (real_images, _) in enumerate(tqdm(dataloader, desc=f"Epoch {epoch + 1}/{NUM_EPOCHS}")):
            real_images = real_images.to(DEVICE)
            batch_size = real_images.size(0)

            # Train Discriminator
            optimizer_D.zero_grad()

            # Generate fake images
            z = torch.randn(batch_size, LATENT_DIM).to(DEVICE)
            fake_images = model.generate(z)

            # Add dynamic noise to inputs
            noise_factor = max(0, 0.05 * (1 - total_steps / 50000))  # Decay noise over time
            real_noisy = real_images + noise_factor * torch.randn_like(real_images)
            fake_noisy = fake_images + noise_factor * torch.randn_like(fake_images)

            # Get discriminator outputs
            real_validity = model.discriminate(real_noisy)
            fake_validity = model.discriminate(fake_noisy.detach())

            # Calculate gradient penalty
            gradient_penalty = compute_gradient_penalty(model.discriminator, real_images, fake_images)

            # Discriminator loss
            d_loss = (adversarial_loss(real_validity, torch.ones_like(real_validity)) +
                      adversarial_loss(fake_validity,
                                       torch.zeros_like(fake_validity))) / 2 + LAMBDA_GP * gradient_penalty

            d_loss.backward()
            optimizer_D.step()
            d_losses.append(d_loss.item())

            # Train Generator
            optimizer_G.zero_grad()

            z = torch.randn(batch_size, LATENT_DIM).to(DEVICE)
            fake_images = model.generate(z)
            fake_validity = model.discriminate(fake_images)

            g_loss = adversarial_loss(fake_validity, torch.ones_like(fake_validity))

            g_loss.backward()
            optimizer_G.step()
            g_losses.append(g_loss.item())

            # Logging
            if total_steps % 100 == 0:
                logger.info(f"[Epoch {epoch + 1}/{NUM_EPOCHS}] [Batch {i + 1}/{len(dataloader)}] "
                            f"[D loss: {d_loss.item():.4f}] [G loss: {g_loss.item():.4f}] "
                            f"[Noise Factor: {noise_factor:.4f}]")

            # Save generated samples
            if total_steps % SAMPLE_INTERVAL == 0:
                with torch.no_grad():
                    fake_samples = model.generate(torch.randn(25, LATENT_DIM).to(DEVICE))
                    save_image(fake_samples, f"output/images/sample_{total_steps}.png", nrow=5, normalize=True)

            total_steps += 1

        # Calculate average losses for the epoch
        avg_g_loss = sum(g_losses) / len(g_losses)
        avg_d_loss = sum(d_losses) / len(d_losses)

        # Update learning rates based on average losses
        scheduler_G.step(avg_g_loss)
        scheduler_D.step(avg_d_loss)

        # Log epoch statistics
        logger.info(f"Epoch {epoch + 1}/{NUM_EPOCHS} - "
                    f"Avg G Loss: {avg_g_loss:.4f}, Avg D Loss: {avg_d_loss:.4f}")

        # Save sample images every 10 epochs
        if (epoch + 1) % 10 == 0:
            save_sample_images(model, epoch + 1, DEVICE)
            logger.info(f"Saved sample images for epoch {epoch + 1}")

        # Save checkpoint if we have the best generator loss so far
        if avg_g_loss < best_g_loss:
            best_g_loss = avg_g_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_G_state_dict': optimizer_G.state_dict(),
                'optimizer_D_state_dict': optimizer_D.state_dict(),
                'scheduler_G_state_dict': scheduler_G.state_dict(),
                'scheduler_D_state_dict': scheduler_D.state_dict(),
                'g_loss': avg_g_loss,
                'd_loss': avg_d_loss,
            }, f"output/checkpoints/stylegan2_best.pth")
            logger.info(f"Saved best model checkpoint with G_loss: {avg_g_loss:.4f}")

        # Regular checkpoint saving
        if (epoch + 1) % CHECKPOINT_INTERVAL == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_G_state_dict': optimizer_G.state_dict(),
                'optimizer_D_state_dict': optimizer_D.state_dict(),
                'scheduler_G_state_dict': scheduler_G.state_dict(),
                'scheduler_D_state_dict': scheduler_D.state_dict(),
                'g_loss': avg_g_loss,
                'd_loss': avg_d_loss,
            }, f"output/checkpoints/stylegan2_epoch_{epoch + 1}.pth")

    # Save final model and generate final samples
    logger.info("Saving final model and samples...")
    final_checkpoint_path = "output/final/stylegan2_final.pth"
    torch.save({
        'epoch': NUM_EPOCHS,
        'model_state_dict': model.state_dict(),
        'optimizer_G_state_dict': optimizer_G.state_dict(),
        'optimizer_D_state_dict': optimizer_D.state_dict(),
        'scheduler_G_state_dict': scheduler_G.state_dict(),
        'scheduler_D_state_dict': scheduler_D.state_dict(),
        'final_g_loss': avg_g_loss,
        'final_d_loss': avg_d_loss,
    }, final_checkpoint_path)

    # Generate final sample images
    save_sample_images(model, NUM_EPOCHS, DEVICE, "output/final")

    logger.info("Training completed. Final model and samples saved.")


if __name__ == "__main__":
    train()