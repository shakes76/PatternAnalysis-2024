import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import R
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
LEARNING_RATE_G = 0.002
LEARNING_RATE_D = 0.0001
BATCH_SIZE = 32
NUM_EPOCHS = 100
LATENT_DIM = 512
IMAGE_SIZE = 256
CHECKPOINT_INTERVAL = 10
SAMPLE_INTERVAL = 500

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

def train():
    # Create output directories
    os.makedirs("output/images", exist_ok=True)
    os.makedirs("output/checkpoints", exist_ok=True)

    # Load the dataset
    dataset = ADNIDataset(root_dir="./dataset/ADNI/train", transform=get_transform())
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

    # Initialize the StyleGAN2 model
    model = StyleGAN2(w_dim=LATENT_DIM, num_layers=7, channels=[512, 512, 512, 512, 256, 128, 64], img_size=256).to(
        DEVICE)
    model.apply(weights_init)

    # Optimizers
    optimizer_G = optim.Adam(model.generator.parameters(), lr=LEARNING_RATE_G, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(model.discriminator.parameters(), lr=LEARNING_RATE_D, betas=(0.5, 0.999))

    # Loss function
    adversarial_loss = nn.BCEWithLogitsLoss()

    # Training loop
    total_steps = 0
    for epoch in range(NUM_EPOCHS):
        for i, (real_images, _) in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")):
            real_images = real_images.to(DEVICE)
            batch_size = real_images.size(0)

            # Train Discriminator
            optimizer_D.zero_grad()

            z = torch.randn(batch_size, LATENT_DIM).to(DEVICE)
            fake_images = model.generate(z)

            real_validity = model.discriminate(real_images)
            fake_validity = model.discriminate(fake_images.detach())

            d_loss = (adversarial_loss(real_validity, torch.ones_like(real_validity)) +
                      adversarial_loss(fake_validity, torch.zeros_like(fake_validity))) / 2

            d_loss.backward()
            optimizer_D.step()

            # Train Generator
            optimizer_G.zero_grad()

            z = torch.randn(batch_size, LATENT_DIM).to(DEVICE)
            fake_images = model.generate(z)
            fake_validity = model.discriminate(fake_images)

            g_loss = adversarial_loss(fake_validity, torch.ones_like(fake_validity))

            g_loss.backward()
            optimizer_G.step()

            # Logging
            if total_steps % 100 == 0:
                logger.info(f"[Epoch {epoch+1}/{NUM_EPOCHS}] [Batch {i+1}/{len(dataloader)}] "
                            f"[D loss: {d_loss.item():.4f}] [G loss: {g_loss.item():.4f}]")

            # Save generated samples
            if total_steps % SAMPLE_INTERVAL == 0:
                save_image(fake_images[:25], f"output/images/sample_{total_steps}.png", nrow=5, normalize=True)

            total_steps += 1

        # Save model checkpoint
        if (epoch + 1) % CHECKPOINT_INTERVAL == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_G_state_dict': optimizer_G.state_dict(),
                'optimizer_D_state_dict': optimizer_D.state_dict(),
            }, f"output/checkpoints/stylegan2_epoch_{epoch+1}.pth")

    logger.info("Training completed.")

if __name__ == "__main__":
    train()
