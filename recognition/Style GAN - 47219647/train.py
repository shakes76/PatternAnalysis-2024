import torch
import torch.nn as nn
import torch.optim as optim
from dataset import data_set_creator
import numpy as np
import os
from torchvision.utils import save_image
from tqdm import tqdm

from modules import *
from dataset import data_set_creator
from params import *

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# Function to generate examples at each step
def generate_examples(gen, steps, n=3):
    gen.eval()
    alpha = 1.0
    for i in range(n):
        with torch.no_grad():
            img = gen(torch.randn(1, Z_DIM).to(DEVICE), alpha, steps)
            if not os.path.exists(f'saved_examples/step{steps}'):
                os.makedirs(f'saved_examples/step{steps}')
            save_image(img * 0.5 + 0.5, f"saved_examples/step{steps}/img_{i}.png")
    gen.train()

# Function to compute gradient penalty for WGAN-GP
def gradient_penalty(disc, real, fake, alpha, train_step, device="cpu"):
    BATCH_SIZE, C, H, W = real.shape
    beta = torch.rand((BATCH_SIZE, 1, 1, 1)).repeat(1, C, H, W).to(device)
    interpolated_images = real * beta + fake.detach() * (1 - beta)
    interpolated_images.requires_grad_(True)

    mixed_scores = disc(interpolated_images, alpha, train_step)
    gradient = torch.autograd.grad(
        inputs=interpolated_images,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True,
    )[0]

    gradient = gradient.view(gradient.shape[0], -1)
    gradient_norm = gradient.norm(2, dim=1)
    gradient_penalty = torch.mean((gradient_norm - 1) ** 2)

    return gradient_penalty




def train_fn(disc, gen, loader, dataset, step, alpha, opt_disc, opt_gen, disc_losses, gen_losses):
    loop = tqdm(loader, leave=True)

    for batch_idx, (real, _) in enumerate(loop):
        real = real.to(DEVICE)
        cur_batch_size = real.shape[0]

        noise = torch.randn(cur_batch_size, Z_DIM).to(DEVICE)

        # Forward passes
        fake = gen(noise, alpha, step)
        disc_real = disc(real, alpha, step)
        disc_fake = disc(fake.detach(), alpha, step)

        # Gradient penalty and discriminator loss
        gp = gradient_penalty(disc, real, fake, alpha, step, device=DEVICE)
        loss_disc = (- (torch.mean(disc_real) - torch.mean(disc_fake))
                     + LAMBDA_GP * gp
                     + (0.001 * torch.mean(disc_real ** 2)))

        # Update discriminator
        disc.zero_grad()
        loss_disc.backward()
        opt_disc.step()

        # Generator loss
        gen_fake = disc(fake, alpha, step)
        loss_gen = -torch.mean(gen_fake)

        # Update generator
        gen.zero_grad()
        loss_gen.backward()
        opt_gen.step()

        # Update alpha (progressive growing blending factor)
        alpha += cur_batch_size / (
            (PROGRESSIVE_EPOCHS[current_image_size] * 0.5) * len(dataset)
        )
        alpha = min(alpha, 1)

        # Append losses to lists
        disc_losses.append(loss_disc.item())
        gen_losses.append(loss_gen.item())

        # Display loss in tqdm loop
        loop.set_postfix(gp=gp.item(), loss_disc=loss_disc.item())

    return alpha

def plot_tsne_style_space(gen, num_samples, Z_DIM, DEVICE):
    latent_vectors = torch.randn(num_samples, Z_DIM).to(DEVICE)
    
    with torch.no_grad():
        style_codes = gen.mapping_network(latent_vectors)

    # Apply t-SNE to the style codes (W space)
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    style_codes_2d = tsne.fit_transform(style_codes.cpu().numpy())

    # Plot the t-SNE embedding of the style codes (W space)
    plt.figure(figsize=(10, 8))
    plt.scatter(style_codes_2d[:, 0], style_codes_2d[:, 1], s=5, cmap='Spectral')
    plt.title('t-SNE embedding of StyleGAN Latent Space (W space)')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.colorbar()
    plt.show()

if __name__ == "__main__":
    disc_losses = []
    gen_losses = []

    # Main training loop over image sizes
    for step, current_image_size in enumerate(IMAGE_SIZES):
        current_batch_size = BATCH_SIZES[current_image_size]
        current_learning_rate = LEARNING_SIZES[current_image_size]

        print(f"Training at image size {current_image_size}x{current_image_size} with batch size {current_batch_size} and learning rate {current_learning_rate}")

        # Initialize generator and discriminator
        gen = Generator(Z_DIM, W_DIM, IN_CHANNELS, img_channels=CHANNELS_IMG).to(DEVICE)
        disc = Discriminator(IN_CHANNELS, img_channels=CHANNELS_IMG).to(DEVICE)

        # Initialize optimizers
        opt_gen = optim.Adam([
            {"params": [param for name, param in gen.named_parameters() if "map" not in name]},
            {"params": gen.map.parameters(), "lr": 1e-5}
        ], lr=current_learning_rate, betas=(0.0, 0.99))

        opt_disc = optim.Adam(disc.parameters(), lr=current_learning_rate, betas=(0.0, 0.99))

        gen.train()
        disc.train()

        # Get data loader for the current image size
        loader, dataset = data_set_creator(image_size=current_image_size, batch_size=current_batch_size)

        alpha = 1e-5  # Start with very low alpha for progressive blending
        for epoch in range(PROGRESSIVE_EPOCHS[current_image_size]):
            print(f"Epoch [{epoch + 1}/{PROGRESSIVE_EPOCHS[current_image_size]}] at image size {current_image_size}x{current_image_size}")

            # Train for one epoch and track the alpha value
            alpha = train_fn(disc, gen, loader, dataset, step, alpha, opt_disc, opt_gen, disc_losses, gen_losses)

        # After training for the current image size, generate examples
        generate_examples(gen, step)
        # Save model, optimizer states, and losses after each epoch
        save_model(gen, disc, opt_gen, opt_disc, epoch, step, disc_losses, gen_losses, file_path=f"model_checkpoint_step{step}_epoch{epoch}.pth")


