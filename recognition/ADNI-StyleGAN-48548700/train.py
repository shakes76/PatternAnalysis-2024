﻿"""
File: train.py
Author: Baibhav Mund (ID: 48548700)

Description:
    Trains a StyleGAN2 model using a generator, discriminator (critic), and 
    mapping network with adversarial training. Features gradient penalty, 
    path length penalty, and AMP for optimized training.

Usage:
    1. Set `DATASET` path and adjust hyperparameters.
    2. Ensure GPU is available; defaults to CPU if not.
    3. Run with `python train.py`.

Parameters:
    - DATASET: Dataset path
    - DEVICE: CPU or GPU
    - EPOCHS, BATCH_SIZE, LEARNING_RATE, Z_DIM, W_DIM, LAMBDA_GP, etc.

Output:
    - Model checkpoints, loss logs (CSV), style code logs (npy) and loss plots (PNG)
"""

from dataset import *  
from modules import *  
import torch 
from tqdm import tqdm 
from torch import optim
import matplotlib.pyplot as plt
import csv
import numpy as np

# Define constants and hyperparameters
DATASET                 = "./train"  # Path to the dataset
CLASS_TYPE              = "AD" # Train either on Alzeihmer's or Normal Data 
DEVICE                  = "cuda" if torch.cuda.is_available() else "cpu"  # Use GPU if available, otherwise use CPU
EPOCHS                  = 151 # Number of training epochs
LEARNING_RATE           = 1e-3  # Learning rate for optimization
BATCH_SIZE              = 32  # Batch size for training
LOG_RESOLUTION          = 7  # Logarithmic resolution used for 128*128 images
Z_DIM                   = 256  # Dimension of the latent space
W_DIM                   = 256  # Dimension of the mapping network output
LAMBDA_GP               = 10  # Weight for the gradient penalty term

# Function to compute the gradient penalty for the discriminator
def gradient_penalty(critic, real, fake, device= DEVICE):
    # Compute gradient penalty for the critic
    BATCH_SIZE, C, H, W = real.shape
    beta = torch.rand((BATCH_SIZE, 1, 1, 1)).repeat(1, C, H, W).to(device)
    interpolated_images = real * beta + fake.detach() * (1 - beta)
    interpolated_images.requires_grad_(True)

    # Calculate discriminator scores
    mixed_scores = critic(interpolated_images)

    # Take the gradient of the scores with respect to the images
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

# Function to generate latent vectors 'w' from random noise
def get_w(batch_size):
    # Generate 'w' from random noise
    z = torch.randn(batch_size, W_DIM).to(DEVICE)
    w = mapping_network(z)
    return w[None, :, :].expand(LOG_RESOLUTION, -1, -1)

# Function to generate noise inputs for the generator
def get_noise(batch_size):
    # Generate noise inputs for the generator
    noise = []
    resolution = 4

    for i in range(LOG_RESOLUTION):
        if i == 0:
            n1 = None
        else:
            n1 = torch.randn(batch_size, 1, resolution, resolution, device=DEVICE)
        n2 = torch.randn(batch_size, 1, resolution, resolution, device=DEVICE)

        noise.append((n1, n2))

        resolution *= 2

    return noise

style_codes_log = []

# Training function for the discriminator and generator
def train_fn(
    critic,
    gen,
    path_length_penalty,
    loader,
    opt_critic,
    opt_gen,
    opt_mapping_network,
    epoch,

):
    loop = tqdm(loader, leave=True)  # Create a tqdm progress bar for training iterations
    
    G_losses = []
    D_losses = []
    
    for batch_idx, (real, _) in enumerate(loop):
        real = real.to(DEVICE)  # Move real data to the specified device
        cur_batch_size = real.shape[0]

        w = get_w(cur_batch_size)  # Generate 'w' from random noise
        noise = get_noise(cur_batch_size)  # Generate noise inputs for the generator
        with torch.amp.autocast('cuda'):  # Use automatic mixed-precision (AMP) for faster training
            fake, style_codes = gen(w, noise, return_style=True)  # Generate fake images
            critic_fake = critic(fake.detach())  # Get critic scores for fake images
            critic_real = critic(real)  # Get critic scores for real images
            gp = gradient_penalty(critic, real, fake, device=DEVICE)  # Compute gradient penalty
            
            loss_critic = (
                -(torch.mean(critic_real) - torch.mean(critic_fake))  # Critic loss
                + LAMBDA_GP * gp  # Gradient penalty term
                + (0.001 * torch.mean(critic_real ** 2))  # Regularization term
            )
        D_losses.append(loss_critic.item())

        critic.zero_grad()  # Reset gradients for the critic
        loss_critic.backward()  # Backpropagate the critic loss
        opt_critic.step()  # Update critic's weights

        gen_fake = critic(fake)  # Get critic scores for fake images
        loss_gen = -torch.mean(gen_fake)  # Generator loss

        if batch_idx % 16 == 0:  # Apply path length penalty every 16 batches
            plp = path_length_penalty(w, fake)
            if not torch.isnan(plp):
                loss_gen = loss_gen + plp  # Update generator loss with path length penalty
        G_losses.append(loss_gen.item())
        mapping_network.zero_grad()  # Reset gradients for the mapping network
        gen.zero_grad()  # Reset gradients for the generator
        loss_gen.backward()  # Backpropagate the generator loss
        opt_gen.step()  # Update generator's weights
        opt_mapping_network.step()  # Update mapping network's weights

        loop.set_postfix(gp=gp.item(), loss_critic=loss_critic.item())

        if epoch % 5 == 0 and batch_idx == 0:
            style_codes_np = torch.cat(style_codes,dim=1).detach().cpu().numpy() # log style codes for embeddings
            style_codes_log.append(style_codes_np)
            
    return (D_losses, G_losses)

# Function to save Gen and critic losses to csv file for each iteration    
def save_losses_to_csv(g_losses, d_losses, filename="losses.csv"):
    with open(filename, mode="a", newline="") as f:
        # Create a CSV writer object to write rows to the file
        writer = csv.writer(f)
        writer.writerow(["Epoch", "Generator Loss", "Discriminator Loss"])
        
        for i in range(len(g_losses)):
            gen_loss = g_losses[i] if i < len(g_losses) else ""
            disc_loss = d_losses[i] if i < len(d_losses) else ""
            writer.writerow([i, gen_loss, disc_loss])


# Initialize the mapping network, generator, and critic on the specified device
mapping_network     = MappingNetwork(Z_DIM, W_DIM).to(DEVICE)  # Initialize mapping network
gen                 = Generator(LOG_RESOLUTION, W_DIM).to(DEVICE)  # Initialize generator
critic              = Discriminator(LOG_RESOLUTION).to(DEVICE)  # Initialize critic

loader = get_loader(DATASET, LOG_RESOLUTION, BATCH_SIZE, CLASS_TYPE)

gen                 = Generator(LOG_RESOLUTION, W_DIM).to(DEVICE)

path_length_penalty = PathLengthPenalty(0.99).to(DEVICE)

opt_gen             = optim.Adam(gen.parameters(), lr=LEARNING_RATE, betas=(0.0, 0.99))
opt_critic          = optim.Adam(critic.parameters(), lr=LEARNING_RATE, betas=(0.0, 0.99))
opt_mapping_network = optim.Adam(mapping_network.parameters(), lr=LEARNING_RATE, betas=(0.0, 0.99))

if __name__ == "__main__":
    # Set the generator, critic, and mapping network to training mode
    gen.train()
    critic.train()
    mapping_network.train()

    # Initialize lists to store total losses for the generator and discriminator (critic)
    Total_G_Losses = []
    Total_D_Losses = []

    # Loop over the number of epochs
    for epoch in range(EPOCHS):
        # Train the generator and discriminator for one epoch, 
        # and return the losses for each batch in G_Losses and D_Losses
        G_Losses, D_Losses = train_fn(
            critic,                  # Critic (discriminator) model
            gen,                     # Generator model
            path_length_penalty,      # Path length penalty function
            loader,                  # DataLoader for the dataset
            opt_critic,              # Optimizer for the critic
            opt_gen,                 # Optimizer for the generator
            opt_mapping_network,      # Optimizer for the mapping network
            epoch,                  # Epoch
        )

        # Extend the total loss lists with the losses from the current epoch
        Total_G_Losses.extend(G_Losses)
        Total_D_Losses.extend(D_Losses)

        # Save the generator and discriminator losses to a CSV file after every epoch
        save_losses_to_csv(Total_G_Losses, Total_D_Losses, filename="total_losses.csv")

        # Save model checkpoints every 5 epochs
        if epoch % 5 == 0:
            # Save the generator's state
            torch.save(gen.state_dict(), f'generator_epoch{epoch}.pt')
            # Save the critic's state
            torch.save(critic.state_dict(), f'critic_epoch{epoch}.pt')

    np.save('style_codes_log.npy', np.array(style_codes_log))    

    plt.figure(figsize=(10,5))
    plt.title("Generator Loss During Training")
    plt.plot(Total_G_Losses, label="G", color="blue")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig('gen_loss.png')

    plt.figure(figsize=(10,5))
    plt.title("Discriminator Loss During Training")
    plt.plot(Total_D_Losses, label="D", color="orange")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig('dis_loss.png')