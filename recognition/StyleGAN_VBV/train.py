import os
import torch
from torch import optim
from tqdm import tqdm
import matplotlib.pyplot as plt
from math import log2
from modules import Generator, Discriminator
from dataset import get_loader
from predict import generate_examples

# Constants
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'  # Use GPU if available
DATASET_PATH = '/home/groups/comp3710/ADNI'  # Path to the ADNI dataset
BATCH_SIZE = 32  # Number of images to process in each training batch
Z_DIM = 512  # Dimensionality of the latent space
W_DIM = 512  # Dimensionality for the style space
CHANNELS_IMG = 3  # Number of image channels (RGB)
IN_CHANNELS = 512  # Number of input channels for the model
LR = 1e-3  # Learning rate for the optimizers
LAMBDA_GP = 10  # Weight for the gradient penalty term
PROGRESSIVE_EPOCHS = [30, 30, 30, 30, 30, 30]  # Number of epochs for each progressive training stage
START_TRAIN_IMG_SIZE = 4  # Starting image size for training

# Function to calculate the gradient penalty for the WGAN-GP
def gradient_penalty(critic, real, fake, alpha, train_step, device="cpu"):
    """ Calculate the gradient penalty for the WGAN-GP """
    BATCH_SIZE, C, H, W = real.shape  # Get dimensions of the real images
    beta = torch.rand((BATCH_SIZE, 1, 1, 1)).repeat(1, C, H, W).to(device)  # Random weighting factor
    # Create interpolated images between real and fake
    interpolated_images = real * beta + fake.detach() * (1 - beta)
    interpolated_images.requires_grad_(True)  # Enable gradient tracking for interpolation

    mixed_scores = critic(interpolated_images, alpha, train_step)  # Get critic scores for interpolated images
    # Calculate gradients with respect to the interpolated images
    gradient = torch.autograd.grad(
        inputs=interpolated_images,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True,
    )[0]
    gradient = gradient.view(gradient.shape[0], -1)  # Flatten gradients
    gradient_norm = gradient.norm(2, dim=1)  # Calculate L2 norm of the gradients
    gradient_penalty = torch.mean((gradient_norm - 1) ** 2)  # Calculate the gradient penalty
    return gradient_penalty  # Return the calculated penalty

# Function to train the generator and discriminator for one step
def train_fn(critic, gen, loader, dataset, step, alpha, opt_critic, opt_gen):
    """Main training function for one training step"""
    loop = tqdm(loader, leave=True)  # Progress bar for loading batches

    # Initialize lists to store losses for plotting
    losses_critic = []
    losses_gen = []

    for batch_idx, (real, _) in enumerate(loop):
        real = real.to(DEVICE)  # Move real images to the device
        cur_batch_size = real.shape[0]  # Current batch size
        noise = torch.randn(cur_batch_size, Z_DIM).to(DEVICE)  # Generate random noise for the generator
        fake = gen(noise, alpha, step)  # Generate fake images

        # Get scores from the critic for real and fake images
        critic_real = critic(real, alpha, step)
        critic_fake = critic(fake.detach(), alpha, step)
        
        # Calculate the gradient penalty
        gp = gradient_penalty(critic, real, fake, alpha, step, DEVICE)
        
        # Calculate the critic loss
        loss_critic = (
            -(torch.mean(critic_real) - torch.mean(critic_fake))  # WGAN loss
            + LAMBDA_GP * gp  # Add gradient penalty
            + (0.001) * torch.mean(critic_real ** 2)  # Regularization term
        )

        critic.zero_grad()  # Clear gradients for the critic
        loss_critic.backward()  # Backpropagate the loss
        opt_critic.step()  # Update the critic's parameters

        # Calculate the generator loss
        gen_fake = critic(fake, alpha, step)
        loss_gen = -torch.mean(gen_fake)  # Negative mean for generator loss

        gen.zero_grad()  # Clear gradients for the generator
        loss_gen.backward()  # Backpropagate the loss
        opt_gen.step()  # Update the generator's parameters

        # Update alpha for progressive training
        alpha += cur_batch_size / (PROGRESSIVE_EPOCHS[step] * 0.5 * len(dataset))
        alpha = min(alpha, 1)  # Ensure alpha doesn't exceed 1

        # Store losses for plotting later
        losses_critic.append(loss_critic.item())
        losses_gen.append(loss_gen.item())

        # Update progress bar with current losses
        loop.set_postfix(gp=gp.item(), loss_critic=loss_critic.item())
    
    return alpha, losses_critic, losses_gen  # Return updated alpha and losses

# Function to save the generator and critic models
def save_model(gen, critic, step):
    """Save the generator and critic models to disk."""

    # Save them
    torch.save(gen.state_dict(), f'generator_step{step}.pth')  # Save generator's state
    torch.save(critic.state_dict(), f'critic_step{step}.pth')  # Save critic's state

# Function to plot and save the loss curves
def plot_loss(losses_critic, losses_gen, step):
    """Generate and save a loss plot for the generator and critic."""

    plt.figure(figsize=(10, 5))  # Set the figure size
    plt.plot(losses_critic, label='Critic Loss')  # Plot critic loss
    plt.plot(losses_gen, label='Generator Loss')  # Plot generator loss
    plt.title(f'Losses at Step {step}')  # Title for the plot
    plt.xlabel('Batch Number')  # X-axis label
    plt.ylabel('Loss')  # Y-axis label
    plt.legend()  # Show the legend
    plt.savefig(f'loss_step{step}.png')  # Save the plot to disk
    plt.close()  # Close the plot to free up memory

# Initialize generator and critic
gen = Generator(Z_DIM, W_DIM, IN_CHANNELS, CHANNELS_IMG).to(DEVICE)  # Create generator
critic = Discriminator(IN_CHANNELS, CHANNELS_IMG).to(DEVICE)  # Create critic

# Set up optimizer for generator and critic
opt_gen = optim.Adam(
    [{'params': [param for name, param in gen.named_parameters() if 'map' not in name]},
     {'params': gen.map.parameters(), 'lr': 1e-5}], lr=LR, betas=(0.0, 0.99))  # Adam for generator
opt_critic = optim.Adam(critic.parameters(), lr=LR, betas=(0.0, 0.99))  # Adam for critic

# Set models to training mode
gen.train()
critic.train()

# Calculate initial step based on starting image size
step = int(log2(START_TRAIN_IMG_SIZE / 4))

# Loop over all specified epochs for each progressive step
for num_epochs in PROGRESSIVE_EPOCHS[step:]:
    alpha = 1e-7  # Initialize alpha for progressive growing
    loader, dataset = get_loader(4 * 2**step)  # Load dataset for current resolution
    print('Current image size: ' + str(4 * 2**step))  # Print current image size

    # Initialize lists to store losses for this step
    losses_critic = []
    losses_gen = []

    # Loop over the specified number of epochs
    for epoch in range(num_epochs):
        print(f'Epoch [{epoch + 1}/{num_epochs}]')  # Print epoch info

        # Train the generator and discriminator, and get updated alpha and losses
        alpha, step_losses_critic, step_losses_gen = train_fn(
            critic, gen, loader, dataset, step, alpha, opt_critic, opt_gen
        )

        # Append losses from this epoch to the total losses
        losses_critic.extend(step_losses_critic)
        losses_gen.extend(step_losses_gen)

    # Save models after each set of epochs
    save_model(gen, critic, step)

    # Plot and save losses after training
    plot_loss(losses_critic, losses_gen, step)

    # Generate and save example images from the generator
    generate_examples(gen, step)
    
    step += 1  # Increment step for the next iteration (increased image size)
