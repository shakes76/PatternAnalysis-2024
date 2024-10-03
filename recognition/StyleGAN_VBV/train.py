import os
import torch
from torch import optim
from tqdm import tqdm
import matplotlib.pyplot as plt
from modules import Generator, Discriminator
from dataset import get_loader

# Constants
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DATASET_PATH = '/home/groups/comp3710/ADNI' 
BATCH_SIZE = 32
Z_DIM = 512
W_DIM = 512
IN_CHANNELS = 512
LR = 1e-3
LAMBDA_GP = 10
PROGRESSIVE_EPOCHS = [30, 30, 30, 30, 30, 30] 

def gradient_penalty(critic, real, fake, alpha, train_step, device="cpu"):
    BATCH_SIZE, C, H, W = real.shape
    beta = torch.rand((BATCH_SIZE, 1, 1, 1)).repeat(1, C, H, W).to(device)
    interpolated_images = real * beta + fake.detach() * (1 - beta)
    interpolated_images.requires_grad_(True)
    mixed_scores = critic(interpolated_images, alpha, train_step)
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

def train_fn(critic, gen, loader, dataset, step, alpha, opt_critic, opt_gen):
    loop = tqdm(loader, leave=True)

    for batch_idx, (real, _) in enumerate(loop):
        real = real.to(DEVICE)
        cur_batch_size = real.shape[0]
        noise = torch.randn(cur_batch_size, Z_DIm).to(DEVICE)
        fake = gen(noise, alpha, step)
        critic_real = critic(real, alpha, step)
        critic_fake = critic(fake.detach(), alpha, step)
        gp = gradient_penalty(critic, real, fake, alpha, step, DEVICE)
        loss_critic = (
            -(torch.mean(critic_real) - torch.mean(critic_fake))
            + LAMBDA_GP * gp
            + (0.001) * torch.mean(critic_real ** 2)
        )

        critic.zero_grad()
        loss_critic.backward()
        opt_critic.step()

        gen_fake = critic(fake, alpha, step)
        loss_gen = -torch.mean(gen_fake)

        gen.zero_grad()
        loss_gen.backward()
        opt_gen.step()

        alpha += cur_batch_size / (
            PROGRESSIVE_EPOCHS[step] * 0.5 * len(dataset)
        )
        alpha = min(alpha, 1)

        loop.set_postfix(
            gp=gp.item(),
            loss_critic=loss_critic.item()
        )
    return alpha

def main():
    # Initialize models
    gen = Generator(Z_DIM, W_DIM, IN_CHANNELS).to(DEVICE)
    critic = Discriminator(IN_CHANNELS).to(DEVICE)
    
    # Optimizers
    opt_gen = optim.Adam(gen.parameters(), lr=LR, betas=(0.0, 0.99))
    opt_critic = optim.Adam(critic.parameters(), lr=LR, betas=(0.0, 0.99))

    step = 0  # Start with the smallest image size
    for num_epochs in PROGRESSIVE_EPOCHS:
        print(f'\nCurrent image size: {4 * 2**step}')
        
        # Load data
        loader = get_loader(DATASET_PATH, batch_size=BATCH_SIZE, img_size=(4 * 2**step, 4 * 2**step))

        for epoch in range(num_epochs):
            print(f'Epoch [{epoch + 1}/{num_epochs}]')
            alpha = 0  # Start alpha for this size
            train_fn(critic, gen, loader, alpha, opt_critic, opt_gen, step)

        # Generate and save examples
        generate_examples(gen, step)

        step += 1  # Move to the next image size

if __name__ == "__main__":
    main()
