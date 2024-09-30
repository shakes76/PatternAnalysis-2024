import os
import torch
from torch import optim
from tqdm import tqdm
from dataset import get_loader
from modules import Generator, Discriminator

# Hyperparameters
Z_DIM = 512
W_DIM = 512
IN_CHANNELS = 512
CHANNELS_IMG = 3
LR = 1e-3
LAMBDA_GP = 10
PROGRESSIVE_EPOCHS = [30] * 6
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Define functions: gradient_penalty, train_fn, generate_examples, main

def gradient_penalty(critic, real, fake, alpha, train_step):
    ...

def train_fn(critic, gen, loader, step, alpha, opt_critic, opt_gen):
    ...

def generate_examples(gen, steps, n=100):
    ...

def main():
    gen = Generator(Z_DIM, W_DIM, IN_CHANNELS, CHANNELS_IMG).to(DEVICE)
    critic = Discriminator(IN_CHANNELS, CHANNELS_IMG).to(DEVICE)
    opt_gen = optim.Adam(gen.parameters(), lr=LR, betas=(0.0, 0.99))
    opt_critic = optim.Adam(critic.parameters(), lr=LR, betas=(0.0, 0.99))

    step = 0
    for num_epochs in PROGRESSIVE_EPOCHS:
        alpha = 1e-7
        loader, dataset = get_loader(4 * 2**step)
        print(f'Current image size: {4 * 2**step}')

        for epoch in range(num_epochs):
            print(f'Epoch [{epoch + 1}/{num_epochs}]')
            alpha = train_fn(critic, gen, loader, step, alpha, opt_critic, opt_gen)

        generate_examples(gen, step)
        step += 1

if __name__ == "__main__":
    main()
