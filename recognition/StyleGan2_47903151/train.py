"""
Contains source code for training, validating, testing, and saving the model.
Plots losses and metrics during training.
"""
import argparse

import torch
from torch import nn, optim
from torchvision import datasets, transforms
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from math import log2, sqrt
import numpy as np
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from modules import *
from dataset import *
import json

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = 300
LEARNING_RATE = 1e-3
BATCH_SIZE = 32
LOG_RESOLUTION = 8  # for 256*256
Z_DIM = 512
W_DIM = 512
LAMBDA_GP = 10

if os.path.exists("params/data.json"):
    with open("params/data.json", 'r') as f:
        json_data = json.load(f)
    total_epochs = json_data["epochs"]
    generator_loss = json_data["G_loss"]
    discriminator_loss = json_data["D_loss"]
else:
    if not os.path.exists("params"):
        os.mkdir("params")
    total_epochs = 0
    generator_loss = []
    discriminator_loss = []
    json_data = {"epochs": 0,
                 "G_loss": [],
                 "D_loss": []}


def generate_examples(gen, epoch, n=100):
    gen.eval()
    alpha = 1.0
    for i in range(n):
        with torch.no_grad():
            w = get_w(1, W_DIM, DEVICE, mapping_network, LOG_RESOLUTION)
            noise = get_noise(1, LOG_RESOLUTION, DEVICE)
            img = gen(w, noise)
            if not os.path.exists(f'saved_examples/epoch{epoch}'):
                os.makedirs(f'saved_examples/epoch{epoch}')
            save_image(img * 0.5 + 0.5, f"saved_examples/epoch{epoch}/img_{i}.png")

    gen.train()

def train_fn(
        critic,
        gen,
        path_length_penalty,
        loader,
        opt_critic,
        opt_gen,
        opt_mapping_network,
):
    loop = tqdm(loader, leave=True)

    for batch_idx, (real, _) in enumerate(loop):
        real = real.to(DEVICE)
        cur_batch_size = real.shape[0]

        w = get_w(cur_batch_size, W_DIM, DEVICE, mapping_network, LOG_RESOLUTION)
        noise = get_noise(cur_batch_size, LOG_RESOLUTION, DEVICE)
        with torch.cuda.amp.autocast():
            fake = gen(w, noise)
            critic_fake = critic(fake.detach())

            critic_real = critic(real)
            gp = gradient_penalty(critic, real, fake, device=DEVICE)
            loss_critic = (
                    -(torch.mean(critic_real) - torch.mean(critic_fake))
                    + LAMBDA_GP * gp
                    + (0.001 * torch.mean(critic_real ** 2))
            )

        critic.zero_grad()
        loss_critic.backward()
        opt_critic.step()

        gen_fake = critic(fake)
        loss_gen = -torch.mean(gen_fake)

        if batch_idx % 16 == 0:
            plp = path_length_penalty(w, fake)
            if not torch.isnan(plp):
                loss_gen = loss_gen + plp
        if batch_idx % 100 == 0 or batch_idx == cur_batch_size-1:
            # Record loss for graph
            generator_loss.append(loss_gen.item())
            discriminator_loss.append(loss_critic.item())

        mapping_network.zero_grad()
        gen.zero_grad()
        loss_gen.backward()
        opt_gen.step()
        opt_mapping_network.step()

        loop.set_postfix(
            gp=gp.item(),
            loss_critic=loss_critic.item(),
        )


if __name__ == "__main__":
    # Get and parse the command line arguments
    parser = argparse.ArgumentParser(description="COMP3506/7505 Assignment Two: Data Structure Tests")

    parser.add_argument("--directory", type=str, help="Directory for the dataset")
    parser.set_defaults(directory="AD_NC")

    args = parser.parse_args()

    loader = get_loader(LOG_RESOLUTION, BATCH_SIZE, args.directory)
    if not os.path.exists('model/stylegan2ANDC'):
        gen = Generator(LOG_RESOLUTION, W_DIM)
        critic = Discriminator(LOG_RESOLUTION)
        mapping_network = MappingNetwork(Z_DIM, W_DIM)
        path_length_penalty = PathLengthPenalty(0.99)
        if not os.path.exists('model'):
            os.mkdir('model')
        os.mkdir('model/stylegan2ANDC')

    else:
        gen = Generator(LOG_RESOLUTION, W_DIM)
        gen.load_state_dict(torch.load("model/stylegan2ANDC/generator.pth"))

        critic = Discriminator(LOG_RESOLUTION)
        critic.load_state_dict(torch.load("model/stylegan2ANDC/discriminator.pth"))

        mapping_network = MappingNetwork(Z_DIM, W_DIM)
        mapping_network.load_state_dict(torch.load("model/stylegan2ANDC/mapping.pth"))

        path_length_penalty = PathLengthPenalty(0.99)
        path_length_penalty.load_state_dict(torch.load("model/stylegan2ANDC/PLP.pth"))

    get = gen.to(DEVICE)
    critic = critic.to(DEVICE)
    mapping_network = mapping_network.to(DEVICE)
    path_length_penalty = path_length_penalty.to(DEVICE)

    opt_gen = optim.Adam(gen.parameters(), lr=LEARNING_RATE, betas=(0.0, 0.99))
    opt_critic = optim.Adam(critic.parameters(), lr=LEARNING_RATE, betas=(0.0, 0.99))
    opt_mapping_network = optim.Adam(mapping_network.parameters(), lr=LEARNING_RATE, betas=(0.0, 0.99))

    gen.train()
    critic.train()
    mapping_network.train()


    for epoch in range(EPOCHS):
        train_fn(
            critic,
            gen,
            path_length_penalty,
            loader,
            opt_critic,
            opt_gen,
            opt_mapping_network,
        )
        # Saving model every epoch
        torch.save(gen.state_dict(), "model/stylegan2ANDC/generator.pth")
        torch.save(critic.state_dict(), "model/stylegan2ANDC/discriminator.pth")
        torch.save(mapping_network.state_dict(), "model/stylegan2ANDC/mapping.pth")
        torch.save(path_length_penalty.state_dict(), "model/stylegan2ANDC/PLP.pth")
        if total_epochs % 10 == 0:
            generate_examples(gen, epoch, 12)
        total_epochs += 1
        json_data["epochs"] += 1
        json_data["G_loss"] = generator_loss
        json_data["D_loss"] = discriminator_loss
        # Writing to json file to remember num. epochs
        with open("params/data.json", "w") as f:
            json.dump(json_data, f)

    plt.figure(figsize=(10, 5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(generator_loss, label="G")
    plt.plot(discriminator_loss, label="D")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    if not os.path.exists("training"):
        os.mkdir("training")
    plt.savefig("training/training_loss.png")

