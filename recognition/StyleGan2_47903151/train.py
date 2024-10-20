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
from constants import *

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


def generate_examples(gen, mapping_net, epoch, n=10, display=False):
    gen.eval()
    # alpha = 1.0
    if display:
        fig = plt.figure(figsize=(10, 7))
        plt.title(f"Epoch {epoch}")
        plt.axis('off')
        rows, columns = 2, 2
        for i in range(4):
            with torch.no_grad():
                w = get_w(1, W_DIM, DEVICE, mapping_net, LOG_RESOLUTION)
                noise = get_noise(1, LOG_RESOLUTION, DEVICE)
                img = gen(w, noise)
                img = img[0]
                fig.add_subplot(rows, columns, i+1)
                plt.imshow(img.permute(1, 2, 0))
                plt.axis('off')
        plt.show()
    else:
        for i in range(n):
            with torch.no_grad():  # turn off gradient calculation to speed up generation
                w = get_w(1, W_DIM, DEVICE, mapping_net, LOG_RESOLUTION)
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

def plot_loss(g_loss, d_loss):
    plt.figure(figsize=(10, 5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(g_loss, label="G")
    plt.plot(d_loss, label="D")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    if not os.path.exists("training"):
        os.mkdir("training")
    plt.savefig("training/training_loss.png")

def load_model(path="model"):
    generator = Generator(LOG_RESOLUTION, W_DIM)
    discriminator = Discriminator(LOG_RESOLUTION)
    mapping_net = MappingNetwork(Z_DIM, W_DIM)
    plp = PathLengthPenalty(0.99)
    optim_gen = optim.Adam(generator.parameters(), lr=LEARNING_RATE, betas=(0.0, 0.99))
    optim_critic = optim.Adam(discriminator.parameters(), lr=LEARNING_RATE, betas=(0.0, 0.99))
    optim_map = optim.Adam(mapping_net.parameters(), lr=LEARNING_RATE, betas=(0.0, 0.99))
    if path is None:
        return generator, discriminator, mapping_net, plp, optim_gen, optim_critic, optim_map
    if not os.path.exists(path):
        os.mkdir(path)
    else:
        try:
            if DEVICE == 'cpu':
                generator.load_state_dict(torch.load(f"{path}/generator.pth", map_location=DEVICE))
                discriminator.load_state_dict(torch.load(f"{path}/discriminator.pth", map_location=DEVICE))
                mapping_net.load_state_dict(torch.load(f"{path}/mapping.pth", map_location=DEVICE))
                plp.load_state_dict(torch.load(f"{path}/PLP.pth", map_location=DEVICE))
                optim_gen.load_state_dict(torch.load(f"{path}/generator_opt.pth", map_location=DEVICE))
                optim_critic.load_state_dict(torch.load(f"{path}/discriminator_opt.pth", map_location=DEVICE))
                optim_map.load_state_dict(torch.load(f"{path}/mapping_opt.pth", map_location=DEVICE))
            else:
                generator.load_state_dict(torch.load(f"{path}/generator.pth"))
                discriminator.load_state_dict(torch.load(f"{path}/discriminator.pth"))
                mapping_net.load_state_dict(torch.load(f"{path}/mapping.pth"))
                plp.load_state_dict(torch.load(f"{path}/PLP.pth"))
                optim_gen.load_state_dict(torch.load(f"{path}/generator_opt.pth"))
                optim_critic.load_state_dict(torch.load(f"{path}/discriminator_opt.pth"))
                optim_map.load_state_dict(torch.load(f"{path}/mapping_opt.pth"))
        except Exception as err:
            print("Failed to load model. Training on new model instead.")
            raise(err)

    return generator, discriminator, mapping_net, plp, optim_gen, optim_critic, optim_map

def save_model(generator: Generator,
               discriminator: Discriminator,
               mapping_net: MappingNetwork,
               plp: PathLengthPenalty,
               optim_gen,
               optim_critic,
               optim_map,
               directory: str = "model"):
    torch.save(generator.state_dict(), f"{directory}/generator.pth")
    torch.save(discriminator.state_dict(), f"{directory}/discriminator.pth")
    torch.save(mapping_net.state_dict(), f"{directory}/mapping.pth")
    torch.save(plp.state_dict(), f"{directory}/PLP.pth")
    torch.save(optim_gen.state_dict(), f"{directory}/generator_opt.pth")
    torch.save(optim_critic.state_dict(), f"{directory}/discriminator_opt.pth")
    torch.save(optim_map.state_dict(), f"{directory}/mapping_opt.pth")




if __name__ == "__main__":
    # Get and parse the command line arguments
    parser = argparse.ArgumentParser(description="COMP3506/7505 Assignment Two: Data Structure Tests")
    parser.add_argument("--dataset_dir", type=str, help="Directory for the dataset")
    parser.add_argument("--model_dir", type=str, help="Directory of the saved model, if any")
    parser.set_defaults(dataset_dir="AD_NC", model_dir="model")
    args = parser.parse_args()

    loader = get_loader(LOG_RESOLUTION, BATCH_SIZE, args.dataset_dir)

    gen, critic, mapping_network, path_length_penalty, opt_gen, opt_critic, opt_mapping_network = \
        load_model(args.model_dir)

    gen = gen.to(DEVICE)
    critic = critic.to(DEVICE)
    mapping_network = mapping_network.to(DEVICE)
    path_length_penalty = path_length_penalty.to(DEVICE)

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
            opt_mapping_network
        )
        # Saving model every epoch
        save_model(gen,
                   critic,
                   mapping_network,
                   path_length_penalty,
                   opt_gen,
                   opt_critic,
                   opt_mapping_network,
                   args.model_dir
                   )

        if total_epochs % 10 == 0:
            generate_examples(gen, mapping_network, total_epochs, 12)
        total_epochs += 1
        json_data["epochs"] += 1
        json_data["G_loss"] = generator_loss
        json_data["D_loss"] = discriminator_loss
        # Writing to json file to remember num. epochs
        with open("params/data.json", "w") as f:
            json.dump(json_data, f)
        plot_loss(generator_loss, discriminator_loss)
