"""
Contains source code for training, validating, testing, and saving the model.
Plots losses and metrics during training.
"""
import argparse
import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from math import log2, sqrt
import numpy as np
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from modules import *
from dataset import *
import json
from constants import *
from predict import load_model, load_optimizers, generate_examples

def train_fn(
        critic: Discriminator,
        gen: Generator,
        mapping_net: MappingNetwork,
        path_length_penalty: PathLengthPenalty,
        loader: DataLoader,
        opt_critic: optim.Adam,
        opt_gen: optim.Adam,
        opt_mapping_network: optim.Adam):
    """
    A single training loop, displaying the training progress each epoch.
    """
    loop = tqdm(loader, leave=True)

    for batch_idx, (real, _) in enumerate(loop):
        real = real.to(DEVICE)
        cur_batch_size = real.shape[0]

        w = get_w(cur_batch_size, W_DIM, DEVICE, mapping_net, LOG_RESOLUTION)
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

        mapping_net.zero_grad()
        gen.zero_grad()
        loss_gen.backward()
        opt_gen.step()
        opt_mapping_network.step()

        loop.set_postfix(
            gp=gp.item(),
            loss_critic=loss_critic.item(),
        )

def plot_loss(g_loss, d_loss):
    """
    Plots the generator lost and discriminator loss graph, and the generator vs. discriminator loss graph before saving
    them.
    :param g_loss: a python list that stores the generator lost
    :param d_loss: a python list that stores the discriminator lost
    :return: None
    """
    fig = plt.figure(figsize=(10, 5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(g_loss, label="G")
    plt.plot(d_loss, label="D")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    if not os.path.exists("training"):
        os.mkdir("training")
    plt.savefig("training/training_loss.png")
    plt.close(fig)
    fig = plt.figure(figsize=(10, 5))
    plt.title("Generator loss vs Discriminator Loss During Training")
    g_d_loss = [g_loss[i]/d_loss[i] for i in range(len(g_loss))]
    plt.plot(g_d_loss)
    plt.xlabel("iterations")
    plt.ylabel("Generator Loss / Discriminator Loss")
    plt.legend()
    plt.savefig("training/training_loss_proportion.png")
    plt.close(fig)

def save_model(generator: Generator,
               discriminator: Discriminator,
               mapping_net: MappingNetwork,
               plp: PathLengthPenalty,
               optim_gen: optim.Adam,
               optim_critic: optim.Adam,
               optim_map: optim.Adam,
               directory: str = "model"):
    """
    Saves the model and optimizers to the given directory
    """
    if not os.path.exists(directory):
        os.mkdir(directory)
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
    parser.add_argument("--json_dir", type=str, help="Directory of json file to store training loss")
    parser.add_argument("--classes", type=str, help="Desired classes to train on. options: all, AD, NC")
    parser.set_defaults(dataset_dir="AD_NC/train", model_dir="model", json_dir="params", classes="all")
    args = parser.parse_args()
    if args.classes == "all":
        loader = get_loader(LOG_RESOLUTION, BATCH_SIZE, args.dataset_dir)
    else:
        loader = get_loader(LOG_RESOLUTION, BATCH_SIZE, args.dataset_dir, classes=args.classes)

    gen, critic, mapping_network, path_length_penalty = load_model(args.model_dir)

    gen = gen.to(DEVICE)
    critic = critic.to(DEVICE)
    mapping_network = mapping_network.to(DEVICE)
    path_length_penalty = path_length_penalty.to(DEVICE)

    opt_gen, opt_critic, opt_mapping_network = load_optimizers(generator=gen,
                                                               discriminator=critic,
                                                               mapping_net=mapping_network,
                                                               path=args.model_dir)

    gen.train()
    critic.train()
    mapping_network.train()

    if os.path.exists(f"{args.json_dir}/data.json"):  # loads the json dict which stores the past losses for plotting
        with open(f"{args.json_dir}/data.json", 'r') as f:
            json_data = json.load(f)
        total_epochs = json_data["epochs"]
        generator_loss = json_data["G_loss"]
        discriminator_loss = json_data["D_loss"]
    else:
        if not os.path.exists(args.json_dir):
            os.mkdir(args.json_dir)
        total_epochs = 0
        generator_loss = []
        discriminator_loss = []
        json_data = {"epochs": 0,
                     "G_loss": [],
                     "D_loss": []}

    for epoch in range(EPOCHS):
        train_fn(
            critic,
            gen,
            mapping_network,
            path_length_penalty,
            loader,
            opt_critic,
            opt_gen,
            opt_mapping_network
        )
        # Saving model every epoch
        save_model(generator=gen,
                   discriminator=critic,
                   mapping_net=mapping_network,
                   plp=path_length_penalty,
                   optim_gen=opt_gen,
                   optim_critic=opt_critic,
                   optim_map=opt_mapping_network,
                   directory=args.model_dir
                   )

        if total_epochs % 10 == 0 or total_epochs == 1 or total_epochs == 5:
            # generates and save examples
            generate_examples(gen, mapping_network, total_epochs, 12)
            # saves model independently so it can be use again later for umap plots
            save_model(gen, critic, mapping_network, path_length_penalty, opt_gen, opt_critic, opt_mapping_network,
                       f"{args.model_dir}_epoch_{total_epochs}")

        total_epochs += 1
        json_data["epochs"] += 1
        json_data["G_loss"] = generator_loss
        json_data["D_loss"] = discriminator_loss
        with open(f"{args.json_dir}/data.json", "w") as f:
            json.dump(json_data, f)
        plot_loss(generator_loss, discriminator_loss)
