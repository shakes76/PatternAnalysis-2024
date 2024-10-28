"""
Containing the source code for training, validating, testing and saving your model. The model
should be imported from “modules.py” and the data loader should be imported from “dataset.py”. Make
sure to plot the losses and metrics during training
"""

# Importing the required libraries
import torch
from torch import optim
from tqdm import tqdm

# Local libraries
from dataset import *
from modules import *
import predict
from config import *
from utils import *

def train(
    generator,
    discriminator,
    journey_penalty,
    optimizer_G,
    optimizer_D,
    dataloader ,
    mapping_net,
    optim_mapping,
    visualize=False):

    # Initialize tqdm progress bar
    # progress_bar = tqdm(dataloader, total=len(dataloader), desc=f"Epoch {epoch+1}/{epochs}", leave=True, dynamic_ncols=True)
    progress_bar = tqdm(dataloader,leave=True,dynamic_ncols=True)

    device = devicer()
    current_G_loss = []
    current_D_loss = []

    for batch_idx, real in enumerate(progress_bar): # Chanes here rea,_ in enumerate(progress_bar)
        real = real.to(device)
        current_batch_size = real.shape[0]
        w = get_w(current_batch_size, mapping_net)
        noise = get_noise(current_batch_size)
        fake = generator(w, noise)
        critic_fake = discriminator(fake.detach())
        critic_real = discriminator(real)
        gp = gradient_penalty(discriminator, real, fake, device=device)
        loss_critic = (
            -(torch.mean(critic_real) - torch.mean(critic_fake))
            + lambda_gp * gp
            + (0.001 * torch.mean(critic_real ** 2))
        )


        current_D_loss.append(loss_critic.item())
        # Normal backward pass
        optimizer_D.zero_grad()
        loss_critic.backward()
        optimizer_D.step()
        gen_fake = discriminator(fake)
        loss_gen = -torch.mean(gen_fake)
    # Apply path length penalty on every 16 Batches
        if batch_idx % 16 == 0:
            plp = journey_penalty(w, fake)
            if not torch.isnan(plp):
                loss_gen = loss_gen + plp

        # Append the observed Generator loss to the list
        current_G_loss.append(loss_gen.item())

        mapping_net.zero_grad()
        generator.zero_grad()
        loss_gen.backward()
        optimizer_G.step()
        optim_mapping.step()

        progress_bar.set_postfix(
            gp=gp.item(),
            loss_critic=loss_critic.item(),
        )

    return (current_D_loss, current_G_loss)


if __name__ == '__main__':

    device=devicer()
    dataloader = get_dataloader()
    generator = Generator(log_resolution,W_DIM).to(device)
    discriminator = Discriminator(log_resolution).to(device)
    mapping_net = MappingNetwork(Z_DIM, W_DIM).to(device)
    journey_penalty = PathLengthPenalty(0.99).to(device)


    optimizer_G = optim.Adam(generator.parameters(), lr=LEARNING_RATE, betas=(0.0, 0.99))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=LEARNING_RATE, betas=(0.0, 0.99))
    optim_mapping = optim.Adam(mapping_net.parameters(), lr=LEARNING_RATE, betas=(0.0, 0.99))

    G_losses = []
    D_losses = []

    generator.train()
    discriminator.train()
    mapping_net.train()

    for epoch in range(EPOCHS):
        G_loss, D_loss = train(generator, discriminator, journey_penalty, optimizer_G, optimizer_D, dataloader, mapping_net, optim_mapping)

        G_losses.append(G_loss)
        D_losses.append(D_loss)

        if epoch % 10 == 0:
            predict.plot_loss(G_losses, D_losses)
    predict.plot_loss(G_losses, D_losses)
