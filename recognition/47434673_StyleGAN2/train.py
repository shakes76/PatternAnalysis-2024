from __future__ import print_function
#%matplotlib inline
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from IPython.display import HTML
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import torch.nn.functional as F
from math import sqrt
from tqdm import tqdm

import modules
import utils
import predict
import train

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


#####################################################
# Training

def train_fn(critic, gen, path_length_penalty, loader, opt_critic, opt_gen, opt_mapping_network,):
    """The training cycle of the StyleGAN2"""
    loop = tqdm(loader, leave=True)

    curr_Gloss = []
    curr_Dloss = []

    for batch_idx, (real, _) in enumerate(loop):
        real = real.to(device)
        cur_batch_size = real.shape[0]

        w = utils.get_w(cur_batch_size, modules.mapping_network, device)
        noise = utils.get_noise(cur_batch_size, device)

        # Use cuda AMP for accelerated training
        with torch.cuda.amp.autocast():
            # Generate fake image using the Generator
            fake = gen(w, noise)

            # Get a critic score for the fake and real image
            critic_fake = critic(fake.detach())
            critic_real = critic(real)

            # Calculate and log gradient penalty
            gp = predict.gradient_penalty(critic, real, fake, device=device)
            loss_critic = (
                -(torch.mean(critic_real) - torch.mean(critic_fake))
                + utils.lambda_gp * gp
                + (0.001 * torch.mean(critic_real ** 2))
            )
        
        # Append the observed Discriminator loss to the list
        curr_Dloss.append(loss_critic.item())

        # Reset gradients for the Discriminator
        # Backpropagate the loss and update the discriminator's weights
        
        critic.zero_grad()
        loss_critic.backward()
        opt_critic.step()

        # Get score for the Discriminator and the Generator
        gen_fake = critic(fake)
        loss_gen = -torch.mean(gen_fake)

        # Apply path length penalty on every 16 Batches
        if batch_idx % 16 == 0:
            plp = path_length_penalty(w, fake)
            if not torch.isnan(plp):
                loss_gen = loss_gen + plp

        # Append the observed Generator loss to the list
        curr_Gloss.append(loss_gen.item())

        '''
        Reset gradients for the mapping network and the generator
        Backpropagate the generator loss
        Update generator's weights and Update mapping network's weights
        '''
        modules.mapping_network.zero_grad()
        gen.zero_grad()
        loss_gen.backward()
        opt_gen.step()
        opt_mapping_network.step()

        loop.set_postfix(
            gp=gp.item(),
            loss_critic=loss_critic.item(),
        )

    return (curr_Dloss, curr_Gloss)

####################################################
# Main function

# Device Config
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Device: ', device)

# Create batch of latent vectors used to visualise the progression of the generator
fixed_noise = torch.randn(64, utils.nz, 1, 1, device=device)

# Dataloader
loader = get_data(DATA, utils.log_resolution, utils.batch_size)

gen = modules.Generator(utils.log_resolution, utils.w_dim).to(device)
critic = modules.Discriminator(utils.log_resolution).to(device)
mapping_network = modules.MappingNetwork(utils.z_dim, utils.w_dim).to(device)
path_length_penalty = modules.PathLengthPenalty(0.99).to(device)

# Initialise Adam optimiser
opt_gen = optim.Adam(gen.parameters(), lr=utils.learning_rate, betas=(0.0, 0.99))
opt_critic = optim.Adam(critic.parameters(), lr=utils.learning_rate, betas=(0.0, 0.99))
opt_mapping_network = optim.Adam(mapping_network.parameters(), lr=utils.learning_rate, betas=(0.0, 0.99))

# Check if there are pre-trained models to load
load_model = False  # Set to True if you want to load a pre-trained model
if load_model:
    gen.load_state_dict(torch.load('/assets/netG.pth'))
    critic.load_state_dict(torch.load('/assets/netD.pth'))
    mapping_network.load_state_dict(torch.load('/assets/netM.pth'))
    print("Loaded pre-trained models.")

if not load_model:
    # Train the following modules
    gen.train()
    critic.train()
    mapping_network.train()

# Keeps a Log of total loss over the training
G_Loss = []
D_Loss = []
img_list = []

# loop over total epcoh.
for epoch in range(utils.epochs):
    curr_Gloss, curr_Dloss = train.train_fn(
        critic,
        gen,
        path_length_penalty,
        loader,
        opt_critic,
        opt_gen,
        opt_mapping_network,
    )

    # Append the current loss to the main list
    G_Loss.extend(curr_Gloss)
    D_Loss.extend(curr_Dloss)

    # Save generator's fake image on every 50th epoch
    if epoch % 10 == 0:
        predict.generate_examples(gen, mapping_network, epoch, device)

    if (epoch % 10 == 0) or (epoch == utils.epoch-1): #and (i == len(loader)-1)):
        fake = gen(fixed_noise).detach().cpu()
        img_list.append(vutils.make_grid(fake, padding=2, normalize=True))


# Save the models after training
torch.save(gen.state_dict(), '/assets/.pth')
torch.save(critic.state_dict(), '/assets/netD.pth')
torch.save(mapping_network.state_dict(), '/assets/netM.pth')
print("Saved models.")

predict.plot_loss(G_Loss, D_Loss)

# Plot the Generator and Discriminator losses
plt.figure(figsize=(10,5))
plt.title("Generator and Discriminator Loss During Training")
plt.plot(G_Loss, label="G")
plt.plot(D_Loss, label="D")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
plt.show()


# Create the animation
fig = plt.figure(figsize=(8,8))
plt.axis("off")
ims = [[plt.imshow(np.transpose(i,(1,2,0)), animated=True)] for i in img_list]
ani = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)

HTML(ani.to_jshtml())


# import sklearn.datasets
# import pandas as pd
# import numpy as np
# import umap
# import umap.plot

# mapper = umap.UMAP().fit()
# umap.plot.points(mapper)


