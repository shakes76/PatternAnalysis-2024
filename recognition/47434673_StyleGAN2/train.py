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
