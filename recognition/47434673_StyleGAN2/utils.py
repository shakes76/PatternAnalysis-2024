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
from math import sqrt #math sqrt is about 7 times faster than numpy sqrt
from tqdm import tqdm

import train
import predict
import modules
import dataset

# Number of workers for dataloader
workers = 2

# Size of z latent vector (i.e. size of generator input)
nz = 100

# Size of feature maps in generator
ngf = 64

# Size of feature maps in discriminator
ndf = 64

# Beta1 hyperparameter for Adam optimizers
beta1 = 0.5

# Number of GPUs available. Use 0 for CPU mode.
ngpu = 1


# Hyper Parameters
epochs = 300           
learning_rate = 0.001   
channels = 1            
batch_size = 32         
image_size = 64         
log_resolution = 7      
image_height = 2**log_resolution    
image_width = 2**log_resolution     
z_dim = 256             
w_dim = 256             
lambda_gp = 10          
interpolation = "bilinear" 
save = "save"





##################################################
# Data augmentation and optimisation

# Samples z (noise) on random and fetches w (latent vectors) from mapping network
def get_w(batch_size, mapping_network, device):

    z = torch.randn(batch_size, w_dim).to(device)
    w = mapping_network(z)
    # Expand w from the generator blocks
    return w[None, :, :].expand(log_resolution, -1, -1)


# Generates random noise for the generator block
def get_noise(batch_size, device):
    
    noise = []
    #noise res starts from 4x4
    resolution = 4

    # For each gen block
    for i in range(log_resolution):
        # First block uses 3x3 conv
        if i == 0:
            n1 = None
        # For rest of conv layer
        else:
            n1 = torch.randn(batch_size, 1, resolution, resolution, device=device)
        n2 = torch.randn(batch_size, 1, resolution, resolution, device=device)

        # add the noise tensors to the lsit
        noise.append((n1, n2))
        # subsequent block has 2x2 res
        resolution *= 2

    return noise

'''
This is an regularization penalty 
:We try to reduce the L2 norm of gradients of the discriminator with respect to images.
'''
def gradient_penalty(critic, real, fake,device="cpu"):
    BATCH_SIZE, C, H, W = real.shape
    beta = torch.rand((BATCH_SIZE, 1, 1, 1)).repeat(1, C, H, W).to(device)
    interpolated_images = real * beta + fake.detach() * (1 - beta)
    interpolated_images.requires_grad_(True)

    # Calculate critic scores
    mixed_scores = critic(interpolated_images)
 
    # Calculates the gradient of scores with respect to the images
    # and we need to create and retain graph since we have to compute gradients
    # with respect to weight on this loss.
    gradient = torch.autograd.grad(
        inputs=interpolated_images,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True,
    )[0]
    # Reshape gradients to calculate the norm
    gradient = gradient.view(gradient.shape[0], -1)
    # Calculate the norm and then the loss
    gradient_norm = gradient.norm(2, dim=1)
    gradient_penalty = torch.mean((gradient_norm - 1) ** 2)

    return gradient_penalty
