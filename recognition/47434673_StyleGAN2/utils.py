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


workers = 2 # Number of workers for dataloader
nz = 100 # Size of z latent vector (i.e. size of generator input)
ngf = 64 # Size of feature maps in generator
ndf = 64 # Size of feature maps in discriminator
beta1 = 0.5 # Beta1 hyperparameter for Adam optimizers
ngpu = 1 # Number of GPUs available. Use 0 for CPU mode.
epochs = 300 # Number of epochs
learning_rate = 0.001 # Learning rate
channels = 1 # 1 Channel for greyscale images, 3 for RGB.
batch_size = 32 # Number of images per training batch
image_size = 64 # Image size is 64 x 64 pixels
log_resolution = 7 # Log of resolution
image_height = 2**log_resolution # asdf
image_width = 2**log_resolution # asdf
z_dim = 256 # asdf
w_dim = 256 # asdf
lambda_gp = 10 # asdf
interpolation = "bilinear" # asdf
save = "save" # asdf


##################################################
# Data augmentation and optimisation


def get_w(batch_size, mapping_network, device):
    """Gets a random noise sample (z) and gets latent vectors (w) from the mapping network"""
    z = torch.randn(batch_size, w_dim).to(device)
    w = mapping_network(z)
    # Expand w from the generator blocks
    return w[None, :, :].expand(log_resolution, -1, -1)


def get_noise(batch_size, device):
    """Generates the random noise used in the generator blocks"""
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


def gradient_penalty(critic, real, fake,device="cpu"):
    """Attempts to reduce the l^2 norm of the gradients of the discriminator with respect to the images. Regularisation penalty"""
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
