"""
Author: Thomas Barros
Date: October 2024

This file contains the methods for data augmentation and optimisation.
Also reduces the L2 norm of the gradients of the discriminator in gradient_penalty method.
"""


from __future__ import print_function
#%matplotlib inline
import torch
import torch.nn.parallel
import torch.utils.data
from config import *


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
    # Noise resolution initially 4x4
    resolution = 4

    # For each generator block
    for i in range(log_resolution):
        # First block uses 3x3 conv
        if i == 0:
            n1 = None
        # For rest of conv layer
        else:
            n1 = torch.randn(batch_size, 1, resolution, resolution, device=device)
        n2 = torch.randn(batch_size, 1, resolution, resolution, device=device)

        # Add the noise tensors to the lsit
        noise.append((n1, n2))
        # Subsequent block has 2x2 resolution
        resolution *= 2

    return noise


def gradient_penalty(discrim, real, fake,device="cpu"):
    """
    Reduces the L2 norm of the gradients of the 
    discriminator with respect to the images.
    """
    BATCH_SIZE, C, H, W = real.shape
    beta = torch.rand((BATCH_SIZE, 1, 1, 1)).repeat(1, C, H, W).to(device)
    interpolated_images = real * beta + fake.detach() * (1 - beta)
    interpolated_images.requires_grad_(True)

    # Calculate discriminator scores
    mixed_scores = discrim(interpolated_images)

    # Calculate the gradient of scores of images
    gradient = torch.autograd.grad(
        inputs=interpolated_images,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True, # For the gradient computation based on the weight of this loss
    )[0]
    gradient = gradient.view(gradient.shape[0], -1) # Reshape gradients
    gradient_norm = gradient.norm(2, dim=1) # Calculate norm
    gradient_penalty = torch.mean((gradient_norm - 1) ** 2) # Calculate loss

    return gradient_penalty
