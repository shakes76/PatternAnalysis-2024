"""
@brief: This file contains the hyperparameters for training a StyleGAN model, including settings for 
blend factor, image channels, latent and style vector dimensions, gradient penalty weight, and progressive 
training steps across multiple image resolutions. It also defines batch sizes and learning rates based on 
image sizes.

@Author: Amlan Nag (s4721964)
"""

import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# Determines the smoothness of the generator and discriminator  
BLEND_FACTOR = 1e-5
# Hyperperameter that is used for EMA 
BETAS = (0.0, 0.99)
# Number of chennels in the image
CHANNELS_IMG = 1
# The number of dimentions in the laten space vector
Z_DIM = 256
# Number of dimentions in the style code vector 
W_DIM = 256
IN_CHANNELS = 256
# Weight used to determine the value of the gradient penalty 
LAMBDA_GP = 10
IMAGE_SIZES = [4, 8, 16, 32, 64, 128, 256]
# Mapping img size to epochs
PROGRESSIVE_EPOCHS = {4: 50, 8: 50, 16: 40, 32: 30, 64: 20, 128: 15, 256: 10}
# Increased batch size from experiments 
BATCH_SIZES = {4: 512, 8: 256, 16: 128, 32: 64, 64: 32, 128: 16, 256: 8}
# Mapping img size to learning rage 
LEARNING_SIZES = {4: 1e-3, 8: 1.2e-3, 16: 1.5e-3, 32: 1.8e-3, 64: 2e-3, 128: 2.5e-3, 256: 3e-3}
