"""
train.py created by Matthew Lockett 46988133
"""
import random
import os
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML
import hyperparameters as hp
from dataset import load_ADNI_dataset

# Force the creation of a folder to save figures if not present 
os.makedirs(hp.SAVED_FIGURES_DIR, exist_ok=True)

# PyTorch Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if not torch.cuda.is_available():
    print("Warning CUDA not Found. Using CPU.")

# Load the ADNI dataset images for training
train_loader = load_ADNI_dataset()

# Plot a sample of images from the ADNI dataset saved as 'adni_sample_images.png'
real_batch = next(iter(train_loader))
plt.figure(figsize=(8,8))
plt.axis("off")
plt.title("Sample Training Images for the ADNI Dataset")
plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=2, normalize=True).cpu(),(1,2,0)))
plt.savefig(os.path.join(hp.SAVED_FIGURES_DIR, "adni_sample_images.png"), bbox_inches='tight', pad_inches=0)
plt.close()