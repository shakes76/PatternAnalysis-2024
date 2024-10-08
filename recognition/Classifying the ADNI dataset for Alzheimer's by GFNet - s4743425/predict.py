"""
This file shows the example usage of the trained model for making predictions and visualising results.
"""

import torch
import os
import matplotlib.pyplot as plt
 #import the trained model architecture
from modules import GFNet 
# Import the data loader
from dataset import dataloader
import numpy as np
import torchvision.utils as vutils

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Directory to save plots
assets_dir = 'assets'
if not os.path.exists(assets_dir):
    os.makedirs(assets_dir)


# Function to load the trained model
def load_model(model_path):
    model = GFNet(
        img_size=256,
        patch_size= 16,
        embed_dim=768,
        num_classes=2,
        in_channels=3,
        drop_rate=0.5,
        depth=2,
        mlp_ratio=4.,
        drop_path_rate=0.6
        ).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()  # Set the model to evaluation mode
    return model