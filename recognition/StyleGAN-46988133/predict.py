"""
predict.py created by Matthew Lockett 46988133
"""
import os
import torch
from modules import *

# PyTorch Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if not torch.cuda.is_available():
    print("Warning CUDA not Found. Using CPU.")

# Setup both the generator and discriminator inference models
gen = Generator().to(device)
disc = Discriminator().to(device)

# Load the generator and discriminator inference models
gen.load_state_dict(torch.load(os.path.join(hp.SAVED_OUTPUT_DIR, "generator_model.pth"), weights_only=True))
disc.load_state_dict(torch.load(os.path.join(hp.SAVED_OUTPUT_DIR, "discriminator_model.pth"), weights_only=True))
gen.eval()
disc.eval()

