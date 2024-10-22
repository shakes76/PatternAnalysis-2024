"""
Imports and utility functions are organised here.
"""


import os
import time
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import umap
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset, random_split
from PIL import Image
from math import log2, sqrt
from settings import *


# set device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def generate_examples(gen, steps, n=100, label=MODEL_LABEL):
    """
    Function used to evaluate the generator by outputting example images. This
    is used during training to monitor progress, and can also be used on a
    fully trained model to generate images.
    """
    gen.eval()
    alpha = 1.0  # for fully trained model
    for i in range(n):
        with torch.no_grad():
            noise = torch.randn(1, Z_DIM).to(DEVICE)
            img = gen(noise, alpha, steps)
            if not os.path.exists(f"{SRC}/saved_examples/{label}/step{steps}"):
                os.makedirs(f"{SRC}/saved_examples/{label}/step{steps}")
            vutils.save_image(img*0.5+0.5, f"{SRC}/saved_examples/{label}/step{steps}/img_{i}.png")
    gen.train()
