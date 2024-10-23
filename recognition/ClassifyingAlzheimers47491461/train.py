import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader as DataLoader
from torchvision.datasets import ImageFolder
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import dataset
import modules

torch.device("cuda")

patch_dim = 16
latent_dim = 768
drop = 0.1
# Matching dimensions from ViT paper
size = 224
