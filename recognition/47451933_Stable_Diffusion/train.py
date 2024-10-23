import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import math
import umap
from utils import *
from models import *
device = torch.device('cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))

num_timesteps = 10
beta_scheduler = NoiseScheduler(num_timesteps=num_timesteps)
alphas_cumprod = beta_scheduler.get_alphas_cumprod()

reducer = umap.UMAP(min_dist=0, n_neighbors=35)

from dataset import *

data = Dataset()

data_loader = data.get_train()