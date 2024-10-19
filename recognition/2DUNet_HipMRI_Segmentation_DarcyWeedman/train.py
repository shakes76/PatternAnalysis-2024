import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import HipMRIDataset
from modules import UNet
from tqdm import tqdm
import numpy as np
import random
from typing import Tuple, Dict

# Set random seeds for reproducibility
def set_seed(seed: int = 42) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True