# train.py

import os
import copy
import time  
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from modules import VisionTransformer  
from dataset import get_data_loaders 
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import random

def set_seed(seed=42):
    """
    Set the seed for reproducibility.
    
    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # If using CUDA
    
    # For CUDA algorithms, ensure deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False