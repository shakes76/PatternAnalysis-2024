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

def train_model(model, dataloaders, criterion, optimizer, device, num_epochs=25, patience=5, save_dir='saved_models'):
    """
    Trains the Vision Transformer model.

    Args:
        model (nn.Module): The Vision Transformer model.
        dataloaders (dict): Dictionary containing 'train' and 'val' DataLoaders.
        criterion (nn.Module): Loss function.
        optimizer (torch.optim.Optimizer): Optimizer.
        device (torch.device): Device to train on.
        num_epochs (int): Number of training epochs.
        patience (int): Number of epochs with no improvement after which training will be stopped.
        save_dir (str): Directory to save the best model and plots.

    Returns:
        nn.Module: The trained model with best validation accuracy.
        dict: Training and validation loss history.
        dict: Training and validation accuracy history.
    """

    os.makedirs(save_dir, exist_ok=True)

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': []
    }

    counter = 0
