import torch
import numpy as np
import matplotlib.pyplot as plt
from modules import GCN  # Assuming GCN is defined in modules.py
import dataset  # Assuming dataset is defined in dataset.py


def train_epoch(model, optimizer, criterion, data):
    """
    Function to train a single epoch of the GCN model.
    """
    model.train()  # Set model to training mode
    optimizer.zero_grad()  # Clear previous gradients
    out = model(data['features'], data['adjacency_matrix'])  # Forward pass

    # Compute loss for training data
    train_loss = criterion(out[data['train_mask']], data['labels'][data['train_mask']])
    train_loss.backward()  # Backpropagation
    optimizer.step()  # Update model parameters

    # Compute validation loss
    val_loss = criterion(out[data['validation_mask']], data['labels'][data['validation_mask']])
    
    return train_loss, val_loss  # Ensure both losses are returned




