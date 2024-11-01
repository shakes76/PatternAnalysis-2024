import os
import torch
import torch.nn as nn
import torch.optim as optim
import time

def train_model():
    """
    Trains, validates, and tests the model. Saves the final model.
    """
    # Set device to GPU if available.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}')