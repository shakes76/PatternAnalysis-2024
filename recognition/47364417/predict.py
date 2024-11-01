import os
import torch

def predict():
    """
    Loads a saved model (or trains a new one if none exists) and tests it on the testing data.
    """
    # Set device to GPU if available.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}')