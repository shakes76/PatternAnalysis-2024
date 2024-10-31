import torch
import torch.optim as optim
import torch.nn as nn
from modules import create_model
from dataset import get_dataloaders

def train_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")