import os
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
from tqdm import tqdm
from modules import VQVAE  # Import your VQVAE model
from dataset import MedicalImageDataset, get_dataloaders  # Import your dataset classes

# Set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
num_epochs = 100
batch_size = 16
learning_rate = 1e-4
num_embeddings = 512  # Number of embeddings in vector quantizer
embedding_dim = 64  # Dimensionality of embedding
commitment_cost = 0.25  # Commitment cost
num_res_layers = 2  # Number of residual layers
hidden_channels = 64  # Hidden dimension of the VQVAE

