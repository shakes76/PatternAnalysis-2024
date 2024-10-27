import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from tqdm import tqdm

# Import your custom modules
from modules import VQVAE
from dataset import get_data_loader 

# Data paths
train_path = "./data/keras_slices_train"
validate_path = "./data/keras_slices_validate"

# Hyperparameters
batch_size = 256
num_training_updates = 1500

num_hiddens = 128
num_residual_hiddens = 32
num_residual_layers = 2

embedding_dim = 64
num_embeddings = 512

commitment_cost = 0.25

learning_rate = 1e-3

# Prepare Data Loaders
train_loader = get_data_loader(train_path, batch_size=batch_size, norm_image=True, early_stop=True)
validate_loader = get_data_loader(validate_path, batch_size=batch_size, norm_image=True, early_stop=True)

# Initialize Model and Optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = VQVAE(num_hiddens, num_residual_layers, num_residual_hiddens, num_embeddings, embedding_dim, commitment_cost).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate, amsgrad=False)

# Training Loop
model.train()
train_recon_errors = []
train_perplexities = []

for update in tqdm(range(num_training_updates), desc="Training Progress"):
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()

        # Forward Pass
        vq_loss, data_recon, perplexity = model(data)

        # Compute Loss and Backpropagation
        recon_error = F.mse_loss(data_recon, data)
        loss = recon_error + vq_loss
        loss.backward()

        # Optimize
        optimizer.step()
    
        # Collect Statistics
        train_recon_errors.append(recon_error.item())
        train_perplexities.append(perplexity.item())

