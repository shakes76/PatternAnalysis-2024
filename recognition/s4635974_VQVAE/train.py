# In your train.py
import torch
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import numpy as np

from dataset import HipMRILoader
import modules


# Hyperparameters
num_epochs = 5
batch_size = 32
lr = 0.0002
num_hiddens = 128
num_residual_hiddens = 32
num_chanels = 1
num_embeddings = 512
dim_embedding = 64
beta = 0.25

# Configure Pytorch
seed = 42
torch.manual_seed(seed)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# Directories for datasets
train_dir = '/home/groups/comp3710/HipMRI_Study_open/keras_slices_data/keras_slices_train'
test_dir = '/home/groups/comp3710/HipMRI_Study_open/keras_slices_data/keras_slices_test'
validate_dir = '/home/groups/comp3710/HipMRI_Study_open/keras_slices_data/keras_slices_validate'

# Define your transformations
transform = transforms.Compose([
    transforms.ToTensor(),  # Convert numpy array to PyTorch tensor
    transforms.Normalize((0.5,), (0.5,)),  # Normalize to [-1, 1]
    # transforms.RandomHorizontalFlip(),  # Random horizontal flip
    # transforms.RandomRotation(15),  # Random rotation within 15 degrees
])

# Get loaders
train_loader, validate_loader, train_variance = HipMRILoader(
    train_dir, validate_dir, test_dir,
    batch_size=batch_size, transform=transform
    ).get_loaders()

# Create model
model = modules.VQVAE(
    num_channels=num_chanels,
    num_hiddens=num_hiddens,
    num_embeddings=num_embeddings,
    dim_embedding=dim_embedding,
    beta=beta)