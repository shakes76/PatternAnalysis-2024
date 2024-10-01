import torch
import torch.nn as nn
import torch.optim as optim
from model import UNet3D

# Define device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if not torch.cuda.is_available():
    print("Warning: No access to CUDA, model being trained on CPU.")

model = UNet3D.to(device)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4) # TODO learning scheduler?

slice_data_path = "/home/groups/comp3710/HipMRI_Study_open/keras_slices_data"
semantic_labels_path = "/home/groups/comp3710/HipMRI_Study_open/semantic_labels_only"
semantic_MR_path = "/home/groups/comp3710/HipMRI_Study_open/semantic_MRs"

temp_slice = None

# TODO load in dataS