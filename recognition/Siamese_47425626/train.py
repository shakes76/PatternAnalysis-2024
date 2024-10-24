import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from dataset import get_data_loaders
from modules import SiameseNetwork
from pytorch_metric_learning import losses
from pytorch_metric_learning.distances import LpDistance
from tqdm import tqdm
import numpy as np

# Local flag to adjust parameters for testing
LOCAL = True

# Parameters
NUM_EPOCHS = 20 if not LOCAL else 1
LEARNING_RATE = 0.001
BATCH_SIZE = 32 if not LOCAL else 2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps")

# Load data loaders without sampler for now
train_loader, val_loader, test_loader = get_data_loaders(batch_size=BATCH_SIZE)

# If LOCAL, use a subset of the train and validation datasets for faster testing
if LOCAL:
    train_subset_indices = np.random.choice(len(train_loader.dataset), size=int(0.01 * len(train_loader.dataset)), replace=False)
    train_subset = Subset(train_loader.dataset, train_subset_indices)
    train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True)

    val_subset_indices = np.random.choice(len(val_loader.dataset), size=int(0.01 * len(val_loader.dataset)), replace=False)
    val_subset = Subset(val_loader.dataset, val_subset_indices)
    val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False)

# Load the model from modules.py
model = SiameseNetwork().to(DEVICE)

# Define distance metric and contrastive loss function
distance = LpDistance(normalize_embeddings=False)
criterion = losses.ContrastiveLoss(pos_margin=0, neg_margin=1, distance=distance)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Learning rate scheduler
scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 0.99 ** epoch)

# Training Loop
for epoch in range(NUM_EPOCHS):
    model.train()  # Set model to training mode
    total_loss = 0

    for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{NUM_EPOCHS}"):
        # Unpack the batch
        img, labels = batch

        # Move to device
        img, labels = img.to(DEVICE), labels.to(DEVICE)

        # Forward pass to get the embeddings
        embeddings = model(img)

        # Calculate contrastive loss
        loss = criterion(embeddings, labels)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    # Step the learning rate scheduler
    scheduler.step()

    # Calculate average loss for the epoch
    avg_loss = total_loss / len(train_loader)

    # Validation Loop
    model.eval()  # Set model to evaluation mode
    with torch.no_grad():
        val_loss = 0
        for val_batch in tqdm(val_loader, desc="Validation"):
            val_img, val_labels = val_batch
            val_img, val_labels = val_img.to(DEVICE), val_labels.to(DEVICE)

            val_embeddings = model(val_img)

            val_loss += criterion(val_embeddings, val_labels).item()

    avg_val_loss = val_loss / len(val_loader)

    # Print summary of training and validation losses for each epoch
    print(f"Epoch [{epoch + 1}/{NUM_EPOCHS}] Summary: Training Loss: {avg_loss:.4f}, Validation Loss: {avg_val_loss:.4f}")
