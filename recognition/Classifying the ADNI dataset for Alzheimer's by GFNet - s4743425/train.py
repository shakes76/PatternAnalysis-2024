"""
This file contains the source code for the training, validating and testing of the model.
the model itself from modules.py is imported and is trained with the data from dataset.py
"""

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os
from modules import *
from dataset import *

# Set device 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set Hyperparameters
num_epochs = 20

learning_rate = 0.001

batch_size = 32

# train the model
def train(model, criterion, optimizer, train_loader, val_loader):
    model.train()
    train_losses = []
    val_losses = []
    val_accuracies = []
    
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        train_loss = running_loss / len(train_loader)
        train_losses.append(train_loss)
        
        # Validate the model at the end of each epoch
#        val_loss, val_acc = validate(model, criterion, val_loader)
#        val_losses.append(val_loss)
#        val_accuracies.append(val_acc)
        
        # Print loss and accuracy for each epoch
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {train_loss:.4f}')