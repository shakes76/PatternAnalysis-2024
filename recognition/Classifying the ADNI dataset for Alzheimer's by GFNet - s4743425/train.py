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
num_epochs = 2

learning_rate = 0.001

batch_size = 64

# train the model
def train(model, criterion, optimizer, train_loader, val_loader):
    print("Start Training ...")
        # Check if train_loader and val_loader have data
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
        val_loss, val_acc = validate(model, criterion, val_loader)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        
        # Print loss and accuracy for each epoch
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')

# for validating
def validate(model, criterion, val_loader):
    val_loss = 0.0
    correct_predictions = 0
    total_predictions = 0
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(val_loader):
            
            inputs, labels = inputs.to(device), labels.to(device)
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
                    # Validation loss
            val_loss += loss.item()

            # Calculate validation accuracy
            _, predicted = torch.max(outputs, 1)
            correct_predictions += (predicted == labels).sum().item()
            total_predictions += labels.size(0)

    val_loss /= len(val_loader)
    val_accuracy = correct_predictions / total_predictions

    return val_loss, val_accuracy

# Main function to start training
def main():
    # Load dataset
    (train_set, val_set, test_set), (train_loader, val_loader, test_loader) = dataloader(batch_size=batch_size)
    #load model
    model = GFNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    #train the model and produce training results
    train(model, criterion, optimizer, train_loader, val_loader)

if __name__ == "__main__":
    main()
