# =====================================================================
# Filename: training_utils.py
# Project: ADNI Alzheimer’s Classification with GFNet
# Author: Siddhant Gaikwad
# Date: 25/10/2024
# Description: This file contains functions for training the GFNet model
#              with early stopping and gradient clipping. It includes 
#              metric tracking for training and validation loss and accuracy,
#              as well as plotting functions for visualizing these metrics.
# =====================================================================

import torch
import torch.optim as optim
import torch.nn as nn
from modules import GFNet
from dataset import get_data_loaders
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train_model_with_val_loss(model, criterion, optimizer, num_epochs, train_loader, val_loader, patience=5, clip_value=1.0):
    
    """
    Trains the GFNet model with early stopping and gradient clipping.
    
    Parameters:
    - model (torch.nn.Module): The GFNet model to be trained.
    - criterion (torch.nn.Module): Loss function, e.g., CrossEntropyLoss.
    - optimizer (torch.optim.Optimizer): Optimizer, e.g., Adam or SGD.
    - num_epochs (int): Number of epochs to train.
    - train_loader (DataLoader): DataLoader for the training data.
    - val_loader (DataLoader): DataLoader for the validation data.
    - patience (int): Early stopping patience (default is 5).
    - clip_value (float): Value for gradient clipping to avoid exploding gradients (default is 1.0).
    
    Returns:
    - train_loss_list (list): Training loss recorded after each epoch.
    - val_loss_list (list): Validation loss recorded after each epoch.
    - train_acc_list (list): Training accuracy recorded after each epoch.
    - val_acc_list (list): Validation accuracy recorded after each epoch.
    """
    
    # Lists to store loss and accuracy for each epoch
    train_loss_list, val_loss_list, train_acc_list, val_acc_list = [], [], [], []
    early_stop, best_val_loss, patience_counter = False, float('inf'), 0

    for epoch in range(num_epochs):
        if early_stop:
            break

        model.train()
        running_loss, correct_train, total_train = 0.0, 0, 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        epoch_train_loss = running_loss / len(train_loader.dataset)
        epoch_train_acc = 100 * correct_train / total_train
        train_loss_list.append(epoch_train_loss)
        train_acc_list.append(epoch_train_acc)

        # Validation phase
        model.eval()
        val_loss, correct_val, total_val = 0.0, 0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()

        epoch_val_loss = val_loss / len(val_loader.dataset)
        epoch_val_acc = 100 * correct_val / total_val
        val_loss_list.append(epoch_val_loss)
        val_acc_list.append(epoch_val_acc)

        print(f'Epoch {epoch+1}/{num_epochs}, Training Loss: {epoch_train_loss:.4f}, Training Accuracy: {epoch_train_acc:.2f}%, Validation Loss: {epoch_val_loss:.4f}, Validation Accuracy: {epoch_val_acc:.2f}%')

    return train_loss_list, val_loss_list, train_acc_list, val_acc_list

def plot_metrics(train_loss, val_loss, train_acc, val_acc):
    
    """
    Plots the training and validation loss and accuracy over epochs.

    Parameters:
    - train_loss (list): List of training loss values per epoch.
    - val_loss (list): List of validation loss values per epoch.
    - train_acc (list): List of training accuracy values per epoch.
    - val_acc (list): List of validation accuracy values per epoch.
    """
    
    epochs = range(1, len(train_loss) + 1)

    plt.figure(figsize=(12, 5))

    #Plot training and validation loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_loss, 'b', label='Training Loss')
    plt.plot(epochs, val_loss, 'r', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.legend()

    #Plot training and validation accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_acc, 'b', label='Training Accuracy')
    plt.plot(epochs, val_acc, 'r', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()