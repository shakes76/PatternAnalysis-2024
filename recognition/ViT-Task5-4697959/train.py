# train.py

import os
import copy
import time  
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from modules import VisionTransformer  
from dataset import get_data_loaders 
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import random

def set_seed(seed=42):
    """
    Set the seed for reproducibility.
    
    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # If using CUDA
    
    # For CUDA algorithms, ensure deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train_model(model, dataloaders, criterion, optimizer, device, num_epochs=25, patience=5, save_dir='saved_models'):
    """
    Trains the Vision Transformer model.

    Args:
        model (nn.Module): The Vision Transformer model.
        dataloaders (dict): Dictionary containing 'train' and 'val' DataLoaders.
        criterion (nn.Module): Loss function.
        optimizer (torch.optim.Optimizer): Optimizer.
        device (torch.device): Device to train on.
        num_epochs (int): Number of training epochs.
        patience (int): Number of epochs with no improvement after which training will be stopped.
        save_dir (str): Directory to save the best model and plots.

    Returns:
        nn.Module: The trained model with best validation accuracy.
        dict: Training and validation loss history.
        dict: Training and validation accuracy history.
    """

    os.makedirs(save_dir, exist_ok=True)

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': []
    }

    counter = 0

    # main training/validation loop
    for epoch in range(num_epochs):
        print(f'Epoch {epoch +1}/{num_epochs}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward
                # Track gradients only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # Backward + optimize only in train
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # Statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            # Compute epoch loss and accuracy
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            history[f'{phase}_loss'].append(epoch_loss)
            history[f'{phase}_acc'].append(epoch_acc.item())

            print(f'{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # Deep copy the model
            if phase == 'val':
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                    torch.save(best_model_wts, os.path.join(save_dir, 'best_vit_model.pth'))
                    print("Best model updated and saved.\n")
                    counter = 0
                else:
                    counter +=1
                    print(f"No improvement for {counter} epochs.\n")
                    if counter >= patience:
                        print("Early stopping triggered.")
                        model.load_state_dict(best_model_wts)
                        return model, history

    print(f'Training complete. Best val Acc: {best_acc:.4f}')

    # Load best model weights
    model.load_state_dict(best_model_wts)

    # Save final model
    torch.save(model.state_dict(), os.path.join(save_dir, 'final_vit_model.pth'))
    print(f"Final model saved at {os.path.join(save_dir, 'final_vit_model.pth')}")

    return model, history

def plot_metrics(history, save_dir='saved_models'):
    """
    Plots training and validation loss and accuracy.

    Args:
        history (dict): Dictionary containing 'train_loss', 'val_loss', 'train_acc', 'val_acc'.
        save_dir (str): Directory to save the plots.
    """
    epochs = range(1, len(history['train_loss']) +1)

    # Plot Loss
    plt.figure(figsize=(10,5))
    plt.plot(epochs, history['train_loss'], label='Train Loss')
    plt.plot(epochs, history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss Over Epochs')
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'loss_curve.png'))
    plt.close()

    # Plot Accuracy
    plt.figure(figsize=(10,5))
    plt.plot(epochs, history['train_acc'], label='Train Accuracy')
    plt.plot(epochs, history['val_acc'], label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Over Epochs')
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'accuracy_curve.png'))
    plt.close()