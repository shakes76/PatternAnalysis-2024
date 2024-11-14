# =====================================================================
# Filename: evaluation.py
# Project: ADNI Alzheimer’s Classification with GFNet
# Author: Siddhant Gaikwad
# Date: 25/10/2024
# Description: This file contains functions for evaluating the GFNet model 
#              on the test dataset. It includes functions for calculating
#              loss and accuracy, generating classification reports, and
#              plotting the confusion matrix for Alzheimer’s classification.
# =====================================================================


import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from dataset import get_data_loaders
from modules import GFNet

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Evaluation of the model
def evaluate_model(model, test_loader, criterion):
    
    """
    Evaluates the model on the test dataset and calculates average test loss.

    Parameters:
    - model (torch.nn.Module): The GFNet model to be evaluated.
    - test_loader (DataLoader): DataLoader for the test dataset.
    - criterion (torch.nn.Module): Loss function, e.g., CrossEntropyLoss.

    Returns:
    - all_preds (list): List of all predicted labels for the test dataset.
    - all_labels (list): List of all true labels for the test dataset.
    - avg_loss (float): The average test loss across the dataset.
    """
    
    model.eval()
    all_preds, all_labels, total_loss = [], [], 0.0
    
    # Disable gradient calculation for evaluation
    with torch.no_grad():  
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass to get model predictions
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
           
    # Calculate average test loss
    avg_loss = total_loss / len(test_loader)
    return all_preds, all_labels, avg_loss

# Plotting the confusion matrix
def plot_confusion_matrix(labels, preds):
    
    """
    Plots the confusion matrix for model predictions vs. true labels.

    Parameters:
    - labels (list): List of true labels for the test dataset.
    - preds (list): List of predicted labels for the test dataset.

    Returns:
    - None: Displays the confusion matrix as a heatmap.
    """
    
    # Generate the confusion matrix
    conf_matrix = confusion_matrix(labels, preds)
    
    # Plotting the confusion matrix as a heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['NC', 'AD'], yticklabels=['NC', 'AD'])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()

