"""
train.py

This module contains the training loop and related functions for training a Siamese Network
on the ISIC 2020 skin lesion dataset. It includes functions for training, validation,
plotting metrics, and the main training loop.

Author: Zain Al-Saffi
Date: 18th October 2024
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from dataset import get_data_loaders
from modules import get_model, get_loss
import numpy as np
from sklearn.metrics import confusion_matrix, roc_auc_score, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import logging

# Set up logging to track training progress
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def train_epoch(model, train_loader, triplet_loss, classifier_loss, optimizer, device, scaler):
    """
    Train the model for one epoch.

    Args:
        model (nn.Module): The Siamese Network model.
        train_loader (DataLoader): DataLoader for the training data.
        triplet_loss (nn.Module): The triplet loss function.
        classifier_loss (nn.Module): The classifier loss function.
        optimizer (torch.optim.Optimizer): The optimizer.
        device (torch.device): The device to train on (CPU or GPU).
        scaler (GradScaler): Gradient scaler for mixed precision training.

    Returns:
        tuple: A tuple containing average loss, final accuracy, and AUC-ROC for the epoch.
    """
    model.train()
    running_loss = 0.0
    all_labels = []
    all_probs = []
    all_preds = []

    pbar = tqdm(train_loader, desc="Training")
    for batch_idx, (anchor, positive, negative, labels) in enumerate(pbar):
        # Move data to the specified device (CPU or GPU)
        anchor, positive, negative, labels = anchor.to(device), positive.to(device), negative.to(device), labels.to(device)
        # Reset gradients for this iteration
        optimizer.zero_grad()  
        
        # Use automatic mixed precision for faster training on compatible GPUs
        with autocast():
            # Forward pass
            anchor_out, positive_out, negative_out = model(anchor, positive, negative)
            
            # Compute losses
            triplet_loss_val = triplet_loss(anchor_out, positive_out, negative_out)
            classifier_out = model.classify(anchor)
            classifier_loss_val = classifier_loss(classifier_out, labels)
            
            # Combine losses
            loss = triplet_loss_val + classifier_loss_val

        # Backward pass with gradient scaling
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        
        # Clip gradients to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()
        
        # Compute probabilities and predictions
        # Probability of positive class (to be used for ROC curve)
        probs = torch.softmax(classifier_out, dim=1)[:, 1]
        _, preds = torch.max(classifier_out, 1)
        
        # Store labels, probabilities, and predictions for metric calculation
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs.detach().cpu().numpy())
        all_preds.extend(preds.cpu().numpy())

        # Calculate and display running accuracy
        running_acc = accuracy_score(all_labels, all_preds)
        pbar.set_postfix({'Loss': running_loss / (batch_idx + 1), 'Acc': f'{running_acc:.4f}'})

    # Calculate final metrics for the epoch
    avg_loss = running_loss / len(train_loader)
    final_acc = accuracy_score(all_labels, all_preds)
    auc_roc = roc_auc_score(all_labels, all_probs)

    return avg_loss, final_acc, auc_roc

def validate(model, val_loader, triplet_loss, classifier_loss, device):
    """
    Validate the model on the validation set.

    Args:
        model (nn.Module): The Siamese Network model.
        val_loader (DataLoader): DataLoader for the validation data.
        triplet_loss (nn.Module): The triplet loss function.
        classifier_loss (nn.Module): The classifier loss function.
        device (torch.device): The device to validate on (CPU or GPU).

    Returns:
        tuple: A tuple containing average loss, final accuracy, and AUC-ROC for the validation set.
    """
    model.eval() 
    running_loss = 0.0
    all_labels = []
    all_probs = []
    all_preds = []

    with torch.no_grad():
        pbar = tqdm(val_loader, desc="Validating")
        for batch_idx, (anchor, positive, negative, labels) in enumerate(pbar):
            # Move data to the specified device for faster computation
            anchor, positive, negative, labels = anchor.to(device), positive.to(device), negative.to(device), labels.to(device)

            # Forward pass
            anchor_out, positive_out, negative_out = model(anchor, positive, negative)
            
            # Compute losses
            triplet_loss_val = triplet_loss(anchor_out, positive_out, negative_out)
            classifier_out = model.classify(anchor)
            classifier_loss_val = classifier_loss(classifier_out, labels)
            
            # Combine losses
            loss = triplet_loss_val + classifier_loss_val

            running_loss += loss.item()
            
            # Compute probabilities and predictions
            probs = torch.softmax(classifier_out, dim=1)[:, 1]
            _, preds = classifier_out.max(1)
            
            # Store labels, probabilities, and predictions for metric calculation
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

            # Calculate and display running accuracy
            running_acc = accuracy_score(all_labels, all_preds)
            pbar.set_postfix({'Loss': running_loss / (batch_idx + 1), 'Acc': f'{running_acc:.4f}'})

    # Calculate final metrics for the validation set
    avg_loss = running_loss / len(val_loader)
    final_acc = accuracy_score(all_labels, all_preds)
    auc_roc = roc_auc_score(all_labels, all_probs)

    return avg_loss, final_acc, auc_roc

def plot_metrics(train_losses, val_losses, train_accs, val_accs, train_aucs, val_aucs):
    """
    Plot training and validation metrics.

    Args:
        train_losses (list): List of training losses for each epoch.
        val_losses (list): List of validation losses for each epoch.
        train_accs (list): List of training accuracies for each epoch.
        val_accs (list): List of validation accuracies for each epoch.
        train_aucs (list): List of training AUC-ROC scores for each epoch.
        val_aucs (list): List of validation AUC-ROC scores for each epoch.
    """
    plt.figure(figsize=(18, 5))

    # Plot Loss
    plt.subplot(1, 3, 1)
    plt.plot(range(1, len(train_losses)+1), train_losses, label='Train Loss')
    plt.plot(range(1, len(val_losses)+1), val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')

    # Plot Accuracy
    plt.subplot(1, 3, 2)
    plt.plot(range(1, len(train_accs)+1), train_accs, label='Train Accuracy')
    plt.plot(range(1, len(val_accs)+1), val_accs, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Training and Validation Accuracy')

    # Plot AUC-ROC
    plt.subplot(1, 3, 3)
    plt.plot(range(1, len(train_aucs)+1), train_aucs, label='Train AUC-ROC')
    plt.plot(range(1, len(val_aucs)+1), val_aucs, label='Validation AUC-ROC')
    plt.xlabel('Epoch')
    plt.ylabel('AUC-ROC')
    plt.legend()
    plt.title('Training and Validation AUC-ROC')

    plt.tight_layout()
    plt.savefig('training_plots.png')
    plt.close()
    logging.info("Training plots saved as 'training_plots.png'")

def main():
    """
    Main function to run the training loop.
    """
    # Hyperparameters
    batch_size = 32
    embedding_dim = 320
    learning_rate = 1e-3 
    num_epochs = 30
    data_dir = 'preprocessed_data/'

    # Early stopping threshold for validation AUC-ROC
    early_stopping_threshold = 0.80
    
    # Set the device 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # Load data
    train_loader, val_loader = get_data_loaders(data_dir=data_dir, batch_size=batch_size)
    logging.info("Data loaded successfully")

    # Initialize model and all its parameters for training
    model = get_model(embedding_dim=embedding_dim).to(device)
    triplet_loss = get_loss(margin=1.0).to(device)
    classifier_loss = nn.CrossEntropyLoss(label_smoothing=0.1).to(device)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, verbose=True)
    scaler = GradScaler()  # For mixed precision training

    # Lists to store metrics for each epoch
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    train_aucs, val_aucs = [], []

    best_val_auc = 0
    for epoch in range(num_epochs):
        logging.info(f"Epoch {epoch+1}/{num_epochs}")

        # Train for one epoch
        train_loss, train_acc, train_auc = train_epoch(model, train_loader, triplet_loss, classifier_loss, optimizer, device, scaler)
        
        # Validate the model
        val_loss, val_acc, val_auc = validate(model, val_loader, triplet_loss, classifier_loss, device)
        
        # Store metrics
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        train_aucs.append(train_auc)
        val_aucs.append(val_auc)

        # Log the results
        logging.info(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Train AUC-ROC: {train_auc:.4f}")
        logging.info(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val AUC-ROC: {val_auc:.4f}")

        # Adjust learning rate based on validation AUC-ROC
        scheduler.step(val_auc)

        # Save the best model
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save(model.state_dict(), 'best_model.pth')
            logging.info(f"New best model saved with validation AUC-ROC: {best_val_auc:.4f}")

        # Early stopping check
        if val_auc >= early_stopping_threshold and epoch >= 15:
            if val_acc > 0.80:
                logging.info(f"Early stopping triggered. Validation AUC-ROC {val_auc:.4f} exceeds threshold of {early_stopping_threshold}")
                break

    logging.info("Training completed")

    # Plot training metrics
    plot_metrics(train_losses, val_losses, train_accs, val_accs, train_aucs, val_aucs)

    # Load the best model and evaluate on the test set
    model.load_state_dict(torch.load('best_model.pth'))
    test_loss, test_acc, test_auc = validate(model, val_loader, triplet_loss, classifier_loss, device)
    logging.info(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}, Test AUC-ROC: {test_auc:.4f}")

    # Generate confusion matrix
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for anchor, _, _, labels in val_loader:
            anchor = anchor.to(device)
            outputs = model.classify(anchor)
            probs = torch.softmax(outputs, dim=1)[:, 1]
            # Threshold for binary classification
            preds = (probs > 0.5).float() 
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['benign', 'malignant'], yticklabels=['benign', 'malignant'])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig('confusion_matrix.png')
    logging.info("Confusion matrix saved as 'confusion_matrix.png'")

if __name__ == '__main__':
    main()