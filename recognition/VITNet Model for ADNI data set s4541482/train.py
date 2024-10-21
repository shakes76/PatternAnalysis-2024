# train.py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from dataset import ADNIDataset
from modules import ViTModel
from focal_loss import FocalLoss
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from collections import defaultdict
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

def train_model():
    # Set device to GPU if available, else CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Hyperparameters
    num_epochs = 20
    learning_rate = 1e-4
    batch_size = 32

    # Paths to data directories
    data_root = r'C:\Users\macke\OneDrive\Desktop\COMP3710 A3\AD_NC'
    train_dir = os.path.join(data_root, 'train')
    test_dir = os.path.join(data_root, 'test')

    # Load all samples to collect patient IDs and labels
    patient_labels = {}
    for label, class_name in enumerate(['NC', 'AD']):
        class_dir = os.path.join(train_dir, class_name)
        if not os.path.isdir(class_dir):
            continue
        for filename in os.listdir(class_dir):
            if '_' in filename:
                patient_id, _ = filename.split('_', 1)
                if patient_id not in patient_labels:
                    patient_labels[patient_id] = label

    # Get unique patient IDs and labels
    patient_ids = list(patient_labels.keys())
    patient_labels_list = [patient_labels[pid] for pid in patient_ids]

    # Split patient IDs into train and validation sets (80-20 split)
    train_ids, val_ids, train_labels, val_labels = train_test_split(
        patient_ids, patient_labels_list, test_size=0.2, stratify=patient_labels_list, random_state=42)

    # Calculate class weights based on training labels
    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(train_labels), y=train_labels)
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device) 
    print(f'Class Weights: {class_weights}')

    # Create dataset instances
    print("Loading datasets...")
    train_dataset = ADNIDataset(train_dir, patient_ids=train_ids, mode='train')
    val_dataset = ADNIDataset(train_dir, patient_ids=val_ids, mode='val')
    test_dataset = ADNIDataset(test_dir, patient_ids=None, mode='test')

    # Define DataLoader instances without sampler
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # Initialize the model and move it to the appropriate device
    model = ViTModel(num_classes=2, freeze=True).to(device)  # Start with frozen layers

    # Define the loss function with Focal Loss using computed class weights
    criterion = FocalLoss(alpha=class_weights, gamma=2, reduction='mean').to(device)

    # Define the optimizer for the classifier head first
    optimizer = AdamW(model.model.head.parameters(), lr=learning_rate, weight_decay=1e-4)

    # Define the learning rate scheduler
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)

    # Early stopping parameters
    early_stop_patience = 5
    no_improvement_epochs = 0
    best_val_loss = float('inf')

    # Lists to store training and validation losses
    train_losses = []
    val_losses = []

    # Training Loop
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0

        print(f"\nEpoch [{epoch+1}/{num_epochs}]")
        batch_progress = tqdm(enumerate(train_loader), total=len(train_loader), desc="Training", leave=False)
        for batch_idx, (images, labels_batch, _) in batch_progress:
            # Move data to the device
            images = images.to(device)
            labels_batch = labels_batch.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(images)

            # Compute the loss
            loss = criterion(outputs, labels_batch)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            # Accumulate loss
            epoch_loss += loss.item()

            # Update progress bar
            batch_progress.set_postfix(loss=loss.item())

        # Calculate average training loss for the epoch
        avg_train_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validation phase
        val_loss, val_acc = evaluate_model(model, val_loader, device, criterion)
        val_losses.append(val_loss)

        # Step the scheduler
        scheduler.step()

        # Print training and validation statistics
        print(f"Epoch [{epoch+1}/{num_epochs}] "
              f"Train Loss: {avg_train_loss:.4f} "
              f"Val Loss: {val_loss:.4f} "
              f"Val Acc: {val_acc:.4f}")

        # Check for improvement
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improvement_epochs = 0
            torch.save(model.state_dict(), 'best_vit_model.pth')
            print("Validation loss decreased, saving model.")
        else:
            no_improvement_epochs += 1
            print(f"No improvement in validation loss for {no_improvement_epochs} epoch(s).")

        # Early stopping condition
        if no_improvement_epochs >= early_stop_patience:
            print("Early stopping triggered")
            break

        # After certain epochs, unfreeze the transformer layers for fine-tuning
        if epoch == 10:
            print("Unfreezing transformer layers for fine-tuning")
            model.unfreeze_layers()
            # Define different learning rates for different parts
            optimizer = AdamW([
                {'params': model.model.patch_embed.parameters(), 'lr': learning_rate/10},
                {'params': model.model.blocks.parameters(), 'lr': learning_rate/10},
                {'params': model.model.norm.parameters(), 'lr': learning_rate/10},
                {'params': model.model.head.parameters(), 'lr': learning_rate},
            ], weight_decay=1e-4)
            # Re-initialize the scheduler for remaining epochs
            scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs - epoch)

    # Plot the training and validation losses over epochs
    plt.figure()
    plt.plot(range(1, len(train_losses)+1), train_losses, label='Training Loss')
    plt.plot(range(1, len(val_losses)+1), val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss Curves')
    plt.savefig('loss_curve.png')
    plt.show()

    # Load the best model for testing
    model.load_state_dict(torch.load('best_vit_model.pth'))
    test_loss, test_acc = evaluate_model(model, test_loader, device, criterion, mode='test')
    print(f'Test Loss: {test_loss:.4f}')
    print(f'Test Accuracy: {test_acc:.4f}')

def evaluate_model(model, data_loader, device, criterion, mode='val'):
    """
    Evaluates the model on the given dataset.

    Args:
        model: Trained model to evaluate.
        data_loader: DataLoader for the dataset.
        device: Device to perform computation on.
        criterion: Loss function.
        mode (str): Mode of evaluation ('val' or 'test').

    Returns:
        avg_loss: Average loss over the dataset.
        accuracy: Classification accuracy on the dataset.
    """
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    patient_predictions = defaultdict(list)
    patient_labels = {}

    with torch.no_grad():
        progress = tqdm(enumerate(data_loader), total=len(data_loader), desc=f"{mode.capitalize()}ing", leave=False)
        for batch_idx, (images, labels, image_paths) in progress:
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)

            # Compute loss
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            # Get predictions
            _, predicted = torch.max(outputs.data, 1)

            # Collect predictions and labels
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            # Aggregate predictions at the patient level
            batch_size = images.size(0)
            for i in range(batch_size):
                image_path = image_paths[i]
                filename = os.path.basename(image_path)
                patient_id = filename.split('_')[0]
                patient_predictions[patient_id].append(predicted[i].cpu().item())
                patient_labels[patient_id] = labels[i].cpu().item()

    # Patient-level aggregation using majority vote
    patient_preds = []
    patient_true = []
    for patient_id, preds in patient_predictions.items():
        # Majority vote
        patient_pred = max(set(preds), key=preds.count)
        patient_preds.append(patient_pred)
        patient_true.append(patient_labels[patient_id])

    # Calculate accuracy
    accuracy = accuracy_score(patient_true, patient_preds)
    avg_loss = total_loss / len(data_loader)

    # Print classification report
    if mode in ['val', 'test']:
        print(f'\nClassification Report ({mode.capitalize()}):')
        print(classification_report(patient_true, patient_preds, target_names=['NC', 'AD']))

    return avg_loss, accuracy

if __name__ == '__main__':
    train_model()
