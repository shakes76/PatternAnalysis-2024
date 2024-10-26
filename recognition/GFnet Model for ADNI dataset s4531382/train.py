"""
Train and save the Global Filter Network model for Alzheimer's disease detection.
"""

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN optimizations for reproducibility

import random
import copy

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision.transforms.functional as TF
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from functools import partial
import matplotlib.pyplot as plt

from dataset import ADNITrainDataset, ADNITestDataset
from model import GlobalFilterNetwork


def visualize_losses(train_losses, val_losses, log_dir):
    """Visualize training and validation losses and save the plot."""
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Loss per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plot_path = os.path.join(log_dir, 'loss_plot.png')
    plt.savefig(plot_path)
    plt.close()
    print(f"Loss plot saved to {plot_path}")


def set_random_seed(seed):
    """Set the random seed for reproducibility across numpy, torch, and random libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # Ensure deterministic behavior in CUDA operations
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_transforms(is_training):
    """Create a composition of image transformations for data augmentation.

    Args:
        is_training (bool): Flag to indicate whether the transformations are for training or testing.

    Returns:
        transforms.Compose: A composition of image transformations.
    """
    transforms_list = [
        transforms.ToTensor(),
    ]
    # Check if the model is in training mode to apply data augmentations
    if is_training:
        # Extend the list of transformations with the following augmentations
        transforms_list.extend([
            # Randomly rotate the image by up to ±10 degrees
            # Helps the model become invariant to slight rotations, which can occur in MRI scans
            transforms.RandomRotation(10),
            
            # Randomly erase a portion of the image with a probability of 0.4
            # The erased area covers between 1% to 10% of the image and has an aspect ratio between 0.5 to 2.0
            # Simulates occlusions or missing data, improving the model's robustness to artifacts in MRI images
            transforms.RandomErasing(p=0.4, scale=(0.01, 0.10), ratio=(0.5, 2.0)),
            
            # Randomly crop the image to a size of 210x210 pixels, then resize back
            # The crop area covers between 95% to 102% of the original size with an aspect ratio between 0.95 to 1.05
            # Helps the model handle slight variations in the field of view and scaling in MRI scans
            transforms.RandomResizedCrop(size=(210, 210), scale=(0.95, 1.02), ratio=(0.95, 1.05)),
            
            # Apply random affine transformations with no rotation, slight translation, and slight scaling
            # Translates the image up to 5% horizontally and vertically and scales between 98% to 102%
            # Enhances the model's ability to recognize structures despite minor shifts and size variations in MRI images
            transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.98, 1.02)),
            
            # Randomly apply elastic deformation with a probability of 0.3
            # ElasticTransform applies random displacement fields controlled by alpha (intensity) and sigma (smoothing)
            # Simulates realistic deformations in MRI data, such as those caused by patient movement or tissue elasticity
            transforms.RandomApply([transforms.ElasticTransform(alpha=10.0, sigma=3.0)], p=0.3)
        ])

    # Append normalization to the transformation list
    # Normalizes the MRI images using the specified mean and standard deviation
    # Ensures that the input data has zero mean and unit variance, which can accelerate training and improve convergence
    transforms_list.append(transforms.Normalize(mean=[0.263], std=[0.271]))

    # Compose all the transformations into a single pipeline
    # This pipeline will be applied sequentially to each input image during training or evaluation
    return transforms.Compose(transforms_list)



def train_epoch(model, data_loader, optimizer, loss_fn, scheduler=None, device="cuda", show_progress=False):
    """Train the model for one epoch using the provided data loader.

    Args:
        model (nn.Module): The neural network model to train.
        data_loader (DataLoader): DataLoader for the training data.
        optimizer (Optimizer): Optimizer for updating the model parameters.
        loss_fn (Loss Function): Loss function to optimize.
        scheduler (Scheduler, optional): Learning rate scheduler. Defaults to None.
        device (str, optional): Device to run the training on. Defaults to "cuda".
        show_progress (bool, optional): Show the progress bar. Defaults to False.

    Returns:
        tuple: Tuple containing the average loss and accuracy over the epoch.
    """
    model.train()
    total_loss = 0.0
    correct_preds = 0
    total_samples = 0

    for images, labels in tqdm(data_loader, disable=not show_progress):
        images, labels = images.to(device), labels.float().to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        logits = model(images)
        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()

        # Update statistics
        total_loss += loss.item()
        predictions = (logits >= 0).float()
        total_samples += labels.size(0)
        correct_preds += (predictions == labels).sum().item()

    # Compute average loss and accuracy
    average_loss = total_loss / len(data_loader)
    accuracy = 100 * correct_preds / total_samples

    # Update learning rate scheduler if provided
    if scheduler is not None:
        scheduler.step()

    return average_loss, accuracy


def evaluate_model(model, data_loader, loss_fn, device="cuda", show_progress=False):
    """Evaluate the model on the validation or test dataset.

    Args:
        model (nn.Module): The neural network model to evaluate.
        data_loader (DataLoader): DataLoader for the validation or test data.
        loss_fn (Loss Function): Loss function to compute loss.
        device (str, optional): Device to run the evaluation on. Defaults to "cuda".
        show_progress (bool, optional): Show the progress bar. Defaults to False.

    Returns:
        tuple: Tuple containing the average loss and accuracy over the dataset.
    """
    model.eval()
    total_loss = 0.0
    correct_preds = 0
    total_samples = 0

    for images, labels in tqdm(data_loader, disable=not show_progress):
        images, labels = images.to(device), labels.float().to(device)

        # Forward pass
        logits = model(images)
        loss = loss_fn(logits, labels)
        total_loss += loss.item()

        # Update statistics
        predictions = (logits >= 0).float()
        total_samples += labels.size(0)
        correct_preds += (predictions == labels).sum().item()

    # Compute average loss and accuracy
    average_loss = total_loss / len(data_loader)
    accuracy = 100 * correct_preds / total_samples
    return average_loss, accuracy


# Define variables instead of using argument parser
data_path = r'C:\Users\macke\OneDrive\Desktop\COMP3710 A3\AD_NC'
show_progress = True
epochs = 1
batch_size = 64
early_stopping = 20
device = "cuda"
train_seed = 1

# Set random seed for reproducibility
set_random_seed(train_seed)

# Determine the device to run on
device = device if torch.cuda.is_available() else "cpu"

# Create log directory
script_dir = os.path.dirname(os.path.abspath(__file__))
log_dir = os.path.join(script_dir, 'logs/GFNet')
os.makedirs(log_dir, exist_ok=True)


if __name__ == "__main__":
    # Prepare data transformations
    train_transforms = create_transforms(is_training=True)
    val_transforms = create_transforms(is_training=False)

    # Load datasets
    train_dataset = ADNITrainDataset(
        data_dir=data_path,
        mode="train",
        transform=train_transforms,
        validation=False,
        random_seed=0,
        split_ratio=0.9
    )

    val_dataset = ADNITrainDataset(
        data_dir=data_path,
        mode="train",
        transform=val_transforms,
        validation=True,
        random_seed=0,
        split_ratio=0.9
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=6,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=6,
        pin_memory=True
    )

    # Initialize the model
    # GlobalFilterNetwork is a neural network architecture that utilizes global filter layers.
    # It is designed to capture long-range dependencies in images efficiently.
    # The model parameters are set according to the 'gfnet-xs' configuration.
    model = GlobalFilterNetwork(
        image_size=210,       # Input image size
        in_channels=1,        # Number of input channels (grayscale images)
        patch_size=14,        # Patch size for splitting the image
        embed_dim=384,        # Embedding dimension
        depth=12,             # Number of layers
        mlp_ratio=4,          # Ratio for MLP hidden dimension
        normalization=partial(nn.LayerNorm, eps=1e-6)  # Normalization layer
    ).to(device)

    # Define loss function and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20, eta_min=1e-6)

    # Initialize variables for training
    best_val_acc = 0.0
    patience_counter = 0  # Counter for early stopping patience

    train_losses = []
    val_losses = []

    # Training loop
    for epoch in range(epochs):
        # Train for one epoch
        train_loss, train_acc = train_epoch(
            model=model,
            data_loader=train_loader,
            optimizer=optimizer,
            loss_fn=criterion,
            scheduler=scheduler,
            device=device,
            show_progress=show_progress
        )

        # Evaluate on validation set
        val_loss, val_acc = evaluate_model(
            model=model,
            data_loader=val_loader,
            loss_fn=criterion,
            device=device,
            show_progress=show_progress
        )

        # Append losses for visualization
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        # Log results
        print(f'Epoch [{epoch+1}/{epochs}]')
        print(f'Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.2f}%')

        # Early Stopping Logic
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            # Save the best model weights
            best_model_state = copy.deepcopy(model.state_dict())
            torch.save(best_model_state, os.path.join(log_dir, 'best_gfnet.pt'))
        else:
            patience_counter += 1

        # Check if early stopping criteria met
        if patience_counter >= early_stopping:
            print(f"Early stopping at epoch {epoch+1}")
            break

    print('Training complete.')

    # Visualize training and validation losses
    visualize_losses(train_losses, val_losses, log_dir)

    # Load the best model
    model.load_state_dict(best_model_state)

    # Testing
    print('Testing')

    # Load the test dataset
    test_dataset = ADNITestDataset(
        data_dir=data_path,
        transform=val_transforms
    )

    model.eval()
    predictions_list = []
    true_labels_list = []

    # Test loop
    for inputs, labels in test_dataset:
        inputs, labels = inputs.to(device), labels.float().to(device)

        outputs = model(inputs)
        outputs = torch.sigmoid(outputs).mean().item()

        preds = 1 if outputs > 0.5 else 0

        # Accumulate predictions and true labels
        predictions_list.append(preds)
        true_labels_list.append(labels.item())

    # Convert lists to numpy arrays for accuracy calculation
    predictions_array = np.array(predictions_list)
    true_labels_array = np.array(true_labels_list)

    # Calculate and print the test accuracy
    if len(true_labels_array) == len(predictions_array):
        test_accuracy = accuracy_score(true_labels_array, predictions_array)
        print(f"Test Accuracy: {test_accuracy}")
    else:
        print(f"Mismatch in sample sizes: {len(true_labels_array)} vs {len(predictions_array)}")
