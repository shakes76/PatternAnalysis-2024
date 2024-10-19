"""
Trains and validates a deep learning model (GFNet) using the ADNI dataset.
"""

import torch
import matplotlib.pyplot as plt
from torch import nn, optim
from torch.utils.data import DataLoader
from dataset import get_adni_dataloader  
from modules import GFNet  
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Hyperparameters
    batch_size = 8
    base_lr = 0.0001  
    max_lr = 0.001    
    num_epochs = 60
    step_size = 5

    T_0 = 10  
    T_mult = 2 

    # Load data
    train_loader, val_loader = get_adni_dataloader(batch_size=batch_size, train=True)

    model = GFNet().to(device)  
    criterion = nn.CrossEntropyLoss()
    # optimizer = optim.Adam(model.parameters(), lr=base_lr)

    # # cosine lr
    # scheduler = optim.lr_scheduler.CyclicLR(optimizer, base_lr=base_lr, max_lr=max_lr, 
    #                                         step_size_up=step_size, mode='triangular')
    optimizer = AdamW(model.parameters(), lr=base_lr)
    scheduler = OneCycleLR(optimizer, max_lr=max_lr, steps_per_epoch=len(train_loader), epochs=num_epochs)

    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []

    for epoch in range(num_epochs):
        train_one_epoch(epoch, model, train_loader, criterion, optimizer, scheduler, train_losses, train_accuracies, device)
        validate_one_epoch(epoch, model, val_loader, criterion, val_losses, val_accuracies, device)

        print(f'Epoch [{epoch + 1}/{num_epochs}], '
              f'Train Loss: {train_losses[-1]:.4f}, Train Acc: {train_accuracies[-1]:.2f}, '
              f'Val Loss: {val_losses[-1]:.4f}, Val Acc: {val_accuracies[-1]:.2f}, '
              f'LR: {scheduler.get_last_lr()[0]:.6f}')

    torch.save(model.state_dict(), 'gfnet_model.pth')
    plot_metrics(num_epochs, train_losses, val_losses, train_accuracies, val_accuracies)

def train_one_epoch(epoch, model, train_loader, criterion, optimizer, scheduler, train_losses, train_accuracies, device):
    """
    Train the model for one epoch.

    Args:
        epoch (int): Current epoch number.
        model (nn.Module): The neural network model.
        train_loader (DataLoader): DataLoader for the training set.
        criterion (nn.Module): Loss function used for training.
        optimizer (Optimizer): Optimizer used for model parameter updates.
        scheduler (Scheduler): Learning rate scheduler.
        train_losses (list): List to store training losses.
        train_accuracies (list): List to store training accuracies.
        device (torch.device): Device for computation (CPU or GPU).
    """

    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        scheduler.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    train_losses.append(running_loss / len(train_loader))
    train_accuracies.append(100 * correct / total)

def validate_one_epoch(epoch, model, val_loader, criterion, val_losses, val_accuracies, device):
    """
    Validate the model for one epoch.

    Args:
        epoch (int): Current epoch number.
        model (nn.Module): The neural network model.
        val_loader (DataLoader): DataLoader for the validation set.
        criterion (nn.Module): Loss function used for validation.
        val_losses (list): List to store validation losses.
        val_accuracies (list): List to store validation accuracies.
        device (torch.device): Device for computation (CPU or GPU).
    """

    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_losses.append(running_loss / len(val_loader))
    val_accuracies.append(100 * correct / total)

def plot_metrics(num_epochs, train_losses, val_losses, train_accuracies, val_accuracies):
    """
    Plot and save training and validation losses and accuracies over epochs.

    Args:
        num_epochs (int): Number of epochs the model was trained for.
        train_losses (list): List of training losses per epoch.
        val_losses (list): List of validation losses per epoch.
        train_accuracies (list): List of training accuracies per epoch.
        val_accuracies (list): List of validation accuracies per epoch.
    """

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(range(num_epochs), train_losses, label='Train Loss')
    plt.plot(range(num_epochs), val_losses, label='Validation Loss')
    plt.title('Loss vs Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(range(num_epochs), train_accuracies, label='Train Accuracy')
    plt.plot(range(num_epochs), val_accuracies, label='Validation Accuracy')
    plt.title('Accuracy vs Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()

    plt.tight_layout()
    plt.savefig(f"Training_vs_validation.png")

if __name__ == '__main__':
    main()
