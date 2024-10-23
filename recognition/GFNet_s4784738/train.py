"""
Trains, validates, tests, and saves a GFNet image classification model.
The model classifies brain images according to whether or not the subject has Alzheimer's disease

Benjamin Thatcher 
s4784738    
"""

import datetime
import os
from matplotlib import pyplot as plt
import time
import torch

from dataset import get_data_loader
from modules import GFNet
from utils import get_parameters, get_path_to_images

from torchvision import transforms
import torchvision.transforms.functional as TF

def train_one_epoch(model: torch.nn.Module, criterion,
                    data_loader, optimizer: torch.optim.Optimizer,
                    device: torch.device):
    """
    Trains the model for one epoch
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in data_loader:
        inputs, labels = inputs.to(device), labels.long().to(device)

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Convert outputs to predicted class (using argmax for multi-class classification)
        _, predicted = torch.max(outputs, 1)
        running_loss += loss.item()
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    # After each epoch, calculate loss and accuracy
    epoch_loss = running_loss / len(data_loader)
    accuracy = 100 * correct / total

    return accuracy, epoch_loss


@torch.no_grad()
def evaluate(data_loader, model, criterion, device):
    """
    Performs evaluation of the model for one epoch
    """
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.long().to(device)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item()

            # Convert outputs to predicted class (using argmax for multi-class classification)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    epoch_loss = test_loss / len(data_loader)
    accuracy = 100 * correct / total
    return accuracy, epoch_loss


def plot_data(train_acc, train_loss, val_acc, val_loss):
    """
    Plots and saves the training and validation loss and accuracy metrics.
    
    train_acc: List of training accuracies
    train_loss: List of training losses
    val_acc: List of validation accuracies
    val_loss: List of validation losses
    """
    # Create images directory if it doesn't exist
    os.makedirs('./assets', exist_ok=True)

    # Plot training accuracy
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_acc, label='Training Accuracy', color='blue')
    plt.title('Training Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig('./assets/training_accuracy.png')

    # Plot training loss
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_loss, label='Training Loss', color='blue')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('./assets/training_loss.png')

    # Plot validation accuracy/loss
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 2)
    plt.plot(val_acc, label='Validation Accuracy', color='blue')
    plt.title('Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig('./assets/validation_accuracy.png')

    # Plot validation loss
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 2)
    plt.plot(val_loss, label='Validation Loss', color='blue')
    plt.title('Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('./assets/validation_loss.png')
    plt.show()
    

def train_GFNet(dataloaders):
    # Model architecture and hyperparameters
    (epochs,
    learning_rate,
    patch_size,
    embed_dim,
    depth,
    mlp_ratio,
    drop_rate,
    drop_path_rate,
    weight_decay,
    t_max,
    ) = get_parameters()

    start_time = time.time()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load training and validation datasets
    data_loader_train, data_loader_val = dataloaders['train']
    data_loader_test = dataloaders['test']

    # Create model with hard-coded parameters
    print(f"Creating GFNet model with img_size: 224x224 (device = {device})")
    model = GFNet(
        img_size=(224, 224),
        patch_size=patch_size,
        in_chans=1, # For greyscale images
        embed_dim=embed_dim,
        depth=depth,
        mlp_ratio=mlp_ratio,
        drop_rate=drop_rate,
        drop_path_rate=drop_path_rate,
        #norm_layer=partial(nn.LayerNorm, eps=1e-6)
    ).to(device)

    if device == 'cuda':
        print(torch.cuda.get_device_name(0))
        print(f"Memory Allocated: {torch.cuda.memory_allocated(0) / 1024 ** 3:.1f} GB")
        print(f"Memory Reserved: {torch.cuda.memory_reserved(0) / 1024 ** 3:.1f} GB")

    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max)

    # Loss criterion
    criterion = torch.nn.CrossEntropyLoss()

    # Track the training and validation accuracy and loss at each epoch
    train_acc, train_loss = [], []
    val_acc, val_loss = [], []

    print(f"Start training for {epochs} epochs")
    for epoch in range(epochs):
        # Train model
        train_a, train_l = train_one_epoch(model, criterion, data_loader_train, optimizer, device)
        # Validate model
        val_a, val_l = evaluate(data_loader_val, model, criterion, device)

        # Track training and validation metrics
        train_acc.append(train_a)
        train_loss.append(train_l)
        val_acc.append(val_a)
        val_loss.append(val_l)

        #lr_scheduler.step(val_a)
        lr_scheduler.step()
        
        print(f'Accuracy of training set (epoch {epoch+1}/{epochs}): {train_a:.1f}%, and loss {train_l:.1f}')
        print(f'Accuracy on validation set (epoch {epoch+1}/{epochs}): {val_a:.1f}%, and loss {val_l:.1f}') 

    print("### Now it's time to run inference on the test dataset ###")
    test_acc, test_loss = evaluate(data_loader_test, model, criterion, device)
    print(f'Accuracy on test set: {test_acc:.1f}, and loss: {test_loss:.1f}\n')
    
    plot_data(train_acc, train_loss, val_acc, val_loss)
    print('Saved loss and accuracy plots in ./assets/')

    print('Saving model...')
    torch.save(model.state_dict(), 'best_model.pth')

    total_time = time.time() - start_time
    print(f'Training time: {str(datetime.timedelta(seconds=int(total_time)))}')


if __name__ == "__main__":
    # Paths to the training and validation datasets
    train_path, test_path = get_path_to_images()
    dataloaders = {
        'train': get_data_loader(train_path, 'train', batch_size = 16, shuffle = True, split=0.2),
        'test': get_data_loader(test_path, 'test', batch_size = 16, shuffle = False, split=0.2)
    }

    train_GFNet(dataloaders)
    print('Finished training')
