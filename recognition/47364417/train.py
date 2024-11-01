import os
import torch
import torch.nn as nn
import torch.optim as optim
import time
from dataset import get_dataloaders
from modules import create_model

def train_model():
    """
    Trains, validates, and tests the model. Saves the final model.
    """
    # Set device to GPU if available.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}')

    data_dir = '/home/groups/comp3710/ADNI/AD_NC'
    output_dir = 'output'
    checkpoints_dir = 'checkpoints'
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(checkpoints_dir, exist_ok=True)

    # Load dataloaders and class names
    dataloaders, class_names = get_dataloaders(data_dir)
    num_classes = len(class_names)
    print(f'Classes: {class_names}')

    # Initialize the model
    model = create_model(num_classes)
    model = model.to(device)

    # Set up the loss function, optimizer, and scheduler.
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    scaler = torch.cuda.amp.GradScaler()

    num_epochs = 13
    train_losses, train_accs = [], []

    for epoch in range(1, num_epochs + 1):
        model.train()
        running_loss, running_corrects, total_samples = 0.0, 0, 0

        for inputs, labels in dataloaders['train']:
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            total_samples += inputs.size(0)

        epoch_loss = running_loss / total_samples
        epoch_acc = running_corrects.double() / total_samples * 100
        train_losses.append(epoch_loss)
        train_accs.append(epoch_acc.item())

        print(f'Epoch [{epoch}/{num_epochs}] - Train loss: {epoch_loss:.4f}, Acc: {epoch_acc:.2f}%')

        model.eval()
        val_running_loss, val_running_corrects, val_total_samples = 0.0, 0, 0

        with torch.no_grad():
            for inputs, labels in dataloaders['val']:
                inputs = inputs.to(device)
                labels = labels.to(device)

                with torch.cuda.amp.autocast():
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                _, preds = torch.max(outputs, 1)
                val_running_loss += loss.item() * inputs.size(0)
                val_running_corrects += torch.sum(preds == labels.data)
                val_total_samples += inputs.size(0)

        val_epoch_loss = val_running_loss / val_total_samples
        val_epoch_acc = val_running_corrects.double() / val_total_samples * 100
        print(f'Validation loss: {val_epoch_loss:.4f}, Acc: {val_epoch_acc:.2f}%')