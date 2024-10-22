"""
Trains, validates, tests, and saves a GFNet image classification model.
The model classifies brain images according to whether or not the subject has Alzheimer's disease

Benjamin Thatcher 
s4784738    
"""

import datetime
import os
from typing import Optional
import numpy as np
from matplotlib import pyplot as plt
import time
import torch
import torch.backends.cudnn as cudnn

from timm.data import Mixup
from timm.models import create_model
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.scheduler import create_scheduler
from timm.optim import create_optimizer
from timm.utils import NativeScaler, accuracy, ModelEma
from functools import partial
import torch.nn as nn

from dataset import get_data_loader
from modules import GFNet
from utils import get_parameters

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

        optimizer.zero_grad()

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

    return epoch_loss, accuracy


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
    return epoch_loss, accuracy
    

def train_GFNet(dataloaders):
    # Model architecture and hyperparameters
    (epochs,
    learning_rate,
    patch_size,
    embed_dim,
    num_classes,
    depth,
    mlp_ratio,
    drop_rate,
    drop_path_rate,
    weight_decay,
    t_max,
    eta_min
    ) = get_parameters()

    start_time = time.time()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    #cudnn.benchmark = True

    # Load training and validation datasets
    data_loader_train, data_loader_val = dataloaders['train']
    data_loader_test = dataloaders['test']

    # Create model with hard-coded parameters
    print(f"Creating GFNet model with img_size: 224x224")
    model = GFNet(
        img_size=(224, 224),
        patch_size=patch_size,
        num_classes=num_classes,
        in_chans=1, # For greyscale images
        embed_dim=embed_dim,
        depth=depth,
        mlp_ratio=mlp_ratio,
        drop_rate=drop_rate,
        drop_path_rate=drop_path_rate,
        norm_layer=partial(nn.LayerNorm, eps=1e-6)
    ).to(device)

    if device == 'cuda':
        print(torch.cuda.get_device_name(0))
        print(f"Memory Allocated: {torch.cuda.memory_allocated(0) / 1024 ** 3:.1f} GB")
        print(f"Memory Reserved: {torch.cuda.memory_reserved(0) / 1024 ** 3:.1f} GB")

    #n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    #print(f"Number of parameters: {n_parameters}")

    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    #optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)

    #loss_scaler = NativeScaler()

    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max, eta_min=eta_min)
    # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    # Loss criterion
    #criterion = LabelSmoothingCrossEntropy(smoothing=smoothing)
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
    
    print('Saving model...')
    torch.save(model.state_dict(), 'best_model.pth')

    total_time = time.time() - start_time
    print(f'Training time: {str(datetime.timedelta(seconds=int(total_time)))}')


if __name__ == "__main__":
    # Paths to the training and validation datasets
    dataloaders = {
        'train': get_data_loader("/home/groups/comp3710/ADNI/AD_NC/train", 'train', batch_size = 64, shuffle = True, split=0.2),
        'test': get_data_loader("/home/groups/comp3710/ADNI/AD_NC/test", 'test', batch_size = 64, shuffle = False, slit=0.2)
        #'train': get_data_loader("../AD_NC/train", 'train', batch_size = 32, shuffle = True, split=0.2),
        #'test': get_data_loader("../AD_NC/test", 'test', batch_size = 32, shuffle = False, split=0.2)
    }

    train_GFNet(dataloaders)
    print('Finished training')
