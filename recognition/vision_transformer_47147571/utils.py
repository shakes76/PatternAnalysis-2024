from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torch
import numpy as np
from dataset import ADNIDataset
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torch.utils.data import DataLoader
from tqdm import tqdm
from modules import GFNet
from functools import partial

def get_transform(train):
    transform_list = [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ]
    if train:
        transform_list.extend([
            transforms.RandomRotation(10),  # rotation
            transforms.ColorJitter(brightness=0.2, contrast=0.2),  # adjust brightness and contrast
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.5),  # Gaussian Blur
            transforms.RandomErasing(p=0.5, scale=(0.02, 0.2)),  # Random Erasing
            transforms.RandomApply([transforms.ElasticTransform()], p=0.5)  # Elastic Transform
        ])
    transform_list.append(transforms.Normalize(mean=[0.5], std=[0.5]))
    return transforms.Compose(transform_list)


def train(model, train_loader, optimizer, criterion, device="cuda", use_tqdm=True):
    """Train the model. We assume the model output logits and train via 
    BCEWithLogitsLoss.
    use_tqdm: Show the progress bar
    
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for inputs, labels in tqdm(train_loader, disable=use_tqdm):
        inputs, labels = inputs.to(device), labels.float().to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # Statistics
        running_loss += loss.item()
        predicted = (outputs >= 0).float()
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(train_loader)
    accuracy = 100 * correct / total
    return epoch_loss, accuracy


def test(model, test_loader, criterion, device="cuda", use_tqdm=True):
    """Test the model. We assume the model output logits and train via 
    BCEWithLogitsLoss.
    use_tqdm: Show the progress bar
    """
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, disable=use_tqdm):
            inputs, labels = inputs.to(device), labels.float().to(device)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item()

            # Statistics
            predicted = (outputs >= 0).float()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    epoch_loss = test_loss / len(test_loader)
    accuracy = 100 * correct / total
    return epoch_loss, accuracy