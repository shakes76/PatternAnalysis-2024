"""
File: train.py
Description: Contains the source code for training, validating, testing and
    saving the model. The model is imported from “modules.py” and the data loader
    is imported from “dataset.py”. Losses and metrics are plotted during training.
Course: COMP3710 Pattern Recognition
Author: Liam Mulhern (S4742847)
Date: 26/10/2024
"""

from torch.utils.data import DataLoader
from typing import Callable

import matplotlib.pyplot as plt
import torch.nn as nn
import torch
import os
import datetime
import csv
import numpy as np

MODEL_DIR = './models/'
CSV_DIR = './models/'
DEBUG = False

best_accuracy = 0

def train(
        model: nn.Module,
        device: torch.device,
        train_loader: DataLoader,
        optimiser: torch.optim.Optimizer,
        epoch: int,
        criterion: nn.Module,
    ) -> float:
    """
        Train 1 generation of the model using the loss function.

        Parameters:
            model: Model to train over batches
            device: The device to train the model on
            train_loader: The data loader to train the model with
            optimiser: The optimisation stratergy to train the model with
            epoch: The current epoch of training
            criterion: The criterion to use for loss

        Returns:
            The loss of the trained generation
    """
    model.train()
    train_loss = 0

    print(f"Epoch: {epoch}")

    # Train each of the batches of data
    for batch_idx, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)

        # Reset the optimiser
        optimiser.zero_grad()

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward and optimize
        loss.backward()
        optimiser.step()

        train_loss += loss.item()

        if DEBUG:
            print (f"Training Batch {batch_idx + 1} Loss: {loss.item()}")

    avg_loss = train_loss / (batch_idx + 1)

    print(f"Training Set: Average Loss: {avg_loss}")

    return avg_loss

def test(
        model: nn.Module,
        device: torch.device,
        test_loader: DataLoader,
        criterion: nn.Module,
        epoch: int,
        name: str,
        save: bool = True,
    ) -> tuple[float, float]:
    """
        Test the model by comparing labelled test data subset to model output 

        Parameters:
            model: Model to train over batches
            device: The device to train the model on
            train_loader: The data loader to train the model with
            optimiser: The optimisation stratergy to train the model with
            epoch: The current epoch of training
            criterion: The criterion to use for loss

        Returns:
            Average loss for the epoch and the accuracy
    """
    global best_accuracy

    # Use evaluation mode so we don't backpropagate or drop
    model.eval()
    test_loss = 0
    correct = 0

    # Turn off gradient descent when we run inference on the model
    with torch.no_grad():
        batch_count = 0

        for data, target in test_loader:
            batch_count += 1
            data, target = data.to(device), target.to(device)

            # Get the predicted classes for this batch
            output = model(data)

            # Calculate the loss for this batch
            test_loss += criterion(output, target).item()

            # Calculate the accuracy for this batch
            _, predicted = torch.max(output.data, 1)
            correct += torch.sum(target==predicted).item()

    # Calculate the average loss and total accuracy for this epoch
    avg_loss = test_loss / batch_count
    accuracy = 100. * correct / len(test_loader.dataset)

    print(f'Validation set: Average loss: {avg_loss}, Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy}%)')

    return avg_loss, accuracy
