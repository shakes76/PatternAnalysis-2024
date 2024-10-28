"""
File: train.py
Description: Contains the source code for training, validating, testing and
    saving the model. The model is imported from “modules.py” and the data loader
    is imported from “dataset.py”. Losses and metrics are plotted during training.
Course: COMP3710 Pattern Recognition
Author: Liam Mulhern (S4742847)
Date: 26/10/2024
"""

import torch
import torch.nn as nn

import os
import datetime
import csv

import dataset
import modules

from torch.utils.data import DataLoader
from torch.functional import Tensor
from typing import Callable
from torch_geometric.data import Data

MODEL_DIR = './models/'
CSV_DIR = './models_csv/'
DATASET_DIR = './dataset/raw'

################
# Global Value #
################

best_accuracy = 0

def run_gnn_training(
        epochs: int,
        batch_size: int,
        learning_rate: float=0.001,
        is_load: bool = True,
        is_save: bool = True
    ) -> None:
    """
        Run the GNN training proccess by loading dataset, creating adjacency matrix,
        and running trainig method on the GNN model for the given number of epochs.

        Parameters:
            epochs: The total number of epochs to train for.
            batch_size: The size of the dataset batches.
            learning_rate: The rate that the model training delta changes.
            is_load: If true, load the saved model data and extend current training;
                Otherwise overwrite current saved model.
            is_save: If true, save the model when the accuracy increase;
                Otherwise the model only stays in memory until the process terminates.
    """
    # Load FLPP dataset
    _, flpp_data, train_mask, test_mask, validate_mask = dataset.load_dataset(DATASET_DIR)

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if not torch.cuda.is_available():
        print("WARNING: CUDA not found; Using CPU")

    # Create the model utilising the class type
    model = modules.GNN(128, 16, 4)

    # Utilise the Adam optimiser and cross entropy loss
    criterion = nn.CrossEntropyLoss()
    optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Run training for GNN
    _run_training(
        num_epochs=epochs,
        model=model,
        device=device,
        flpp_data=flpp_data,
        train_mask=train_mask,
        test_mask=test_mask,
        validate_mask=validate_mask,
        optimiser=optimiser,
        criterion=criterion,
        name='gnn_classifier',
        load=is_load,
        save=is_save
    )

def _train(
        model: nn.Module,
        device: torch.device,
        flpp_data: Data,
        train_mask,
        train_target,
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

    print(f"Epoch: {epoch}")

    # Reset the optimiser
    optimiser.zero_grad()

    # Forward pass
    outputs = model(flpp_data.x, flpp_data.edge_index)
    outputs = outputs[train_mask].to(device)
    loss = criterion(outputs, train_target.to(device))

    # Backward and optimize
    loss.backward()
    optimiser.step()

    print(f"Training Set: Average Loss: {loss.item()}")

    return loss.item()

def _test(
        model: nn.Module,
        device: torch.device,
        criterion: nn.Module,
        flpp_data: Data,
        test_mask: Tensor,
        test_target: Tensor,
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

    # Turn off gradient descent when we run inference on the model
    with torch.no_grad():

        # Get the predicted classes for this batch
        outputs = model(flpp_data.x, flpp_data.edge_index)

        # Calculate the loss for this batch
        test_loss = criterion(outputs[test_mask].to(device), test_target.to(device)).item()

        # Calculate the accuracy by comparing the training data mask
        accuracy = ((outputs.argmax(1)[test_mask] == test_target).float()).mean()

        print(f'Validation set: Average loss: {test_loss}, Accuracy: {accuracy * 100}%')

    return test_loss, float(accuracy)

def _save_model(
        epoch: int,
        model: nn.Module,
        accuracy: int,
        name: str
    ) -> None:
    """
        Save best model to disk if accuracy has improved.

        Parameters:
            epoch: The current epoch of the model
            model: The model with trained values
            accuracy: The accuracy of the model
            name: The name of the model
    """
    global best_accuracy

    # Save the model if its accuracy has improved
    if accuracy >= best_accuracy:
        print(f"Saving {name} : Epoch {epoch} : Accruacy {accuracy}%")

        best_accuracy = accuracy
        state = {
            'model': model.state_dict(),
            'accuracy': accuracy,
            'epoch': epoch
        }

        model_path = os.path.join(MODEL_DIR, f'{name}.pth')
        torch.save(state, model_path) 

def _load_model(
        model: nn.Module,
        name: str
    ) -> int:
    """
        Load the given model from file into the model parameter.

        Parameters:
            model: The model to load with the saved trained values
            name: The name of the model to load

        Returns:
            The start epoch of the loaded training
    """
    global best_accuracy

    model_path = os.path.join(MODEL_DIR, f'{name}.pth')

    loader = torch.load(model_path)
    model.load_state_dict(loader['model'])

    best_accuracy = loader['accuracy']
    start_epoch = loader['epoch']

    print(f"Loaded {name}: Epoch {start_epoch} : Accuracy: {best_accuracy}%")

    return start_epoch

def _run_training(
        num_epochs: int,
        model: nn.Module,
        device: torch.device,
        flpp_data: Data,
        train_mask,
        test_mask,
        validate_mask,
        optimiser: torch.optim.Optimizer,
        criterion: nn.Module,
        name: str,
        save: bool = True,
        load: bool = True,
        train_function: Callable | None = _train,
        test_function: Callable | None = _test,
    ) -> None:
    """
        Run the training for the model for the given number of epochs.

        Parameters:
            num_epochs: The total number of epochs to run the training for
            model: The model used to train
            device: The torch device used to train the model
            train_loader: The dataloader for the training data
            test_loader: The dataloader for the test data
            optimiser: The optimisation stratergy for the training
            criterion: The loss criterion used to evaluated the model's training gradient
            name: The name of the model being trained
            save: If true, saves the model after each iteration of improved accuracy
            load: If true, loads the saved model on start
            train_function: The function used to train the model (defaults to "train.train()")
            test_function: The function used to test  the model (defaults to "train.test()")
    """
    print(f"Training: {name}")

    start_epoch = 0

    csv_path = os.path.join(CSV_DIR, f'{name}.csv')

    if not os.path.isdir(MODEL_DIR):
        os.mkdir(MODEL_DIR)
    if not os.path.isdir(CSV_DIR):
        os.mkdir(CSV_DIR)

    # Load save states of the model from disk
    if load:
        start_epoch = _load_model(model, name)
    elif os.path.exists(csv_path):
        os.remove(csv_path)

    train_target = flpp_data.y[train_mask]
    test_target = flpp_data.y[test_mask]
    #target_validate = flpp_data.y[validate_mask]

    model.to(device)

    # Train the model for the given number of epochs from the start epoch
    for epoch in range(start_epoch, start_epoch + num_epochs):
        start_time = datetime.datetime.now()
        train_loss = 0
        test_accuracy = 0
        test_loss = 0

        # Train the model with the given training function
        if train_function is not None:
            train_loss = train_function(
                model=model,
                device=device,
                flpp_data=flpp_data,
                train_mask=train_mask,
                train_target=train_target,
                optimiser=optimiser,
                epoch=epoch,
                criterion=criterion
            )

        # Test the model with the given test function
        if test_function is not None:
            test_loss, test_accuracy = test_function(
                model=model,
                device=device,
                flpp_data=flpp_data,
                test_mask=test_mask,
                test_target=test_target,
                criterion=criterion
            )

        if save:
            _save_model(epoch, model, test_accuracy, name)

        # Calculate the time taken to train the epoch
        end_time = datetime.datetime.now()
        training_duration = (end_time - start_time).total_seconds()

        # Write training iteration to disk
        with open(csv_path, mode='a', newline='') as file:
            csv_writer = csv.writer(file)
            csv_writer.writerow([epoch, train_loss, test_loss, test_accuracy, training_duration])

