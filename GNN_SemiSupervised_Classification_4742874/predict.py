"""
File: predict.py
Description: Runs inference on the trained GNN classification model.
    Print out any results and provide visualisations where of TSNE and UMAP
    embeddings.
Course: COMP3710 Pattern Recognition
Author: Liam Mulhern (S4742847)
Date: 26/10/2024
"""

import torch
import torch.nn as nn

import os

from torch.utils.data import DataLoader

MODEL_DIR = './models/'
CSV_DIR = './models_csv/'

def run_inference(
        model: nn.Module,
        device: torch.device,
        test_dataloader: DataLoader,
        name: str,
        index: int,
        labels: list,
    ) -> None:
    """
        Run inference on the given model and predict the outcome at the given index.

        Parameters:
            model: The trained model to run inference on
            device: The device to move the model to
            test_dataloader: The dataloader to input to the model
            name: The name of the model to load
            index: The index of the test data to load into the model
            labels: The labels of the model outputs
    """

    model_path = os.path.join(MODEL_DIR, f'{name}.pth')
    loader = torch.load(model_path)
    model.load_state_dict(loader['model'])

    # Turn off gradient descent when we run inference on the model
    with torch.no_grad():
        for data, target in test_dataloader:
            data, target = data.to(device), target.to(device)

            # Get the predicted classes for this batch
            output = model(data)

            # Get the maximum output tensor (i.e predicted label)
            _, predicted = torch.max(output.data, 1)

            predicted = predicted[index].cpu().numpy()

            print(f"Predicted index: {predicted}, Labelled index: {target[index]}, Label: {labels[predicted]}")
