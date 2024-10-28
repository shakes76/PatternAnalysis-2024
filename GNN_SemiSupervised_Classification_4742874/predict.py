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

import dataset
import train
import modules

MODEL_DIR = './models/'
CSV_DIR = './models_csv/'

def run_inference(
        index: int,
    ) -> None:
    """
        Run inference on the given model and predict the outcome at the given index.

        Parameters:
            index: The index of the test data to load into the model
    """
    _, flpp_data, _, _, validate_mask = dataset.load_dataset(train.DATASET_DIR)

    validate_target = flpp_data.y[validate_mask]

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if not torch.cuda.is_available():
        print("WARNING: CUDA not found; Using CPU")

    model = modules.GNN(128, 16, 4)
    train._load_model(model, 'gnn_classifier')

    model.to(device)
    model.eval()

    # Turn off gradient descent when we run inference on the model
    with torch.no_grad():

        # Get the predicted classes for this batch
        outputs = model(flpp_data.x, flpp_data.edge_index)

        # Get the maximum output tensor (i.e predicted label)
        _, predicted = torch.max(outputs.data, 1)

        predicted = predicted[index].cpu().numpy()

        # Calculate the accuracy of all the predictions
        accuracy = ((outputs.argmax(1)[validate_mask] == validate_target).float()).mean()

        print(f"Predicted index: {predicted}, Labelled index: {validate_target[index]}, Label: {dataset.FLPP_CATEGORIES[predicted]}, Accuracy: {accuracy}")
