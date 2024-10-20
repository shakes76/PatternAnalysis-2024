import torch
import numpy as np
import matplotlib.pyplot as plt
from modules import GCN  
import dataset 

# Call the preprocessing function and assign returned data
adjacency_matrix, train_mask, validation_mask, test_mask, feature_matrix, labels = dataset.preprocess_adjacency_matrix()

# Initialize the data dictionary with the returned values
data = {
    'features': feature_matrix,
    'adjacency_matrix': adjacency_matrix,
    'labels': labels,
    'train_mask': train_mask,
    'validation_mask': validation_mask,
    'test_mask': test_mask
}

# Initialize GCN model
model = GCN(feature_matrix.size(1), 16, len(torch.unique(labels)), 0.5)


def train_epoch(model, optimizer, criterion, data, clip=1.0):
    # 1. Set the model to training mode
    model.train()
    
    # 2. Clear the previous gradients
    optimizer.zero_grad()

    # 3. Forward pass, compute outputs
    outputs = model(data['features'], data['adjacency_matrix'])

    # 4. Compute the training loss
    train_loss = criterion(outputs[data['train_mask']], data['labels'][data['train_mask']])

    # 5. Backward pass and update model weights
    train_loss.backward()
    
    # Gradient clipping
    torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
    
    optimizer.step()

    # 6. Compute the validation loss
    validation_loss = evaluate_validation_loss(model, criterion, data)

    # 7. Return the training and validation losses
    return train_loss, validation_loss


def evaluate_validation_loss(model, criterion, data):
    model.eval()  # Set the model to evaluation mode
    
    with torch.no_grad():  # Disable gradient computation
        outputs = model(data['features'], data['adjacency_matrix'])
        validation_loss = criterion(outputs[data['validation_mask']], data['labels'][data['validation_mask']])

    return validation_loss


def test_accuracy(model, mask, data):
    model.eval()  # Set the model to evaluation mode

    with torch.no_grad():  # Disable gradient computation
        # Forward pass, compute outputs
        outputs = model(data['features'], data['adjacency_matrix'])

        # Get model predictions
        predictions = torch.argmax(outputs, dim=1)

        # Select correct predictions based on the mask
        correct_predictions = (predictions[mask] == data['labels'][mask])

        # Compute accuracy
        accuracy = correct_predictions.sum().item() / mask.sum().item()

    return accuracy


