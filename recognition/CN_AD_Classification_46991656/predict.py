# Provides example usage of the trained model.
# Requires the trained model to already exist in the current directory.


import os
import torch
import torch.nn as nn
from dataset import get_data_loaders
from modules import get_vit_model

# Move model to the appropriate device (GPU or CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def predict(model, test_loader):
    model.eval()  # Set the model to evaluation mode
    all_predictions = []
    all_labels = []

    with torch.no_grad():  # Disable gradient calculation
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)  # Move data to GPU if available
            outputs = model(images)  # Forward pass
            _, predicted = torch.max(outputs.data, 1)  # Get predictions

            all_predictions.extend(predicted.cpu().numpy())  # Store predictions
            all_labels.extend(labels.cpu().numpy())  # Store true labels

    return all_predictions, all_labels

def calculate_accuracy(predictions, labels):
    correct = sum(p == l for p, l in zip(predictions, labels))  # Count correct predictions
    accuracy = correct / len(labels) * 100  # Calculate percentage accuracy
    return accuracy

def main():
    # Get test data loader
    _, test_loader = get_data_loaders(batch_size=32)

    # Initialize model and load state_dict
    model = get_vit_model()
    model.load_state_dict(torch.load('gfnet_adni_model.pth', map_location=device))
    model.to(device)  # Move model to the appropriate device

    print("Starting predictions...")
    predictions, labels = predict(model, test_loader)

    # Calculate accuracy
    accuracy = calculate_accuracy(predictions, labels)
    print(f"Accuracy of the model: {accuracy:.2f}%")

if __name__ == '__main__':
    main()