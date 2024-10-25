"""
This file contains the code used to perform inference. The model, which should
be saved as .pth file, is tested on the ADNI test set.

Author: Kevin Gu
"""
import torch
import time

from dataset import load_adni_data
from utils import get_device, get_dataset_root
from train import initialise_model

MODEL_PATH = "model_checkpoint.pth"

# Testing loop
def test_existing_model():
    """
    Test the model by first importing the relevant pth file.

    No parameters or returns.
    """
    device = get_device()
    root_dir = get_dataset_root()

    model = initialise_model(device)
    
    if device == 'cpu':
        checkpoint = torch.load(MODEL_PATH, map_location=torch.device('cpu'))
    else:
        checkpoint = torch.load(MODEL_PATH)
    
    # Load the values from checkpoint
    model.load_state_dict(checkpoint['model_state_dict'])

    testing_start = time.time()
    model.eval()  # Set the model to evaluation mode
    correct = 0
    total = 0

    test_loader = load_adni_data(root_dir=root_dir, testing=True)

    with torch.no_grad():  # No need to compute gradients during evaluation
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)  # Get predicted classes
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    testing_time = time.time() - testing_start
    
    print(f"Testing took {testing_time} seconds or {testing_time / 60} minutes")
    print("Accuracy", accuracy)


if __name__ == "__main__":
    test_existing_model()
