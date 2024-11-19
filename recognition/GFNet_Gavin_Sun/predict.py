"""
Evaluates a trained model on a test dataset, displays predictions, 
and visualizes results.
"""

import torch
from torch.utils.data import DataLoader
from dataset import get_adni_dataloader, ADNIDataset, ADNI_ROOT_PATH, TEST_TRANSFORM  # Import data loader and dataset classes
from modules import GFNet  # Import your model class
import random
import matplotlib.pyplot as plt
import math


def evaluate_model(model, device, test_loader):
    """
    Evaluate the model's accuracy on the test dataset.
    
    Args:
        model: Trained model to be evaluated.
        device: Torch device (CPU or GPU).
        test_loader (DataLoader): DataLoader for the test set.
    """
    model.eval()  
    correct = 0
    total = 0

    with torch.no_grad(): 
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'Accuracy on the test set: {accuracy:.2f}%')


def do_predictions(model, device: torch.device, num_predictions: int = 9, show_plot: bool = True):
    """
    Randomly selects a number of samples from the test dataset,
    makes predictions using the model, and displays.

    Args:
        model: Trained model for making predictions.
        device (torch.device): Torch device (CPU or GPU) for computation.
        num_predictions (int): Number of predictions to visualize (default: 9).
        show_plot (bool): Flag to display the plot (default: True).
    """

    test_dataset = ADNIDataset(ADNI_ROOT_PATH, train=False, transform=TEST_TRANSFORM)
    nrc = math.ceil(math.sqrt(num_predictions))
    fig, axes = plt.subplots(nrows=nrc, ncols=nrc, squeeze=False, figsize=(math.ceil(224 * nrc / 100), math.ceil(224 * nrc / 100)))

    for i in range(num_predictions):
        # Randomly select samples
        idx = random.randint(0, len(test_dataset) - 1)
        image, label = test_dataset[idx]
        image_tensor = image.unsqueeze(0).to(device)  
        output = model(image_tensor)
        _, predicted = torch.max(output, 1)
        pred = int(predicted[0].item())
        label_strs = {0: "NC", 1: "AD"} # Label mappings

        image = image.permute(1, 2, 0).cpu().numpy() # Prepare image for display
        ax = axes[i // nrc, i % nrc]
        ax.imshow(image)
        ax.set_title(f"Pred: {label_strs[pred]}, True: {label_strs[label]}")
        ax.axis('off')

    for i in range(num_predictions, nrc * nrc):
        ax = axes[i // nrc, i % nrc]
        ax.axis('off')

    plt.subplots_adjust(wspace=0.2, hspace=0.2)
    plt.savefig("predictions.png")
    if show_plot:
        plt.show()


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    model = GFNet().to(device)
    model.load_state_dict(torch.load('gfnet_model.pth')) # Load trained model weight
    print("Model loaded successfully.")

    batch_size = 8  
    test_loader = get_adni_dataloader(batch_size=batch_size, train=False)  # Get test data

    evaluate_model(model, device, test_loader)

    do_predictions(model, device, num_predictions=9, show_plot=True) # Make predictions and visualize

if __name__ == "__main__":
    main()
