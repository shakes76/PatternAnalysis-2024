import torch
from torch.utils.data import DataLoader
from dataset import get_adni_dataloader, ADNIDataset, ADNI_ROOT_PATH, TEST_TRANSFORM  # Import data loader and dataset classes
from modules import GFNet  # Import your model class
import random
import matplotlib.pyplot as plt
import math

def evaluate_model(model, device, test_loader):
    """
    Function to evaluate the model's accuracy on the test dataset.
    Args:
        model: Trained model to be evaluated.
        device: Torch device (CPU or GPU).
        test_loader (DataLoader): DataLoader for the test set.
    """
    model.eval()  # Set the model to evaluation mode
    correct = 0
    total = 0

    with torch.no_grad():  # Disable gradient computation
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
    Function to perform sample predictions on the test dataset and visualize them.
    Args:
        model: Trained model used for making predictions.
        device: Torch device (CPU or GPU).
        num_predictions (int): Number of predictions to display.
        show_plot (bool): Whether to display the plot with predictions.
    """
    test_dataset = ADNIDataset(ADNI_ROOT_PATH, train=False, transform=TEST_TRANSFORM)
    nrc = math.ceil(math.sqrt(num_predictions))
    fig, axes = plt.subplots(nrows=nrc, ncols=nrc, squeeze=False, figsize=(math.ceil(224 * nrc / 100), math.ceil(224 * nrc / 100)))

    for i in range(num_predictions):
        idx = random.randint(0, len(test_dataset) - 1)
        image, label = test_dataset[idx]
        image_tensor = image.unsqueeze(0).to(device)  # Add batch dimension
        output = model(image_tensor)
        _, predicted = torch.max(output, 1)
        pred = int(predicted[0].item())
        label_strs = {0: "NC", 1: "AD"}

        image = image.permute(1, 2, 0).cpu().numpy()  # Convert tensor to numpy array for visualization
        ax = axes[i // nrc, i % nrc]
        ax.imshow(image)
        ax.set_title(f"Pred: {label_strs[pred]}, True: {label_strs[label]}")
        ax.axis('off')

    # Hide any unused subplots
    for i in range(num_predictions, nrc * nrc):
        ax = axes[i // nrc, i % nrc]
        ax.axis('off')

    plt.subplots_adjust(wspace=0.2, hspace=0.2)
    plt.savefig("predictions.png")
    if show_plot:
        plt.show()

def main():
    """Main execution function."""
    # Set device to GPU if available, otherwise fallback to CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Load the model
    model = GFNet().to(device)
    model.load_state_dict(torch.load('gfnet_model.pth'))
    print("Model loaded successfully.")

    # Load the test dataset
    batch_size = 8  # Adjust batch size if needed
    test_loader = get_adni_dataloader(batch_size=batch_size, train=False)

    # Evaluate the model accuracy on the test set
    evaluate_model(model, device, test_loader)

    # Make and display a few sample predictions
    do_predictions(model, device, num_predictions=9, show_plot=True)

if __name__ == "__main__":
    main()
