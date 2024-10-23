"""
Tests the model and prints out predictions for the test set

@author Damian Bellew
"""

import torchvision.utils
from utils import *
from modules import *
from dataset import *

import torch
import torch.utils
import torchvision
import matplotlib.pyplot as plt

def test_model(device, model, test_loader, criterion):
    """
    Evaluates the model on the test dataset and computes the average Dice score.
    
    device (torch.device): The device (CPU or GPU) to run the model on.
    model (nn.Module): The 3D U-Net model to be evaluated.
    test_loader (DataLoader): DataLoader for the test dataset.
    criterion (nn.Module): Loss function (Dice loss) used for evaluation.
    """
    model.eval()
    dice_score = 0.0
    cumulative_loss_per_class = torch.zeros(NUM_CLASSES).to(device)

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            # Prediction
            outputs = model(images)
            labels = labels.long().view(-1)
            labels = F.one_hot(labels, num_classes=NUM_CLASSES)

            # Compute loss
            loss, loss_per_class = criterion(outputs, labels)
            dice_score += loss.item()

            loss_per_class = torch.tensor(loss_per_class).to(device)
            cumulative_loss_per_class += loss_per_class

    # Compute the average test loss and Dice score
    avg_dice_score = dice_score / len(test_loader)
    avg_loss_per_class = cumulative_loss_per_class / len(test_loader)

    print(f'Average Dice Score: {avg_dice_score}')
    print(f'Average Class Losses: {avg_loss_per_class.tolist()}')


def save_dice_loss_graph(dice_losses):
    """
    Saves a graph of the Dice loss over the epochs during training.

    dice_losses (list): A list of Dice loss values recorded during training.
    """
    plt.plot(dice_losses)
    plt.xlabel('Epoch')
    plt.ylabel('Dice Loss')
    plt.title('Dice Loss vs Epoch')
    plt.savefig(DICE_LOSS_GRAPH_PATH)

def save_prediction_images(device, model, test_loader, num_images=5):
    """
    Saves input images, actual labels, and model predictions from the test dataset.
    
    device (torch.device): The device (CPU or GPU) to run the model on.
    model (nn.Module): The 3D U-Net model used for prediction.
    test_loader (DataLoader): DataLoader for the test dataset.
    """
    model.eval()
    total_saved_images = 0
    
    with torch.no_grad():
        for idx, (images, labels) in enumerate(test_loader):
            if total_saved_images >= num_images:
                break
            images = images.to(device)
            labels = labels.to(device)

            # Prediction
            output = torch.softmax(model(images), dim=1)
            predicted = torch.argmax(output, dim=1)

            # Select the middle slice
            slice_idx = images.shape[2] // 2  

            image_slice = images[0, 0, slice_idx, :, :] 
            label_slice = labels[0, 0, slice_idx, :, :]  
            predicted_slice = predicted[0, slice_idx, :, :] 

            # Save the images
            torchvision.utils.save_image(image_slice.unsqueeze(0), f'{ORIGINAL_IMAGES_PATH}/image_{idx}_input.png')
            torchvision.utils.save_image(label_slice.unsqueeze(0).float(), f'{ORIGINAL_LABELS_PATH}/image_{idx}_label.png')
            torchvision.utils.save_image(predicted_slice.unsqueeze(0).float(), f'{PREDICTION_PATH}/image_{idx}_output.png')


if __name__ == "__main__":

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if not torch.cuda.is_available():
        print("WARNING: CUDA is not available. Running on CPU")

    # Data loaders
    _, test_loader = get_data_loaders()

    # Load saved model
    model = Unet3D(IN_DIM, NUM_CLASSES, NUM_FILTERS).to(device)
    model.load_state_dict(torch.load(MODEL_PATH))

    # Test the model
    test_model(device, model, test_loader, DiceLoss())
    