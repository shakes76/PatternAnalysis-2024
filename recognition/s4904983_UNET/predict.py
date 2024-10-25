""" 
File: predict.py
Author: Øystein Kvandal
Description: Contains the prediction function for the 2D UNET. Test the model on the test data.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from modules import UNet2D
from train import dice_coefficient
from dataset import MRIDataLoader, MRIDataset
import matplotlib.pyplot as plt
from tqdm import tqdm


def load_model(model_path, device):
    """ 
    Load the model from the specified path.

    Args:
        model_path (str): The path to the saved model.
        device (torch.device): The device to load the model on.

    Returns:
        nn.Module: The model.
    """
    model = UNet2D()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def predict_image(model, image, device):
    """ 
    Predict the segmentation mask for the input image.

    Args:
        model (nn.Module): The model.
        image (torch.Tensor): The input image.
        device (torch.device): The device to run the model on.

    Returns:
        torch.Tensor: The predicted segmentation mask.
        float: The dice coefficient.
    """
    with torch.no_grad():
        image = image.to(device)
        pred = model(image)
        return pred
    

def plot_prediction(input_img, prediction, target, dice_coefficient):
    """ 
    Plot the input image, prediction and target segmentation masks.

    Args:
        input (torch.Tensor): The input image.
        prediction (torch.Tensor): The predicted segmentation mask.
        target (torch.Tensor): The target segmentation mask.
    """
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 8, 1)
    plt.imshow(input_img, cmap='gray')
    plt.title('Input Image')
    plt.axis('off')
    for i in range(6):
        plt.subplot(1, 8, i+2)
        plt.imshow(prediction[i], cmap='gray')
        if i == 3:
            plt.title(f'Mask {i}')
        plt.axis('off')
    plt.axis('off')
    plt.subplot(1, 8, 8)
    plt.imshow(target, cmap='gray')
    plt.title('Target')
    plt.axis('off')
    ### Show the plot
    # plt.show()
    ### Save the plot
    # plt.savefig('C:/Users/oykva/OneDrive - NTNU/Semester 7/PatRec/Project/figures/prediction.png') # Local path
    plt.savefig('/home/Student/s4904983/COMP3710/project/figures/prediction.png') # Rangpur path


def test_model():
    """ 
    Test the model on the test data.

    Args:
        model (nn.Module): The model.
        images (torch.Tensor): The test images.
        segmentations (torch.Tensor): The test segmentations.
        device (torch.device): The device to run the model on.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Directory where model is saved
    model_path = 'C:/Users/oykva/OneDrive - NTNU/Semester 7/PatRec/Project/models/unet_model_ep10.pth' # Local path
    # model_dir = '/home/Student/s4904983/COMP3710/project/models/unet_model_ep10.pth' # Rangpur path
    print("Loading model")
    model = load_model(model_path, device)

    # Load test data
    print("Loading data")
    TestDataLoader = MRIDataLoader("test", batch_size=1)
    print("Model and test data loaded")

    # Predict and calculate dice coefficient
    dice_coefficients = []
    predictions = []

    for i, (image, label) in enumerate(tqdm(TestDataLoader)):
        # image = image[None, None, :, :]
        pred = predict_image(model, image, device)
        predictions.append(pred)
        dice_coefficients.append(dice_coefficient(pred, label))


    # Print average dice coefficient
    avg_dice = sum(dice_coefficients) / len(dice_coefficients)
    print("Average Dice Coefficient: {:.4f}".format(avg_dice))

    min_dice = min(dice_coefficients)
    print("Minimum Dice Coefficient: {:.4f}".format(min_dice))

    # Plot five random predictions
    for _ in range(5):
        idx = torch.randint(0, len(TestDataLoader.dataset), (1,)).item()
        input_image, target = TestDataLoader.dataset[idx]
        prediction = predictions[idx]
        dice = dice_coefficients[idx]
        plot_prediction(input_image.squeeze(0), prediction[0], target.squeeze(0), dice)

    return


if __name__ == '__main__':
    test_model()