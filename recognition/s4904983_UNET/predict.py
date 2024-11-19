""" 
File: predict.py
Author: Øystein Kvandal
Description: Contains the prediction function for the 2D UNET. Test the model on the test data.
"""

import os
import torch
from modules import UNet2D, DiceLoss
import dataset as ds
from dataset import MRIDataLoader, MRIDataset
import numpy as np
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
    """
    with torch.no_grad():
        image = image.to(device)
        pred = model(image)
        return pred
    

def plot_prediction(input_img, prediction, target, filename):
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
    if ds.IS_RANGPUR_ENV:
        save_dir = '/home/Student/s4904983/COMP3710/project/figures/' + filename + '.png' # Rangpur path
    else:
        save_dir = 'C:/Users/oykva/OneDrive - NTNU/Semester 7/PatRec/Project/predictions/' + filename + '.png' # Local path

    if not os.path.exists(os.path.dirname(save_dir)):
        os.makedirs(os.path.dirname(save_dir))

    plt.savefig(save_dir)


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
    if ds.IS_RANGPUR_ENV:
        model_dir = '/home/Student/s4904983/COMP3710/project/models/unet_model_ep20.pth' # Rangpur path
    else:    
        model_dir = 'C:/Users/oykva/OneDrive - NTNU/Semester 7/PatRec/Project/results_a/unet_model_ep10_a.pth' # Local path
    
    print("Loading model")
    model = load_model(model_dir, device)
    model.eval()

    # Load test data
    print("Loading data")
    TestDataLoader = MRIDataLoader("test", batch_size=1, shuffle=True)
    print("Model and test data loaded")

    # Predict and calculate dice coefficient
    dice_loss = DiceLoss()
    dice_coefficients = []
    dice_per_class = []
    min_dice_per_class = []
    predictions = []

    # Plot a random sample of the predictions
    rand_idx = np.random.choice(len(TestDataLoader.dataset), 20)

    for i, (image, label) in enumerate(tqdm(TestDataLoader)):
        # image = image[None, None, :, :]
        pred = predict_image(model, image, device)
        predictions.append(pred)
        dice = dice_loss(pred, label.long(), return_dice=True)
        classwise_dice = dice_loss(pred, label.long(), return_dice=True, separate_classes=True)
        dice_coefficients.append(dice)
        dice_per_class.append(classwise_dice)

        if i in rand_idx:
            plot_prediction(image[0,0,:,:], pred[0], label[0,0,:,:], f'prediction_{i}')
    # Print average dice coefficient
    avg_dice = sum(dice_coefficients) / len(dice_coefficients)
    print("Average Dice Coefficient: {:.4f}".format(avg_dice))

    min_dice = min(dice_coefficients)
    print("Minimum Dice Coefficient: {:.4f}".format(min_dice))

    avg_class_dice = sum([d[0] for d in dice_per_class]) / len(dice_per_class)
    for i in range(6):
        print(f"Average Dice Coefficient for class {i}: {avg_class_dice[i]}")

    return


if __name__ == '__main__':
    test_model()