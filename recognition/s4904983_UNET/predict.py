""" 
File: predict.py
Author: Øystein Kvandal
Description: Contains the prediction function for the 2D UNET. Test the model on the test data.
"""

import torch
from modules import UNet2D, DiceLoss
import dataset as ds
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
    

def plot_prediction(input_img, prediction, target):
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
    plt.show()
    ### Save the plot
    if ds.IS_RANGPUR_ENV:
        plt.savefig('/home/Student/s4904983/COMP3710/project/figures/prediction.png') # Rangpur path
    else:
        plt.savefig('C:/Users/oykva/OneDrive - NTNU/Semester 7/PatRec/Project/predictions/prediction.png') # Local path


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
        model_dir = 'C:/Users/oykva/OneDrive - NTNU/Semester 7/PatRec/Project/models/unet_model_ep20.pth' # Local path
    
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
    predictions = []

    for i, (image, label) in enumerate(tqdm(TestDataLoader)):
        # image = image[None, None, :, :]
        pred = predict_image(model, image, device)
        predictions.append(pred)
        dice = dice_loss(pred, label.long(), return_dice=True)
        dice_coefficients.append(dice)


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
        plot_prediction(input_image.squeeze(0), prediction[0], target.squeeze(0))

    return


if __name__ == '__main__':
    test_model()