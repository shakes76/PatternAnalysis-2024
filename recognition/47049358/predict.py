"""
showing example usage of your trained model. Print out any results and / or provide visualisations
where applicable
"""
# libraries 
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import torchvision.transforms as transforms


# import from local files  
from train import dice_coefficient, trained_model
from dataset import test_loader

"""
    Tests improved unet on trained model. 
    Calcualtes dice coeficient for each image and corresponding ground truth. 

    Parameters:
    - model (nn.Module): The trained model to be tested.
    - test_loader (DataLoader): DataLoader for the test dataset.
    - device (str): The device (e.g., 'cuda' or 'cpu') to run the evaluation on.

    Returns:
    - dice_scores (list): List of Dice coefficients for each image in the test dataset.
"""
def test(model, test_loader, device):
    model.eval()  # Set the model to evaluation mode
    
    dice_scores = [] # stores dice scores. 

    with torch.no_grad():
        for test_inputs, test_masks in test_loader:
            inputs, targets = test_inputs[np.newaxis, :], test_masks[np.newaxis, :]

            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            dice = dice_coefficient(outputs, targets)
            dice_scores.append(dice.item())

    return dice_scores

"""
    Visualises model image, predictions and ground truth on first three images from test loader.

    Parameters:
    - model (nn.Module): The trained model used for making predictions.
    - test_loader (DataLoader): DataLoader for the test dataset.
    - device (str): The device (e.g., 'cuda' or 'cpu') to run the visualization on.
    - num_images (int): The number of images to visualize (default is 3).

    """
def visualise_predictions(model, test_loader, device, num_images=3):
    model.eval()  # Set the model to evaluation mode

    image_count = 0  # Keep track of the number of images processed

    with torch.no_grad():
        for data in test_loader:
            inputs, targets = data
            inputs, targets = inputs.to(device), targets.to(device)
            # get prediction 
            outputs = model(inputs)

            # Convert PyTorch tensors to NumPy arrays
            input_image = inputs[0].cpu().numpy()  
            target_image = targets[0].cpu().numpy()
            predicted_image = outputs[0].cpu().numpy()

            # Create a side-by-side visualization for three images, prediction, ground truth. 
            plt.figure(figsize=(12, 4))
            plt.subplot(1, 3, 1)
            plt.title("Input Image")
            plt.imshow(input_image[0], cmap='gray')  

            plt.subplot(1, 3, 2)
            plt.title("Model Prediction")
            plt.imshow(predicted_image[0], cmap='gray')  

            plt.subplot(1, 3, 3)
            plt.title("Ground Truth")
            plt.imshow(target_image[0], cmap='gray')  

            plt.show()

            image_count += 1

            if image_count >= num_images:
                break

"""
    Plots dice coefficients of the whole test dataset.
    Takes an array of dice scores as input. 
"""
def plot_dice(dice):
    x_values = np.arange(len(dice))  # Generate x-values as indices
    plt.figure(figsize=(8, 6))
    plt.plot(x_values, dice, marker='o', linestyle='-')
    plt.xlabel("Image Index")
    plt.ylabel("Dice Coefficient")
    plt.title("Dice Coefficient across test inputs")
    plt.grid(True)
    plt.show()


"""
    Driver method 

"""
if __name__ == "__main__":
    # connect to gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # set up data transform. 
    data_transform = transforms.Compose([
        transforms.ToTensor(),
        #transforms.Normalize(mean=[0.7071, 0.5821, 0.5360], std=[0.1561, 0.1644, 0.1795])
    ])

    # perform predictions
    dice_scores = test(trained_model, test_loader, device)
    average_dice = np.mean(dice_scores)
    print(f"Average Dice Coefficient: {average_dice:.4f}")

    # plot dice scores across the dataset.
    plot_dice(dice_scores)

    # plot three examples of images, prediction and truth. 
    visualise_predictions(trained_model, test_loader,device)


