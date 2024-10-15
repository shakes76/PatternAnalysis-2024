"""
showing example usage of your trained model. Print out any results and / or provide visualisations
where applicable
"""
# libraries 
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader


# import from local files  
from train import DiceCoefficientLoss, trained_model
from dataset import X_test, y_test, Prostate3dDataset

BATCH_SIZE = 2

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
    
    test_scores = [] # stores dice scores.
    seg_0_scores = []
    seg_1_scores = []
    seg_2_scores = []
    seg_3_scores = []
    seg_4_scores = []
    seg_5_scores = []

    with torch.no_grad():
        
        criterion = DiceCoefficientLoss()

        for inputs, masks in test_loader:
            inputs, masks = inputs.to(device), masks.to(device)
            outputs = model(inputs)
            test_loss, dice_coefs = criterion(outputs, masks)

            for i in range(len(dice_coefs)):
                if i == 0:
                    seg_0_scores.append(dice_coefs[i])
                elif i == 1:
                    seg_1_scores.append(dice_coefs[i])
                elif i == 2:
                    seg_2_scores.append(dice_coefs[i])
                elif i == 3:
                    seg_3_scores.append(dice_coefs[i])
                elif i == 4:
                    seg_4_scores.append(dice_coefs[i])
                else:
                    seg_5_scores.append(dice_coefs[i])
                
            test_scores.append(test_loss.item())

    return test_scores, seg_0_scores, seg_1_scores, seg_2_scores, seg_3_scores, seg_4_scores, seg_5_scores

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
        for inputs, targets in test_loader:
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

            plt.savefig('input_image.png')

            plt.subplot(1, 3, 2)
            plt.title("Model Prediction")
            plt.imshow(predicted_image[0], cmap='gray') 

            plt.savefig('model_prediction.png') 

            plt.subplot(1, 3, 3)
            plt.title("Ground Truth")
            plt.imshow(target_image[0], cmap='gray')  

            plt.show()

            plt.savefig('ground_truth.png')

            image_count += 1

            if image_count >= num_images:
                break

"""
    Plots dice coefficients of the whole test dataset.
    Takes an array of dice scores as input. 
"""
def plot_dice(dice, segment_scores):
    x_values = np.arange(len(dice))  # Generate x-values as indices
    plt.figure(figsize=(8, 6))

    # Plot overall dice scores
    plt.plot(x_values, dice_scores, label='Overall Dice Scores')
    
    # Plot segment scores
    for i, segment_score in enumerate(segment_scores):
        plt.plot(x_values, segment_score, label=f'Segment {i} Dice Scores')


    plt.xlabel("Image Index")
    plt.ylabel("Dice Coefficient")
    plt.title("Dice Coefficient across test inputs")
    plt.legend()
    plt.grid(True)
    plt.savefig('dice_scores_test.png')
    plt.close()


"""
    Driver method 

"""
if __name__ == "__main__":
    # connect to gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_set = Prostate3dDataset(X_test, y_test)
    test_loader = DataLoader(dataset = test_set, batch_size = BATCH_SIZE)

    # perform predictions
    dice_scores, s0, s1, s2, s3, s4, s5 = test(trained_model, test_loader, device)

    average_dice = np.mean(dice_scores)
    print(f"Average Dice Coefficient: {average_dice:.4f}")

    average_s0 = np.mean(s0)
    print(f"Segment 0 Dice Coefficient: {average_s0:.4f}")

    average_s1 = np.mean(s1)
    print(f"Segment 1 Dice Coefficient: {average_s1:.4f}")

    average_s2 = np.mean(s2)
    print(f"Segment 2 Dice Coefficient: {average_s2:.4f}")

    average_s3 = np.mean(s3)
    print(f"Segment 3 Dice Coefficient: {average_s3:.4f}")

    average_s4 = np.mean(s4)
    print(f"Segment 4 Dice Coefficient: {average_s4:.4f}")

    average_s5 = np.mean(s5)
    print(f"Segment 5 Dice Coefficient: {average_s5:.4f}")

    segment_scores = [s0, s1, s2, s3, s4, s5]

    # plot dice scores across the dataset.
    plot_dice(dice_scores, segment_scores)

    # plot three examples of images, prediction and truth. 
    # visualise_predictions(trained_model, test_loader,device)


