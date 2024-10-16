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
from train import *
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
def test(model, test_loader, criterion, device):
    model.eval()  # Set the model to evaluation mode
    
    test_scores = [] # stores dice scores.
    seg_0_scores = []
    seg_1_scores = []
    seg_2_scores = []
    seg_3_scores = []
    seg_4_scores = []
    seg_5_scores = []

    with torch.no_grad():
        
        criterion = criterion

        for inputs, masks in test_loader:
            inputs, masks = inputs.to(device), masks.to(device)
            outputs = model(inputs)
            test_loss, dice_coefs, _ = criterion(outputs, masks)

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
    Plots dice coefficients of the whole test dataset.
    Takes an array of dice scores as input. 
"""
def plot_dice(dice, criterion, segment_scores):
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
    plt.savefig(f'dice_scores_test_{str(criterion)}.png')
    plt.close()


"""
    Driver method 

"""
if __name__ == "__main__":
    # connect to gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    criterion = loss_map.get(LOSS_IDX)

    test_set = Prostate3dDataset(X_test, y_test)
    test_loader = DataLoader(dataset = test_set, batch_size = BATCH_SIZE)

    # perform predictions
    dice_scores, s0, s1, s2, s3, s4, s5 = test(model = trained_model, test_loader = test_loader, criterion = criterion,
                                               device = device)

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
    plot_dice(dice_scores, criterion, segment_scores)


