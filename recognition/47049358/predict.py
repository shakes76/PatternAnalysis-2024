"""
The file contains a method to visualise and/or measure the performance of the trained model
on unseen data.
"""
# libraries 
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from time import time
from monai.losses import DiceLoss
from monai.data import DataLoader, Dataset
from monai.transforms import (AsDiscrete, Compose, CastToType)

# import from local files  
from train import trained_model, CRITERION, compute_dice_segments, DEVICE
from dataset import test_dict, test_transforms

__author__ = "Ryuto Hisamoto"

__license__ = "Apache"
__version__ = "1.0.0"
__maintainer__ = "Ryuto Hisamoto"
__email__ = "s4704935@student.uq.edu.au"
__status__ = "Committed"

def visualise_ground_truths(images: list, ground_truths: list, criterion):
    """ Visualises the ground truths and their images by overlaying them on the same 3 x 3 plot.

    Args:
        images (list): Images to overlay labels on.
        ground_truths (list): Labels to overlay on top of images.
        criterion (callable): Loss function used during the training to name the plot.

    Returns:
        None: The function only plots, so it does not return any value.
    """

    # Create a 3x3 grid of subplots
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))

    # Plot the images
    for i in range(3):
        for j in range(3):

            idx = i * 3 + j

            # Original image

            image = images[idx]

            axes[i, j].imshow(image, cmap='gray')
            axes[i, j].axis('off')
            axes[i, j].set_title(f'Image {idx+1}')

            # Ground truth mask

            ground_truth = ground_truths[idx]
            num_masks = ground_truth.shape[0]

            mask_gt = np.zeros((ground_truth.shape[1], ground_truth.shape[2]), dtype = np.uint8)

            for k in range(num_masks):
                mask_gt += (k + 1) * ground_truth[k, : , : ]
            axes[i, j].imshow(mask_gt, cmap='jet', alpha=0.3)

    # Show the plot
    plt.tight_layout()
    plt.savefig(f'ground_truths_{criterion}.png')
    plt.close()

def visualise_predictions(images: list, predictions: list, criterion):
    """Visualises the predictions and their images by overlaying them on the same 3 x 3 plot.

    Args:
        images (list): A list of images to lay predicted labels on
        predictions (list): A list of predicted labels proeuced by the model
        criterion (callable): Loss function used during the training to name the plot.

    Returns:
        None: The function only plots, so it does not return any value.
    """

    # Create a 3x3 grid of subplots
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))

    # Plot the images
    for i in range(3):
        for j in range(3):

            idx = i * 3 + j

            # Original image

            image = images[idx]

            axes[i, j].imshow(image, cmap='gray')
            axes[i, j].axis('off')
            axes[i, j].set_title(f'Image {idx+1}')

            mask_pred = predictions[idx]

            axes[i, j].imshow(mask_pred, cmap='jet', alpha=0.3)

    # Show the plot
    plt.tight_layout()
    plt.savefig(f'predictions_{criterion}.png')
    plt.close()

def test(model: nn.Module, test_loader: DataLoader, device: torch.device | str):
    """The function which tests the model on unseen data stored in a DataLoader

    Args:
        model (nn.Module): A trained model that is to be tested.
        test_loader (DataLoader): DataLoader instance which contains image data and their labels for the model
        to compare its performance against.
        device (torch.device | str): A device the training is based on. 

    Returns:
        tuple: A tuple containing:
            - np.array: An array of overall dice score for each test image and labels
            - np.array: An array of segment 0 dice score for each test image and labels
            - np.array: An array of segment 1 dice score for each test image and labels
            - np.array: An array of segment 2 dice score for each test image and labels
            - np.array: An array of segment 3 dice score for each test image and labels
            - np.array: An array of segment 4 dice score for each test image and labels
            - np.array: An array of segment 5 dice score for each test image and labels
    """

    model.to(device)
    model.eval()  # Set the model to evaluation mode

    criterion = DiceLoss(batch = True)
    
    test_dice_coefs = np.array([]) # stores dice scores.
    seg_0_dice_coef = np.array([])
    seg_1_dice_coef = np.array([])
    seg_2_dice_coef = np.array([])
    seg_3_dice_coef = np.array([])
    seg_4_dice_coef = np.array([])
    seg_5_dice_coef = np.array([])

    images = []
    ground_truths = []
    predictions = []

    output_transform = Compose(
    [
        AsDiscrete(to_onehot=6),
        CastToType(dtype=torch.uint8),
    ]
)

    with torch.no_grad():

        for i, batch_data in enumerate(test_loader):
            inputs, labels = (
                batch_data["image"].to(device),
                batch_data["label"].to(device),
            )
            outputs = model(inputs)
            outputs = output_transform(torch.argmax(outputs, dim=1))[np.newaxis, : , : , : , :]
            segment_coefs = compute_dice_segments(outputs, labels, device)
            dice_loss = criterion(outputs, labels).item()

            test_dice = 1 - dice_loss

            if len(images) < 9:
                image = inputs[0, 0 , : , : , 50].cpu().numpy()
                images.append(image)
                mask = labels[0, : , : , : , 50].cpu().numpy().astype(np.uint8)
                ground_truths.append(mask)
                prediction = torch.argmax(outputs[0, : , : , : , 50 ], dim = 0).cpu().numpy().astype(np.uint8)
                predictions.append(prediction)

            seg_0_dice_coef = np.append(seg_0_dice_coef, segment_coefs[0].item())
            seg_1_dice_coef = np.append(seg_1_dice_coef, segment_coefs[1].item())
            seg_2_dice_coef = np.append(seg_2_dice_coef, segment_coefs[2].item())
            seg_3_dice_coef = np.append(seg_3_dice_coef, segment_coefs[3].item())
            seg_4_dice_coef = np.append(seg_4_dice_coef, segment_coefs[4].item())
            seg_5_dice_coef = np.append(seg_5_dice_coef, segment_coefs[5].item())
        
            print(f'Test No.{i} - Overall Dice Coefficient: {test_dice}')
                
            test_dice_coefs = np.append(test_dice_coefs, test_dice)
    
    visualise_ground_truths(images, ground_truths, CRITERION)
    visualise_predictions(images, predictions, CRITERION)

    return test_dice_coefs, seg_0_dice_coef, seg_1_dice_coef, seg_2_dice_coef, seg_3_dice_coef, seg_4_dice_coef, seg_5_dice_coef

def plot_dice(criterion, segment_coefs: np.array):
    """ A method that plots a bar chart to visualise the performance of model on unseen data
    for each label. It is meant to demonstrated how accurately the model produces segmentations
    for each lebel.

    Args:
        criterion (callable): Loss function used during the training to name the plot.
        segment_coefs (np.array): an array containing dice scores for each segment at corresponding indices.
    """

    x_values = np.arange(len(segment_coefs))  # Generate x-values as indices

    # Plot overall dice scores
    plt.bar(x_values, segment_coefs)

    plt.xlabel("Segment No.")
    plt.ylabel("Dice Score")
    plt.title("Dice Score for Each Segment")
    plt.legend()
    plt.grid(True)
    plt.savefig(f'dice_coefs_test_{criterion}.png')
    plt.close()


if __name__ == "__main__":
    # connect to gpu

    test_set = Dataset(test_dict, test_transforms)
    test_loader = DataLoader(dataset = test_set, batch_size = 1)

    print('> Start Testing')

    start = time()

    # perform predictions
    dice_coefs, s0, s1, s2, s3, s4, s5 = test(model = trained_model, test_loader = test_loader,
                                               device = DEVICE)
    
    end = time()

    elapsed_time = end - start
    
    print(f"> Test completed in {elapsed_time:.2f} seconds")

    average_dice = np.mean(dice_coefs)
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

    segment_coefs = np.array([average_s0, average_s1, average_s2, average_s3,
                      average_s4, average_s5])

    # plot dice scores across the dataset.
    plot_dice(CRITERION, segment_coefs)