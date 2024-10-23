"""
showing example usage of your trained model. Print out any results and / or provide visualisations
where applicable
"""
# libraries 
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from time import time
from monai.losses import DiceLoss
from monai.data import DataLoader, Dataset
from monai.transforms import (AsDiscrete, Compose, CastToType)

# import from local files  
from train import trained_model, CRITERION, compute_dice_segments
from dataset import test_dict, test_transforms

def visualise_ground_truths(images, ground_truths, criterion):

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
    
    return fig, axes

def visualise_predictions(images, predictions, criterion):

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
    
    return fig, axes

def test(model, test_loader, device):
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
            segment_coefs = compute_dice_segments(outputs, labels)
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

"""
    Plots dice coefficients of the whole test dataset.
    Takes an array of dice scores as input. 
"""

def plot_dice(criterion, segment_coefs):

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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_set = Dataset(test_dict, test_transforms)
    test_loader = DataLoader(dataset = test_set, batch_size = 1)

    print('> Start Testing')

    start = time()

    # perform predictions
    dice_coefs, s0, s1, s2, s3, s4, s5 = test(model = trained_model, test_loader = test_loader,
                                               device = device)
    
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


