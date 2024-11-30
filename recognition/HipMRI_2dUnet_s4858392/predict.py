import os
import torch
import nibabel as nib
import numpy as np
from torch.utils.data import DataLoader
from modules import uNet
from newdataset import NIFTIDataset
import torch.nn.functional as F


# Hyperparameters
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "models/unet_model.pth"  
OUTPUT_DIR = "/home/Student/s4858392/PAR/results" 
GROUND_TRUTH_DIR = "/home/groups/comp3710/HipMRI_Study_open/keras_slices_data/keras_slices_seg_test"

def load_model(model_path):
    model = uNet(in_channels=1, out_channels=1).to(DEVICE)
    model.load_state_dict(torch.load(model_path))
    model.eval()  # Set the model to evaluation mode
    return model


def predict(model, dataset_loader):
    model.eval()
    predictions = []
    with torch.no_grad():
        for batch in dataset_loader:
            images, _ = batch
            if isinstance(images, list):
                images = torch.stack(images)

            images = images.to(DEVICE)
            if images.dim() == 3:  # Check if it's a 3D tensor
                images = images.unsqueeze(1)  # Add a channel dimension if necessary
            
            outputs = model(images)
            outputs = torch.sigmoid(outputs)  # Apply sigmoid to get probabilities
            
            # Assuming binary segmentation; thresholding to get binary masks
            binary_mask = (outputs > 0.5).float()  # Threshold at 0.5
            predictions.append(binary_mask.cpu().numpy())
    
    return predictions

def save_predictions(predictions, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for i, pred in enumerate(predictions):
        img = nib.Nifti1Image(pred[0], np.eye(4))  
        nib.save(img, os.path.join(output_dir, f"predicted_mask_{i}.nii.gz"))

def load_ground_truth(ground_truth_dir):
    ground_truth_masks = []
    for file_name in os.listdir(ground_truth_dir):
        if file_name.endswith(".nii.gz"):
            img = nib.load(os.path.join(ground_truth_dir, file_name)).get_fdata()
            ground_truth_masks.append(img)
    return ground_truth_masks

def dice_score(pred, target):
    smooth = 1e-6 
    pred = pred.flatten()
    target = target.flatten()
    intersection = np.sum(pred * target)
    return (2. * intersection + smooth) / (np.sum(pred) + np.sum(target) + smooth)


if __name__ == '__main__':
    # Load the trained model
    model = load_model(MODEL_PATH)

    # Prepare your new dataset (adjust the path to your new images)
    test_image_dir = "//home/groups/comp3710/HipMRI_Study_open/keras_slices_data/keras_slices_test"
    test_dataset = NIFTIDataset(imageDir=test_image_dir)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    ground_truth_masks = load_ground_truth(GROUND_TRUTH_DIR)

    # Make predictions
    predictions = predict(model, test_loader)

    # Save predictions to a directory
    save_predictions(predictions, output_dir="/home/Student/s4858392/PAR/results")
    print("Predictions saved to 'results' directory.")

    dice_scores = []
    for pred, gt in zip(predictions, ground_truth_masks):
        score = dice_score(pred[0], gt) 
        dice_scores.append(score)
        print(f"Dice Score: {score:.4f}")

    avg_dice_score = np.mean(dice_scores)
    print(f"Average Dice Score: {avg_dice_score:.4f}")