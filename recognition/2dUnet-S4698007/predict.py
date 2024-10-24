import torch
import nibabel as nib
import numpy as np
import os
import matplotlib.pyplot as plt
import argparse
from modules import UNet  # Import the U-Net model from the modules.py file

def load_model(model_path):
    """Loads the trained U-Net model from the specified path."""
    model = UNet(n_classes=6)  # Instantiate the U-Net model with the number of classes
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))  # Load model weights
    model.eval()  # Set the model to evaluation mode
    return model

def predict_image(model, image):
    """Runs the model on a single image and returns the predicted mask."""
    with torch.no_grad():  # Disable gradient calculation for inference
        # Convert image to a tensor and add batch and channel dimensions
        image_tensor = torch.tensor(image, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        prediction = model(image_tensor)  # Get model prediction
        predicted_mask = torch.argmax(prediction, dim=1).squeeze(0).cpu().numpy()  # Get the predicted class labels
    return predicted_mask

def load_mri_image(image_path):
    """Loads and preprocesses the MRI image from a NIfTI file."""
    img = nib.load(image_path)  # Load the NIfTI image
    img_data = img.get_fdata()  # Get the data from the NIfTI file
    img_slice = img_data[:, :, img_data.shape[2] // 2]  # Select the central slice
    return img_slice  # Return the slice directly without preprocessing

def save_prediction(predicted_mask, output_path):
    """Saves the predicted mask as a PNG image."""
    plt.imshow(predicted_mask, cmap='gray')  # Display the predicted mask
    plt.axis('off')  # Turn off axis labels
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)  # Save the figure
    plt.close()  # Close the plot to free up memory

def dice_score(predicted, target, num_classes):
    """Calculates the Dice score for each class."""
    dice_scores = []  # Initialize a list to store Dice scores for each class
    for i in range(num_classes):  # Loop over each class
        # Calculate intersection and union for the Dice score
        intersection = np.sum((predicted == i) & (target == i))
        union = np.sum(predicted == i) + np.sum(target == i)
        if union == 0:
            dice = 1.0  # Perfect match when both predicted and target have no pixels for the class
        else:
            dice = 2.0 * intersection / union  # Compute Dice score
        dice_scores.append(dice)  # Append the Dice score to the list
    return dice_scores

def process_and_evaluate(model, image_path, label_path, output_path, num_classes):
    """Loads the image and label, predicts the mask, saves the result, and computes the Dice score."""
    image = load_mri_image(image_path)  # Load and preprocess the MRI image
    label = load_mri_image(label_path)  # Load and preprocess the ground truth label
    
    predicted_mask = predict_image(model, image)  # Get the predicted segmentation mask
    
    # Save the predicted segmentation mask
    save_prediction(predicted_mask, output_path)

    # Calculate Dice score for each class
    dice_scores = dice_score(predicted_mask, label, num_classes)
    
    return dice_scores  # Return the Dice scores

def main(args):
    """Main function to run inference and evaluate the model on a test image."""
    model = load_model(args.model)  # Load the trained model
    
    num_classes = 6  # Set the number of classes for segmentation
    
    # Paths for the test image and label
    img_path = args.test_image  # Single test image path
    label_path = args.test_label  # Single label image path
    output_path = args.output  # Path to save the predicted mask
    
    # Process the image, make predictions, and evaluate the results
    dice_scores = process_and_evaluate(model, img_path, label_path, output_path, num_classes)
    
    # Print Dice scores for the current image
    print(f"Dice scores for the test image: {dice_scores}")

if __name__ == '__main__':
    # Set up command-line argument parsing
    parser = argparse.ArgumentParser(description="Run inference on a single MRI image using a trained U-Net model.")
    parser.add_argument('--test_image', type=str, required=True, help="Path to the test MRI NIfTI file.")
    parser.add_argument('--test_label', type=str, required=True, help="Path to the ground truth label for the test image.")
    parser.add_argument('--output', type=str, required=True, help="Path to save the output segmentation mask.")
    parser.add_argument('--model', type=str, required=True, help="Path to the trained U-Net model.")
    
    args = parser.parse_args()  # Parse the command-line arguments
    main(args)  # Run the main function

